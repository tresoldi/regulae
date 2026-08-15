#ifndef REGULAE_INTERNAL_H
#define REGULAE_INTERNAL_H

#include "regulae.h"

#include <stddef.h>

#define RG_DEFAULT_GAP_COST 0.5
#define RG_DEFAULT_CHUNK_PENALTY 0.25
#define RG_CHUNK_COMPLEXITY_PENALTY 1e-9
/* Two alignments can carry mathematically identical cost; which one the DP
 * keeps then comes down to the order floating-point error happens to fall in.
 * A later candidate must beat the incumbent by more than this to displace it,
 * so ties resolve toward the first candidate enumerated (smallest source span,
 * then smallest target span). Three orders of magnitude below the chunk
 * complexity penalty, so real tie-breaks still decide. */
#define RG_TIE_EPSILON 1e-12
/* The longest span a reordering is looked for over. Attested metathesis is
 * local: two or three segments transpose. */
#define RG_MAX_REORDER_SPAN 8
/* How much of the search is charged for. The split penalty gains
 * gamma * 2 * ln(candidates); gamma = 1 is the full extended-BIC term.
 *
 * 0.5 is not a taste: it is the largest value at which no sound law in
 * testdata/soundlaws/ is lost. At 0.75 the multi-lect stage stops committing
 * on the palatalization corpus; at 1.0 lenition finds only two of its three
 * stops, and Grimm -- which is unconditioned and should commit nothing --
 * looks identical to a corpus whose conditioning has been suppressed. Below
 * 0.25 the spurious commits the term exists to stop come back. */
#ifndef RG_SEARCH_PENALTY_GAMMA
#define RG_SEARCH_PENALTY_GAMMA 0.5
#endif
/* Stages the pairwise pipeline reports: initial prior, segment EM, displacement
 * aggregation, context discovery, chunk promotion, tonal aggregation,
 * cross-dimensional discovery, long-range discovery. */
#define RG_PAIRWISE_STAGE_COUNT 8
/* Multi-lect stages beyond the per-pair work: reconciliation, class discovery,
 * cross-dimensional lifting. */
#define RG_MULTILECT_STAGE_COUNT 3

/* Dirichlet prior mass alpha(t|s) over the corpus grapheme inventory, derived
 * from merkmal distances by a softmax. Held separately from the observed
 * counts so the posterior can be formed as (alpha + n) / (beta + N(s)). */
typedef struct rg_segment_prior_row {
    char *source;
    char *target;
    double alpha;
} rg_segment_prior_row;

/* log Z(s): the log partition function of the prior softmax for source s.
 * Subtracted from the segment cost so costs stay comparable across sources. */
typedef struct rg_log_normalizer_row {
    char *source;
    double value;
    /* How much mass answers to this grapheme as a *target*. The forward
     * denominator is the total for a source; a symmetric score needs the other
     * one too, and this is already the only table keyed by a single grapheme
     * and binary-searched, so it carries both. */
    double target_total;
} rg_log_normalizer_row;

struct rg_pairwise_model {
    rg_segment_prior_row *segment_priors;
    size_t segment_prior_count;
    rg_log_normalizer_row *log_normalizers;
    size_t log_normalizer_count;
    double concentration;
    double segment_weight;
    double displacement_weight;
    double tone_weight;
    rg_segment_count_row *segment_counts;
    size_t segment_count_count;
    rg_conditioned_segment_count_row *conditioned_segment_counts;
    size_t conditioned_segment_count_count;
    /* Decisions committed so far, so a rule can record where in the sequence
     * it was settled. Publication sorts the rows by key and would otherwise
     * lose it. */
    int decision_count;
    /* Whether any conditioned row names the target's environment. The DP builds
     * a target-side context for every cell it costs, and that is pure waste
     * while no rule can consult it -- which is the whole of EM, before any
     * conditioning has been discovered. */
    int has_target_conditioned;
    rg_chunk_row *chunks;
    size_t chunk_count;
    rg_cross_dimensional_row *cross_dimensional_rows;
    size_t cross_dimensional_count;
    rg_displacement_row *displacement_rows;
    size_t displacement_row_count;
    rg_tonal_count_row *tonal_counts;
    size_t tonal_count_count;
};

typedef struct rg_multi_pair_model_owned {
    char *lect_a;
    char *lect_b;
    rg_pairwise_model *model;
    rg_multi_pair_model_row view;
} rg_multi_pair_model_owned;

typedef struct rg_multi_class_owned {
    rg_multi_class_row view;
} rg_multi_class_owned;

typedef struct rg_multi_cross_dimensional_owned {
    rg_multi_cross_dimensional_row view;
} rg_multi_cross_dimensional_owned;

/* One reconciled position tuple and the classes it realises. Reconciliation
 * already knows which aligned positions formed each class; retaining it is what
 * lets a consumer point at the evidence for a class rather than guess at it by
 * matching graphemes, which cannot tell a conditioned class from the
 * unconditioned one over the same segments. */
typedef struct rg_class_position {
    size_t cognate_index;
    size_t *lect_indices;
    size_t *positions;
    size_t segment_count;
    int *class_ids;
    size_t class_id_count;
    size_t class_id_cap;
} rg_class_position;

struct rg_multi_model {
    rg_class_position *class_positions;
    size_t class_position_count;
    char **lect_ids;
    size_t lect_count;
    rg_multi_pair_model_owned *pair_models;
    size_t pair_model_count;
    rg_multi_class_owned *unconditioned_classes;
    size_t unconditioned_class_count;
    rg_multi_class_owned *conditioned_classes;
    size_t conditioned_class_count;
    rg_multi_cross_dimensional_owned *cross_dimensional_rows;
    size_t cross_dimensional_count;
    size_t unpaired_set_count;
    rg_corpus_fit fit;
};

/* Class ids realised at a given position pair, for a link of an alignment
 * between two lects of one cognate set. Returns the number written to out,
 * capped at capacity. */
size_t rg_model_classes_at_internal(
    const rg_multi_model *model,
    size_t cognate_index,
    size_t lect_a,
    size_t position_a,
    size_t lect_b,
    size_t position_b,
    int *out,
    size_t capacity
);

/* The one place the `const` on library-owned storage comes off, for freeing it
 * or for handing it to a clear helper; see memory.c for why it has to. */
void *rg_owned_internal(const void *owned);
void rg_free_owned_internal(const void *owned);
char *rg_strdup_internal(const char *value);
char *rg_strndup_internal(const char *text, size_t length);

/* Whether a link's target is the same multiset of graphemes as its source in a
 * different order -- a reordering rather than a set of substitutions.
 *
 * The alignment DP is monotone, so it can only express this by pairing the
 * spans positionally, which reads as "s corresponds to k and k to s". That is
 * two false correspondences describing one event, and it is what a reader
 * takes away unless the reordering is recognised as such. Fills `pairing` with
 * the target index each source index answers to. */
int rg_link_is_reordering_internal(
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    size_t *pairing
);
void rg_segment_clear_internal(rg_segment *segment);
rg_status rg_segment_copy_internal(const rg_segment *src, rg_segment *out);
void rg_context_spec_clear_internal(rg_context_spec *context);
rg_status rg_context_spec_copy_internal(const rg_context_spec *src, rg_context_spec *out);
rg_status rg_feature_constraint_copy_internal(
    const rg_feature_constraint *src,
    rg_feature_constraint *out
);
rg_status rg_feature_constraint_array_copy_internal(
    const rg_feature_constraint *src,
    size_t count,
    const rg_feature_constraint **out
);
void rg_distance_constraint_array_clear_internal(
    const rg_distance_constraint *items,
    size_t count
);
void rg_feature_constraint_array_clear_internal(
    const rg_feature_constraint *items,
    size_t count
);
void rg_link_clear_internal(rg_link *link);
rg_status rg_link_copy_chunks_internal(
    rg_link *link,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count
);
/* A candidate conditioning predicate. slot names the rg_context_spec field the
 * constraint applies to ("preceding", "following@2", "self_stress", ...);
 * feature is the feature name, or the position name for the "position" slot. */
typedef struct rg_split_candidate {
    const char *slot;
    const char *feature;
    const char *value;
} rg_split_candidate;

/* What a candidate must clear before its split can be committed.
 *
 * The bar belongs on the candidate rather than on the search because the two
 * kinds of predicate do not deserve the same one. There are a handful of
 * immediate neighbours and around a hundred long-range and existential axes,
 * and the more candidates a search considers the likelier one of them fits an
 * outcome by chance -- so the same evidence has to buy less from the larger
 * pool. Carrying the bar per candidate is what lets a single search weigh both
 * kinds at once, which a rule conditioned by position *and* by something at a
 * distance requires and two separate passes can never express. */
typedef struct rg_split_gate {
    double min_obs;
    double delta_threshold;
    double min_dominant_fraction;
} rg_split_gate;

int rg_predicate_holds_internal(
    const rg_context_spec *context,
    const rg_split_candidate *candidate
);
rg_status rg_context_extend_internal(
    const rg_context_spec *base_context,
    const rg_split_candidate *candidate,
    rg_context_spec *out
);
rg_status rg_context_from_candidate_internal(
    const rg_split_candidate *candidate,
    rg_context_spec *out
);

/* One context per segment position of the form, each a 1-segment window, so
 * observation contexts match the contexts the DP attaches to 1-to-1 links. */
rg_status rg_form_position_contexts_internal(
    const rg_context *ctx,
    const rg_form *form,
    rg_context_spec **out,
    size_t *out_count
);
void rg_context_spec_array_free_internal(rg_context_spec *contexts, size_t count);

/* Which conditioning predicates distinguish anything in the corpus at hand.
 *
 * There is no fixed list of feature names anywhere in regulae. A grapheme's
 * features are whatever the merkmal system in use reports for it, and the
 * searchable vocabulary is derived from the corpus's own inventory. That is
 * what lets one code path serve `distinctive`, which names the features a
 * segment *has*, and `phoible`, which names every feature with a value; and it
 * is why widening the vocabulary is a question about the data rather than a
 * question about whose phonology the author had in mind.
 *
 * Two filters. Contrastive: some segment carries the predicate and some does
 * not, since a predicate true of everything partitions nothing. Distinct: no
 * two predicates that separate this corpus's inventory identically both
 * survive, because keeping synonyms widens the argmax without widening what
 * can be found. */
typedef struct rg_feature_vocabulary {
    rg_feature_constraint *entries;
    size_t count;
} rg_feature_vocabulary;

rg_status rg_feature_vocabulary_build_internal(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    rg_feature_vocabulary *out
);
rg_status rg_feature_vocabulary_build_from_sets_internal(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    rg_feature_vocabulary *out
);
void rg_feature_vocabulary_clear_internal(rg_feature_vocabulary *vocabulary);


rg_status rg_context_displacement_internal(
    const rg_context *ctx,
    const char *source,
    const char *target,
    const rg_feature_displacement **out,
    size_t *out_count
);

rg_status rg_context_constraints_internal(
    const rg_context *ctx,
    const char *grapheme,
    const rg_feature_constraint **out,
    size_t *out_count
);

/* JSON transport for the CLI's --json output and the WebAssembly adapter. Not
 * the interchange schema: that is M8's job, and a single MAP correspondence
 * system is explicitly not claim-capable data. */
/* Threads one progress counter through a whole training run so a caller sees a
 * single monotonic fraction, rather than each lect pair restarting at zero. */
typedef struct rg_progress_state {
    rg_progress_fn fn;
    void *user_data;
    size_t completed;
    size_t total;
    int cancelled;
} rg_progress_state;

void rg_progress_init_internal(rg_progress_state *state, const rg_train_options *options, size_t total);
/* Reports one finished stage. Returns non-zero once the caller has asked to
 * stop, and keeps returning it so unwinding callers all see the same answer. */
int rg_progress_step_internal(rg_progress_state *state, const char *stage);

rg_status rg_train_pairwise_internal(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_progress_state *progress,
    rg_pairwise_model **out
);

char *rg_json_from_multi_model_internal(
    const rg_context *ctx,
    const rg_multi_model *model,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    int include_alignments,
    int include_outliers
);
char *rg_json_error_internal(rg_status status, const char *detail);
rg_status rg_json_read_train_options_internal(
    const char *text,
    rg_train_options *out,
    char *error_detail,
    size_t error_detail_size
);

void rg_context_note_unknown_grapheme_internal(const rg_context *ctx, const char *grapheme);
rg_status rg_context_refusal_status_internal(const rg_context *ctx, const char *grapheme);

rg_status rg_context_features_internal(
    const rg_context *ctx,
    const char *grapheme,
    const rg_feature_set **out
);

/* `out_inferred`, when not NULL, reports that the form had no nucleus of its
 * own -- no vowel and no syllabic consonant -- and was given one so that the
 * syllable predicates have something to hold of. Vowelless words are real
 * (Nuxalk, Berber) and their analyses differ; making the guess silently is
 * what was wrong with it. */
rg_status rg_compute_syllable_breaks_internal(
    const rg_context *ctx,
    const rg_form *form,
    size_t **out,
    size_t *out_count,
    int *out_inferred
);
int rg_segment_posterior_internal(
    const rg_pairwise_model *model,
    const char *source,
    const char *target,
    const rg_context_spec *link_context,
    const rg_context_spec *target_context,
    double *out
);
double rg_segment_log_normalizer_internal(const rg_pairwise_model *model, const char *source);

/* How readable a promoted chunk is as a single historical process, in [0, 1].
 * `rows` is the promoted set the chunk belongs to, which the score consults to
 * see whether this chunk merely wraps a smaller one. */
rg_status rg_chunk_transparency_internal(
    const rg_context *ctx,
    const rg_train_options *options,
    const rg_pairwise_model *model,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    const rg_chunk_row *rows,
    size_t row_count,
    double *out
);

/* Half of -log P(t|s) plus half of -log P(s|t), with no log-Z offset. Returns 0
 * when the pair is unknown to the model. */
int rg_segment_symmetric_raw_cost_internal(
    const rg_pairwise_model *model,
    const char *source,
    const char *target,
    double *out
);
size_t rg_segment_vocab_size_internal(const rg_pairwise_model *model);
rg_status rg_score_link_with_context_model_internal(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    const rg_train_options *options,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    const rg_context_spec *link_context,
    const rg_context_spec *target_context,
    double *out
);

/* The Wilson interval at RG_DEFAULT_ALPHA, which is what every published table
 * uses. Folds away the status return, since the alpha is a compile-time
 * constant that cannot be unsupported. */
rg_uncertainty_estimate rg_wilson_default_internal(double count, double total);

/* A percentile interval over resampled rates. Implemented since the port and
 * unreachable until 2026-08-15, when bootstrap_n stopped being an option that
 * did nothing. */
rg_status rg_percentile_interval(
    const double *samples,
    size_t sample_count,
    double estimate,
    double total,
    double alpha,
    rg_uncertainty_estimate *out
);

#endif
