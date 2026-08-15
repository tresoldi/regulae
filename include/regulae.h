#ifndef REGULAE_H
#define REGULAE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(REGULAE_SHARED)
#  if defined(REGULAE_BUILDING_LIBRARY)
#    define RG_API __declspec(dllexport)
#  else
#    define RG_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define RG_API __attribute__((visibility("default")))
#else
#  define RG_API
#endif

#define RG_VERSION_MAJOR 0
#define RG_VERSION_MINOR 1
#define RG_VERSION_PATCH 0
#define RG_VERSION_STRING "0.1.0"
#define RG_ABI_VERSION 22
#define RG_DEFAULT_MAX_CHUNK_SIZE 3
/* merkmal's own default. It reads the same graphemes and returns the same
 * feature labels as "descriptive", but scores through its own dimensions, and
 * on BDPA gold alignments it is the one merkmal cannot distinguish from SCA.
 * rg_context_use_system takes any system the registry holds. */
#define RG_DEFAULT_FEATURE_SYSTEM "distinctive"

typedef struct rg_context rg_context;
typedef struct rg_feature_set rg_feature_set;
typedef struct rg_alignment rg_alignment;
typedef struct rg_pairwise_model rg_pairwise_model;
typedef struct rg_multi_model rg_multi_model;
typedef struct rg_corpus rg_corpus;

typedef enum rg_status {
    RG_OK = 0,
    RG_ERR_INVALID_ARGUMENT,
    RG_ERR_IO,
    RG_ERR_PARSE,
    RG_ERR_MERKMAL,
    RG_ERR_UNKNOWN_GRAPHEME,
    RG_ERR_UNSUPPORTED_OPTION,
    RG_ERR_OOM,
    RG_ERR_CANCELLED,
    /* The token is CLDF/CLTS markup rather than a transcription of a sound:
     * `<?>`, `<<...>>`, and the boundary marks `+`, `_` and `#`. This is a
     * documented gap in the source data, not a sound the feature system fails
     * to cover, and the two call for different responses from the user. */
    RG_ERR_SOURCE_MARKER
} rg_status;

/* Reports training progress. stage names the pipeline step just finished;
 * completed and total count steps, so completed/total is a usable fraction.
 * Return true to abort the run, which surfaces as RG_ERR_CANCELLED.
 * Called from the training thread, between stages, never mid-stage. */
typedef bool (*rg_progress_fn)(
    const char *stage,
    size_t completed,
    size_t total,
    void *user_data
);

typedef struct rg_bic_config {
    double delta_bic_threshold;
    int min_split_observations;
    int max_split_depth;
    int min_chunk_observations;
    double long_range_delta_bic_threshold;
    int long_range_min_split_observations;
    double long_range_min_dominant_fraction;
    int cross_dim_max_iterations;
    int cross_dim_min_rule_count;
    /* A floor on P(target_value | environment). Defaults to 0.0, which is not
     * an oversight: a fixed fraction is not a measure of conditioning. With
     * two possible values 0.5 is chance; with ten it is overwhelming evidence,
     * and a threshold that rejects a value occurring at 0.4 against a base
     * rate of 0.05 throws away exactly the conditioned splits this stage
     * exists to find. Whether an environment conditions anything is decided by
     * cross_dim_delta_bic_threshold against the complementary environment.
     * Set this only to suppress weak rules in a report. */
    double cross_dim_min_rule_confidence;
    double cross_dim_delta_bic_threshold;
    bool multi_lect_bic_small_sample_correction;
    double multi_lect_min_commit_scale;
    /* How much of the search a split is charged for, on top of its parameter.
     * The penalty gains `search_penalty_gamma * 2 * ln(candidates)`: BIC prices
     * one added term, but the term that survives is the best of many, and the
     * maximum of a hundred candidates clears its bar by chance far more often
     * than one does.
     *
     * 0.5 is the largest fixed value at which no sound law in
     * testdata/soundlaws/ is lost. Setting `permutation_count` and
     * `tune_search_penalty` replaces it with a value measured from the corpus
     * itself. */
    double search_penalty_gamma;
} rg_bic_config;

/* The feature system is not here. It belongs to the context -- see
 * rg_context_use_system -- because it decides what a grapheme means before any
 * training option is consulted. This struct carried a `feature_system` field
 * until ABI 21 that nothing read: setting it selected nothing and reported
 * nothing. */
typedef struct rg_train_options {
    int max_chunk_size;
    double temperature;
    double concentration;
    int max_iter;
    double convergence_eps;
    double segment_weight;
    double displacement_weight;
    double tone_weight;
    double chunk_min_transparency;
    rg_bic_config bic;
    int bootstrap_n;
    int bootstrap_seed;
    /* Shuffled-baseline runs for the fit summary. Each one is a full training
     * run on a corpus whose pairings have been permuted, so this multiplies
     * training time; 0 (the default) skips it. */
    int permutation_count;
    int permutation_seed;
    /* Set the search charge from the corpus's own shuffled baseline rather
     * than from bic.search_penalty_gamma: the shuffles are trained with no
     * charge at all, and the level their spurious rules reach becomes the bar
     * the real run has to clear. Requires permutation_count > 0.
     *
     * This buys precision with recall, and the trade is real: a corpus whose
     * genuine conditioning is weak relative to its own noise will lose rules
     * that the fixed 0.5 keeps. It is off by default for that reason. */
    bool tune_search_penalty;
    rg_progress_fn progress;
    void *progress_user_data;
} rg_train_options;

typedef struct rg_segment {
    const char *grapheme;
    const char *tone;
    const char *length;
    const char *stress;
} rg_segment;

typedef struct rg_form {
    const char *lect_id;
    const rg_segment *segments;
    size_t segment_count;
    const int *syllable_breaks;
    size_t syllable_break_count;
    const int *morpheme_breaks;
    size_t morpheme_break_count;
} rg_form;

typedef struct rg_feature_constraint {
    const char *feature;
    const char *value;
} rg_feature_constraint;

typedef struct rg_distance_constraint {
    int offset;
    rg_feature_constraint constraint;
} rg_distance_constraint;

typedef struct rg_context_spec {
    const char *position;
    const rg_feature_constraint *preceding;
    size_t preceding_count;
    const rg_feature_constraint *following;
    size_t following_count;
    /* Where the segment sits in its own morpheme -- "initial", "final",
     * "internal", or "only" -- derived from the morpheme boundaries the caller
     * supplied on the form. Absent when the form carries none. */
    const char *morphological;
    /* Which morpheme, counted from the start of the word: "0", "1", ... A rule
     * stated in it does not travel between a suffixing language and a
     * prefixing one, where the same index is a different thing. */
    const char *morpheme_index;
    const rg_distance_constraint *preceding_at_distance;
    size_t preceding_at_distance_count;
    const rg_distance_constraint *following_at_distance;
    size_t following_at_distance_count;
    const rg_feature_constraint *somewhere_preceding;
    size_t somewhere_preceding_count;
    const rg_feature_constraint *somewhere_following;
    size_t somewhere_following_count;
    const rg_feature_constraint *same_syllable;
    size_t same_syllable_count;
    const rg_feature_constraint *next_syllable;
    size_t next_syllable_count;
    const rg_feature_constraint *previous_syllable;
    size_t previous_syllable_count;
    /* Constraints on the segment the environment is *about*, rather than on
     * its neighbours. A conditioned correspondence rarely needs this -- the
     * segment is already the key -- but a cross-dimensional rule does: the
     * Middle Chinese register split conditions a target tone on the preceding
     * onset's voicing *and* on the source segment's own tone, and the second
     * of those is a statement about the segment itself. Suprasegmentals appear
     * here as ordinary features named "tone", "length" and "stress". */
    const rg_feature_constraint *self;
    size_t self_count;
    const rg_feature_constraint *self_stress;
    size_t self_stress_count;
    const rg_feature_constraint *preceding_stress;
    size_t preceding_stress_count;
    const rg_feature_constraint *following_stress;
    size_t following_stress_count;
} rg_context_spec;

typedef struct rg_feature_displacement {
    const char *feature;
    const char *from_value;
    const char *to_value;
} rg_feature_displacement;

/* How an interval was produced. A consumer cannot otherwise tell a closed-form
 * interval from a resampled one, and the two answer different questions: the
 * Wilson interval asks how much the count alone constrains the rate, the
 * bootstrap asks how much the rate moves when the corpus is resampled. */
typedef enum rg_uncertainty_method {
    /* No observations backed the estimate, so the interval is the whole
     * range. Not an error: an unobserved rate is unconstrained, not zero. */
    RG_UNCERTAINTY_NONE = 0,
    RG_UNCERTAINTY_WILSON = 1,
    RG_UNCERTAINTY_BOOTSTRAP = 2
} rg_uncertainty_method;

/* A two-sided interval on a rate in [0, 1] — the conditional probability the
 * count represents — plus the provenance needed to read it. n is the
 * denominator behind the point estimate, not the number of bootstrap samples. */
/* Whether a rule stands above what the search finds in this corpus with the
 * correspondences taken out of it.
 *
 * Every conditioned rule already reports a count, a contrast, a delta-BIC, a
 * search margin and an interval, and the corpus reports what its own shuffles
 * reach. Putting those together was left to the reader, which is a synthesis
 * a reader should not have to do on twenty-five rules -- and is exactly the
 * judgement the tool is for. */
typedef enum rg_rule_standing {
    /* No shuffled baseline was run, so there is nothing to stand above.
     * Not a verdict: pass permutation_count to get one. */
    RG_RULE_STANDING_UNMEASURED = 0,
    /* The rule's evidence carries a heavier search charge than the level the
     * same search reaches on the shuffled corpus. */
    RG_RULE_STANDING_ABOVE_NOISE = 1,
    /* It does not. The rule may still be true; what it is not is
     * distinguishable from an artefact of having looked. */
    RG_RULE_STANDING_WITHIN_NOISE = 2
} rg_rule_standing;

RG_API const char *rg_rule_standing_string(rg_rule_standing standing);

typedef struct rg_uncertainty_estimate {
    double estimate;
    double lower;
    double upper;
    double n;
    double alpha;
    rg_uncertainty_method method;
    /* Set when the row's environment was chosen by the same data the interval
     * is computed from. The interval then says how well the rate is pinned
     * *given* that environment, and not whether the environment is real -- for
     * which see `search_margin` against `rg_corpus_fit.null_search_margin`. */
    bool post_selection;
} rg_uncertainty_estimate;

typedef struct rg_link {
    const rg_segment *source;
    size_t source_count;
    const rg_segment *target;
    size_t target_count;
    rg_context_spec context;
    const rg_feature_displacement *feature_displacement;
    size_t feature_displacement_count;
    double confidence;
} rg_link;

typedef struct rg_form_pair {
    rg_form source;
    rg_form target;
    double weight;
} rg_form_pair;

typedef struct rg_cognate_form {
    const char *lect_id;
    rg_form form;
} rg_cognate_form;

typedef struct rg_cognate_set {
    const char *cognate_id;
    const rg_cognate_form *forms;
    size_t form_count;
    double confidence;
} rg_cognate_set;

typedef struct rg_segment_count_row {
    const char *source;
    const char *target;
    double count;
    double source_total;
    /* How much mass answers to this target overall. Alignment is scored
     * symmetrically -- the geometric mean of the two conditional directions --
     * and that needs both denominators, so both are published. */
    double target_total;
    rg_uncertainty_estimate uncertainty;
} rg_segment_count_row;

/* One observed feature-displacement vector: the whole set of features gained
 * and lost between a source and target segment, counted as a unit. */
typedef struct rg_displacement_row {
    const rg_feature_displacement *items;
    size_t item_count;
    double count;
    double total;
    rg_uncertainty_estimate uncertainty;
} rg_displacement_row;

typedef struct rg_tonal_count_row {
    const char *source_tone;
    const char *target_tone;
    double count;
    double source_total;
    rg_uncertainty_estimate uncertainty;
} rg_tonal_count_row;

/* A correspondence conditioned on an environment, and which form the
 * environment is read from.
 *
 * A sound change is conditioned by the environment it happened in, which lives
 * in the ancestor -- and regulae refuses to decide which lect that is. So it
 * looks from both: a rule may name the source form's environment or the
 * target's, and `context_is_target` says which. The distinction is not
 * cosmetic. A change is only *visible* from the side that has the split: where
 * Greek reflects Grassmann's Law, Proto-Indo-European tʰ answers to Greek t in
 * some words and tʰ in others, while from the Greek side each segment has one
 * source and there is nothing to condition. Looking from one side only leaves
 * half of every pair's conditioning unreachable, and which half depends on
 * which lect happened to sort first. */
/* A conditioned correspondence, and the comparison that bought it.
 *
 * A conditioning claim is a comparison: this environment against its
 * complement. Publishing the environment's own count without the complement's
 * makes the row unreadable -- "s ~ r between vowels, count 14" says nothing
 * until you know what s does elsewhere. contrast_count is the same
 * correspondence in the observations where the environment does not hold, and
 * contrast_total is that side's denominator. delta_bic is what the split
 * scored: negative means it paid for its parameter. */
typedef struct rg_conditioned_segment_count_row {
    const char *source;
    const char *target;
    rg_context_spec context;
    bool context_is_target;
    double count;
    double source_total;
    /* Where this rule sits in the decision list.
     *
     * Discovery is greedy and each rule is committed against what the earlier
     * ones left unexplained, so the rules are ordered and the order carries
     * meaning: a later rule refines, or applies within, what an earlier one
     * did not settle. The Middle Chinese register split reads as three
     * decisions in sequence -- source tone accounts for one class, then the
     * onset's voicing splits what it left -- and as an unordered set it reads
     * as three unrelated facts, one of them at confidence 0.50.
     *
     * Published tables are sorted by key so lookups can binary-search them,
     * which destroys that order; this preserves it. Rules committed by one
     * decision share an index. -1 where the row was not committed by a search:
     * an unconditioned class is aggregated, not decided. */
    int decision_index;
    double contrast_count;
    double contrast_total;
    double delta_bic;
    /* How heavy a search charge this rule's evidence could carry and still
     * commit. Comparable across corpora, and comparable against the same
     * number measured on the corpus shuffled -- see rg_corpus_fit. */
    double search_margin;
    /* Set once the shuffled baseline has been measured; see rg_rule_standing. */
    rg_rule_standing standing;
    rg_uncertainty_estimate uncertainty;
} rg_conditioned_segment_count_row;

typedef struct rg_chunk_row {
    const rg_segment *source;
    size_t source_count;
    const rg_segment *target;
    size_t target_count;
    double cost;
    double count;
    /* Set when the target is the source's own segments in another order: a
     * reordering, not a set of substitutions. The segment table records the
     * matching pairs -- s answering s, k answering k -- so without this the
     * model would say nothing happened. */
    bool reordering;
    /* How readable this chunk is as one historical process, in [0, 1]. A chunk
     * that is short, balanced, decomposes without gaps and does not merely
     * wrap a smaller promoted chunk scores high; a long lopsided bundle of
     * unrelated changes that happened to pay for itself jointly scores low.
     * Always computed. `rg_train_options.chunk_min_transparency` is what acts
     * on it, and its default of zero drops nothing.
     *
     * A heuristic, and advisory: it is a claim about how easy the chunk is to
     * read, not about whether it is right. */
    double transparency;
    rg_uncertainty_estimate uncertainty;
} rg_chunk_row;

/* A claim that a source-side feature conditions a target-side dimension: in
 * the environment (source_feature at source_position), the target dimension
 * takes target_value.
 *
 * A conditioning environment is only conditioning if the complementary
 * environment behaves differently, so every row carries the contrast it was
 * measured against. `count`/`source_count`/`confidence` describe the
 * environment; `contrast_count`/`contrast_source_count`/`contrast_confidence`
 * describe everywhere else. Reading `confidence` alone will mislead: a rule
 * holding at 0.9 where the contrast also holds at 0.9 states the ambient
 * distribution, not a conditioned split. The row is published only when the
 * environment raises the value above its contrast.
 *
 * `delta_bic` is for the environment as a whole, not for this value: the
 * likelihood gain from modelling the target dimension separately inside and
 * outside the environment, penalised by the parameters that costs. It is
 * negative for every published row, and more negative is stronger. */
/* A rule where something about the source form conditions a suprasegmental
 * value on the target.
 *
 * The environment is an rg_context_spec, the same type a conditioned
 * correspondence uses, so it can name more than one predicate. It has to: the
 * Middle Chinese register split conditions the target tone on the preceding
 * onset's voicing *and* on the source segment's own tone, and neither alone
 * predicts it above chance. A single-predicate row reported that rule at
 * confidence 0.50 and looked like a weak finding rather than half of one. */
typedef struct rg_cross_dimensional_row {
    rg_context_spec source_environment;
    const char *target_dimension;
    const char *target_value;
    int target_position_offset;
    double count;
    double source_count;
    double confidence;
    double contrast_count;
    double contrast_source_count;
    double contrast_confidence;
    double delta_bic;
    int decision_index;
    double search_margin;
    rg_rule_standing standing;
    rg_uncertainty_estimate uncertainty;
} rg_cross_dimensional_row;

typedef struct rg_multi_pair_model_row {
    const char *lect_a;
    const char *lect_b;
    const rg_pairwise_model *model;
} rg_multi_pair_model_row;

typedef struct rg_multi_class_row {
    int class_id;
    const char *const *lect_ids;
    const char *const *graphemes;
    const rg_context_spec *contexts;
    size_t segment_count;
    double count;
    double confidence;
    /* The same segment tuple where the environment does not hold, and the
     * delta-BIC the split scored. Zero on an unconditioned class, which has no
     * environment and so no complement to compare against. */
    double contrast_count;
    double delta_bic;
    int decision_index;
    /* How heavy a search charge this class's evidence could carry and still
     * commit; read against rg_corpus_fit's null_search_margin. Zero on an
     * unconditioned class, which was not committed by a search. */
    double search_margin;
    rg_rule_standing standing;
    const char *const *supporting_cognates;
    size_t supporting_cognate_count;
    rg_uncertainty_estimate uncertainty;
} rg_multi_class_row;

typedef struct rg_multi_cross_dimensional_row {
    const char *source_lect;
    const char *target_lect;
    rg_context_spec source_environment;
    const char *target_dimension;
    const char *target_value;
    int target_position_offset;
    double count;
    double source_count;
    double confidence;
    double contrast_count;
    double contrast_source_count;
    double contrast_confidence;
    double delta_bic;
    int decision_index;
    double search_margin;
    rg_rule_standing standing;
    rg_uncertainty_estimate uncertainty;
} rg_multi_cross_dimensional_row;

typedef struct rg_cognate_outlier_row {
    const char *cognate_id;
    int pair_count;
    double cost_per_segment;
    double z_score;
} rg_cognate_outlier_row;

/* How well the model fits the corpus it was trained on, and what the same
 * training does to data with the correspondences taken out of it.
 *
 * The class counts are not a measure of relatedness. They rise when the signal
 * is removed: a corpus whose pairings have been shuffled has no correspondences
 * left to find, and greedy splitting over a large candidate inventory finds
 * more environments in it, not fewer. `cost_per_segment` is the number that
 * separates the two -- it is strongly negative on real cognates and near zero
 * on shuffled ones -- and the permutation baseline is what makes it readable,
 * since its scale depends on the corpus.
 *
 * The baseline is off by default because it costs one full training run per
 * shuffle. `permutation_count == 0` means it was not run and every `null_`
 * field is zero. */
typedef struct rg_corpus_fit {
    double cost_per_segment;
    size_t scored_set_count;
    size_t unconditioned_class_count;
    size_t conditioned_class_count;
    size_t permutation_count;
    double null_cost_per_segment_mean;
    double null_cost_per_segment_sd;
    /* Standard deviations between the observed fit and the shuffled baseline.
     * Strongly negative means the corpus aligns far better than chance. Zero
     * when no baseline was run, or when the baseline had no spread. */
    double cost_per_segment_z;
    double null_unconditioned_class_mean;
    double null_conditioned_class_mean;
    /* The search margin the shuffled corpus reached, at the quantile named
     * below: the level a rule has to clear to be saying more than the search
     * itself does. Rules at or under it were findable in data with no
     * correspondences left in it. */
    double null_search_margin;
    double null_search_margin_quantile;
    /* Forms with no nucleus of their own -- no vowel and no syllabic consonant
     * -- which were given one so the syllable predicates have something to
     * hold of. Vowelless words are real and their analyses differ, so a
     * syllable-conditioned rule on a corpus with many of these is resting on a
     * guess. */
    size_t inferred_nucleus_form_count;
    size_t syllabified_form_count;
    /* How many conditioned rules stand above the shuffled baseline, of how
     * many that were measured. Zero and zero when no baseline was run. */
    size_t rules_above_noise;
    size_t rules_measured;
} rg_corpus_fit;

RG_API const char *rg_version_string(void);
RG_API int rg_version_major(void);
RG_API int rg_version_minor(void);
RG_API int rg_version_patch(void);
RG_API uint32_t rg_abi_version(void);
RG_API const char *rg_status_string(rg_status status);
RG_API void rg_string_free(char *value);

RG_API void rg_bic_config_init_defaults(rg_bic_config *config);
RG_API void rg_train_options_init_defaults(rg_train_options *options);

/* The significance level every table in a trained model is published at.
 * 0.05 is a 95% interval. */
#define RG_DEFAULT_ALPHA 0.05

/* Wilson score interval on the rate count/total. Accepts fractional counts,
 * because confidence-weighted training produces them. total <= 0 yields the
 * full [0, 1] range with method RG_UNCERTAINTY_NONE. Returns
 * RG_ERR_UNSUPPORTED_OPTION for an alpha outside {0.10, 0.05, 0.01}. */
RG_API rg_status rg_wilson_interval(
    double count,
    double total,
    double alpha,
    rg_uncertainty_estimate *out
);

/* Distribution-free percentile interval over samples, each a rate drawn from
 * one resampled training run. total is the denominator behind the point
 * estimate and is recorded, not used. Samples are clamped to [0, 1]. An empty
 * sample set yields the full [0, 1] range with method RG_UNCERTAINTY_NONE.
 * Returns RG_ERR_UNSUPPORTED_OPTION for an unsupported alpha. */
RG_API rg_status rg_percentile_interval(
    const double *samples,
    size_t sample_count,
    double estimate,
    double total,
    double alpha,
    rg_uncertainty_estimate *out
);

/* Stable lowercase name of an interval method: "none", "wilson", "bootstrap".
 * Borrowed and static. */
RG_API const char *rg_uncertainty_method_string(rg_uncertainty_method method);

/* How a written word is cut into segments.
 *
 * Neither reading is universally right, so the choice belongs to whoever knows
 * the corpus. Orthographic honours the tie bar, which is how a transcription
 * says "this is one segment": "t͡ʃ" is one and untied "tʃ" is two. Longest
 * match instead asks the feature system what it recognises, which reads untied
 * "tʃ" and "kp" as single segments -- and, on the same rule, "kk", "dr", "ng"
 * and "st", which are sequences in most corpora regulae is pointed at. Latin
 * "bukka" comes out as b/u/kk/a under it. */
typedef enum rg_segmentation {
    RG_SEGMENT_ORTHOGRAPHIC = 0,
    RG_SEGMENT_SYSTEM_LONGEST_MATCH
} rg_segmentation;

RG_API rg_status rg_context_new_builtin(rg_context **out);
RG_API rg_status rg_context_set_segmentation(rg_context *ctx, rg_segmentation mode);
RG_API rg_segmentation rg_context_segmentation(const rg_context *ctx);
RG_API void rg_context_free(rg_context *ctx);
RG_API rg_status rg_context_use_system(rg_context *ctx, const char *system_name);
RG_API rg_status rg_context_system_name(const rg_context *ctx, const char **out);
RG_API rg_status rg_context_is_segment(const rg_context *ctx, const char *grapheme, bool *out);
/* Names the grapheme behind the most recent RG_ERR_UNKNOWN_GRAPHEME, and the
 * feature system that rejected it. Both are borrowed and valid until the next
 * failure or until the context is freed; grapheme is null if none has failed.
 * "unknown grapheme" without saying which one is unactionable on a corpus of
 * any size. */
RG_API void rg_context_last_error(
    const rg_context *ctx,
    const char **grapheme,
    const char **feature_system
);

/* Why a grapheme was refused. status is what a feature lookup would return:
 * RG_ERR_UNKNOWN_GRAPHEME for a sound the system does not cover,
 * RG_ERR_SOURCE_MARKER for CLDF/CLTS markup, RG_ERR_PARSE for a token the
 * system recognises and rejects, RG_OK for one that resolves.
 *
 * valid_prefix_bytes is the longest prefix that does resolve, which localises
 * the problem and is usually the repair; it is 0 when nothing resolves and the
 * whole length when nothing is wrong. offending_offset is the byte offset just
 * past that prefix, and offending holds the character there, empty when there
 * is none. There is deliberately no nearest-valid-grapheme suggestion: that
 * would be a guess presented as an answer. */
typedef struct rg_grapheme_diagnosis {
    rg_status status;
    size_t valid_prefix_bytes;
    size_t offending_offset;
    char offending[8];
} rg_grapheme_diagnosis;

/* Diagnoses one grapheme. Returns RG_OK unless the arguments are unusable: a
 * refused grapheme is the normal case and is reported in out->status. */
RG_API rg_status rg_context_diagnose(
    const rg_context *ctx,
    const char *grapheme,
    rg_grapheme_diagnosis *out
);

/* The diagnosis behind the most recent refusal, alongside the grapheme
 * rg_context_last_error names. Returns false when nothing has been refused. */
RG_API bool rg_context_last_diagnosis(
    const rg_context *ctx,
    rg_grapheme_diagnosis *out
);

RG_API rg_status rg_context_segment_distance(
    const rg_context *ctx,
    const char *a,
    const char *b,
    double *out
);
RG_API rg_status rg_context_grapheme_features(
    const rg_context *ctx,
    const char *grapheme,
    rg_feature_set **out
);

RG_API rg_status rg_context_segment_word(
    const rg_context *ctx,
    const char *word,
    rg_segment **out,
    size_t *out_count
);
RG_API void rg_segments_free(rg_segment *segments, size_t count);

RG_API size_t rg_feature_set_size(const rg_feature_set *features);
RG_API const char *rg_feature_set_get(const rg_feature_set *features, size_t index);
RG_API void rg_feature_set_free(rg_feature_set *features);

RG_API void rg_context_spec_init_empty(rg_context_spec *context);
RG_API size_t rg_context_spec_constraint_count(const rg_context_spec *context);
RG_API rg_status rg_context_spec_is_subset(
    const rg_context_spec *subset,
    const rg_context_spec *other,
    bool *out
);

RG_API rg_status rg_score_link(
    const rg_context *ctx,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    double *out
);
RG_API rg_status rg_score_link_with_model(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    const rg_train_options *options,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    double *out
);
RG_API rg_status rg_compute_displacement(
    const rg_context *ctx,
    rg_segment source,
    rg_segment target,
    rg_feature_displacement **out,
    size_t *out_count
);
RG_API void rg_feature_displacement_free(rg_feature_displacement *items, size_t count);

RG_API rg_status rg_compute_syllable_breaks(
    const rg_context *ctx,
    const rg_form *form,
    int **out,
    size_t *out_count
);
RG_API void rg_syllable_breaks_free(int *breaks);

RG_API rg_status rg_align_forms(
    const rg_context *ctx,
    const rg_form *source,
    const rg_form *target,
    int max_chunk_size,
    rg_alignment **out
);
RG_API rg_status rg_align_forms_with_model(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    const rg_train_options *options,
    const rg_form *source,
    const rg_form *target,
    int max_chunk_size,
    rg_alignment **out
);
RG_API void rg_alignment_free(rg_alignment *alignment);
RG_API size_t rg_alignment_link_count(const rg_alignment *alignment);
RG_API const rg_link *rg_alignment_link_at(const rg_alignment *alignment, size_t index);
RG_API rg_status rg_alignment_cost(const rg_context *ctx, const rg_alignment *alignment, double *out);
RG_API rg_status rg_alignment_cost_with_model(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    const rg_train_options *options,
    const rg_alignment *alignment,
    double *out
);

RG_API rg_status rg_train_pairwise_segment_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model **out
);
RG_API rg_status rg_train_pairwise(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model **out
);
RG_API void rg_pairwise_model_free(rg_pairwise_model *model);
RG_API size_t rg_pairwise_model_segment_count_row_count(const rg_pairwise_model *model);
RG_API const rg_segment_count_row *rg_pairwise_model_segment_count_row_at(
    const rg_pairwise_model *model,
    size_t index
);
RG_API size_t rg_pairwise_model_displacement_row_count(const rg_pairwise_model *model);
RG_API const rg_displacement_row *rg_pairwise_model_displacement_row_at(
    const rg_pairwise_model *model,
    size_t index
);
RG_API size_t rg_pairwise_model_tonal_count_row_count(const rg_pairwise_model *model);
RG_API const rg_tonal_count_row *rg_pairwise_model_tonal_count_row_at(
    const rg_pairwise_model *model,
    size_t index
);
RG_API size_t rg_pairwise_model_conditioned_segment_count_row_count(const rg_pairwise_model *model);
RG_API const rg_conditioned_segment_count_row *rg_pairwise_model_conditioned_segment_count_row_at(
    const rg_pairwise_model *model,
    size_t index
);
RG_API size_t rg_pairwise_model_chunk_row_count(const rg_pairwise_model *model);
RG_API const rg_chunk_row *rg_pairwise_model_chunk_row_at(
    const rg_pairwise_model *model,
    size_t index
);
RG_API size_t rg_pairwise_model_cross_dimensional_row_count(const rg_pairwise_model *model);
RG_API const rg_cross_dimensional_row *rg_pairwise_model_cross_dimensional_row_at(
    const rg_pairwise_model *model,
    size_t index
);

typedef struct rg_tsv_load_options {
    const char *cognate_id_column;
    const char *lect_id_column;
    const char *segments_column;
    const char *alignment_column;
    const char *confidence_column;
    /* Per-segment tone, whitespace-separated and positionally parallel to the
     * segments cell: "1 0 3" tones the first, second and third segment. "-" or
     * an empty token leaves that segment untoned, which is how a tone-bearing
     * corpus marks its consonants. Defaults to the "tone" column when present.
     *
     * Tone gets its own column rather than riding inline on the grapheme
     * because merkmal's segmenter does not merge Chao digits — "pa1" segments
     * as p, a, 1, and the stray digit is then an unknown grapheme. A separate
     * column also keeps the tone out of feature lookup entirely, which is
     * where it belongs: tone is a suprasegmental, not part of the grapheme. */
    const char *tone_column;
    /* Per-segment stress, laid out like tone: one value per segment, "-" for a
     * segment that carries none. Defaults to the "stress" column when present.
     *
     * A corpus can also write the IPA marks in the word, where the segmenter
     * lifts them onto the syllable nucleus. The column is for corpora that
     * record stress as an annotation rather than in the transcription, and for
     * anything the marks cannot express. */
    const char *stress_column;
    /* Morpheme boundaries as segment indices, default "breaks". */
    const char *morpheme_breaks_column;
    /* Per-segment length values, default "length". A corpus may instead write
     * length into the grapheme as `aː`, which merkmal reads as its own segment
     * carrying the `long` feature; that is the right shape where length is
     * contrastive. This column is for treating it as a dimension -- something
     * that happens to a vowel -- so a lengthening can be found as a rule
     * rather than as a correspondence between two different vowels. */
    const char *length_column;
    /* Syllable boundaries as segment indices, default "syllables". Supplying
     * them overrides the sonority syllabifier, which is language-agnostic by
     * design and will be wrong wherever a language's phonotactics are not. */
    const char *syllable_breaks_column;
} rg_tsv_load_options;

/* Wide-format TSV: one row per cognate, one column per lect, cells holding
 * whole unsegmented words. This is the shape linguistic data is actually
 * written in; the generic long-format loader wants pre-segmented input.
 * A "<lect>_breaks" column supplies morpheme boundary indices for that lect. */
typedef struct rg_wide_load_options {
    const char *cognate_id_column;
    const char *confidence_column;
    const char *const *lect_columns;
    size_t lect_column_count;
} rg_wide_load_options;

typedef struct rg_gled_load_options {
    const char *family;
    const char *const *doculects;
    size_t doculect_count;
    int min_lects;
} rg_gled_load_options;

typedef struct rg_arcaverborum_load_options {
    const char *dataset;
    const char *family;
    const char *const *language_ids;
    size_t language_id_count;
    int min_lects;
    /* Zero selects the delimiter from the file extension when loading a path,
     * and comma when parsing text. */
    char delimiter;
} rg_arcaverborum_load_options;

/* Why a load failed, and where.
 *
 * The caller owns the storage, which is the whole point. A failed load returns
 * no corpus to hang a message on, and the answer until ABI 22 was a
 * process-wide buffer: not thread-safe, and stale between calls, because the
 * parse_* entry points never cleared it and a successful parse after a failed
 * load still reported the failure.
 *
 * Every load and parse entry point clears this on entry when one is supplied,
 * and passing null asks for no reporting. line is 1-based, and 0 when the
 * failure is not tied to one. message is empty when the loader recorded no
 * detail beyond the status -- "parse error" with no line is unactionable on a
 * corpus of any size, which is the first thing a new user meets. */
typedef struct rg_load_diagnosis {
    size_t line;
    char message[256];
} rg_load_diagnosis;

RG_API rg_status rg_corpus_load_tsv(
    const char *path,
    const rg_tsv_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
);
RG_API rg_status rg_corpus_load_wide_tsv(
    const rg_context *ctx,
    const char *path,
    const rg_wide_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
);
RG_API rg_status rg_corpus_load_gled(
    const char *path,
    const rg_gled_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
);
RG_API rg_status rg_corpus_load_arcaverborum(
    const char *path,
    const rg_arcaverborum_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
);
RG_API rg_status rg_corpus_from_pairs(
    const rg_form_pair *pairs,
    size_t pair_count,
    const char *lect_a,
    const char *lect_b,
    const char *cognate_id_prefix,
    rg_corpus **out
);
/* Parse variants take the corpus as text rather than a path, so a caller that
 * already holds the data needs no temporary file. The WebAssembly build links
 * without a filesystem and uses these exclusively. */
RG_API rg_status rg_corpus_parse_tsv(
    const char *text,
    const rg_tsv_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
);
RG_API rg_status rg_corpus_parse_wide_tsv(
    const rg_context *ctx,
    const char *text,
    const rg_wide_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
);
RG_API rg_status rg_corpus_parse_gled(
    const char *text,
    const rg_gled_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
);
RG_API rg_status rg_corpus_parse_arcaverborum(
    const char *text,
    const rg_arcaverborum_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
);

RG_API void rg_corpus_free(rg_corpus *corpus);
RG_API size_t rg_corpus_cognate_count(const rg_corpus *corpus);
/* Cognate sets in which some lect contributed more than one reflex, and how
 * many extra sets the corpus was expanded into as a result.
 *
 * A doublet is a fact about a language, not an error in a file: it is 3.2% of
 * cognate-set members across the Lexibank datasets with expert judgements, and
 * near 10% in some Austronesian ones. The corpus carries one set per
 * combination of reflexes, each with its share of the original confidence, so
 * both reflexes are counted and neither is invented. These say how often that
 * happened. */
RG_API size_t rg_corpus_doublet_set_count(const rg_corpus *corpus);
RG_API size_t rg_corpus_doublet_expansion_count(const rg_corpus *corpus);

RG_API const rg_cognate_set *rg_corpus_cognates(const rg_corpus *corpus);
RG_API const rg_cognate_set *rg_corpus_cognate_at(const rg_corpus *corpus, size_t index);

/* Formatting helpers. The strings these produce are diagnostics meant to be
 * read; iterate the model accessors for anything a program depends on. Every
 * returned string is caller-owned and freed with rg_string_free. */
typedef struct rg_format_model_options {
    int top_segments;
    int top_displacements;
    double min_count;
    int top_chunks;
    int top_classes;
} rg_format_model_options;

RG_API void rg_format_model_options_init_defaults(rg_format_model_options *options);
RG_API char *rg_format_segments(const rg_segment *segments, size_t count);
RG_API char *rg_format_link(const rg_link *link);
RG_API char *rg_format_alignment(const rg_alignment *alignment);
RG_API char *rg_format_pairwise_model(
    const rg_pairwise_model *model,
    const rg_format_model_options *options
);
RG_API char *rg_format_multi_model(
    const rg_multi_model *model,
    const rg_format_model_options *options
);
RG_API char *rg_describe_multi_class(
    const rg_multi_model *model,
    const char *lect_id,
    const char *grapheme
);

/* Renders a trained model as JSON. Caller owns the string (rg_string_free).
 * This is a debug/inspection snapshot, labelled as such in the payload; it is
 * not the interchange format, which carries ensembles rather than a single
 * maximum-a-posteriori system. */
RG_API char *rg_model_to_json(
    const rg_context *ctx,
    const rg_multi_model *model,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    bool include_alignments,
    bool include_outliers
);

/* Renders an error as the same JSON envelope a successful call uses, so a
 * caller has one shape to handle. Caller owns the string. */
RG_API char *rg_error_to_json(rg_status status, const char *detail);

/* Reads a flat options object. Unknown keys are rejected rather than ignored,
 * with the offending key named in error_detail, which may be null. */
RG_API rg_status rg_train_options_from_json(
    const char *text,
    rg_train_options *out,
    char *error_detail,
    size_t error_detail_size
);

/* Renders a segmented word, so a caller can show how input will be read. */
RG_API char *rg_segments_to_json(const rg_segment *segments, size_t count);

RG_API rg_status rg_train_model(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    rg_multi_model **out
);
RG_API void rg_multi_model_free(rg_multi_model *model);
RG_API size_t rg_multi_model_lect_count(const rg_multi_model *model);
/* Cognate sets that carried fewer than two forms, and so contributed no
 * correspondence. Not an error: a form whose cognates are in lects the corpus
 * did not sample has nothing to align against, and every cognate-coded
 * wordlist has some. Worth reading as a proportion of the corpus, because a
 * high one means the lect sample, not the method, is deciding the result. */
RG_API size_t rg_multi_model_unpaired_set_count(const rg_multi_model *model);
/* Borrowed; valid while the model lives. Never NULL for a trained model. */
RG_API const rg_corpus_fit *rg_multi_model_fit(const rg_multi_model *model);
RG_API const char *rg_multi_model_lect_at(const rg_multi_model *model, size_t index);
RG_API size_t rg_multi_model_pair_model_count(const rg_multi_model *model);
RG_API const rg_multi_pair_model_row *rg_multi_model_pair_model_at(
    const rg_multi_model *model,
    size_t index
);
RG_API size_t rg_multi_model_unconditioned_class_count(const rg_multi_model *model);
RG_API const rg_multi_class_row *rg_multi_model_unconditioned_class_at(
    const rg_multi_model *model,
    size_t index
);
RG_API size_t rg_multi_model_conditioned_class_count(const rg_multi_model *model);
RG_API const rg_multi_class_row *rg_multi_model_conditioned_class_at(
    const rg_multi_model *model,
    size_t index
);
RG_API size_t rg_multi_model_cross_dimensional_row_count(const rg_multi_model *model);
RG_API const rg_multi_cross_dimensional_row *rg_multi_model_cross_dimensional_row_at(
    const rg_multi_model *model,
    size_t index
);
RG_API rg_status rg_find_cognate_outliers(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_multi_model *model,
    const rg_train_options *options,
    int top_k,
    int max_chunk_size,
    rg_cognate_outlier_row **out,
    size_t *out_count
);
RG_API void rg_cognate_outlier_rows_free(rg_cognate_outlier_row *rows, size_t count);

#ifdef __cplusplus
}
#endif

#endif
