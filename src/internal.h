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

struct rg_multi_model {
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
};

char *rg_strdup_internal(const char *value);
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

int rg_predicate_holds_internal(
    const rg_context_spec *context,
    const rg_split_candidate *candidate
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

extern const char *const rg_context_feature_names[];
extern const size_t rg_context_feature_name_count;

rg_status rg_context_constraints_internal(
    const rg_context *ctx,
    const char *grapheme,
    const rg_feature_constraint **out,
    size_t *out_count
);

/* JSON transport for the CLI's --json output and the WebAssembly adapter. Not
 * the interchange schema: that is M8's job, and a single MAP correspondence
 * system is explicitly not claim-capable data. */
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

rg_status rg_context_features_internal(
    const rg_context *ctx,
    const char *grapheme,
    const rg_feature_set **out
);

rg_status rg_compute_syllable_breaks_internal(
    const rg_context *ctx,
    const rg_form *form,
    size_t **out,
    size_t *out_count
);
int rg_segment_posterior_internal(
    const rg_pairwise_model *model,
    const char *source,
    const char *target,
    const rg_context_spec *link_context,
    double *out
);
double rg_segment_log_normalizer_internal(const rg_pairwise_model *model, const char *source);
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
    double *out
);

#endif
