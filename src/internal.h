#ifndef REGULAE_INTERNAL_H
#define REGULAE_INTERNAL_H

#include "regulae.h"

#include <stddef.h>

#define RG_DEFAULT_GAP_COST 0.5
#define RG_DEFAULT_CHUNK_PENALTY 0.25
#define RG_CHUNK_COMPLEXITY_PENALTY 1e-9

struct rg_pairwise_model {
    rg_segment_count_row *segment_counts;
    size_t segment_count_count;
    rg_conditioned_segment_count_row *conditioned_segment_counts;
    size_t conditioned_segment_count_count;
    rg_chunk_row *chunks;
    size_t chunk_count;
    rg_cross_dimensional_row *cross_dimensional_rows;
    size_t cross_dimensional_count;
    rg_displacement_count_row *displacement_counts;
    size_t displacement_count_count;
    rg_tonal_count_row *tonal_counts;
    size_t tonal_count_count;
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
