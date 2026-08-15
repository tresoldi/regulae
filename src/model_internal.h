#ifndef REGULAE_MODEL_INTERNAL_H
#define REGULAE_MODEL_INTERNAL_H

/* Shared between the pairwise pipeline's stages, and between them only.
 *
 * model.c ran to 4,800 lines and 101 static functions, and the stages inside
 * it are the ones AGENTS.md calls load-bearing: initial prior, segment EM,
 * displacement aggregation, context discovery, chunk promotion, tonal
 * aggregation, cross-dimensional discovery, long-range discovery. They are now
 * one file each, and this header is what they had been sharing implicitly by
 * sitting in the same translation unit.
 *
 * It is deliberately not `internal.h`. That header is the surface between
 * modules -- context, search, loaders, format -- and none of them has any
 * business with a chunk candidate or a split search. Everything here is
 * private to the pairwise pipeline, and anything that stops being shared
 * should go back to `static` in the file that uses it. */

#include "internal.h"

/* The published-table machinery in model.c, used by the stages that fill those
 * tables. These keep their names rather than gaining an `_internal` suffix:
 * that suffix marks the surface *between modules* in internal.h, and these are
 * private to the pipeline and hidden by the visibility preset. Renaming them
 * would make a pure move read as an API change. */
void segment_array_clear(const rg_segment *segments, size_t count);
rg_status segment_array_copy(const rg_segment *segments, size_t count, const rg_segment **out);
int segment_array_equal(const rg_segment *a, size_t a_count, const rg_segment *b, size_t b_count);
void chunk_row_clear(rg_chunk_row *row);
int chunk_row_cmp(const void *a, const void *b);

/* Chunk promotion, called by the pipeline driver. */
rg_status promote_chunk_rows(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model
);

void cross_dimensional_row_clear(rg_cross_dimensional_row *row);
int cross_dimensional_row_cmp(const void *a, const void *b);

/* Cross-dimensional discovery, called by the pipeline driver. */
rg_status discover_cross_dimensional_rows(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model,
    const rg_feature_vocabulary *vocabulary
);

/* The conditioned-correspondence table, and the two discovery stages that
 * fill it. */
rg_status add_conditioned_segment_count(
    rg_conditioned_segment_count_row **rows,
    size_t *count,
    size_t *cap,
    const char *source,
    const char *target,
    const rg_context_spec *context,
    int context_is_target,
    double weight,
    double source_total,
    double contrast_count,
    double contrast_total,
    double delta_bic,
    double search_margin,
    int decision_index
);
double source_total_for_rows(const rg_segment_count_row *rows, size_t count, const char *source);
int conditioned_count_row_cmp(const void *a, const void *b);
rg_status discover_immediate_context_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model,
    const rg_feature_vocabulary *vocabulary
);
rg_status discover_long_range_context_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model,
    const rg_feature_vocabulary *vocabulary
);

/* model_tables.c: allocation, ordering and publication of the rows every
 * stage writes into. */
void string_array_free(char **items, size_t count);
void segment_count_row_clear(rg_segment_count_row *row);
void displacement_row_clear(rg_displacement_row *row);
int tonal_row_cmp(const void *a, const void *b);
int count_row_cmp(const void *a, const void *b);
int displacement_row_cmp(const void *a, const void *b);
rg_status add_segment_count(
    rg_segment_count_row **rows,
    size_t *count,
    size_t *cap,
    const char *source,
    const char *target,
    double weight
);
rg_status add_displacement_vector(
    rg_displacement_row **rows,
    size_t *count,
    size_t *cap,
    const rg_feature_displacement *items,
    size_t item_count,
    double weight
);
rg_status add_tonal_count(
    rg_tonal_count_row **rows,
    size_t *count,
    size_t *cap,
    const char *source_tone,
    const char *target_tone,
    double weight
);
void fill_totals(rg_segment_count_row *rows, size_t count);
void publish_target_totals(rg_pairwise_model *model, const rg_segment_count_row *rows, size_t count);
void fill_displacement_total(rg_displacement_row *rows, size_t count);
void fill_tonal_source_totals(rg_tonal_count_row *rows, size_t count);

#endif
