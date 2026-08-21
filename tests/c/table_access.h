#ifndef REGULAE_TEST_TABLE_ACCESS_H
#define REGULAE_TEST_TABLE_ACCESS_H

#include "regulae.h"

/* Per-row access to the published tables, for tests.
 *
 * The library hands out each table whole -- the rows and how many -- because
 * that is what its own consumers do with them: sort an index over a table,
 * scan it, walk it in decision order. It used to export a count function and an
 * index function per table instead, twenty-two of them, each a null check and
 * an array index, and every caller wrote its own per-row loop.
 *
 * Tests want the other shape. An assertion is about row i, or about how many
 * rows there are, and reads worst when it has to declare a pointer and a count
 * first. So the convenience lives here, on top of the published interface,
 * rather than in it: the bulk accessor is the primitive, and indexing is a
 * caller's business.
 *
 * Bounds-checked, returning NULL past the end, so a test that walks off a table
 * fails on a null dereference rather than reading whatever follows it.
 */

#define RG_TEST_TABLE(name, rowtype, handle, bulk)                          \
    static inline size_t name##_count(const handle *model) {                       \
        size_t n = 0;                                                       \
        bulk(model, &n);                                                    \
        return n;                                                           \
    }                                                                       \
    static inline const rowtype *name##_at(const handle *model, size_t index) {    \
        size_t n = 0;                                                       \
        const rowtype *rows = bulk(model, &n);                              \
        return index < n ? &rows[index] : 0;                                \
    }

RG_TEST_TABLE(rg_pairwise_model_segment_count_row, rg_segment_count_row,
              rg_pairwise_model, rg_pairwise_model_segment_counts)
RG_TEST_TABLE(rg_pairwise_model_displacement_row, rg_displacement_row,
              rg_pairwise_model, rg_pairwise_model_displacements)
RG_TEST_TABLE(rg_pairwise_model_tonal_count_row, rg_tonal_count_row,
              rg_pairwise_model, rg_pairwise_model_tonal_counts)
RG_TEST_TABLE(rg_pairwise_model_null_correspondence_row, rg_segment_count_row,
              rg_pairwise_model, rg_pairwise_model_null_correspondences)
RG_TEST_TABLE(rg_pairwise_model_conditioned_segment_count_row, rg_conditioned_segment_count_row,
              rg_pairwise_model, rg_pairwise_model_conditioned_segment_counts)
RG_TEST_TABLE(rg_pairwise_model_chunk_row, rg_chunk_row,
              rg_pairwise_model, rg_pairwise_model_chunks)
RG_TEST_TABLE(rg_pairwise_model_cross_dimensional_row, rg_cross_dimensional_row,
              rg_pairwise_model, rg_pairwise_model_cross_dimensional_rows)
RG_TEST_TABLE(rg_multi_model_unconditioned_class, rg_multi_class_row,
              rg_multi_model, rg_multi_model_unconditioned_classes)
RG_TEST_TABLE(rg_multi_model_conditioned_class, rg_multi_class_row,
              rg_multi_model, rg_multi_model_conditioned_classes)
RG_TEST_TABLE(rg_multi_model_cross_dimensional_row, rg_multi_cross_dimensional_row,
              rg_multi_model, rg_multi_model_cross_dimensional_rows)

#undef RG_TEST_TABLE

static inline size_t rg_multi_model_lect_count(const rg_multi_model *model) {
    size_t n = 0;
    rg_multi_model_lects(model, &n);
    return n;
}

static inline const char *rg_multi_model_lect_at(const rg_multi_model *model, size_t index) {
    size_t n = 0;
    const char *const *names = rg_multi_model_lects(model, &n);
    return index < n ? names[index] : 0;
}

#endif
