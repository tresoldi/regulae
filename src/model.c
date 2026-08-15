#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void string_array_free(char **items, size_t count) {
    size_t i;
    if (items == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        free(items[i]);
    }
    free(items);
}

static void segment_count_row_clear(rg_segment_count_row *row) {
    if (row == 0) {
        return;
    }
    free((char *)row->source);
    free((char *)row->target);
    row->source = 0;
    row->target = 0;
    row->count = 0.0;
    row->source_total = 0.0;
    row->uncertainty = rg_wilson_default_internal(0.0, 0.0);
}

static void conditioned_segment_count_row_clear(rg_conditioned_segment_count_row *row) {
    if (row == 0) {
        return;
    }
    free((char *)row->source);
    free((char *)row->target);
    rg_context_spec_clear_internal(&row->context);
    row->source = 0;
    row->target = 0;
    row->count = 0.0;
    row->source_total = 0.0;
    row->uncertainty = rg_wilson_default_internal(0.0, 0.0);
}

static void segment_array_clear(const rg_segment *segments, size_t count) {
    size_t i;
    if (segments == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        rg_segment_clear_internal((rg_segment *)&segments[i]);
    }
    free((rg_segment *)segments);
}

static rg_status segment_array_copy(const rg_segment *segments, size_t count, const rg_segment **out) {
    rg_segment *copy;
    size_t i;
    rg_status status;
    if (out == 0 || (count > 0 && segments == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    if (count == 0) {
        return RG_OK;
    }
    copy = (rg_segment *)calloc(count, sizeof(*copy));
    if (copy == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < count; i++) {
        status = rg_segment_copy_internal(&segments[i], &copy[i]);
        if (status != RG_OK) {
            segment_array_clear(copy, i);
            return status;
        }
    }
    *out = copy;
    return RG_OK;
}

static void chunk_row_clear(rg_chunk_row *row) {
    if (row == 0) {
        return;
    }
    segment_array_clear(row->source, row->source_count);
    segment_array_clear(row->target, row->target_count);
    row->source = 0;
    row->source_count = 0;
    row->target = 0;
    row->target_count = 0;
    row->cost = 0.0;
    row->count = 0.0;
    row->uncertainty = rg_wilson_default_internal(0.0, 0.0);
}

static void cross_dimensional_row_clear(rg_cross_dimensional_row *row) {
    if (row == 0) {
        return;
    }
    free((char *)row->source_feature);
    free((char *)row->source_value);
    free((char *)row->source_position);
    free((char *)row->target_dimension);
    free((char *)row->target_value);
    row->source_feature = 0;
    row->source_value = 0;
    row->source_position = 0;
    row->target_dimension = 0;
    row->target_value = 0;
    row->target_position_offset = 0;
    row->count = 0.0;
    row->source_count = 0.0;
    row->confidence = 0.0;
    row->uncertainty = rg_wilson_default_internal(0.0, 0.0);
}

static void displacement_row_clear(rg_displacement_row *row) {
    size_t i;
    if (row == 0) {
        return;
    }
    for (i = 0; i < row->item_count; i++) {
        free((char *)row->items[i].feature);
        free((char *)row->items[i].from_value);
        free((char *)row->items[i].to_value);
    }
    free((rg_feature_displacement *)row->items);
    memset(row, 0, sizeof(*row));
}

void rg_pairwise_model_free(rg_pairwise_model *model) {
    size_t i;
    if (model == 0) {
        return;
    }
    for (i = 0; i < model->segment_count_count; i++) {
        segment_count_row_clear(&model->segment_counts[i]);
    }
    free(model->segment_counts);
    for (i = 0; i < model->conditioned_segment_count_count; i++) {
        conditioned_segment_count_row_clear(&model->conditioned_segment_counts[i]);
    }
    free(model->conditioned_segment_counts);
    for (i = 0; i < model->chunk_count; i++) {
        chunk_row_clear(&model->chunks[i]);
    }
    free(model->chunks);
    for (i = 0; i < model->cross_dimensional_count; i++) {
        cross_dimensional_row_clear(&model->cross_dimensional_rows[i]);
    }
    free(model->cross_dimensional_rows);
    for (i = 0; i < model->displacement_row_count; i++) {
        displacement_row_clear(&model->displacement_rows[i]);
    }
    free(model->displacement_rows);
    for (i = 0; i < model->tonal_count_count; i++) {
        free((char *)model->tonal_counts[i].source_tone);
        free((char *)model->tonal_counts[i].target_tone);
    }
    free(model->tonal_counts);
    for (i = 0; i < model->segment_prior_count; i++) {
        free(model->segment_priors[i].source);
        free(model->segment_priors[i].target);
    }
    free(model->segment_priors);
    for (i = 0; i < model->log_normalizer_count; i++) {
        free(model->log_normalizers[i].source);
    }
    free(model->log_normalizers);
    free(model);
}

static int count_row_cmp(const void *a, const void *b) {
    const rg_segment_count_row *ra = (const rg_segment_count_row *)a;
    const rg_segment_count_row *rb = (const rg_segment_count_row *)b;
    int c = strcmp(ra->source, rb->source);
    if (c != 0) {
        return c;
    }
    return strcmp(ra->target, rb->target);
}

static int nullable_strcmp(const char *a, const char *b) {
    return strcmp(a == 0 ? "" : a, b == 0 ? "" : b);
}

static int constraint_list_cmp(
    const rg_feature_constraint *a, size_t a_count,
    const rg_feature_constraint *b, size_t b_count
) {
    size_t i;
    if (a_count != b_count) {
        return a_count < b_count ? -1 : 1;
    }
    for (i = 0; i < a_count; i++) {
        int c = nullable_strcmp(a[i].feature, b[i].feature);
        if (c != 0) {
            return c;
        }
        c = nullable_strcmp(a[i].value, b[i].value);
        if (c != 0) {
            return c;
        }
    }
    return 0;
}

static int distance_list_cmp(
    const rg_distance_constraint *a, size_t a_count,
    const rg_distance_constraint *b, size_t b_count
) {
    size_t i;
    if (a_count != b_count) {
        return a_count < b_count ? -1 : 1;
    }
    for (i = 0; i < a_count; i++) {
        int c;
        if (a[i].offset != b[i].offset) {
            return a[i].offset < b[i].offset ? -1 : 1;
        }
        c = nullable_strcmp(a[i].constraint.feature, b[i].constraint.feature);
        if (c != 0) {
            return c;
        }
        c = nullable_strcmp(a[i].constraint.value, b[i].constraint.value);
        if (c != 0) {
            return c;
        }
    }
    return 0;
}

/* A total order over everything a context can express. Ordering on a summary of
 * the environment -- position and constraint count -- leaves rows that differ
 * only in which constraint they carry comparing equal, and qsort is free to
 * return them in either order. It did: the same corpus published the same rows
 * in a different order under the native and the WebAssembly build, which moved
 * every class id downstream. */
static int context_spec_cmp(const rg_context_spec *a, const rg_context_spec *b) {
    int c = nullable_strcmp(a->position, b->position);
    if (c != 0) {
        return c;
    }
    c = nullable_strcmp(a->morphological, b->morphological);
    if (c != 0) {
        return c;
    }
    c = constraint_list_cmp(a->preceding, a->preceding_count, b->preceding, b->preceding_count);
    if (c != 0) {
        return c;
    }
    c = constraint_list_cmp(a->following, a->following_count, b->following, b->following_count);
    if (c != 0) {
        return c;
    }
    c = distance_list_cmp(a->preceding_at_distance, a->preceding_at_distance_count,
                          b->preceding_at_distance, b->preceding_at_distance_count);
    if (c != 0) {
        return c;
    }
    c = distance_list_cmp(a->following_at_distance, a->following_at_distance_count,
                          b->following_at_distance, b->following_at_distance_count);
    if (c != 0) {
        return c;
    }
    c = constraint_list_cmp(a->somewhere_preceding, a->somewhere_preceding_count,
                            b->somewhere_preceding, b->somewhere_preceding_count);
    if (c != 0) {
        return c;
    }
    c = constraint_list_cmp(a->somewhere_following, a->somewhere_following_count,
                            b->somewhere_following, b->somewhere_following_count);
    if (c != 0) {
        return c;
    }
    c = constraint_list_cmp(a->same_syllable, a->same_syllable_count,
                            b->same_syllable, b->same_syllable_count);
    if (c != 0) {
        return c;
    }
    c = constraint_list_cmp(a->next_syllable, a->next_syllable_count,
                            b->next_syllable, b->next_syllable_count);
    if (c != 0) {
        return c;
    }
    c = constraint_list_cmp(a->previous_syllable, a->previous_syllable_count,
                            b->previous_syllable, b->previous_syllable_count);
    if (c != 0) {
        return c;
    }
    c = constraint_list_cmp(a->self_stress, a->self_stress_count,
                            b->self_stress, b->self_stress_count);
    if (c != 0) {
        return c;
    }
    c = constraint_list_cmp(a->preceding_stress, a->preceding_stress_count,
                            b->preceding_stress, b->preceding_stress_count);
    if (c != 0) {
        return c;
    }
    return constraint_list_cmp(a->following_stress, a->following_stress_count,
                               b->following_stress, b->following_stress_count);
}

static int conditioned_count_row_cmp(const void *a, const void *b) {
    const rg_conditioned_segment_count_row *ra = (const rg_conditioned_segment_count_row *)a;
    const rg_conditioned_segment_count_row *rb = (const rg_conditioned_segment_count_row *)b;
    int c = strcmp(ra->source, rb->source);
    if (c != 0) {
        return c;
    }
    c = strcmp(ra->target, rb->target);
    if (c != 0) {
        return c;
    }
    if (ra->context_is_target != rb->context_is_target) {
        return ra->context_is_target < rb->context_is_target ? -1 : 1;
    }
    c = (int)rg_context_spec_constraint_count(&ra->context) - (int)rg_context_spec_constraint_count(&rb->context);
    if (c != 0) {
        return c;
    }
    return context_spec_cmp(&ra->context, &rb->context);
}

static int segment_equal(const rg_segment *a, const rg_segment *b) {
    return strcmp(a->grapheme == 0 ? "" : a->grapheme, b->grapheme == 0 ? "" : b->grapheme) == 0 &&
        strcmp(a->tone == 0 ? "" : a->tone, b->tone == 0 ? "" : b->tone) == 0 &&
        strcmp(a->length == 0 ? "" : a->length, b->length == 0 ? "" : b->length) == 0 &&
        strcmp(a->stress == 0 ? "" : a->stress, b->stress == 0 ? "" : b->stress) == 0;
}

static int segment_array_equal(const rg_segment *a, size_t a_count, const rg_segment *b, size_t b_count) {
    size_t i;
    if (a_count != b_count) {
        return 0;
    }
    for (i = 0; i < a_count; i++) {
        if (!segment_equal(&a[i], &b[i])) {
            return 0;
        }
    }
    return 1;
}

static int chunk_row_cmp(const void *a, const void *b) {
    const rg_chunk_row *ra = (const rg_chunk_row *)a;
    const rg_chunk_row *rb = (const rg_chunk_row *)b;
    size_t i;
    size_t n = ra->source_count < rb->source_count ? ra->source_count : rb->source_count;
    for (i = 0; i < n; i++) {
        int c = strcmp(ra->source[i].grapheme == 0 ? "" : ra->source[i].grapheme, rb->source[i].grapheme == 0 ? "" : rb->source[i].grapheme);
        if (c != 0) {
            return c;
        }
    }
    if (ra->source_count != rb->source_count) {
        return ra->source_count < rb->source_count ? -1 : 1;
    }
    n = ra->target_count < rb->target_count ? ra->target_count : rb->target_count;
    for (i = 0; i < n; i++) {
        int c = strcmp(ra->target[i].grapheme == 0 ? "" : ra->target[i].grapheme, rb->target[i].grapheme == 0 ? "" : rb->target[i].grapheme);
        if (c != 0) {
            return c;
        }
    }
    if (ra->target_count != rb->target_count) {
        return ra->target_count < rb->target_count ? -1 : 1;
    }
    return 0;
}

static int cross_dimensional_row_cmp(const void *a, const void *b) {
    const rg_cross_dimensional_row *ra = (const rg_cross_dimensional_row *)a;
    const rg_cross_dimensional_row *rb = (const rg_cross_dimensional_row *)b;
    int c = strcmp(ra->source_feature, rb->source_feature);
    if (c != 0) {
        return c;
    }
    c = strcmp(ra->source_value, rb->source_value);
    if (c != 0) {
        return c;
    }
    c = strcmp(ra->source_position, rb->source_position);
    if (c != 0) {
        return c;
    }
    c = strcmp(ra->target_dimension, rb->target_dimension);
    if (c != 0) {
        return c;
    }
    c = strcmp(ra->target_value, rb->target_value);
    if (c != 0) {
        return c;
    }
    if (ra->target_position_offset != rb->target_position_offset) {
        return ra->target_position_offset < rb->target_position_offset ? -1 : 1;
    }
    return 0;
}

static rg_status add_segment_count(
    rg_segment_count_row **rows,
    size_t *count,
    size_t *cap,
    const char *source,
    const char *target,
    double weight
) {
    size_t i;
    rg_segment_count_row *next;
    for (i = 0; i < *count; i++) {
        if (strcmp((*rows)[i].source, source) == 0 && strcmp((*rows)[i].target, target) == 0) {
            (*rows)[i].count += weight;
            return RG_OK;
        }
    }
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 16 : *cap * 2;
        next = (rg_segment_count_row *)realloc(*rows, next_cap * sizeof(**rows));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *rows = next;
        *cap = next_cap;
    }
    (*rows)[*count].source = rg_strdup_internal(source);
    (*rows)[*count].target = rg_strdup_internal(target);
    (*rows)[*count].count = weight;
    (*rows)[*count].source_total = 0.0;
    if ((*rows)[*count].source == 0 || (*rows)[*count].target == 0) {
        segment_count_row_clear(&(*rows)[*count]);
        return RG_ERR_OOM;
    }
    (*count)++;
    return RG_OK;
}

static double source_total_for_rows(const rg_segment_count_row *rows, size_t count, const char *source) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (strcmp(rows[i].source, source) == 0) {
            return rows[i].source_total;
        }
    }
    return 0.0;
}

static rg_status add_conditioned_segment_count(
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
    double delta_bic
) {
    size_t i;
    rg_conditioned_segment_count_row *next;
    for (i = 0; i < *count; i++) {
        int a_subset_b = 0;
        int b_subset_a = 0;
        if ((*rows)[i].context_is_target == context_is_target &&
            strcmp((*rows)[i].source, source) == 0 &&
            strcmp((*rows)[i].target, target) == 0 &&
            rg_context_spec_is_subset(&(*rows)[i].context, context, &a_subset_b) == RG_OK &&
            rg_context_spec_is_subset(context, &(*rows)[i].context, &b_subset_a) == RG_OK &&
            a_subset_b && b_subset_a) {
            (*rows)[i].count += weight;
            (*rows)[i].source_total = source_total;
            (*rows)[i].contrast_count = contrast_count;
            (*rows)[i].contrast_total = contrast_total;
            (*rows)[i].delta_bic = delta_bic;
            (*rows)[i].uncertainty = rg_wilson_default_internal((*rows)[i].count, source_total);
            return RG_OK;
        }
    }
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 8 : *cap * 2;
        next = (rg_conditioned_segment_count_row *)realloc(*rows, next_cap * sizeof(**rows));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *rows = next;
        *cap = next_cap;
    }
    memset(&(*rows)[*count], 0, sizeof((*rows)[*count]));
    (*rows)[*count].source = rg_strdup_internal(source);
    (*rows)[*count].target = rg_strdup_internal(target);
    (*rows)[*count].context_is_target = context_is_target;
    (*rows)[*count].count = weight;
    (*rows)[*count].source_total = source_total;
    (*rows)[*count].contrast_count = contrast_count;
    (*rows)[*count].contrast_total = contrast_total;
    (*rows)[*count].delta_bic = delta_bic;
    (*rows)[*count].uncertainty = rg_wilson_default_internal(weight, source_total);
    if ((*rows)[*count].source == 0 || (*rows)[*count].target == 0) {
        conditioned_segment_count_row_clear(&(*rows)[*count]);
        return RG_ERR_OOM;
    }
    if (rg_context_spec_copy_internal(context, &(*rows)[*count].context) != RG_OK) {
        conditioned_segment_count_row_clear(&(*rows)[*count]);
        return RG_ERR_OOM;
    }
    (*count)++;
    return RG_OK;
}


static int displacement_vector_equal(
    const rg_feature_displacement *a,
    size_t a_count,
    const rg_feature_displacement *b,
    size_t b_count
) {
    size_t i;
    if (a_count != b_count) {
        return 0;
    }
    for (i = 0; i < a_count; i++) {
        if (strcmp(a[i].feature, b[i].feature) != 0 ||
            strcmp(a[i].from_value, b[i].from_value) != 0 ||
            strcmp(a[i].to_value, b[i].to_value) != 0) {
            return 0;
        }
    }
    return 1;
}

static int displacement_row_cmp(const void *a, const void *b) {
    const rg_displacement_row *ra = (const rg_displacement_row *)a;
    const rg_displacement_row *rb = (const rg_displacement_row *)b;
    size_t i;
    for (i = 0; i < ra->item_count && i < rb->item_count; i++) {
        int c = strcmp(ra->items[i].feature, rb->items[i].feature);
        if (c != 0) {
            return c;
        }
        c = strcmp(ra->items[i].from_value, rb->items[i].from_value);
        if (c != 0) {
            return c;
        }
        c = strcmp(ra->items[i].to_value, rb->items[i].to_value);
        if (c != 0) {
            return c;
        }
    }
    if (ra->item_count != rb->item_count) {
        return ra->item_count < rb->item_count ? -1 : 1;
    }
    return 0;
}

/* Counts one whole displacement vector. The vector, not the individual feature
 * changes, is the unit of observation: "voiced lost and fricative gained" is a
 * different event from either change alone. */
static rg_status add_displacement_vector(
    rg_displacement_row **rows,
    size_t *count,
    size_t *cap,
    const rg_feature_displacement *items,
    size_t item_count,
    double weight
) {
    size_t i;
    rg_feature_displacement *copy;
    for (i = 0; i < *count; i++) {
        if (displacement_vector_equal((*rows)[i].items, (*rows)[i].item_count, items, item_count)) {
            (*rows)[i].count += weight;
            return RG_OK;
        }
    }
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 8 : *cap * 2;
        rg_displacement_row *next = (rg_displacement_row *)realloc(*rows, next_cap * sizeof(**rows));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *rows = next;
        *cap = next_cap;
    }
    memset(&(*rows)[*count], 0, sizeof((*rows)[*count]));
    copy = (rg_feature_displacement *)calloc(item_count == 0 ? 1 : item_count, sizeof(*copy));
    if (copy == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < item_count; i++) {
        copy[i].feature = rg_strdup_internal(items[i].feature);
        copy[i].from_value = rg_strdup_internal(items[i].from_value);
        copy[i].to_value = rg_strdup_internal(items[i].to_value);
        if (copy[i].feature == 0 || copy[i].from_value == 0 || copy[i].to_value == 0) {
            size_t j;
            for (j = 0; j <= i; j++) {
                free((char *)copy[j].feature);
                free((char *)copy[j].from_value);
                free((char *)copy[j].to_value);
            }
            free(copy);
            return RG_ERR_OOM;
        }
    }
    (*rows)[*count].items = copy;
    (*rows)[*count].item_count = item_count;
    (*rows)[*count].count = weight;
    (*count)++;
    return RG_OK;
}

static int tonal_row_cmp(const void *a, const void *b) {
    const rg_tonal_count_row *ra = (const rg_tonal_count_row *)a;
    const rg_tonal_count_row *rb = (const rg_tonal_count_row *)b;
    int c = strcmp(ra->source_tone, rb->source_tone);
    if (c != 0) {
        return c;
    }
    return strcmp(ra->target_tone, rb->target_tone);
}

static rg_status add_tonal_count(
    rg_tonal_count_row **rows,
    size_t *count,
    size_t *cap,
    const char *source_tone,
    const char *target_tone,
    double weight
) {
    size_t i;
    rg_tonal_count_row *next;
    const char *src = source_tone == 0 ? "" : source_tone;
    const char *tgt = target_tone == 0 ? "" : target_tone;
    if (src[0] == '\0' && tgt[0] == '\0') {
        return RG_OK;
    }
    for (i = 0; i < *count; i++) {
        if (strcmp((*rows)[i].source_tone, src) == 0 && strcmp((*rows)[i].target_tone, tgt) == 0) {
            (*rows)[i].count += weight;
            return RG_OK;
        }
    }
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 8 : *cap * 2;
        next = (rg_tonal_count_row *)realloc(*rows, next_cap * sizeof(**rows));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *rows = next;
        *cap = next_cap;
    }
    (*rows)[*count].source_tone = rg_strdup_internal(src);
    (*rows)[*count].target_tone = rg_strdup_internal(tgt);
    (*rows)[*count].count = weight;
    (*rows)[*count].source_total = 0.0;
    if ((*rows)[*count].source_tone == 0 || (*rows)[*count].target_tone == 0) {
        free((char *)(*rows)[*count].source_tone);
        free((char *)(*rows)[*count].target_tone);
        return RG_ERR_OOM;
    }
    (*count)++;
    return RG_OK;
}

static void fill_totals(rg_segment_count_row *rows, size_t count) {
    size_t i;
    size_t j;
    for (i = 0; i < count; i++) {
        double source_total = 0.0;
        double target_total = 0.0;
        for (j = 0; j < count; j++) {
            if (strcmp(rows[i].source, rows[j].source) == 0) {
                source_total += rows[j].count;
            }
            if (strcmp(rows[i].target, rows[j].target) == 0) {
                target_total += rows[j].count;
            }
        }
        rows[i].source_total = source_total;
        rows[i].target_total = target_total;
        rows[i].uncertainty = rg_wilson_default_internal(rows[i].count, source_total);
    }
}

/* Publishes the target totals onto the normalizer table, which the scorer can
 * binary-search by a single grapheme. Recomputed with the counts, since EM
 * rebuilds them every iteration. */
static void publish_target_totals(rg_pairwise_model *model, const rg_segment_count_row *rows, size_t count) {
    size_t i;
    for (i = 0; i < model->log_normalizer_count; i++) {
        model->log_normalizers[i].target_total = 0.0;
    }
    for (i = 0; i < count; i++) {
        size_t low = 0;
        size_t high = model->log_normalizer_count;
        while (low < high) {
            size_t mid = low + (high - low) / 2;
            int c = strcmp(model->log_normalizers[mid].source, rows[i].target);
            if (c == 0) {
                model->log_normalizers[mid].target_total += rows[i].count;
                break;
            }
            if (c < 0) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
    }
}

static void fill_displacement_total(rg_displacement_row *rows, size_t count) {
    double total = 0.0;
    size_t i;
    for (i = 0; i < count; i++) {
        total += rows[i].count;
    }
    for (i = 0; i < count; i++) {
        rows[i].total = total;
        rows[i].uncertainty = rg_wilson_default_internal(rows[i].count, total);
    }
}

static void fill_tonal_source_totals(rg_tonal_count_row *rows, size_t count) {
    size_t i;
    size_t j;
    for (i = 0; i < count; i++) {
        double total = 0.0;
        for (j = 0; j < count; j++) {
            if (strcmp(rows[i].source_tone, rows[j].source_tone) == 0) {
                total += rows[j].count;
            }
        }
        rows[i].source_total = total;
        rows[i].uncertainty = rg_wilson_default_internal(rows[i].count, total);
    }
}

typedef struct context_observation {
    char *source;
    char *target;
    rg_context_spec context;
    double weight;
} context_observation;

typedef rg_split_candidate split_candidate;

static const char *const cross_dimensional_feature_names[] = {
    "back",
    "close",
    "consonant",
    "fricative",
    "front",
    "long",
    "nasal",
    "open",
    "sonorant",
    "stop",
    "voiced",
    "voiceless",
    "vowel"
};

static void context_observation_clear(context_observation *obs) {
    if (obs == 0) {
        return;
    }
    free(obs->source);
    free(obs->target);
    rg_context_spec_clear_internal(&obs->context);
    obs->source = 0;
    obs->target = 0;
    obs->weight = 0.0;
}

/* `swap` records the link from the target's point of view: the target grapheme
 * becomes the thing being conditioned and the source grapheme the outcome, and
 * the caller supplies the target form's context. Everything downstream --
 * bucketing, splitting, refinement -- then works unchanged, and the roles are
 * put back when the row is written. */
static rg_status append_context_observation_as(
    context_observation **items,
    size_t *count,
    size_t *cap,
    const rg_link *link,
    const rg_context_spec *context,
    int swap,
    double weight
) {
    context_observation *next;
    rg_status status;
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 32 : *cap * 2;
        next = (context_observation *)realloc(*items, next_cap * sizeof(**items));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *items = next;
        *cap = next_cap;
    }
    memset(&(*items)[*count], 0, sizeof((*items)[*count]));
    (*items)[*count].source = rg_strdup_internal(
        swap ? link->target[0].grapheme : link->source[0].grapheme);
    (*items)[*count].target = rg_strdup_internal(
        swap ? link->source[0].grapheme : link->target[0].grapheme);
    (*items)[*count].weight = weight;
    if ((*items)[*count].source == 0 || (*items)[*count].target == 0) {
        context_observation_clear(&(*items)[*count]);
        return RG_ERR_OOM;
    }
    status = rg_context_spec_copy_internal(context, &(*items)[*count].context);
    if (status != RG_OK) {
        context_observation_clear(&(*items)[*count]);
        return status;
    }
    (*count)++;
    return RG_OK;
}

static int context_has_constraint(const rg_feature_constraint *items, size_t count, const char *feature, const char *value) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (strcmp(items[i].feature, feature) == 0 && strcmp(items[i].value, value) == 0) {
            return 1;
        }
    }
    return 0;
}

static int context_has_distance_constraint(const rg_distance_constraint *items, size_t count, int offset, const char *feature, const char *value) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (items[i].offset == offset &&
            strcmp(items[i].constraint.feature, feature) == 0 &&
            strcmp(items[i].constraint.value, value) == 0) {
            return 1;
        }
    }
    return 0;
}

int rg_predicate_holds_internal(const rg_context_spec *context, const rg_split_candidate *candidate) {
    if (strcmp(candidate->slot, "following") == 0) {
        return context_has_constraint(context->following, context->following_count, candidate->feature, candidate->value);
    }
    if (strcmp(candidate->slot, "preceding") == 0) {
        return context_has_constraint(context->preceding, context->preceding_count, candidate->feature, candidate->value);
    }
    if (strcmp(candidate->slot, "position") == 0) {
        return context->position != 0 && strcmp(context->position, candidate->feature) == 0;
    }
    if (strcmp(candidate->slot, "preceding@2") == 0) {
        return context_has_distance_constraint(context->preceding_at_distance, context->preceding_at_distance_count, 2, candidate->feature, candidate->value);
    }
    if (strcmp(candidate->slot, "preceding@3") == 0) {
        return context_has_distance_constraint(context->preceding_at_distance, context->preceding_at_distance_count, 3, candidate->feature, candidate->value);
    }
    if (strcmp(candidate->slot, "following@2") == 0) {
        return context_has_distance_constraint(context->following_at_distance, context->following_at_distance_count, 2, candidate->feature, candidate->value);
    }
    if (strcmp(candidate->slot, "following@3") == 0) {
        return context_has_distance_constraint(context->following_at_distance, context->following_at_distance_count, 3, candidate->feature, candidate->value);
    }
    if (strcmp(candidate->slot, "somewhere_preceding") == 0) {
        return context_has_constraint(context->somewhere_preceding, context->somewhere_preceding_count, candidate->feature, candidate->value);
    }
    if (strcmp(candidate->slot, "somewhere_following") == 0) {
        return context_has_constraint(context->somewhere_following, context->somewhere_following_count, candidate->feature, candidate->value);
    }
    if (strcmp(candidate->slot, "same_syllable") == 0) {
        return context_has_constraint(context->same_syllable, context->same_syllable_count, candidate->feature, candidate->value);
    }
    if (strcmp(candidate->slot, "next_syllable") == 0) {
        return context_has_constraint(context->next_syllable, context->next_syllable_count, candidate->feature, candidate->value);
    }
    if (strcmp(candidate->slot, "previous_syllable") == 0) {
        return context_has_constraint(context->previous_syllable, context->previous_syllable_count, candidate->feature, candidate->value);
    }
    if (strcmp(candidate->slot, "self_stress") == 0) {
        return context_has_constraint(context->self_stress, context->self_stress_count, candidate->feature, candidate->value);
    }
    if (strcmp(candidate->slot, "preceding_stress") == 0) {
        return context_has_constraint(context->preceding_stress, context->preceding_stress_count, candidate->feature, candidate->value);
    }
    if (strcmp(candidate->slot, "following_stress") == 0) {
        return context_has_constraint(context->following_stress, context->following_stress_count, candidate->feature, candidate->value);
    }
    return 0;
}

rg_status rg_context_from_candidate_internal(const rg_split_candidate *candidate, rg_context_spec *out) {
    rg_feature_constraint constraint;
    rg_distance_constraint distance;
    rg_status status;
    rg_context_spec_init_empty(out);
    if (strcmp(candidate->slot, "position") == 0) {
        out->position = rg_strdup_internal(candidate->feature);
        return out->position == 0 ? RG_ERR_OOM : RG_OK;
    }
    constraint.feature = candidate->feature;
    constraint.value = candidate->value;
    if (strcmp(candidate->slot, "following") == 0) {
        status = rg_feature_constraint_array_copy_internal(&constraint, 1, &out->following);
        if (status == RG_OK) {
            out->following_count = 1;
        }
        return status;
    }
    if (strcmp(candidate->slot, "preceding") == 0) {
        status = rg_feature_constraint_array_copy_internal(&constraint, 1, &out->preceding);
        if (status == RG_OK) {
            out->preceding_count = 1;
        }
        return status;
    }
    if (strcmp(candidate->slot, "somewhere_preceding") == 0) {
        status = rg_feature_constraint_array_copy_internal(&constraint, 1, &out->somewhere_preceding);
        if (status == RG_OK) {
            out->somewhere_preceding_count = 1;
        }
        return status;
    }
    if (strcmp(candidate->slot, "somewhere_following") == 0) {
        status = rg_feature_constraint_array_copy_internal(&constraint, 1, &out->somewhere_following);
        if (status == RG_OK) {
            out->somewhere_following_count = 1;
        }
        return status;
    }
    if (strcmp(candidate->slot, "same_syllable") == 0) {
        status = rg_feature_constraint_array_copy_internal(&constraint, 1, &out->same_syllable);
        if (status == RG_OK) {
            out->same_syllable_count = 1;
        }
        return status;
    }
    if (strcmp(candidate->slot, "next_syllable") == 0) {
        status = rg_feature_constraint_array_copy_internal(&constraint, 1, &out->next_syllable);
        if (status == RG_OK) {
            out->next_syllable_count = 1;
        }
        return status;
    }
    if (strcmp(candidate->slot, "previous_syllable") == 0) {
        status = rg_feature_constraint_array_copy_internal(&constraint, 1, &out->previous_syllable);
        if (status == RG_OK) {
            out->previous_syllable_count = 1;
        }
        return status;
    }
    if (strcmp(candidate->slot, "self_stress") == 0) {
        status = rg_feature_constraint_array_copy_internal(&constraint, 1, &out->self_stress);
        if (status == RG_OK) {
            out->self_stress_count = 1;
        }
        return status;
    }
    if (strcmp(candidate->slot, "preceding_stress") == 0) {
        status = rg_feature_constraint_array_copy_internal(&constraint, 1, &out->preceding_stress);
        if (status == RG_OK) {
            out->preceding_stress_count = 1;
        }
        return status;
    }
    if (strcmp(candidate->slot, "following_stress") == 0) {
        status = rg_feature_constraint_array_copy_internal(&constraint, 1, &out->following_stress);
        if (status == RG_OK) {
            out->following_stress_count = 1;
        }
        return status;
    }
    if (strcmp(candidate->slot, "preceding@2") == 0 || strcmp(candidate->slot, "preceding@3") == 0) {
        distance.offset = strcmp(candidate->slot, "preceding@2") == 0 ? 2 : 3;
        distance.constraint = constraint;
        {
            rg_distance_constraint *items = (rg_distance_constraint *)calloc(1, sizeof(*items));
            if (items == 0) {
                return RG_ERR_OOM;
            }
            items[0].offset = distance.offset;
            status = rg_feature_constraint_copy_internal(&distance.constraint, &items[0].constraint);
            if (status != RG_OK) {
                free(items);
                return status;
            }
            out->preceding_at_distance = items;
            out->preceding_at_distance_count = 1;
        }
        return RG_OK;
    }
    if (strcmp(candidate->slot, "following@2") == 0 || strcmp(candidate->slot, "following@3") == 0) {
        distance.offset = strcmp(candidate->slot, "following@2") == 0 ? 2 : 3;
        distance.constraint = constraint;
        {
            rg_distance_constraint *items = (rg_distance_constraint *)calloc(1, sizeof(*items));
            if (items == 0) {
                return RG_ERR_OOM;
            }
            items[0].offset = distance.offset;
            status = rg_feature_constraint_copy_internal(&distance.constraint, &items[0].constraint);
            if (status != RG_OK) {
                free(items);
                return status;
            }
            out->following_at_distance = items;
            out->following_at_distance_count = 1;
        }
        return RG_OK;
    }
    return RG_ERR_INVALID_ARGUMENT;
}

typedef struct target_mass {
    const char *target;
    double mass;
} target_mass;

static rg_status add_target_mass(target_mass **items, size_t *count, size_t *cap, const char *target, double weight) {
    size_t i;
    target_mass *next;
    for (i = 0; i < *count; i++) {
        if (strcmp((*items)[i].target, target) == 0) {
            (*items)[i].mass += weight;
            return RG_OK;
        }
    }
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 4 : *cap * 2;
        next = (target_mass *)realloc(*items, next_cap * sizeof(**items));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *items = next;
        *cap = next_cap;
    }
    (*items)[*count].target = target;
    (*items)[*count].mass = weight;
    (*count)++;
    return RG_OK;
}

static int string_ptr_cmp(const void *a, const void *b) {
    const char *const *sa = (const char *const *)a;
    const char *const *sb = (const char *const *)b;
    return strcmp(*sa, *sb);
}

static rg_status append_unique_source(const char ***items, size_t *count, size_t *cap, const char *source) {
    size_t i;
    const char **next;
    for (i = 0; i < *count; i++) {
        if (strcmp((*items)[i], source) == 0) {
            return RG_OK;
        }
    }
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 16 : *cap * 2;
        next = (const char **)realloc(*items, next_cap * sizeof(**items));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *items = next;
        *cap = next_cap;
    }
    (*items)[(*count)++] = source;
    return RG_OK;
}

/* ---- pairwise context discovery ---------------------------------------- */


static const char *const split_positions[] = {"initial", "medial", "final"};

static const char *const stress_slot_names[] = {"self_stress", "preceding_stress", "following_stress"};


static const char *const long_range_slot_names[] = {
    "same_syllable",
    "next_syllable",
    "previous_syllable",
    "preceding@2",
    "preceding@3",
    "following@2",
    "following@3",
    "somewhere_preceding",
    "somewhere_following"
};

typedef struct stress_inventory {
    char **values;
    size_t count;
    size_t cap;
} stress_inventory;

static void stress_inventory_clear(stress_inventory *inventory) {
    size_t i;
    for (i = 0; i < inventory->count; i++) {
        free(inventory->values[i]);
    }
    free(inventory->values);
    memset(inventory, 0, sizeof(*inventory));
}

static rg_status stress_inventory_add(stress_inventory *inventory, const char *value) {
    size_t i;
    size_t insert_at;
    for (i = 0; i < inventory->count; i++) {
        if (strcmp(inventory->values[i], value) == 0) {
            return RG_OK;
        }
    }
    if (inventory->count == inventory->cap) {
        size_t next_cap = inventory->cap == 0 ? 4 : inventory->cap * 2;
        char **next = (char **)realloc(inventory->values, next_cap * sizeof(*next));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        inventory->values = next;
        inventory->cap = next_cap;
    }
    insert_at = inventory->count;
    while (insert_at > 0 && strcmp(inventory->values[insert_at - 1], value) > 0) {
        insert_at--;
    }
    if (insert_at < inventory->count) {
        memmove(&inventory->values[insert_at + 1], &inventory->values[insert_at],
                (inventory->count - insert_at) * sizeof(*inventory->values));
    }
    inventory->values[insert_at] = rg_strdup_internal(value);
    if (inventory->values[insert_at] == 0) {
        return RG_ERR_OOM;
    }
    inventory->count++;
    return RG_OK;
}

static rg_status collect_observed_stress(stress_inventory *inventory, const rg_context_spec *context) {
    const rg_feature_constraint *slots[3];
    size_t counts[3];
    size_t s;
    slots[0] = context->self_stress;
    counts[0] = context->self_stress_count;
    slots[1] = context->preceding_stress;
    counts[1] = context->preceding_stress_count;
    slots[2] = context->following_stress;
    counts[2] = context->following_stress_count;
    for (s = 0; s < 3; s++) {
        size_t i;
        for (i = 0; i < counts[s]; i++) {
            if (slots[s][i].feature != 0 && strcmp(slots[s][i].feature, "stress") == 0 && slots[s][i].value != 0) {
                rg_status status = stress_inventory_add(inventory, slots[s][i].value);
                if (status != RG_OK) {
                    return status;
                }
            }
        }
    }
    return RG_OK;
}

/* Candidate axes not already constrained by base_context. A feature is dropped
 * from the preceding/following slots once that slot constrains it, the position
 * axis disappears once a position is fixed, and a stress value disappears once
 * that slot already carries it. */
static size_t immediate_candidates_for(
    const rg_context_spec *base_context,
    const stress_inventory *stress,
    const rg_feature_vocabulary *vocabulary,
    split_candidate *out,
    size_t capacity
) {
    size_t count = 0;
    size_t i;
    size_t s;
    for (i = 0; i < 2 * rg_context_feature_name_count; i++) {
        split_candidate generated;
        const split_candidate *candidate = &generated;
        int skip = 0;
        size_t feature_index = i / 2;
        if (!vocabulary->contrastive[feature_index]) {
            continue;
        }
        generated.slot = (i % 2) == 0 ? "preceding" : "following";
        generated.feature = rg_context_feature_names[feature_index];
        generated.value = "+";
        if (strcmp(candidate->slot, "following") == 0) {
            size_t j;
            for (j = 0; j < base_context->following_count; j++) {
                if (strcmp(base_context->following[j].feature, candidate->feature) == 0) {
                    skip = 1;
                    break;
                }
            }
        } else if (strcmp(candidate->slot, "preceding") == 0) {
            size_t j;
            for (j = 0; j < base_context->preceding_count; j++) {
                if (strcmp(base_context->preceding[j].feature, candidate->feature) == 0) {
                    skip = 1;
                    break;
                }
            }
        }
        if (!skip && count < capacity) {
            out[count++] = *candidate;
        }
    }
    if (base_context->position == 0 || base_context->position[0] == '\0') {
        for (i = 0; i < sizeof(split_positions) / sizeof(split_positions[0]); i++) {
            if (count < capacity) {
                out[count].slot = "position";
                out[count].feature = split_positions[i];
                out[count].value = "+";
                count++;
            }
        }
    }
    for (s = 0; s < sizeof(stress_slot_names) / sizeof(stress_slot_names[0]); s++) {
        const rg_feature_constraint *existing = 0;
        size_t existing_count = 0;
        if (s == 0) {
            existing = base_context->self_stress;
            existing_count = base_context->self_stress_count;
        } else if (s == 1) {
            existing = base_context->preceding_stress;
            existing_count = base_context->preceding_stress_count;
        } else {
            existing = base_context->following_stress;
            existing_count = base_context->following_stress_count;
        }
        for (i = 0; i < stress->count; i++) {
            int skip = 0;
            size_t j;
            for (j = 0; j < existing_count; j++) {
                if (existing[j].feature != 0 && strcmp(existing[j].feature, "stress") == 0 &&
                    strcmp(existing[j].value, stress->values[i]) == 0) {
                    skip = 1;
                    break;
                }
            }
            if (!skip && count < capacity) {
                out[count].slot = stress_slot_names[s];
                out[count].feature = "stress";
                out[count].value = stress->values[i];
                count++;
            }
        }
    }
    return count;
}

static size_t long_range_candidates(
    const rg_feature_vocabulary *vocabulary,
    split_candidate *out,
    size_t capacity
) {
    size_t count = 0;
    size_t s;
    size_t f;
    for (s = 0; s < sizeof(long_range_slot_names) / sizeof(long_range_slot_names[0]); s++) {
        for (f = 0; f < rg_context_feature_name_count; f++) {
            if (!vocabulary->contrastive[f]) {
                continue;
            }
            if (count < capacity) {
                out[count].slot = long_range_slot_names[s];
                out[count].feature = rg_context_feature_names[f];
                out[count].value = "+";
                count++;
            }
        }
    }
    return count;
}

/* base_context extended with one more constraint. Contexts are immutable by
 * convention, so this always allocates a fresh value. */
/* Conjoins one more predicate onto a context. The multi-lect stage needs the
 * same operation the pairwise refinement does, and a conditioning environment
 * built from two predicates is one context, not two rules. */
rg_status rg_context_extend_internal(
    const rg_context_spec *base_context,
    const split_candidate *candidate,
    rg_context_spec *out
) {
    rg_feature_constraint *merged = 0;
    const rg_feature_constraint **slot = 0;
    size_t *slot_count = 0;
    const rg_feature_constraint *existing = 0;
    size_t existing_count = 0;
    rg_feature_constraint addition;
    rg_status status;

    status = rg_context_spec_copy_internal(base_context, out);
    if (status != RG_OK) {
        return status;
    }
    if (strcmp(candidate->slot, "position") == 0) {
        free((char *)out->position);
        out->position = rg_strdup_internal(candidate->feature);
        if (out->position == 0) {
            rg_context_spec_clear_internal(out);
            return RG_ERR_OOM;
        }
        return RG_OK;
    }
    addition.feature = candidate->feature;
    addition.value = candidate->value;

#define PICK(name, field)                                     \
    if (strcmp(candidate->slot, name) == 0) {                 \
        slot = &out->field;                                   \
        slot_count = &out->field##_count;                     \
        existing = out->field;                                \
        existing_count = out->field##_count;                  \
    }

    PICK("preceding", preceding)
    PICK("following", following)
    PICK("somewhere_preceding", somewhere_preceding)
    PICK("somewhere_following", somewhere_following)
    PICK("same_syllable", same_syllable)
    PICK("next_syllable", next_syllable)
    PICK("previous_syllable", previous_syllable)
    PICK("self_stress", self_stress)
    PICK("preceding_stress", preceding_stress)
    PICK("following_stress", following_stress)

#undef PICK

    if (slot == 0) {
        rg_distance_constraint *items;
        int offset;
        size_t base_count;
        const rg_distance_constraint *base_items;
        size_t i;
        int preceding;
        if (strncmp(candidate->slot, "preceding@", 10) == 0) {
            preceding = 1;
            offset = atoi(candidate->slot + 10);
            base_items = out->preceding_at_distance;
            base_count = out->preceding_at_distance_count;
        } else if (strncmp(candidate->slot, "following@", 10) == 0) {
            preceding = 0;
            offset = atoi(candidate->slot + 10);
            base_items = out->following_at_distance;
            base_count = out->following_at_distance_count;
        } else {
            rg_context_spec_clear_internal(out);
            return RG_ERR_INVALID_ARGUMENT;
        }
        items = (rg_distance_constraint *)calloc(base_count + 1, sizeof(*items));
        if (items == 0) {
            rg_context_spec_clear_internal(out);
            return RG_ERR_OOM;
        }
        for (i = 0; i < base_count; i++) {
            items[i].offset = base_items[i].offset;
            if (rg_feature_constraint_copy_internal(&base_items[i].constraint, &items[i].constraint) != RG_OK) {
                rg_distance_constraint_array_clear_internal(items, i);
                rg_context_spec_clear_internal(out);
                return RG_ERR_OOM;
            }
        }
        items[base_count].offset = offset;
        if (rg_feature_constraint_copy_internal(&addition, &items[base_count].constraint) != RG_OK) {
            rg_distance_constraint_array_clear_internal(items, base_count);
            rg_context_spec_clear_internal(out);
            return RG_ERR_OOM;
        }
        if (preceding) {
            rg_distance_constraint_array_clear_internal((rg_distance_constraint *)out->preceding_at_distance, out->preceding_at_distance_count);
            out->preceding_at_distance = items;
            out->preceding_at_distance_count = base_count + 1;
        } else {
            rg_distance_constraint_array_clear_internal((rg_distance_constraint *)out->following_at_distance, out->following_at_distance_count);
            out->following_at_distance = items;
            out->following_at_distance_count = base_count + 1;
        }
        return RG_OK;
    }

    merged = (rg_feature_constraint *)calloc(existing_count + 1, sizeof(*merged));
    if (merged == 0) {
        rg_context_spec_clear_internal(out);
        return RG_ERR_OOM;
    }
    {
        size_t i;
        for (i = 0; i < existing_count; i++) {
            if (rg_feature_constraint_copy_internal(&existing[i], &merged[i]) != RG_OK) {
                rg_feature_constraint_array_clear_internal(merged, i);
                rg_context_spec_clear_internal(out);
                return RG_ERR_OOM;
            }
        }
        if (rg_feature_constraint_copy_internal(&addition, &merged[existing_count]) != RG_OK) {
            rg_feature_constraint_array_clear_internal(merged, existing_count);
            rg_context_spec_clear_internal(out);
            return RG_ERR_OOM;
        }
    }
    rg_feature_constraint_array_clear_internal(existing, existing_count);
    *slot = merged;
    *slot_count = existing_count + 1;
    return RG_OK;
}

/* Negative log-likelihood of a group under a single unconditioned
 * correspondence. Target keys are summed in sorted order so repeated runs
 * produce bit-identical results. */
static double observation_group_cost(const context_observation *const *rows, size_t count) {
    target_mass *targets = 0;
    size_t target_count = 0;
    size_t target_cap = 0;
    double total = 0.0;
    double cost = 0.0;
    size_t i;
    size_t j;

    for (i = 0; i < count; i++) {
        if (add_target_mass(&targets, &target_count, &target_cap, rows[i]->target, rows[i]->weight) != RG_OK) {
            free(targets);
            return 0.0;
        }
        total += rows[i]->weight;
    }
    if (total <= 0.0) {
        free(targets);
        return 0.0;
    }
    /* add_target_mass appends in first-seen order; sort by target for a
     * deterministic summation order. */
    for (i = 1; i < target_count; i++) {
        target_mass key = targets[i];
        j = i;
        while (j > 0 && strcmp(targets[j - 1].target, key.target) > 0) {
            targets[j] = targets[j - 1];
            j--;
        }
        targets[j] = key;
    }
    for (i = 0; i < target_count; i++) {
        double p = targets[i].mass / total;
        if (p > 0.0) {
            cost += -targets[i].mass * log(p);
        }
    }
    free(targets);
    return cost;
}

static double observation_total_weight(const context_observation *const *rows, size_t count) {
    double total = 0.0;
    size_t i;
    for (i = 0; i < count; i++) {
        total += rows[i]->weight;
    }
    return total;
}

static double observation_dominant_fraction(const context_observation *const *rows, size_t count) {
    target_mass *targets = 0;
    size_t target_count = 0;
    size_t target_cap = 0;
    double total = 0.0;
    double mode = 0.0;
    size_t i;
    for (i = 0; i < count; i++) {
        if (add_target_mass(&targets, &target_count, &target_cap, rows[i]->target, rows[i]->weight) != RG_OK) {
            free(targets);
            return 0.0;
        }
        total += rows[i]->weight;
    }
    for (i = 0; i < target_count; i++) {
        if (targets[i].mass > mode) {
            mode = targets[i].mass;
        }
    }
    free(targets);
    return total > 0.0 ? mode / total : 0.0;
}

/* Writes one conditioned entry per observed target in this group. Repeating a
 * (source, target, context) key replaces the previous count rather than adding
 * to it, because each commit states the mass of that group outright. */
static rg_status commit_observation_group(
    rg_pairwise_model *model,
    const char *source,
    const context_observation *const *rows,
    size_t count,
    const rg_context_spec *context,
    int target_side,
    double bucket_total,
    const context_observation *const *contrast_rows,
    size_t contrast_row_count,
    double delta_bic
) {
    target_mass *targets = 0;
    target_mass *contrast_targets = 0;
    size_t contrast_target_count = 0;
    size_t contrast_target_cap = 0;
    double contrast_total = 0.0;
    size_t target_count = 0;
    size_t target_cap = 0;
    size_t conditioned_cap = model->conditioned_segment_count_count;
    /* On the target side the bucket key is a target grapheme, which the segment
     * table is not keyed by, so the denominator is the bucket's own mass. */
    double source_total = target_side
        ? bucket_total
        : source_total_for_rows(model->segment_counts, model->segment_count_count, source);
    size_t i;
    rg_status status = RG_OK;

    for (i = 0; i < count; i++) {
        status = add_target_mass(&targets, &target_count, &target_cap, rows[i]->target, rows[i]->weight);
        if (status != RG_OK) {
            free(targets);
            return status;
        }
    }
    /* The same correspondence where the environment does not hold. A rule
     * published without it cannot be read. */
    for (i = 0; i < contrast_row_count; i++) {
        status = add_target_mass(&contrast_targets, &contrast_target_count, &contrast_target_cap,
                                 contrast_rows[i]->target, contrast_rows[i]->weight);
        if (status != RG_OK) {
            free(targets);
            free(contrast_targets);
            return status;
        }
        contrast_total += contrast_rows[i]->weight;
    }
    for (i = 0; i < target_count && status == RG_OK; i++) {
        size_t existing;
        int replaced = 0;
        double contrast_count = 0.0;
        for (existing = 0; existing < contrast_target_count; existing++) {
            if (strcmp(contrast_targets[existing].target, targets[i].target) == 0) {
                contrast_count = contrast_targets[existing].mass;
                break;
            }
        }
        for (existing = 0; existing < model->conditioned_segment_count_count; existing++) {
            rg_conditioned_segment_count_row *row = &model->conditioned_segment_counts[existing];
            int a_subset_b = 0;
            int b_subset_a = 0;
            const char *want_source = target_side ? targets[i].target : source;
            const char *want_target = target_side ? source : targets[i].target;
            if (row->context_is_target != target_side ||
                strcmp(row->source, want_source) != 0 ||
                strcmp(row->target, want_target) != 0) {
                continue;
            }
            if (rg_context_spec_is_subset(&row->context, context, &a_subset_b) == RG_OK &&
                rg_context_spec_is_subset(context, &row->context, &b_subset_a) == RG_OK &&
                a_subset_b && b_subset_a) {
                row->count = targets[i].mass;
                row->source_total = source_total;
                row->contrast_count = contrast_count;
                row->contrast_total = contrast_total;
                row->delta_bic = delta_bic;
                row->uncertainty = rg_wilson_default_internal(row->count, source_total);
                replaced = 1;
                break;
            }
        }
        if (replaced) {
            continue;
        }
        if (target_side) {
            model->has_target_conditioned = 1;
        }
        status = add_conditioned_segment_count(
            &model->conditioned_segment_counts,
            &model->conditioned_segment_count_count,
            &conditioned_cap,
            target_side ? targets[i].target : source,
            target_side ? source : targets[i].target,
            context,
            target_side,
            targets[i].mass,
            source_total,
            contrast_count,
            contrast_total,
            delta_bic
        );
    }
    free(targets);
    free(contrast_targets);
    return status;
}

typedef struct split_search {
    const context_observation **yes;
    const context_observation **no;
    const context_observation **best_yes;
    const context_observation **best_no;
    split_candidate *candidates;
    size_t capacity;
} split_search;

static void split_search_clear(split_search *search) {
    free(search->yes);
    free(search->no);
    free(search->best_yes);
    free(search->best_no);
    free(search->candidates);
    memset(search, 0, sizeof(*search));
}

static rg_status split_search_init(split_search *search, size_t observation_capacity, size_t candidate_capacity) {
    memset(search, 0, sizeof(*search));
    search->capacity = candidate_capacity;
    search->yes = (const context_observation **)calloc(observation_capacity == 0 ? 1 : observation_capacity, sizeof(*search->yes));
    search->no = (const context_observation **)calloc(observation_capacity == 0 ? 1 : observation_capacity, sizeof(*search->no));
    search->best_yes = (const context_observation **)calloc(observation_capacity == 0 ? 1 : observation_capacity, sizeof(*search->best_yes));
    search->best_no = (const context_observation **)calloc(observation_capacity == 0 ? 1 : observation_capacity, sizeof(*search->best_no));
    search->candidates = (split_candidate *)calloc(candidate_capacity == 0 ? 1 : candidate_capacity, sizeof(*search->candidates));
    if (search->yes == 0 || search->no == 0 || search->best_yes == 0 || search->best_no == 0 || search->candidates == 0) {
        split_search_clear(search);
        return RG_ERR_OOM;
    }
    return RG_OK;
}

/* Finds the BIC-best split of rows over the given candidates. Returns 1 when a
 * split beats the threshold, filling best_yes/best_no and best_candidate. */
static int find_best_split(
    split_search *search,
    const context_observation *const *rows,
    size_t count,
    const split_candidate *candidates,
    const rg_split_gate *gates,
    size_t candidate_count,
    double penalty,
    split_candidate *best_candidate,
    size_t *best_yes_count,
    size_t *best_no_count,
    double *best_delta_bic
) {
    double baseline = observation_group_cost(rows, count);
    /* Charge for the search, not only for the parameter.
     *
     * BIC prices one added term against the likelihood it buys. The term that
     * survives here is not one term: it is the best of candidate_count of them,
     * and the maximum of a hundred candidates beats its bar by chance far more
     * often than one candidate does. Permuting a corpus's pairings -- which
     * removes every correspondence there is to find -- used to *raise* the
     * number of committed rules, which is what an unpriced argmax looks like.
     *
     * 2*ln(candidates) is the same currency as the BIC penalty and is the
     * standard extended-BIC shape for a large model space. It is not a
     * substitute for the shuffled baseline, which measures the inflation this
     * only models. */
    double search_penalty = candidate_count > 1 ? RG_SEARCH_PENALTY_GAMMA * 2.0 * log((double)candidate_count) : 0.0;
    double best_margin = 0.0;
    int found = 0;
    size_t ci;

    for (ci = 0; ci < candidate_count; ci++) {
        double min_obs = gates[ci].min_obs;
        double delta_threshold = gates[ci].delta_threshold;
        double min_dominant_fraction = gates[ci].min_dominant_fraction;
        double margin;
        size_t yes_count = 0;
        size_t no_count = 0;
        size_t i;
        double split_cost;
        double delta_bic;
        for (i = 0; i < count; i++) {
            if (rg_predicate_holds_internal(&rows[i]->context, &candidates[ci])) {
                search->yes[yes_count++] = rows[i];
            } else {
                search->no[no_count++] = rows[i];
            }
        }
        if (observation_total_weight(search->yes, yes_count) < min_obs ||
            observation_total_weight(search->no, no_count) < min_obs) {
            continue;
        }
        if (min_dominant_fraction > 0.0 &&
            observation_dominant_fraction(search->yes, yes_count) < min_dominant_fraction) {
            continue;
        }
        split_cost = observation_group_cost(search->yes, yes_count) + observation_group_cost(search->no, no_count);
        delta_bic = -2.0 * (baseline - split_cost) + penalty + search_penalty;
        margin = delta_threshold - delta_bic;
        /* Two predicates can carve the same partition and so clear their bar by
         * the same amount. Requiring a later candidate to beat the incumbent by
         * more than the tie epsilon hands the tie to candidate order, which is
         * the same everywhere, rather than to the last bit of a log. */
        if (margin > best_margin + RG_TIE_EPSILON) {
            best_margin = margin;
            *best_delta_bic = delta_bic;
            *best_candidate = candidates[ci];
            memcpy(search->best_yes, search->yes, yes_count * sizeof(*search->yes));
            memcpy(search->best_no, search->no, no_count * sizeof(*search->no));
            *best_yes_count = yes_count;
            *best_no_count = no_count;
            found = 1;
        }
    }
    return found;
}

/* Recursively refines an already-committed conditioned entry, looking for one
 * further conditioning axis that improves BIC. */
/* Deepens a committed split by conjoining a second predicate within the group
 * that satisfied the first.
 *
 * The candidates offered here are the union of every axis, not the immediate
 * neighbours the enclosing stage happened to search. A conditioning
 * environment is not obliged to be built out of one kind of predicate:
 * Grassmann's Law is word-initial *and* followed somewhere by an aspirate, and
 * assimilation is regularly "before X" *and* "after Y". Searching one kind at
 * a time can state either half and never the conjunction.
 *
 * Candidates the base context already implies need no filtering: every row in
 * the group satisfies them, so the split has an empty complement and its own
 * minimum rejects it. */
static rg_status refine_split(
    rg_pairwise_model *model,
    const char *source,
    const context_observation *const *rows,
    size_t count,
    const rg_context_spec *base_context,
    int depth,
    int max_depth,
    double min_obs,
    const split_candidate *candidates,
    const rg_split_gate *gates,
    size_t candidate_count,
    double penalty,
    const stress_inventory *stress,
    size_t observation_capacity,
    int target_side,
    double bucket_total
) {
    split_search search;
    split_candidate best;
    size_t yes_count = 0;
    size_t no_count = 0;
    rg_context_spec yes_context;
    double delta_bic = 0.0;
    rg_status status;

    (void)stress;
    if (depth >= max_depth || observation_total_weight(rows, count) < min_obs) {
        return RG_OK;
    }
    status = split_search_init(&search, observation_capacity, 1);
    if (status != RG_OK) {
        return status;
    }
    if (!find_best_split(&search, rows, count, candidates, gates, candidate_count,
                         penalty, &best, &yes_count, &no_count, &delta_bic)) {
        split_search_clear(&search);
        return RG_OK;
    }
    status = rg_context_extend_internal(base_context, &best, &yes_context);
    if (status == RG_OK) {
        status = commit_observation_group(model, source, search.best_yes, yes_count, &yes_context,
                                          target_side, bucket_total,
                                          search.best_no, no_count, delta_bic);
        if (status == RG_OK) {
            status = refine_split(
                model,
                source,
                search.best_yes,
                yes_count,
                &yes_context,
                depth + 1,
                max_depth,
                min_obs,
                candidates,
                gates,
                candidate_count,
                penalty,
                stress,
                observation_capacity,
                target_side,
                bucket_total
            );
        }
        rg_context_spec_clear_internal(&yes_context);
    }
    split_search_clear(&search);
    return status;
}

/* Sequential greedy splitting for one source grapheme: repeatedly take the best
 * BIC-improving split of the remaining observations, commit it (plus a
 * refinement of its YES side when immediate axes are in play), and continue on
 * the NO side. */
static rg_status commit_splits_for_source(
    rg_pairwise_model *model,
    const char *source,
    const context_observation *const *rows,
    size_t count,
    const split_candidate *top_candidates,
    const rg_split_gate *top_gates,
    size_t top_candidate_count,
    const split_candidate *all_candidates,
    const rg_split_gate *all_gates,
    size_t all_candidate_count,
    const stress_inventory *stress,
    int max_depth,
    double min_obs,
    double penalty,
    int target_side
) {
    split_search search;
    const context_observation **remaining;
    size_t remaining_count = count;
    int committed = 0;
    rg_status status;

    status = split_search_init(&search, count, 64 + 3 * stress->count);
    if (status != RG_OK) {
        return status;
    }
    remaining = (const context_observation **)calloc(count == 0 ? 1 : count, sizeof(*remaining));
    if (remaining == 0) {
        split_search_clear(&search);
        return RG_ERR_OOM;
    }
    memcpy(remaining, rows, count * sizeof(*remaining));

    while (committed < max_depth * 4 && observation_total_weight(remaining, remaining_count) >= min_obs) {
        split_candidate best;
        size_t yes_count = 0;
        size_t no_count = 0;
        double delta_bic = 0.0;
        rg_context_spec yes_context;
        rg_context_spec empty;

        if (!find_best_split(&search, remaining, remaining_count, top_candidates, top_gates,
                             top_candidate_count, penalty, &best, &yes_count, &no_count,
                             &delta_bic)) {
            break;
        }
        rg_context_spec_init_empty(&empty);
        status = rg_context_extend_internal(&empty, &best, &yes_context);
        rg_context_spec_clear_internal(&empty);
        if (status != RG_OK) {
            break;
        }
        status = commit_observation_group(model, source, search.best_yes, yes_count, &yes_context,
                                          target_side, observation_total_weight(rows, count),
                                          search.best_no, no_count, delta_bic);
        if (status == RG_OK) {
            status = refine_split(
                model,
                source,
                search.best_yes,
                yes_count,
                &yes_context,
                1,
                max_depth,
                min_obs,
                all_candidates,
                all_gates,
                all_candidate_count,
                penalty,
                stress,
                count,
                target_side,
                observation_total_weight(rows, count)
            );
        }
        rg_context_spec_clear_internal(&yes_context);
        if (status != RG_OK) {
            break;
        }
        memcpy(remaining, search.best_no, no_count * sizeof(*remaining));
        remaining_count = no_count;
        committed++;
    }
    free(remaining);
    split_search_clear(&search);
    return status;
}

/* Flattens the corpus alignments into (source, target, context, weight) rows.
 * When decompose_chunks is set, multi-segment links are broken down through a
 * one-segment sub-alignment and each resulting 1-to-1 pair inherits the outer
 * link's context, matching the Go reference. */
static rg_status flatten_context_observations(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model,
    int decompose_chunks,
    int target_side,
    context_observation **out,
    size_t *out_count,
    double *out_total
) {
    context_observation *observations = 0;
    size_t observation_count = 0;
    size_t observation_cap = 0;
    double n_total = 0.0;
    int max_chunk_size = RG_DEFAULT_MAX_CHUNK_SIZE;
    rg_context_spec *target_contexts = 0;
    size_t target_context_count = 0;
    size_t target_position = 0;
    size_t i;
    rg_status status = RG_OK;

    *out = 0;
    *out_count = 0;
    *out_total = 0.0;
    if (options != 0 && options->max_chunk_size > 0) {
        max_chunk_size = options->max_chunk_size;
    }
    for (i = 0; i < pair_count && status == RG_OK; i++) {
        rg_alignment *alignment = 0;
        size_t j;
        double weight = pairs[i].weight == 0.0 ? 1.0 : pairs[i].weight;
        if (weight <= 0.0) {
            continue;
        }
        status = rg_align_forms_with_model(ctx, model, options, &pairs[i].source, &pairs[i].target, max_chunk_size, &alignment);
        if (status != RG_OK) {
            break;
        }
        /* The same alignments either way. Reversing the pair would produce
         * different ones, and the question is what the model's own alignment
         * looks like from the other side, not what a differently-trained model
         * would do. */
        if (target_side) {
            status = rg_form_position_contexts_internal(ctx, &pairs[i].target,
                                                        &target_contexts, &target_context_count);
            if (status != RG_OK) {
                rg_alignment_free(alignment);
                break;
            }
        }
        target_position = 0;
        for (j = 0; j < rg_alignment_link_count(alignment) && status == RG_OK; j++) {
            const rg_link *link = rg_alignment_link_at(alignment, j);
            size_t this_target = target_position;
            target_position += link->target_count;
            if (link->source_count == 1 && link->target_count == 1) {
                if (target_side) {
                    if (this_target < target_context_count) {
                        status = append_context_observation_as(
                            &observations, &observation_count, &observation_cap, link,
                            &target_contexts[this_target], 1, weight);
                        n_total += weight;
                    }
                } else {
                    status = append_context_observation_as(
                        &observations, &observation_count, &observation_cap, link,
                        &link->context, 0, weight);
                    n_total += weight;
                }
                continue;
            }
            if (target_side && this_target >= target_context_count) {
                continue;
            }
            if (!decompose_chunks || link->source_count == 0 || link->target_count == 0) {
                continue;
            }
            {
                rg_form sub_source;
                rg_form sub_target;
                rg_alignment *sub = 0;
                size_t k;
                memset(&sub_source, 0, sizeof(sub_source));
                memset(&sub_target, 0, sizeof(sub_target));
                sub_source.lect_id = "_sub";
                sub_source.segments = link->source;
                sub_source.segment_count = link->source_count;
                sub_target.lect_id = "_sub";
                sub_target.segments = link->target;
                sub_target.segment_count = link->target_count;
                status = rg_align_forms_with_model(ctx, model, options, &sub_source, &sub_target, 1, &sub);
                if (status != RG_OK) {
                    break;
                }
                for (k = 0; k < rg_alignment_link_count(sub) && status == RG_OK; k++) {
                    const rg_link *sub_link = rg_alignment_link_at(sub, k);
                    rg_link merged;
                    if (sub_link->source_count != 1 || sub_link->target_count != 1) {
                        continue;
                    }
                    /* The outer link's context is what conditions this pair. */
                    /* The outer link's context is what conditions this pair.
                     * On the target side that is the target form's view at the
                     * chunk's first position -- the source side uses the outer
                     * link's own span context, and dropping chunks instead
                     * would lose precisely the correspondences chunk promotion
                     * found interesting. */
                    merged = *sub_link;
                    merged.context = link->context;
                    status = append_context_observation_as(
                        &observations, &observation_count, &observation_cap, &merged,
                        target_side ? &target_contexts[this_target] : &link->context,
                        target_side, weight);
                    n_total += weight;
                }
                rg_alignment_free(sub);
            }
        }
        rg_alignment_free(alignment);
        if (target_side) {
            rg_context_spec_array_free_internal(target_contexts, target_context_count);
            target_contexts = 0;
            target_context_count = 0;
        }
    }
    if (target_side) {
        rg_context_spec_array_free_internal(target_contexts, target_context_count);
    }
    if (status != RG_OK) {
        for (i = 0; i < observation_count; i++) {
            context_observation_clear(&observations[i]);
        }
        free(observations);
        return status;
    }
    *out = observations;
    *out_count = observation_count;
    *out_total = n_total;
    return RG_OK;
}

/* Shared driver for the two context-discovery passes. */
static rg_status discover_context_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model,
    int long_range,
    int target_side,
    const rg_feature_vocabulary *vocabulary,
    context_observation *observations,
    size_t observation_count,
    double n_total
) {
    const char **sources = 0;
    size_t source_count = 0;
    size_t source_cap = 0;
    stress_inventory stress;
    split_candidate *long_range_list = 0;
    size_t long_range_count = 0;
    const context_observation **rows = 0;
    size_t i;
    rg_status status;
    double min_obs;
    double delta_threshold;
    double dominant_fraction = 0.0;
    int max_depth = 3;
    split_candidate *immediate_list = 0;
    size_t immediate_count = 0;
    rg_split_gate *immediate_gates = 0;
    rg_split_gate *long_gates = 0;
    split_candidate *all_list = 0;
    rg_split_gate *all_gates = 0;
    size_t all_count = 0;
    double immediate_min_obs = 2.0;
    double immediate_delta = -1.0;
    double long_min_obs = 5.0;
    double long_delta = -5.0;
    double long_dominant = 0.6;

    if (ctx == 0 || model == 0 || (pair_count > 0 && pairs == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    memset(&stress, 0, sizeof(stress));
    if (options != 0) {
        max_depth = options->bic.max_split_depth > 0 ? options->bic.max_split_depth : 3;
        if (options->bic.min_split_observations > 0) {
            immediate_min_obs = (double)options->bic.min_split_observations;
        }
        immediate_delta = options->bic.delta_bic_threshold;
        if (options->bic.long_range_min_split_observations > 0) {
            long_min_obs = (double)options->bic.long_range_min_split_observations;
        }
        long_delta = options->bic.long_range_delta_bic_threshold;
        long_dominant = options->bic.long_range_min_dominant_fraction;
    }
    min_obs = long_range ? long_min_obs : immediate_min_obs;
    delta_threshold = long_range ? long_delta : immediate_delta;
    dominant_fraction = long_range ? long_dominant : 0.0;
    (void)min_obs;
    (void)delta_threshold;
    (void)dominant_fraction;

    status = RG_OK;
    if (observation_count == 0 || n_total <= 0.0) {
        return RG_OK;
    }

    for (i = 0; i < observation_count && status == RG_OK; i++) {
        status = append_unique_source(&sources, &source_count, &source_cap, observations[i].source);
        if (status == RG_OK && !long_range) {
            status = collect_observed_stress(&stress, &observations[i].context);
        }
    }
    /* Both lists are built whatever this stage leads with: the stage decides
     * which kind of predicate opens a split, and refinement may then conjoin
     * either kind onto it. */
    if (status == RG_OK) {
        size_t immediate_cap =
            2 * rg_context_feature_name_count +
            sizeof(split_positions) / sizeof(split_positions[0]) +
            3 * stress.count + 8;
        size_t long_cap =
            sizeof(long_range_slot_names) / sizeof(long_range_slot_names[0]) *
            rg_context_feature_name_count;
        rg_context_spec empty;
        immediate_list = (split_candidate *)calloc(immediate_cap, sizeof(*immediate_list));
        if (long_range_list == 0) {
            long_range_list = (split_candidate *)calloc(long_cap, sizeof(*long_range_list));
            if (long_range_list != 0) {
                long_range_count = long_range_candidates(vocabulary, long_range_list, long_cap);
            }
        }
        if (immediate_list == 0 || long_range_list == 0) {
            status = RG_ERR_OOM;
        } else {
            rg_context_spec_init_empty(&empty);
            immediate_count = immediate_candidates_for(&empty, &stress, vocabulary, immediate_list, immediate_cap);
            rg_context_spec_clear_internal(&empty);
        }
    }
    if (status == RG_OK) {
        size_t total = immediate_count + long_range_count;
        size_t k;
        all_list = (split_candidate *)calloc(total == 0 ? 1 : total, sizeof(*all_list));
        all_gates = (rg_split_gate *)calloc(total == 0 ? 1 : total, sizeof(*all_gates));
        immediate_gates = (rg_split_gate *)calloc(immediate_count == 0 ? 1 : immediate_count, sizeof(*immediate_gates));
        long_gates = (rg_split_gate *)calloc(long_range_count == 0 ? 1 : long_range_count, sizeof(*long_gates));
        if (all_list == 0 || all_gates == 0 || immediate_gates == 0 || long_gates == 0) {
            status = RG_ERR_OOM;
        } else {
            for (k = 0; k < immediate_count; k++) {
                immediate_gates[k].min_obs = immediate_min_obs;
                immediate_gates[k].delta_threshold = immediate_delta;
                immediate_gates[k].min_dominant_fraction = 0.0;
                all_list[k] = immediate_list[k];
                all_gates[k] = immediate_gates[k];
            }
            for (k = 0; k < long_range_count; k++) {
                long_gates[k].min_obs = long_min_obs;
                long_gates[k].delta_threshold = long_delta;
                long_gates[k].min_dominant_fraction = long_dominant;
                all_list[immediate_count + k] = long_range_list[k];
                all_gates[immediate_count + k] = long_gates[k];
            }
            all_count = total;
        }
    }
    rows = (const context_observation **)calloc(observation_count, sizeof(*rows));
    if (rows == 0) {
        status = RG_ERR_OOM;
    }
    if (status == RG_OK) {
        qsort(sources, source_count, sizeof(*sources), string_ptr_cmp);
    }

    for (i = 0; i < source_count && status == RG_OK; i++) {
        size_t row_count = 0;
        size_t j;
        size_t distinct_targets = 0;
        double mass;

        for (j = 0; j < observation_count; j++) {
            if (strcmp(observations[j].source, sources[i]) == 0) {
                rows[row_count++] = &observations[j];
            }
        }
        for (j = 0; j < row_count; j++) {
            size_t k;
            int seen = 0;
            for (k = 0; k < j; k++) {
                if (strcmp(rows[k]->target, rows[j]->target) == 0) {
                    seen = 1;
                    break;
                }
            }
            if (!seen) {
                distinct_targets++;
            }
        }
        mass = observation_total_weight(rows, row_count);
        if (distinct_targets < 2 || mass < 4.0) {
            continue;
        }
        status = commit_splits_for_source(
            model,
            sources[i],
            rows,
            row_count,
            long_range ? long_range_list : immediate_list,
            long_range ? long_gates : immediate_gates,
            long_range ? long_range_count : immediate_count,
            all_list,
            all_gates,
            all_count,
            &stress,
            max_depth,
            long_range ? long_min_obs : immediate_min_obs,
            log(n_total),
            target_side
        );
    }

    free(sources);
    free(rows);
    free(long_range_list);
    free(immediate_list);
    free(immediate_gates);
    free(long_gates);
    free(all_list);
    free(all_gates);
    stress_inventory_clear(&stress);
    if (status == RG_OK && model->conditioned_segment_count_count > 1) {
        qsort(model->conditioned_segment_counts, model->conditioned_segment_count_count,
              sizeof(*model->conditioned_segment_counts), conditioned_count_row_cmp);
    }
    return status;
}

/* Both sides, flattened before either is committed.
 *
 * The order matters and must not: committing the source-side rules first would
 * change the alignments the target-side pass then reads, so which side ran
 * first would leave a mark on the result -- and which side runs first is
 * decided by which lect the corpus happened to name first. Reading both views
 * off the same model is what makes the two passes commute. */
static rg_status discover_both_sides(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model,
    const rg_feature_vocabulary *vocabulary,
    int long_range
) {
    context_observation *observations[2] = {0, 0};
    size_t counts[2] = {0, 0};
    double totals[2] = {0.0, 0.0};
    int side;
    size_t i;
    rg_status status = RG_OK;

    for (side = 0; side < 2 && status == RG_OK; side++) {
        status = flatten_context_observations(
            ctx, pairs, pair_count, options, model, long_range ? 0 : 1, side,
            &observations[side], &counts[side], &totals[side]
        );
    }
    for (side = 0; side < 2 && status == RG_OK; side++) {
        status = discover_context_counts(ctx, pairs, pair_count, options, model,
                                         long_range, side, vocabulary,
                                         observations[side], counts[side], totals[side]);
    }
    for (side = 0; side < 2; side++) {
        for (i = 0; i < counts[side]; i++) {
            context_observation_clear(&observations[side][i]);
        }
        free(observations[side]);
    }
    return status;
}

static rg_status discover_immediate_context_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model,
    const rg_feature_vocabulary *vocabulary
) {
    /* Both directions. A change is only visible from the side that has the
     * split: where the daughter reflects a conditioned change, the ancestor's
     * segment answers to two daughter segments and the ancestor's environment
     * is what separates them, while from the daughter's side each segment has
     * one source and there is nothing to condition. Looking from one side only
     * left half of every pair's conditioning unreachable, and which half
     * depended on which lect happened to sort first. */
    return discover_both_sides(ctx, pairs, pair_count, options, model, vocabulary, 0);
}

/* Long-range discovery runs after cross-dimensional discovery and adds more
 * specific overlays alongside the immediate-neighbour entries. */
static rg_status discover_long_range_context_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model,
    const rg_feature_vocabulary *vocabulary
) {
    return discover_both_sides(ctx, pairs, pair_count, options, model, vocabulary, 1);
}
typedef struct chunk_candidate {
    const rg_segment *source;
    size_t source_count;
    const rg_segment *target;
    size_t target_count;
    double count;
} chunk_candidate;

static void chunk_candidate_clear(chunk_candidate *candidate) {
    if (candidate == 0) {
        return;
    }
    segment_array_clear(candidate->source, candidate->source_count);
    segment_array_clear(candidate->target, candidate->target_count);
    candidate->source = 0;
    candidate->source_count = 0;
    candidate->target = 0;
    candidate->target_count = 0;
    candidate->count = 0.0;
}

static int spans_break(size_t start, size_t end, const int *breaks, size_t break_count) {
    size_t i;
    for (i = 0; i < break_count; i++) {
        if (start < (size_t)breaks[i] && (size_t)breaks[i] < end) {
            return 1;
        }
    }
    return 0;
}

static rg_status append_segments_from_link_span(
    const rg_alignment *alignment,
    size_t start_link,
    size_t end_link,
    int source_side,
    const rg_segment **out,
    size_t *out_count
) {
    rg_segment *segments = 0;
    size_t count = 0;
    size_t i;
    size_t pos = 0;
    rg_status status;
    *out = 0;
    *out_count = 0;
    for (i = start_link; i <= end_link; i++) {
        const rg_link *link = rg_alignment_link_at(alignment, i);
        count += source_side ? link->source_count : link->target_count;
    }
    if (count == 0) {
        return RG_OK;
    }
    segments = (rg_segment *)calloc(count, sizeof(*segments));
    if (segments == 0) {
        return RG_ERR_OOM;
    }
    for (i = start_link; i <= end_link; i++) {
        const rg_link *link = rg_alignment_link_at(alignment, i);
        const rg_segment *chunk = source_side ? link->source : link->target;
        size_t chunk_count = source_side ? link->source_count : link->target_count;
        size_t j;
        for (j = 0; j < chunk_count; j++) {
            status = rg_segment_copy_internal(&chunk[j], &segments[pos]);
            if (status != RG_OK) {
                segment_array_clear(segments, pos);
                return status;
            }
            pos++;
        }
    }
    *out = segments;
    *out_count = count;
    return RG_OK;
}

static rg_status add_chunk_candidate(
    chunk_candidate **items,
    size_t *count,
    size_t *cap,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    double weight
) {
    size_t i;
    chunk_candidate *next;
    rg_status status;
    for (i = 0; i < *count; i++) {
        if (segment_array_equal((*items)[i].source, (*items)[i].source_count, source, source_count) &&
            segment_array_equal((*items)[i].target, (*items)[i].target_count, target, target_count)) {
            (*items)[i].count += weight;
            return RG_OK;
        }
    }
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 16 : *cap * 2;
        next = (chunk_candidate *)realloc(*items, next_cap * sizeof(**items));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *items = next;
        *cap = next_cap;
    }
    memset(&(*items)[*count], 0, sizeof((*items)[*count]));
    status = segment_array_copy(source, source_count, &(*items)[*count].source);
    if (status != RG_OK) {
        return status;
    }
    status = segment_array_copy(target, target_count, &(*items)[*count].target);
    if (status != RG_OK) {
        chunk_candidate_clear(&(*items)[*count]);
        return status;
    }
    (*items)[*count].source_count = source_count;
    (*items)[*count].target_count = target_count;
    (*items)[*count].count = weight;
    (*count)++;
    return RG_OK;
}

static double chunk_source_total(const chunk_candidate *items, size_t count, const chunk_candidate *candidate, size_t *target_variants) {
    size_t i;
    double total = 0.0;
    size_t variants = 0;
    for (i = 0; i < count; i++) {
        if (segment_array_equal(items[i].source, items[i].source_count, candidate->source, candidate->source_count)) {
            total += items[i].count;
            variants++;
        }
    }
    *target_variants = variants;
    return total;
}

/* Raw compositional cost of a chunk: the negative log probability of producing
 * it from independent segment-level draws under the best one-segment
 * decomposition. No log-Z offset, so it is directly comparable with the
 * promoted cost below. */
static rg_status compositional_chunk_cost_raw(
    const rg_context *ctx,
    const rg_train_options *options,
    const rg_pairwise_model *model,
    const chunk_candidate *candidate,
    double *out
) {
    rg_form sub_source;
    rg_form sub_target;
    rg_alignment *sub = 0;
    size_t vocab_size;
    double gap_cost_per_segment;
    double cost = 0.0;
    size_t i;
    rg_status status;
    rg_pairwise_model no_chunks;

    *out = INFINITY;
    memset(&sub_source, 0, sizeof(sub_source));
    memset(&sub_target, 0, sizeof(sub_target));
    sub_source.lect_id = "_sub_src";
    sub_source.segments = candidate->source;
    sub_source.segment_count = candidate->source_count;
    sub_target.lect_id = "_sub_tgt";
    sub_target.segments = candidate->target;
    sub_target.segment_count = candidate->target_count;

    /* A borrowed shallow view with the chunk table emptied: the decomposition
     * must not reuse chunks that are themselves under evaluation. */
    no_chunks = *model;
    no_chunks.chunks = 0;
    no_chunks.chunk_count = 0;

    status = rg_align_forms_with_model(ctx, &no_chunks, options, &sub_source, &sub_target, 1, &sub);
    if (status != RG_OK) {
        return status;
    }
    vocab_size = rg_segment_vocab_size_internal(model);
    if (vocab_size < 1) {
        vocab_size = 1;
    }
    gap_cost_per_segment = vocab_size > 1 ? log((double)vocab_size) : 1.0;

    for (i = 0; i < rg_alignment_link_count(sub); i++) {
        const rg_link *link = rg_alignment_link_at(sub, i);
        if (link->source_count == 1 && link->target_count == 1) {
            rg_context_spec empty;
            double posterior = 0.0;
            rg_context_spec_init_empty(&empty);
            if (!rg_segment_posterior_internal(model, link->source[0].grapheme, link->target[0].grapheme, &empty, 0, &posterior) ||
                posterior <= 0.0) {
                rg_context_spec_clear_internal(&empty);
                rg_alignment_free(sub);
                return RG_OK;
            }
            rg_context_spec_clear_internal(&empty);
            cost += -log(posterior);
        } else {
            size_t span = link->source_count > link->target_count ? link->source_count : link->target_count;
            cost += gap_cost_per_segment * (double)span;
        }
    }
    rg_alignment_free(sub);
    *out = cost;
    return RG_OK;
}

/* Laplace-smoothed MLE cost of the chunk under the candidate counts. */
static double promoted_chunk_cost(
    const chunk_candidate *items,
    size_t count,
    const chunk_candidate *candidate,
    double alpha
) {
    size_t variants = 0;
    double source_total = chunk_source_total(items, count, candidate, &variants);
    double probability;
    if (variants == 0) {
        return INFINITY;
    }
    probability = (candidate->count + alpha) / (source_total + alpha * (double)variants);
    if (probability <= 0.0) {
        return INFINITY;
    }
    return -log(probability);
}

/* Enumerates contiguous sub-alignments as candidate chunks and promotes the
 * ones whose BIC improves. Each candidate is judged independently against the
 * unchanged segment table, so promotion order cannot matter. */
static rg_status promote_chunk_rows(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model
) {
    chunk_candidate *candidates = 0;
    size_t candidate_count = 0;
    size_t candidate_cap = 0;
    rg_chunk_row *rows = 0;
    size_t row_count = 0;
    size_t row_cap = 0;
    double n_observations = 0.0;
    double min_chunk_obs = 2.0;
    size_t i;
    int max_chunk_size = RG_DEFAULT_MAX_CHUNK_SIZE;
    rg_status status = RG_OK;

    if (options != 0) {
        if (options->max_chunk_size > 0) {
            max_chunk_size = options->max_chunk_size;
        }
        if (options->bic.min_chunk_observations > 0) {
            min_chunk_obs = (double)options->bic.min_chunk_observations;
        }
        if (options->chunk_min_transparency > 0.0) {
            /* The transparency screen needs the chunk-diagnostics analyzer,
             * which is not ported yet; refusing beats silently skipping it. */
            return RG_ERR_UNSUPPORTED_OPTION;
        }
    }

    for (i = 0; i < pair_count && status == RG_OK; i++) {
        rg_alignment *alignment = 0;
        size_t link_count;
        size_t *source_starts = 0;
        size_t *source_ends = 0;
        size_t *target_starts = 0;
        size_t *target_ends = 0;
        size_t a;
        size_t source_pos = 0;
        size_t target_pos = 0;
        double weight = pairs[i].weight == 0.0 ? 1.0 : pairs[i].weight;
        if (weight <= 0.0) {
            continue;
        }
        status = rg_align_forms_with_model(ctx, model, options, &pairs[i].source, &pairs[i].target, max_chunk_size, &alignment);
        if (status != RG_OK) {
            break;
        }
        link_count = rg_alignment_link_count(alignment);
        for (a = 0; a < link_count; a++) {
            const rg_link *link = rg_alignment_link_at(alignment, a);
            if (link->source_count == 1 && link->target_count == 1) {
                n_observations += weight;
            }
        }
        if (link_count == 0) {
            rg_alignment_free(alignment);
            continue;
        }
        source_starts = (size_t *)calloc(link_count, sizeof(*source_starts));
        source_ends = (size_t *)calloc(link_count, sizeof(*source_ends));
        target_starts = (size_t *)calloc(link_count, sizeof(*target_starts));
        target_ends = (size_t *)calloc(link_count, sizeof(*target_ends));
        if (source_starts == 0 || source_ends == 0 || target_starts == 0 || target_ends == 0) {
            free(source_starts);
            free(source_ends);
            free(target_starts);
            free(target_ends);
            rg_alignment_free(alignment);
            status = RG_ERR_OOM;
            break;
        }
        for (a = 0; a < link_count; a++) {
            const rg_link *link = rg_alignment_link_at(alignment, a);
            source_starts[a] = source_pos;
            target_starts[a] = target_pos;
            source_pos += link->source_count;
            target_pos += link->target_count;
            source_ends[a] = source_pos;
            target_ends[a] = target_pos;
        }
        for (a = 0; a < link_count && status == RG_OK; a++) {
            size_t b;
            for (b = a; b < link_count && status == RG_OK; b++) {
                const rg_segment *source_chunk = 0;
                const rg_segment *target_chunk = 0;
                size_t source_chunk_count = 0;
                size_t target_chunk_count = 0;
                status = append_segments_from_link_span(alignment, a, b, 1, &source_chunk, &source_chunk_count);
                if (status != RG_OK) {
                    break;
                }
                status = append_segments_from_link_span(alignment, a, b, 0, &target_chunk, &target_chunk_count);
                if (status != RG_OK) {
                    segment_array_clear(source_chunk, source_chunk_count);
                    break;
                }
                if (source_chunk_count > (size_t)max_chunk_size || target_chunk_count > (size_t)max_chunk_size) {
                    segment_array_clear(source_chunk, source_chunk_count);
                    segment_array_clear(target_chunk, target_chunk_count);
                    break;
                }
                if (source_chunk_count == 0 || target_chunk_count == 0 ||
                    source_chunk_count + target_chunk_count < 3 ||
                    spans_break(source_starts[a], source_ends[b], pairs[i].source.morpheme_breaks, pairs[i].source.morpheme_break_count) ||
                    spans_break(target_starts[a], target_ends[b], pairs[i].target.morpheme_breaks, pairs[i].target.morpheme_break_count)) {
                    segment_array_clear(source_chunk, source_chunk_count);
                    segment_array_clear(target_chunk, target_chunk_count);
                    continue;
                }
                status = add_chunk_candidate(
                    &candidates, &candidate_count, &candidate_cap,
                    source_chunk, source_chunk_count, target_chunk, target_chunk_count, weight
                );
                segment_array_clear(source_chunk, source_chunk_count);
                segment_array_clear(target_chunk, target_chunk_count);
            }
        }
        free(source_starts);
        free(source_ends);
        free(target_starts);
        free(target_ends);
        rg_alignment_free(alignment);
    }

    if (n_observations <= 0.0) {
        n_observations = 1.0;
    }

    for (i = 0; i < candidate_count && status == RG_OK; i++) {
        double compositional = 0.0;
        double promoted;
        double reduction;
        double delta_bic;
        double log_z_sum = 0.0;
        size_t k_params;
        size_t j;
        if (candidates[i].count < min_chunk_obs) {
            continue;
        }
        status = compositional_chunk_cost_raw(ctx, options, model, &candidates[i], &compositional);
        if (status != RG_OK) {
            break;
        }
        promoted = promoted_chunk_cost(candidates, candidate_count, &candidates[i], 1.0);
        if (isinf(compositional) || isinf(promoted)) {
            continue;
        }
        reduction = candidates[i].count * (compositional - promoted);
        k_params = candidates[i].source_count > candidates[i].target_count
            ? candidates[i].source_count
            : candidates[i].target_count;
        delta_bic = -2.0 * reduction + (double)k_params * log(n_observations);
        if (delta_bic >= 0.0) {
            continue;
        }
        for (j = 0; j < candidates[i].source_count; j++) {
            log_z_sum += rg_segment_log_normalizer_internal(model, candidates[i].source[j].grapheme);
        }
        if (row_count == row_cap) {
            size_t next_cap = row_cap == 0 ? 8 : row_cap * 2;
            rg_chunk_row *next = (rg_chunk_row *)realloc(rows, next_cap * sizeof(*next));
            if (next == 0) {
                status = RG_ERR_OOM;
                break;
            }
            rows = next;
            row_cap = next_cap;
        }
        memset(&rows[row_count], 0, sizeof(rows[row_count]));
        status = segment_array_copy(candidates[i].source, candidates[i].source_count, &rows[row_count].source);
        if (status != RG_OK) {
            break;
        }
        status = segment_array_copy(candidates[i].target, candidates[i].target_count, &rows[row_count].target);
        if (status != RG_OK) {
            chunk_row_clear(&rows[row_count]);
            break;
        }
        rows[row_count].source_count = candidates[i].source_count;
        rows[row_count].target_count = candidates[i].target_count;
        rows[row_count].cost = promoted - log_z_sum;
        rows[row_count].count = candidates[i].count;
        rows[row_count].uncertainty = rg_wilson_default_internal(candidates[i].count, n_observations);
        row_count++;
    }

    for (i = 0; i < candidate_count; i++) {
        chunk_candidate_clear(&candidates[i]);
    }
    free(candidates);
    if (status != RG_OK) {
        for (i = 0; i < row_count; i++) {
            chunk_row_clear(&rows[i]);
        }
        free(rows);
        return status;
    }
    if (row_count > 1) {
        qsort(rows, row_count, sizeof(*rows), chunk_row_cmp);
    }
    for (i = 0; i < model->chunk_count; i++) {
        chunk_row_clear(&model->chunks[i]);
    }
    free(model->chunks);
    model->chunks = rows;
    model->chunk_count = row_count;
    return RG_OK;
}

static const char *relative_position_name(int offset) {
    switch (offset) {
    case -1:
        return "relative_-1";
    case 0:
        return "relative_0";
    case 1:
        return "relative_+1";
    default:
        return "";
    }
}

/* Whether the position an environment names exists at all. "The preceding
 * segment is not voiced" is a claim about a preceding segment, and word-initial
 * position is not a voiceless onset: counting the edge as the negative side
 * merges a positional environment into a featural one, and a rule stated over
 * that union cannot be read as either. Positions off the end of the form are
 * excluded from both sides of the contrast rather than assigned to one. */
static int context_position_exists(const rg_form *form, int index) {
    return form != 0 && index >= 0 && (size_t)index < form->segment_count;
}

static int segment_has_context_feature(const rg_context *ctx, const rg_form *form, int index, const char *feature) {
    const rg_feature_set *features = 0;
    rg_status status;
    int found = 0;
    if (ctx == 0 || form == 0 || feature == 0 || index < 0 || (size_t)index >= form->segment_count) {
        return 0;
    }
    if (form->segments[index].grapheme == 0) {
        return 0;
    }
    status = rg_context_features_internal(ctx, form->segments[index].grapheme, &features);
    if (status != RG_OK) {
        return 0;
    }
    {
        size_t i;
        for (i = 0; i < rg_feature_set_size(features); i++) {
            const char *item = rg_feature_set_get(features, i);
            if (item != 0 && strcmp(item, feature) == 0) {
                found = 1;
                break;
            }
        }
    }
    return found;
}

typedef struct tone_mass {
    char *tone;
    double count;
} tone_mass;

static void tone_masses_clear(tone_mass *items, size_t count) {
    size_t i;
    if (items == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        free(items[i].tone);
    }
    free(items);
}

static rg_status add_tone_mass(tone_mass **items, size_t *count, size_t *cap, const char *tone, double weight) {
    size_t i;
    tone_mass *next;
    if (tone == 0 || tone[0] == '\0') {
        return RG_OK;
    }
    for (i = 0; i < *count; i++) {
        if (strcmp((*items)[i].tone, tone) == 0) {
            (*items)[i].count += weight;
            return RG_OK;
        }
    }
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 4 : *cap * 2;
        next = (tone_mass *)realloc(*items, next_cap * sizeof(**items));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *items = next;
        *cap = next_cap;
    }
    (*items)[*count].tone = rg_strdup_internal(tone);
    if ((*items)[*count].tone == 0) {
        return RG_ERR_OOM;
    }
    (*items)[*count].count = weight;
    (*count)++;
    return RG_OK;
}

static rg_status append_cross_dimensional_row(
    rg_cross_dimensional_row **rows,
    size_t *count,
    size_t *cap,
    const char *source_feature,
    const char *source_value,
    const char *source_position,
    const char *target_dimension,
    const char *target_value,
    int target_position_offset,
    double rule_count,
    double source_count,
    double contrast_count,
    double contrast_source_count,
    double delta_bic
) {
    rg_cross_dimensional_row *next;
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 8 : *cap * 2;
        next = (rg_cross_dimensional_row *)realloc(*rows, next_cap * sizeof(**rows));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *rows = next;
        *cap = next_cap;
    }
    memset(&(*rows)[*count], 0, sizeof((*rows)[*count]));
    (*rows)[*count].source_feature = rg_strdup_internal(source_feature);
    (*rows)[*count].source_value = rg_strdup_internal(source_value);
    (*rows)[*count].source_position = rg_strdup_internal(source_position);
    (*rows)[*count].target_dimension = rg_strdup_internal(target_dimension);
    (*rows)[*count].target_value = rg_strdup_internal(target_value);
    (*rows)[*count].target_position_offset = target_position_offset;
    (*rows)[*count].count = rule_count;
    (*rows)[*count].source_count = source_count;
    (*rows)[*count].confidence = source_count > 0.0 ? rule_count / source_count : 0.0;
    (*rows)[*count].contrast_count = contrast_count;
    (*rows)[*count].contrast_source_count = contrast_source_count;
    (*rows)[*count].contrast_confidence =
        contrast_source_count > 0.0 ? contrast_count / contrast_source_count : 0.0;
    (*rows)[*count].delta_bic = delta_bic;
    (*rows)[*count].uncertainty = rg_wilson_default_internal(rule_count, source_count);
    if ((*rows)[*count].source_feature == 0 ||
        (*rows)[*count].source_value == 0 ||
        (*rows)[*count].source_position == 0 ||
        (*rows)[*count].target_dimension == 0 ||
        (*rows)[*count].target_value == 0) {
        cross_dimensional_row_clear(&(*rows)[*count]);
        return RG_ERR_OOM;
    }
    (*count)++;
    return RG_OK;
}

/* Negative log-likelihood of a tone distribution, in nats. Summed in sorted
 * tone order so the total does not depend on which tone was seen first. */
static double tone_group_cost(const tone_mass *items, size_t count) {
    double total = 0.0;
    double cost = 0.0;
    size_t i;
    size_t *order;

    for (i = 0; i < count; i++) {
        total += items[i].count;
    }
    if (total <= 0.0) {
        return 0.0;
    }
    order = (size_t *)malloc(count * sizeof(*order));
    if (order == 0) {
        return 0.0;
    }
    for (i = 0; i < count; i++) {
        order[i] = i;
    }
    for (i = 1; i < count; i++) {
        size_t key = order[i];
        size_t j = i;
        while (j > 0 && strcmp(items[order[j - 1]].tone, items[key].tone) > 0) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = key;
    }
    for (i = 0; i < count; i++) {
        double mass = items[order[i]].count;
        if (mass > 0.0) {
            cost += -mass * log(mass / total);
        }
    }
    free(order);
    return cost;
}

/* BIC for "this value's rate differs between the two sides", on the 2x2 table
 * of (value, not-value) by (inside, outside). The environment passing its own
 * test says the distribution moved; it does not say which value moved, and on
 * a dimension with several values most of them did not. Without this, an
 * environment that genuinely conditions one tone also publishes every other
 * tone that drifted upward inside it. Negative means the difference is worth
 * its parameter. */
static double value_split_delta_bic(
    double here_mass,
    double here_total,
    double there_mass,
    double there_total
) {
    double total = here_total + there_total;
    double pooled = (here_mass + there_mass) / total;
    double gain;
    if (total <= 0.0 || here_total <= 0.0 || there_total <= 0.0) {
        return 0.0;
    }
    if (pooled <= 0.0 || pooled >= 1.0) {
        return 0.0;
    }
    /* Binomial log-likelihood, one rate per side against one rate pooled.
     * 0 * log(0) is 0, which is why each term is guarded rather than summed
     * blind. */
    gain = 0.0;
    {
        double sides[2][2];
        size_t i;
        sides[0][0] = here_mass;
        sides[0][1] = here_total;
        sides[1][0] = there_mass;
        sides[1][1] = there_total;
        for (i = 0; i < 2; i++) {
            double mass = sides[i][0];
            double side_total = sides[i][1];
            double rate = mass / side_total;
            if (mass > 0.0) {
                gain += mass * (log(rate) - log(pooled));
            }
            if (side_total - mass > 0.0) {
                gain += (side_total - mass) * (log(1.0 - rate) - log(1.0 - pooled));
            }
        }
    }
    return -2.0 * gain + log(total);
}

static double tone_mass_of(const tone_mass *items, size_t count, const char *tone) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (strcmp(items[i].tone, tone) == 0) {
            return items[i].count;
        }
    }
    return 0.0;
}

static double tone_total(const tone_mass *items, size_t count) {
    double total = 0.0;
    size_t i;
    for (i = 0; i < count; i++) {
        total += items[i].count;
    }
    return total;
}

/* One aligned link whose target segment carries tone: the evidence
 * cross-dimensional discovery reasons over. `environments` is a bitmask over
 * the candidate environment table, so testing membership costs a shift rather
 * than a feature lookup, and `live` goes to zero once a committed rule
 * accounts for this observation. */
typedef struct xdim_observation {
    const char *tone;
    double weight;
    unsigned long long environments;
    /* Environments whose position exists for this observation. An environment
     * the form is too short to have is neither satisfied nor contradicted. */
    unsigned long long defined;
    int live;
} xdim_observation;

typedef struct xdim_environment {
    const char *feature;
    int offset;
} xdim_environment;

typedef struct xdim_scored {
    double delta_bic;
    tone_mass *inside;
    size_t inside_count;
    tone_mass *outside;
    size_t outside_count;
    double inside_total;
    double outside_total;
    int usable;
} xdim_scored;

static void xdim_scored_clear(xdim_scored *scored) {
    tone_masses_clear(scored->inside, scored->inside_count);
    tone_masses_clear(scored->outside, scored->outside_count);
    memset(scored, 0, sizeof(*scored));
}

/* Partitions the live observations on one environment and scores the split the
 * way every other stage in this pipeline scores one: the likelihood gain from
 * modelling the target dimension separately inside and outside, against what
 * the extra parameters cost under BIC. */
static rg_status xdim_score_environment(
    const xdim_observation *observations,
    size_t observation_count,
    size_t env_index,
    double min_count,
    xdim_scored *out
) {
    tone_mass *pooled = 0;
    size_t pooled_count = 0;
    size_t pooled_cap = 0;
    size_t inside_cap = 0;
    size_t outside_cap = 0;
    size_t i;
    rg_status status = RG_OK;

    memset(out, 0, sizeof(*out));
    for (i = 0; i < observation_count && status == RG_OK; i++) {
        const xdim_observation *observation = &observations[i];
        if (!observation->live) {
            continue;
        }
        if (!((observation->defined >> env_index) & 1ULL)) {
            continue;
        }
        if ((observation->environments >> env_index) & 1ULL) {
            status = add_tone_mass(&out->inside, &out->inside_count, &inside_cap,
                                   observation->tone, observation->weight);
        } else {
            status = add_tone_mass(&out->outside, &out->outside_count, &outside_cap,
                                   observation->tone, observation->weight);
        }
        if (status == RG_OK) {
            status = add_tone_mass(&pooled, &pooled_count, &pooled_cap,
                                   observation->tone, observation->weight);
        }
    }
    if (status != RG_OK) {
        tone_masses_clear(pooled, pooled_count);
        xdim_scored_clear(out);
        return status;
    }
    out->inside_total = tone_total(out->inside, out->inside_count);
    out->outside_total = tone_total(out->outside, out->outside_count);
    /* Both sides of the contrast must be attested, or there is no split to
     * test: a predicate holding of every segment partitions nothing. */
    if (out->inside_total >= min_count && out->outside_total >= min_count && pooled_count > 1) {
        double pooled_total = out->inside_total + out->outside_total;
        double baseline = tone_group_cost(pooled, pooled_count);
        double split = tone_group_cost(out->inside, out->inside_count) +
                       tone_group_cost(out->outside, out->outside_count);
        out->delta_bic = -2.0 * (baseline - split) +
                         (double)(pooled_count - 1) * log(pooled_total);
        out->usable = 1;
    }
    tone_masses_clear(pooled, pooled_count);
    return RG_OK;
}

/* Cross-dimensional discovery: does a feature on the source side condition a
 * dimension on the target side?
 *
 * The claim is a conditioned split, so it is tested the way this pipeline
 * tests one. An environment earns a rule only when
 *
 *   1. the complementary environment exists and is attested. A predicate that
 *      holds of every segment partitions nothing, and "the preceding segment
 *      is a consonant" on a corpus of CV syllables is not an environment; it
 *      is a description of the corpus.
 *   2. modelling the target dimension separately inside and outside beats
 *      modelling it once by more than the extra parameters cost under BIC.
 *      This is the criterion, and the code shape, of context discovery.
 *
 * and a value inside that environment is reported only when the environment
 * raises it above its rate in the contrast. That last condition is what makes
 * the output read as historical linguistics rather than as a frequency table:
 * a rule says the environment *did something*, and on a two-valued dimension
 * two rules naming one environment can no longer contradict each other.
 *
 * Environments are then committed one at a time, best first, against what the
 * already-committed rules have not accounted for. Without that, every
 * correlated framing of one fact commits separately -- on a corpus of CV
 * syllables `consonant`, `sonorant`, `nasal` and `voiced` at the same offset
 * are four descriptions of the same coda, and a reader has no way to tell that
 * they are one finding. Explaining the residue is what this stage is for; it
 * runs after the segmental and tonal baselines for the same reason.
 *
 * The stage previously committed on P(value | environment) >= 0.5 with no
 * contrast at all, which reported the ambient distribution as though it were a
 * rule, and on a two-valued dimension emitted both values at once. */
static rg_status discover_cross_dimensional_rows(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model
) {
    rg_cross_dimensional_row *rows = 0;
    size_t row_count = 0;
    size_t row_cap = 0;
    rg_status status = RG_OK;
    double min_count = 3.0;
    double min_confidence = 0.0;
    double delta_threshold = -1.0;
    int max_iterations = 5;
    int max_chunk_size = RG_DEFAULT_MAX_CHUNK_SIZE;
    const size_t feature_count =
        sizeof(cross_dimensional_feature_names) / sizeof(cross_dimensional_feature_names[0]);
    xdim_environment environments[sizeof(cross_dimensional_feature_names) /
                                  sizeof(cross_dimensional_feature_names[0]) * 3];
    size_t env_count = 0;
    xdim_observation *observations = 0;
    size_t observation_count = 0;
    size_t observation_cap = 0;
    int *committed = 0;
    size_t feature_i;
    size_t pair_i;
    int iteration;

    if (ctx == 0 || model == 0 || (pair_count > 0 && pairs == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    if (options != 0) {
        if (options->bic.cross_dim_min_rule_count > 0) {
            min_count = (double)options->bic.cross_dim_min_rule_count;
        }
        if (options->bic.cross_dim_min_rule_confidence > 0.0) {
            min_confidence = options->bic.cross_dim_min_rule_confidence;
        }
        if (options->bic.cross_dim_delta_bic_threshold != 0.0) {
            delta_threshold = options->bic.cross_dim_delta_bic_threshold;
        }
        if (options->bic.cross_dim_max_iterations > 0) {
            max_iterations = options->bic.cross_dim_max_iterations;
        }
        if (options->max_chunk_size > 0) {
            max_chunk_size = options->max_chunk_size;
        }
    }

    /* The environment table, in a fixed order: equal-scoring candidates are
     * resolved by taking the first, so this order is load-bearing. */
    for (feature_i = 0; feature_i < feature_count; feature_i++) {
        int offset;
        for (offset = -1; offset <= 1; offset++) {
            environments[env_count].feature = cross_dimensional_feature_names[feature_i];
            environments[env_count].offset = offset;
            env_count++;
        }
    }

    /* One alignment pass for every environment, rather than one per
     * environment: the membership of each link is recorded as a bitmask as the
     * corpus is walked. */
    for (pair_i = 0; pair_i < pair_count && status == RG_OK; pair_i++) {
        rg_alignment *alignment = 0;
        size_t link_i;
        size_t src_pos = 0;
        size_t tgt_pos = 0;
        double weight = pairs[pair_i].weight == 0.0 ? 1.0 : pairs[pair_i].weight;
        if (weight <= 0.0) {
            continue;
        }
        status = rg_align_forms_with_model(ctx, model, options, &pairs[pair_i].source,
                                           &pairs[pair_i].target, max_chunk_size, &alignment);
        if (status != RG_OK) {
            break;
        }
        for (link_i = 0; link_i < rg_alignment_link_count(alignment); link_i++) {
            const rg_link *link = rg_alignment_link_at(alignment, link_i);
            if (link->source_count == 1 && link->target_count == 1) {
                const char *tone = pairs[pair_i].target.segments[tgt_pos].tone;
                if (tone != 0 && tone[0] != '\0') {
                    size_t env_i;
                    if (observation_count == observation_cap) {
                        size_t next_cap = observation_cap == 0 ? 64 : observation_cap * 2;
                        xdim_observation *next = (xdim_observation *)realloc(
                            observations, next_cap * sizeof(*next));
                        if (next == 0) {
                            status = RG_ERR_OOM;
                            break;
                        }
                        observations = next;
                        observation_cap = next_cap;
                    }
                    memset(&observations[observation_count], 0, sizeof(observations[observation_count]));
                    observations[observation_count].tone = tone;
                    observations[observation_count].weight = weight;
                    observations[observation_count].live = 1;
                    for (env_i = 0; env_i < env_count; env_i++) {
                        int index = (int)src_pos + environments[env_i].offset;
                        if (!context_position_exists(&pairs[pair_i].source, index)) {
                            continue;
                        }
                        observations[observation_count].defined |= 1ULL << env_i;
                        if (segment_has_context_feature(ctx, &pairs[pair_i].source, index,
                                                        environments[env_i].feature)) {
                            observations[observation_count].environments |= 1ULL << env_i;
                        }
                    }
                    observation_count++;
                }
            }
            src_pos += link->source_count;
            tgt_pos += link->target_count;
        }
        rg_alignment_free(alignment);
    }

    if (status == RG_OK) {
        committed = (int *)calloc(env_count, sizeof(*committed));
        if (committed == 0) {
            status = RG_ERR_OOM;
        }
    }

    for (iteration = 0; iteration < max_iterations && status == RG_OK; iteration++) {
        xdim_scored best;
        size_t best_env = 0;
        int found = 0;
        size_t env_i;

        memset(&best, 0, sizeof(best));
        for (env_i = 0; env_i < env_count && status == RG_OK; env_i++) {
            xdim_scored scored;
            if (committed[env_i]) {
                continue;
            }
            status = xdim_score_environment(observations, observation_count, env_i, min_count, &scored);
            if (status != RG_OK) {
                break;
            }
            if (scored.usable && scored.delta_bic < delta_threshold &&
                (!found || scored.delta_bic < best.delta_bic - RG_TIE_EPSILON)) {
                if (found) {
                    xdim_scored_clear(&best);
                }
                best = scored;
                best_env = env_i;
                found = 1;
            } else {
                xdim_scored_clear(&scored);
            }
        }
        if (status != RG_OK || !found) {
            xdim_scored_clear(&best);
            break;
        }

        {
            /* A split is a two-sided statement, and both sides are findings:
             * "voiced onsets give tone 4" is half of the tonogenesis, and
             * "voiceless onsets give tone 1" is the other half. Reporting only
             * the side the feature holds on would describe a merger as though
             * it were a one-way change. The complement is published under
             * source_value "-". */
            int side;
            int emitted = 0;
            for (side = 0; side < 2 && status == RG_OK; side++) {
                const tone_mass *here = side == 0 ? best.inside : best.outside;
                size_t here_count = side == 0 ? best.inside_count : best.outside_count;
                const tone_mass *there = side == 0 ? best.outside : best.inside;
                size_t there_count = side == 0 ? best.outside_count : best.inside_count;
                double here_total = side == 0 ? best.inside_total : best.outside_total;
                double there_total = side == 0 ? best.outside_total : best.inside_total;
                const char *source_value = side == 0 ? "+" : "-";
                size_t tone_i;
                for (tone_i = 0; tone_i < here_count && status == RG_OK; tone_i++) {
                    double here_mass = here[tone_i].count;
                    double there_mass = tone_mass_of(there, there_count, here[tone_i].tone);
                    double confidence = here_mass / here_total;
                    double contrast = there_mass / there_total;
                    if (here_mass < min_count || confidence < min_confidence) {
                        continue;
                    }
                    /* This side must raise the value, not merely contain it,
                     * and the rise must be worth stating. */
                    if (confidence <= contrast) {
                        continue;
                    }
                    if (value_split_delta_bic(here_mass, here_total, there_mass, there_total) >=
                        delta_threshold) {
                        continue;
                    }
                    status = append_cross_dimensional_row(
                        &rows, &row_count, &row_cap,
                        environments[best_env].feature, source_value,
                        relative_position_name(environments[best_env].offset),
                        "tone", here[tone_i].tone, 0,
                        here_mass, here_total,
                        there_mass, there_total,
                        best.delta_bic);
                    if (status == RG_OK) {
                        size_t obs_i;
                        /* Retire what this rule accounts for. What it
                         * mispredicts stays live, so a later rule can still
                         * explain the residue. */
                        for (obs_i = 0; obs_i < observation_count; obs_i++) {
                            int inside = ((observations[obs_i].environments >> best_env) & 1ULL) != 0;
                            if (!((observations[obs_i].defined >> best_env) & 1ULL)) {
                                continue;
                            }
                            if (observations[obs_i].live && inside == (side == 0) &&
                                strcmp(observations[obs_i].tone, here[tone_i].tone) == 0) {
                                observations[obs_i].live = 0;
                            }
                        }
                        emitted = 1;
                    }
                }
            }
            committed[best_env] = 1;
            xdim_scored_clear(&best);
            if (!emitted) {
                /* The split was real but named no value either side raises;
                 * nothing was retired, so stop rather than spin. */
                break;
            }
        }
    }

    free(observations);
    free(committed);
    if (status != RG_OK) {
        size_t i;
        for (i = 0; i < row_count; i++) {
            cross_dimensional_row_clear(&rows[i]);
        }
        free(rows);
        return status;
    }
    if (row_count > 1) {
        qsort(rows, row_count, sizeof(*rows), cross_dimensional_row_cmp);
    }
    model->cross_dimensional_rows = rows;
    model->cross_dimensional_count = row_count;
    return RG_OK;
}


/* ---- initial prior model ------------------------------------------------ */

static rg_status append_vocab(char ***vocab, size_t *count, size_t *cap, const char *grapheme) {
    size_t i;
    size_t insert_at;
    if (grapheme == 0) {
        return RG_OK;
    }
    for (i = 0; i < *count; i++) {
        if (strcmp((*vocab)[i], grapheme) == 0) {
            return RG_OK;
        }
    }
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 32 : *cap * 2;
        char **next = (char **)realloc(*vocab, next_cap * sizeof(*next));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *vocab = next;
        *cap = next_cap;
    }
    insert_at = *count;
    while (insert_at > 0 && strcmp((*vocab)[insert_at - 1], grapheme) > 0) {
        insert_at--;
    }
    if (insert_at < *count) {
        memmove(&(*vocab)[insert_at + 1], &(*vocab)[insert_at], (*count - insert_at) * sizeof(**vocab));
    }
    (*vocab)[insert_at] = rg_strdup_internal(grapheme);
    if ((*vocab)[insert_at] == 0) {
        return RG_ERR_OOM;
    }
    (*count)++;
    return RG_OK;
}

static int grapheme_is_known(const rg_context *ctx, const char *grapheme) {
    const rg_feature_set *features = 0;
    return rg_context_features_internal(ctx, grapheme, &features) == RG_OK;
}

/* Builds the starting model: no counts, a merkmal-derived Dirichlet prior over
 * the corpus grapheme inventory, and empty displacement, chunk and tonal
 * tables. The vocabulary is sorted so the softmax sums in a fixed order;
 * floating-point non-associativity would otherwise make iteration order
 * visible in the prior values. */
static rg_status build_initial_prior_model(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model **out
) {
    rg_pairwise_model *model;
    char **vocab = 0;
    size_t vocab_count = 0;
    size_t vocab_cap = 0;
    double *logits = 0;
    double *exps = 0;
    size_t *targets = 0;
    size_t prior_cap = 0;
    size_t normalizer_cap = 0;
    size_t i;
    rg_status status = RG_OK;
    double temperature = options->temperature == 0.0 ? 1.0 : options->temperature;
    double concentration = options->concentration;

    *out = 0;
    for (i = 0; i < pair_count && status == RG_OK; i++) {
        size_t j;
        double weight = pairs[i].weight == 0.0 ? 1.0 : pairs[i].weight;
        if (weight <= 0.0) {
            continue;
        }
        for (j = 0; j < pairs[i].source.segment_count && status == RG_OK; j++) {
            status = append_vocab(&vocab, &vocab_count, &vocab_cap, pairs[i].source.segments[j].grapheme);
        }
        for (j = 0; j < pairs[i].target.segment_count && status == RG_OK; j++) {
            status = append_vocab(&vocab, &vocab_count, &vocab_cap, pairs[i].target.segments[j].grapheme);
        }
    }
    model = (rg_pairwise_model *)calloc(1, sizeof(*model));
    if (model == 0 || status != RG_OK) {
        string_array_free(vocab, vocab_count);
        free(model);
        return status == RG_OK ? RG_ERR_OOM : status;
    }
    model->concentration = concentration;
    model->segment_weight = options->segment_weight;
    model->displacement_weight = options->displacement_weight;
    model->tone_weight = options->tone_weight;

    logits = (double *)calloc(vocab_count == 0 ? 1 : vocab_count, sizeof(*logits));
    exps = (double *)calloc(vocab_count == 0 ? 1 : vocab_count, sizeof(*exps));
    targets = (size_t *)calloc(vocab_count == 0 ? 1 : vocab_count, sizeof(*targets));
    if (logits == 0 || exps == 0 || targets == 0) {
        free(logits);
        free(exps);
        free(targets);
        string_array_free(vocab, vocab_count);
        rg_pairwise_model_free(model);
        return RG_ERR_OOM;
    }

    for (i = 0; i < vocab_count && status == RG_OK; i++) {
        size_t j;
        size_t logit_count = 0;
        double max_logit = 0.0;
        double total = 0.0;
        if (!grapheme_is_known(ctx, vocab[i])) {
            continue;
        }
        for (j = 0; j < vocab_count; j++) {
            double distance = 0.0;
            if (!grapheme_is_known(ctx, vocab[j])) {
                continue;
            }
            if (rg_context_segment_distance(ctx, vocab[i], vocab[j], &distance) != RG_OK) {
                continue;
            }
            logits[logit_count] = -temperature * distance;
            targets[logit_count] = j;
            logit_count++;
        }
        if (logit_count == 0) {
            continue;
        }
        max_logit = logits[0];
        for (j = 1; j < logit_count; j++) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
            }
        }
        for (j = 0; j < logit_count; j++) {
            exps[j] = exp(logits[j] - max_logit);
            total += exps[j];
        }
        if (total <= 0.0) {
            continue;
        }
        if (model->log_normalizer_count == normalizer_cap) {
            size_t next_cap = normalizer_cap == 0 ? 32 : normalizer_cap * 2;
            rg_log_normalizer_row *next = (rg_log_normalizer_row *)realloc(model->log_normalizers, next_cap * sizeof(*next));
            if (next == 0) {
                status = RG_ERR_OOM;
                break;
            }
            model->log_normalizers = next;
            normalizer_cap = next_cap;
        }
        model->log_normalizers[model->log_normalizer_count].source = rg_strdup_internal(vocab[i]);
        if (model->log_normalizers[model->log_normalizer_count].source == 0) {
            status = RG_ERR_OOM;
            break;
        }
        model->log_normalizers[model->log_normalizer_count].value = max_logit + log(total);
        /* The reverse direction has no observations yet; publish_target_totals
         * fills this in once counts exist. The row comes from realloc, so the
         * first cost evaluation reads whatever it is left as. */
        model->log_normalizers[model->log_normalizer_count].target_total = 0.0;
        model->log_normalizer_count++;

        for (j = 0; j < logit_count && status == RG_OK; j++) {
            if (model->segment_prior_count == prior_cap) {
                size_t next_cap = prior_cap == 0 ? 64 : prior_cap * 2;
                rg_segment_prior_row *next = (rg_segment_prior_row *)realloc(model->segment_priors, next_cap * sizeof(*next));
                if (next == 0) {
                    status = RG_ERR_OOM;
                    break;
                }
                model->segment_priors = next;
                prior_cap = next_cap;
            }
            model->segment_priors[model->segment_prior_count].source = rg_strdup_internal(vocab[i]);
            model->segment_priors[model->segment_prior_count].target = rg_strdup_internal(vocab[targets[j]]);
            if (model->segment_priors[model->segment_prior_count].source == 0 ||
                model->segment_priors[model->segment_prior_count].target == 0) {
                free(model->segment_priors[model->segment_prior_count].source);
                free(model->segment_priors[model->segment_prior_count].target);
                status = RG_ERR_OOM;
                break;
            }
            model->segment_priors[model->segment_prior_count].alpha = concentration * (exps[j] / total);
            model->segment_prior_count++;
        }
    }

    free(logits);
    free(exps);
    free(targets);
    string_array_free(vocab, vocab_count);
    if (status != RG_OK) {
        rg_pairwise_model_free(model);
        return status;
    }
    *out = model;
    return RG_OK;
}

/* Rebuilds the segment counts from the 1-to-1 links of the current alignments,
 * leaving the prior, the normalizers, and every other table untouched. This is
 * the M-step; the displacement, chunk and tonal tables stay empty until their
 * own stages run. */
static rg_status update_segment_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model
) {
    rg_segment_count_row *rows = 0;
    size_t row_count = 0;
    size_t row_cap = 0;
    size_t i;
    int max_chunk_size = options->max_chunk_size > 0 ? options->max_chunk_size : RG_DEFAULT_MAX_CHUNK_SIZE;
    rg_status status = RG_OK;

    for (i = 0; i < pair_count && status == RG_OK; i++) {
        rg_alignment *alignment = 0;
        size_t j;
        double weight = pairs[i].weight == 0.0 ? 1.0 : pairs[i].weight;
        if (weight <= 0.0) {
            continue;
        }
        status = rg_align_forms_with_model(ctx, model, options, &pairs[i].source, &pairs[i].target, max_chunk_size, &alignment);
        if (status != RG_OK) {
            break;
        }
        for (j = 0; j < rg_alignment_link_count(alignment) && status == RG_OK; j++) {
            const rg_link *link = rg_alignment_link_at(alignment, j);
            if (link->source_count == 1 && link->target_count == 1) {
                status = add_segment_count(&rows, &row_count, &row_cap,
                                           link->source[0].grapheme, link->target[0].grapheme, weight);
            }
        }
        rg_alignment_free(alignment);
    }
    if (status != RG_OK) {
        for (i = 0; i < row_count; i++) {
            segment_count_row_clear(&rows[i]);
        }
        free(rows);
        return status;
    }
    fill_totals(rows, row_count);
    publish_target_totals(model, rows, row_count);
    qsort(rows, row_count, sizeof(*rows), count_row_cmp);
    for (i = 0; i < model->segment_count_count; i++) {
        segment_count_row_clear(&model->segment_counts[i]);
    }
    free(model->segment_counts);
    model->segment_counts = rows;
    model->segment_count_count = row_count;
    return RG_OK;
}

/* Re-aligns under the post-EM model and builds the feature displacement
 * distribution from every 1-to-1 link. */
static rg_status aggregate_displacement_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model
) {
    rg_displacement_row *rows = 0;
    size_t row_count = 0;
    size_t row_cap = 0;
    size_t i;
    int max_chunk_size = options->max_chunk_size > 0 ? options->max_chunk_size : RG_DEFAULT_MAX_CHUNK_SIZE;
    rg_status status = RG_OK;

    for (i = 0; i < pair_count && status == RG_OK; i++) {
        rg_alignment *alignment = 0;
        size_t j;
        double weight = pairs[i].weight == 0.0 ? 1.0 : pairs[i].weight;
        if (weight <= 0.0) {
            continue;
        }
        status = rg_align_forms_with_model(ctx, model, options, &pairs[i].source, &pairs[i].target, max_chunk_size, &alignment);
        if (status != RG_OK) {
            break;
        }
        for (j = 0; j < rg_alignment_link_count(alignment) && status == RG_OK; j++) {
            const rg_link *link = rg_alignment_link_at(alignment, j);
            if (link->source_count != 1 || link->target_count != 1) {
                continue;
            }
            status = add_displacement_vector(
                &rows,
                &row_count,
                &row_cap,
                link->feature_displacement,
                link->feature_displacement_count,
                weight
            );
        }
        rg_alignment_free(alignment);
    }
    if (status != RG_OK) {
        for (i = 0; i < row_count; i++) {
            displacement_row_clear(&rows[i]);
        }
        free(rows);
        return status;
    }
    fill_displacement_total(rows, row_count);
    qsort(rows, row_count, sizeof(*rows), displacement_row_cmp);
    for (i = 0; i < model->displacement_row_count; i++) {
        displacement_row_clear(&model->displacement_rows[i]);
    }
    free(model->displacement_rows);
    model->displacement_rows = rows;
    model->displacement_row_count = row_count;
    return RG_OK;
}

/* Re-aligns under the post-chunk-promotion model and counts each 1-to-1 link's
 * tonal correspondence. The table stays empty for non-tonal corpora. */
static rg_status aggregate_tonal_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model
) {
    rg_tonal_count_row *rows = 0;
    size_t row_count = 0;
    size_t row_cap = 0;
    size_t i;
    int any_toned = 0;
    int max_chunk_size = options->max_chunk_size > 0 ? options->max_chunk_size : RG_DEFAULT_MAX_CHUNK_SIZE;
    rg_status status = RG_OK;

    for (i = 0; i < pair_count && status == RG_OK; i++) {
        rg_alignment *alignment = 0;
        size_t j;
        double weight = pairs[i].weight == 0.0 ? 1.0 : pairs[i].weight;
        if (weight <= 0.0) {
            continue;
        }
        status = rg_align_forms_with_model(ctx, model, options, &pairs[i].source, &pairs[i].target, max_chunk_size, &alignment);
        if (status != RG_OK) {
            break;
        }
        for (j = 0; j < rg_alignment_link_count(alignment) && status == RG_OK; j++) {
            const rg_link *link = rg_alignment_link_at(alignment, j);
            const char *source_tone;
            const char *target_tone;
            if (link->source_count != 1 || link->target_count != 1) {
                continue;
            }
            source_tone = link->source[0].tone == 0 ? "" : link->source[0].tone;
            target_tone = link->target[0].tone == 0 ? "" : link->target[0].tone;
            if (source_tone[0] == '\0' && target_tone[0] == '\0') {
                continue;
            }
            any_toned = 1;
            status = add_tonal_count(&rows, &row_count, &row_cap, source_tone, target_tone, weight);
        }
        rg_alignment_free(alignment);
    }
    if (status != RG_OK || !any_toned) {
        for (i = 0; i < row_count; i++) {
            free((char *)rows[i].source_tone);
            free((char *)rows[i].target_tone);
        }
        free(rows);
        return status;
    }
    fill_tonal_source_totals(rows, row_count);
    qsort(rows, row_count, sizeof(*rows), tonal_row_cmp);
    for (i = 0; i < model->tonal_count_count; i++) {
        free((char *)model->tonal_counts[i].source_tone);
        free((char *)model->tonal_counts[i].target_tone);
    }
    free(model->tonal_counts);
    model->tonal_counts = rows;
    model->tonal_count_count = row_count;
    return RG_OK;
}

static rg_status corpus_cost_with_model(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    const rg_pairwise_model *model,
    double *out
) {
    size_t i;
    rg_status status = RG_OK;
    int max_chunk_size = RG_DEFAULT_MAX_CHUNK_SIZE;
    double total = 0.0;
    if (ctx == 0 || (pair_count > 0 && pairs == 0) || model == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    if (options != 0 && options->max_chunk_size > 0) {
        max_chunk_size = options->max_chunk_size;
    }
    for (i = 0; i < pair_count; i++) {
        rg_alignment *alignment = 0;
        double cost = 0.0;
        double weight = pairs[i].weight == 0.0 ? 1.0 : pairs[i].weight;
        if (weight <= 0.0) {
            continue;
        }
        status = rg_align_forms_with_model(ctx, model, options, &pairs[i].source, &pairs[i].target, max_chunk_size, &alignment);
        if (status != RG_OK) {
            return status;
        }
        status = rg_alignment_cost_with_model(ctx, model, options, alignment, &cost);
        rg_alignment_free(alignment);
        if (status != RG_OK) {
            return status;
        }
        total += weight * cost;
    }
    *out = total;
    return RG_OK;
}

rg_status rg_train_pairwise_internal(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_progress_state *progress,
    rg_pairwise_model **out
) {
    rg_pairwise_model *model = 0;
    rg_train_options defaults;
    const rg_train_options *opts = options;
    int max_iter;
    int iter;
    double prev_cost = INFINITY;
    double eps;
    rg_feature_vocabulary vocabulary;
    rg_status status;

    vocabulary.contrastive = 0;
    vocabulary.contrastive_count = 0;
    if (ctx == 0 || out == 0 || (pair_count > 0 && pairs == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    if (opts == 0) {
        rg_train_options_init_defaults(&defaults);
        opts = &defaults;
    }
    max_iter = opts->max_iter <= 0 ? 1 : opts->max_iter;
    eps = opts->convergence_eps <= 0.0 ? 1e-4 : opts->convergence_eps;

    /* Stage order is load-bearing and mirrors the Go pipeline: initial prior,
     * segment EM, displacement aggregation, immediate context discovery, chunk
     * promotion, tonal aggregation, cross-dimensional discovery, long-range
     * context discovery. Context splits need stable segment correspondences,
     * chunk promotion would otherwise steal observations, and cross-dimensional
     * rules should explain what the segmental and tonal baselines leave over. */
    status = build_initial_prior_model(ctx, pairs, pair_count, opts, &model);
    if (status != RG_OK) {
        return status;
    }
    if (rg_progress_step_internal(progress, "initial prior")) {
        rg_pairwise_model_free(model);
        return RG_ERR_CANCELLED;
    }
    for (iter = 0; iter < max_iter; iter++) {
        double cost = 0.0;
        status = corpus_cost_with_model(ctx, pairs, pair_count, opts, model, &cost);
        if (status != RG_OK) {
            rg_pairwise_model_free(model);
            return status;
        }
        if (!isinf(prev_cost)) {
            double denom = fabs(prev_cost) < 1.0 ? 1.0 : fabs(prev_cost);
            if (fabs(prev_cost - cost) / denom < eps) {
                break;
            }
        }
        prev_cost = cost;
        status = update_segment_counts(ctx, pairs, pair_count, opts, model);
        if (status != RG_OK) {
            rg_pairwise_model_free(model);
            return status;
        }
    }
    if (rg_progress_step_internal(progress, "segment em")) {
        rg_pairwise_model_free(model);
        return RG_ERR_CANCELLED;
    }

    /* The searchable vocabulary is a property of the corpus, not of the source
     * file, so it is derived once and every discovery stage searches the same
     * list. Built here rather than earlier because nothing before this point
     * uses it, and every stage from here on unwinds through RUN_STAGE, which
     * frees it. */
    status = rg_feature_vocabulary_build_internal(ctx, pairs, pair_count, &vocabulary);
    if (status != RG_OK) {
        rg_pairwise_model_free(model);
        return status;
    }

    /* Cancellation is checked between stages, where the unwind path is already
     * a plain free of the partly-built model. */
#define RUN_STAGE(name, call)                          \
    do {                                               \
        status = (call);                               \
        if (status != RG_OK) {                         \
            rg_feature_vocabulary_clear_internal(&vocabulary); \
            rg_pairwise_model_free(model);             \
            return status;                             \
        }                                              \
        if (rg_progress_step_internal(progress, name)) { \
            rg_feature_vocabulary_clear_internal(&vocabulary); \
            rg_pairwise_model_free(model);             \
            return RG_ERR_CANCELLED;                   \
        }                                              \
    } while (0)

    RUN_STAGE("displacement aggregation", aggregate_displacement_counts(ctx, pairs, pair_count, opts, model));
    RUN_STAGE("context discovery", discover_immediate_context_counts(ctx, pairs, pair_count, opts, model, &vocabulary));
    RUN_STAGE("chunk promotion", promote_chunk_rows(ctx, pairs, pair_count, opts, model));
    RUN_STAGE("tonal aggregation", aggregate_tonal_counts(ctx, pairs, pair_count, opts, model));
    RUN_STAGE("cross-dimensional discovery", discover_cross_dimensional_rows(ctx, pairs, pair_count, opts, model));
    RUN_STAGE("long-range discovery", discover_long_range_context_counts(ctx, pairs, pair_count, opts, model, &vocabulary));

#undef RUN_STAGE

    rg_feature_vocabulary_clear_internal(&vocabulary);
    *out = model;
    return RG_OK;
}

rg_status rg_train_pairwise_segment_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model **out
) {
    rg_progress_state progress;
    rg_progress_init_internal(&progress, options, RG_PAIRWISE_STAGE_COUNT);
    return rg_train_pairwise_internal(ctx, pairs, pair_count, options, &progress, out);
}

rg_status rg_train_pairwise(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model **out
) {
    return rg_train_pairwise_segment_counts(ctx, pairs, pair_count, options, out);
}

size_t rg_pairwise_model_displacement_row_count(const rg_pairwise_model *model) {
    if (model == 0) {
        return 0;
    }
    return model->displacement_row_count;
}

const rg_displacement_row *rg_pairwise_model_displacement_row_at(
    const rg_pairwise_model *model,
    size_t index
) {
    if (model == 0 || index >= model->displacement_row_count) {
        return 0;
    }
    return &model->displacement_rows[index];
}

size_t rg_pairwise_model_tonal_count_row_count(const rg_pairwise_model *model) {
    if (model == 0) {
        return 0;
    }
    return model->tonal_count_count;
}

const rg_tonal_count_row *rg_pairwise_model_tonal_count_row_at(
    const rg_pairwise_model *model,
    size_t index
) {
    if (model == 0 || index >= model->tonal_count_count) {
        return 0;
    }
    return &model->tonal_counts[index];
}

size_t rg_pairwise_model_segment_count_row_count(const rg_pairwise_model *model) {
    if (model == 0) {
        return 0;
    }
    return model->segment_count_count;
}

const rg_segment_count_row *rg_pairwise_model_segment_count_row_at(
    const rg_pairwise_model *model,
    size_t index
) {
    if (model == 0 || index >= model->segment_count_count) {
        return 0;
    }
    return &model->segment_counts[index];
}

size_t rg_pairwise_model_conditioned_segment_count_row_count(const rg_pairwise_model *model) {
    if (model == 0) {
        return 0;
    }
    return model->conditioned_segment_count_count;
}

const rg_conditioned_segment_count_row *rg_pairwise_model_conditioned_segment_count_row_at(
    const rg_pairwise_model *model,
    size_t index
) {
    if (model == 0 || index >= model->conditioned_segment_count_count) {
        return 0;
    }
    return &model->conditioned_segment_counts[index];
}

size_t rg_pairwise_model_chunk_row_count(const rg_pairwise_model *model) {
    if (model == 0) {
        return 0;
    }
    return model->chunk_count;
}

const rg_chunk_row *rg_pairwise_model_chunk_row_at(
    const rg_pairwise_model *model,
    size_t index
) {
    if (model == 0 || index >= model->chunk_count) {
        return 0;
    }
    return &model->chunks[index];
}

size_t rg_pairwise_model_cross_dimensional_row_count(const rg_pairwise_model *model) {
    if (model == 0) {
        return 0;
    }
    return model->cross_dimensional_count;
}

const rg_cross_dimensional_row *rg_pairwise_model_cross_dimensional_row_at(
    const rg_pairwise_model *model,
    size_t index
) {
    if (model == 0 || index >= model->cross_dimensional_count) {
        return 0;
    }
    return &model->cross_dimensional_rows[index];
}
