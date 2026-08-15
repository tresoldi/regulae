#include "model_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void string_array_free(char **items, size_t count) {
    size_t i;
    if (items == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        free(items[i]);
    }
    free(items);
}

void segment_count_row_clear(rg_segment_count_row *row) {
    if (row == 0) {
        return;
    }
    rg_free_owned_internal(row->source);
    rg_free_owned_internal(row->target);
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
    rg_free_owned_internal(row->source);
    rg_free_owned_internal(row->target);
    rg_context_spec_clear_internal(&row->context);
    row->source = 0;
    row->target = 0;
    row->count = 0.0;
    row->source_total = 0.0;
    row->uncertainty = rg_wilson_default_internal(0.0, 0.0);
}

void segment_array_clear(const rg_segment *segments, size_t count) {
    size_t i;
    if (segments == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        rg_segment_clear_internal(rg_owned_internal(&segments[i]));
    }
    rg_free_owned_internal(segments);
}

rg_status segment_array_copy(const rg_segment *segments, size_t count, const rg_segment **out) {
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

void chunk_row_clear(rg_chunk_row *row) {
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

void cross_dimensional_row_clear(rg_cross_dimensional_row *row) {
    if (row == 0) {
        return;
    }
    rg_context_spec_clear_internal(&row->source_environment);
    rg_free_owned_internal(row->target_dimension);
    rg_free_owned_internal(row->target_value);

    row->target_dimension = 0;
    row->target_value = 0;
    row->target_position_offset = 0;
    row->count = 0.0;
    row->source_count = 0.0;
    row->confidence = 0.0;
    row->uncertainty = rg_wilson_default_internal(0.0, 0.0);
}

void displacement_row_clear(rg_displacement_row *row) {
    size_t i;
    if (row == 0) {
        return;
    }
    for (i = 0; i < row->item_count; i++) {
        rg_free_owned_internal(row->items[i].feature);
        rg_free_owned_internal(row->items[i].from_value);
        rg_free_owned_internal(row->items[i].to_value);
    }
    rg_free_owned_internal(row->items);
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
        rg_free_owned_internal(model->tonal_counts[i].source_tone);
        rg_free_owned_internal(model->tonal_counts[i].target_tone);
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

int count_row_cmp(const void *a, const void *b) {
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
    c = nullable_strcmp(a->morpheme_index, b->morpheme_index);
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
    c = constraint_list_cmp(a->self, a->self_count, b->self, b->self_count);
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

int conditioned_count_row_cmp(const void *a, const void *b) {
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

int segment_array_equal(const rg_segment *a, size_t a_count, const rg_segment *b, size_t b_count) {
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

int chunk_row_cmp(const void *a, const void *b) {
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

int cross_dimensional_row_cmp(const void *a, const void *b) {
    const rg_cross_dimensional_row *ra = (const rg_cross_dimensional_row *)a;
    const rg_cross_dimensional_row *rb = (const rg_cross_dimensional_row *)b;
    int c = strcmp(ra->target_dimension, rb->target_dimension);
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

rg_status add_segment_count(
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

double source_total_for_rows(const rg_segment_count_row *rows, size_t count, const char *source) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (strcmp(rows[i].source, source) == 0) {
            return rows[i].source_total;
        }
    }
    return 0.0;
}

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
) {
    size_t i;
    rg_conditioned_segment_count_row *next;
    for (i = 0; i < *count; i++) {
        bool a_subset_b = false;
        bool b_subset_a = false;
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
            (*rows)[i].search_margin = search_margin;
            /* The earliest decision that reached this row keeps it: a later
             * refinement restating the same correspondence did not discover
             * it. */
            if ((*rows)[i].decision_index < 0 || decision_index < (*rows)[i].decision_index) {
                (*rows)[i].decision_index = decision_index;
            }
            (*rows)[i].uncertainty = rg_wilson_default_internal((*rows)[i].count, source_total);
            (*rows)[i].uncertainty.post_selection = 1;
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
    (*rows)[*count].search_margin = search_margin;
    (*rows)[*count].decision_index = decision_index;
    (*rows)[*count].search_margin = search_margin;
    (*rows)[*count].uncertainty = rg_wilson_default_internal(weight, source_total);
    (*rows)[*count].uncertainty.post_selection = 1;
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

int displacement_row_cmp(const void *a, const void *b) {
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
rg_status add_displacement_vector(
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
                rg_free_owned_internal(copy[j].feature);
                rg_free_owned_internal(copy[j].from_value);
                rg_free_owned_internal(copy[j].to_value);
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

int tonal_row_cmp(const void *a, const void *b) {
    const rg_tonal_count_row *ra = (const rg_tonal_count_row *)a;
    const rg_tonal_count_row *rb = (const rg_tonal_count_row *)b;
    int c = strcmp(ra->source_tone, rb->source_tone);
    if (c != 0) {
        return c;
    }
    return strcmp(ra->target_tone, rb->target_tone);
}

rg_status add_tonal_count(
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
        rg_free_owned_internal((*rows)[*count].source_tone);
        rg_free_owned_internal((*rows)[*count].target_tone);
        return RG_ERR_OOM;
    }
    (*count)++;
    return RG_OK;
}

void fill_totals(rg_segment_count_row *rows, size_t count) {
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
void publish_target_totals(rg_pairwise_model *model, const rg_segment_count_row *rows, size_t count) {
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

void fill_displacement_total(rg_displacement_row *rows, size_t count) {
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

void fill_tonal_source_totals(rg_tonal_count_row *rows, size_t count) {
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


