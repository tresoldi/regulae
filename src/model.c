#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static rg_uncertainty_estimate wilson_interval(double count, double total) {
    double z = 1.959963984540054;
    rg_uncertainty_estimate out;
    if (total <= 0.0) {
        out.estimate = 0.0;
        out.lower = 0.0;
        out.upper = 0.0;
        return out;
    }
    {
        double p = count / total;
        double z2 = z * z;
        double denom = 1.0 + z2 / total;
        double center = (p + z2 / (2.0 * total)) / denom;
        double margin = z * sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denom;
        out.estimate = p;
        out.lower = center - margin;
        out.upper = center + margin;
        if (out.lower < 0.0) {
            out.lower = 0.0;
        }
        if (out.upper > 1.0) {
            out.upper = 1.0;
        }
    }
    return out;
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
    row->uncertainty = wilson_interval(0.0, 0.0);
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
    row->uncertainty = wilson_interval(0.0, 0.0);
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
    row->uncertainty = wilson_interval(0.0, 0.0);
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
    row->uncertainty = wilson_interval(0.0, 0.0);
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
    for (i = 0; i < model->displacement_count_count; i++) {
        free((char *)model->displacement_counts[i].feature);
        free((char *)model->displacement_counts[i].from_value);
        free((char *)model->displacement_counts[i].to_value);
    }
    free(model->displacement_counts);
    for (i = 0; i < model->tonal_count_count; i++) {
        free((char *)model->tonal_counts[i].source_tone);
        free((char *)model->tonal_counts[i].target_tone);
    }
    free(model->tonal_counts);
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
    c = strcmp(ra->context.position == 0 ? "" : ra->context.position, rb->context.position == 0 ? "" : rb->context.position);
    if (c != 0) {
        return c;
    }
    c = (int)rg_context_spec_constraint_count(&ra->context) - (int)rg_context_spec_constraint_count(&rb->context);
    if (c != 0) {
        return c;
    }
    return 0;
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
    double weight,
    double source_total
) {
    size_t i;
    rg_conditioned_segment_count_row *next;
    for (i = 0; i < *count; i++) {
        int a_subset_b = 0;
        int b_subset_a = 0;
        if (strcmp((*rows)[i].source, source) == 0 &&
            strcmp((*rows)[i].target, target) == 0 &&
            rg_context_spec_is_subset(&(*rows)[i].context, context, &a_subset_b) == RG_OK &&
            rg_context_spec_is_subset(context, &(*rows)[i].context, &b_subset_a) == RG_OK &&
            a_subset_b && b_subset_a) {
            (*rows)[i].count += weight;
            (*rows)[i].source_total = source_total;
            (*rows)[i].uncertainty = wilson_interval((*rows)[i].count, source_total);
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
    (*rows)[*count].count = weight;
    (*rows)[*count].source_total = source_total;
    (*rows)[*count].uncertainty = wilson_interval(weight, source_total);
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

static int displacement_row_cmp(const void *a, const void *b) {
    const rg_displacement_count_row *ra = (const rg_displacement_count_row *)a;
    const rg_displacement_count_row *rb = (const rg_displacement_count_row *)b;
    int c = strcmp(ra->feature, rb->feature);
    if (c != 0) {
        return c;
    }
    c = strcmp(ra->from_value, rb->from_value);
    if (c != 0) {
        return c;
    }
    return strcmp(ra->to_value, rb->to_value);
}

static rg_status add_displacement_count(
    rg_displacement_count_row **rows,
    size_t *count,
    size_t *cap,
    const char *feature,
    const char *from_value,
    const char *to_value,
    double weight
) {
    size_t i;
    rg_displacement_count_row *next;
    for (i = 0; i < *count; i++) {
        if (strcmp((*rows)[i].feature, feature) == 0 &&
            strcmp((*rows)[i].from_value, from_value) == 0 &&
            strcmp((*rows)[i].to_value, to_value) == 0) {
            (*rows)[i].count += weight;
            return RG_OK;
        }
    }
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 16 : *cap * 2;
        next = (rg_displacement_count_row *)realloc(*rows, next_cap * sizeof(**rows));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *rows = next;
        *cap = next_cap;
    }
    (*rows)[*count].feature = rg_strdup_internal(feature);
    (*rows)[*count].from_value = rg_strdup_internal(from_value);
    (*rows)[*count].to_value = rg_strdup_internal(to_value);
    (*rows)[*count].count = weight;
    (*rows)[*count].total = 0.0;
    if ((*rows)[*count].feature == 0 || (*rows)[*count].from_value == 0 || (*rows)[*count].to_value == 0) {
        free((char *)(*rows)[*count].feature);
        free((char *)(*rows)[*count].from_value);
        free((char *)(*rows)[*count].to_value);
        return RG_ERR_OOM;
    }
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

static void fill_source_totals(rg_segment_count_row *rows, size_t count) {
    size_t i;
    size_t j;
    for (i = 0; i < count; i++) {
        double total = 0.0;
        for (j = 0; j < count; j++) {
            if (strcmp(rows[i].source, rows[j].source) == 0) {
                total += rows[j].count;
            }
        }
        rows[i].source_total = total;
        rows[i].uncertainty = wilson_interval(rows[i].count, total);
    }
}

static void fill_displacement_total(rg_displacement_count_row *rows, size_t count) {
    size_t i;
    double total = 0.0;
    for (i = 0; i < count; i++) {
        total += rows[i].count;
    }
    for (i = 0; i < count; i++) {
        rows[i].total = total;
        rows[i].uncertainty = wilson_interval(rows[i].count, total);
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
        rows[i].uncertainty = wilson_interval(rows[i].count, total);
    }
}

typedef struct context_observation {
    char *source;
    char *target;
    rg_context_spec context;
    double weight;
} context_observation;

typedef struct split_candidate {
    const char *slot;
    const char *feature;
    const char *value;
} split_candidate;

static const split_candidate immediate_split_candidates[] = {
    {"following", "vowel", "+"},
    {"following", "front", "+"},
    {"following", "back", "+"},
    {"following", "close", "+"},
    {"following", "open", "+"},
    {"following", "long", "+"},
    {"preceding", "vowel", "+"},
    {"preceding", "front", "+"},
    {"preceding", "back", "+"},
    {"preceding", "voiced", "+"},
    {"preceding", "voiceless", "+"},
    {"preceding", "consonant", "+"},
    {"preceding", "long", "+"},
    {"position", "initial", "+"},
    {"position", "medial", "+"},
    {"position", "final", "+"},
    {"preceding@2", "front", "+"},
    {"preceding@2", "back", "+"},
    {"preceding@2", "close", "+"},
    {"preceding@2", "open", "+"},
    {"preceding@2", "voiced", "+"},
    {"preceding@2", "voiceless", "+"},
    {"preceding@2", "long", "+"},
    {"preceding@3", "front", "+"},
    {"preceding@3", "back", "+"},
    {"preceding@3", "close", "+"},
    {"preceding@3", "open", "+"},
    {"preceding@3", "voiced", "+"},
    {"preceding@3", "voiceless", "+"},
    {"preceding@3", "long", "+"},
    {"following@2", "front", "+"},
    {"following@2", "back", "+"},
    {"following@2", "close", "+"},
    {"following@2", "open", "+"},
    {"following@2", "voiced", "+"},
    {"following@2", "voiceless", "+"},
    {"following@2", "long", "+"},
    {"following@3", "front", "+"},
    {"following@3", "back", "+"},
    {"following@3", "close", "+"},
    {"following@3", "open", "+"},
    {"following@3", "voiced", "+"},
    {"following@3", "voiceless", "+"},
    {"following@3", "long", "+"},
    {"somewhere_preceding", "front", "+"},
    {"somewhere_preceding", "back", "+"},
    {"somewhere_preceding", "close", "+"},
    {"somewhere_preceding", "open", "+"},
    {"somewhere_preceding", "voiced", "+"},
    {"somewhere_preceding", "voiceless", "+"},
    {"somewhere_preceding", "long", "+"},
    {"somewhere_following", "front", "+"},
    {"somewhere_following", "back", "+"},
    {"somewhere_following", "close", "+"},
    {"somewhere_following", "open", "+"},
    {"somewhere_following", "voiced", "+"},
    {"somewhere_following", "voiceless", "+"},
    {"somewhere_following", "long", "+"}
};

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

static rg_status append_context_observation(
    context_observation **items,
    size_t *count,
    size_t *cap,
    const rg_link *link,
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
    (*items)[*count].source = rg_strdup_internal(link->source[0].grapheme);
    (*items)[*count].target = rg_strdup_internal(link->target[0].grapheme);
    (*items)[*count].weight = weight;
    if ((*items)[*count].source == 0 || (*items)[*count].target == 0) {
        context_observation_clear(&(*items)[*count]);
        return RG_ERR_OOM;
    }
    status = rg_context_spec_copy_internal(&link->context, &(*items)[*count].context);
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

static int predicate_holds(const rg_context_spec *context, const split_candidate *candidate) {
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
    return 0;
}

static rg_status context_from_candidate(const split_candidate *candidate, rg_context_spec *out) {
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

static double group_cost_for_candidate(
    const context_observation *observations,
    size_t observation_count,
    const char *source,
    const split_candidate *candidate,
    int want_yes,
    double *mass_out,
    size_t *target_count_out
) {
    target_mass *targets = 0;
    size_t target_count = 0;
    size_t target_cap = 0;
    size_t i;
    double total = 0.0;
    double cost = 0.0;
    for (i = 0; i < observation_count; i++) {
        int yes;
        if (strcmp(observations[i].source, source) != 0) {
            continue;
        }
        yes = predicate_holds(&observations[i].context, candidate);
        if (yes != want_yes) {
            continue;
        }
        if (add_target_mass(&targets, &target_count, &target_cap, observations[i].target, observations[i].weight) != RG_OK) {
            free(targets);
            *mass_out = 0.0;
            *target_count_out = 0;
            return INFINITY;
        }
        total += observations[i].weight;
    }
    for (i = 0; i < target_count; i++) {
        double p = total <= 0.0 ? 0.0 : targets[i].mass / total;
        if (p > 0.0) {
            cost += -targets[i].mass * log(p);
        }
    }
    free(targets);
    *mass_out = total;
    *target_count_out = target_count;
    return cost;
}

static double baseline_group_cost(
    const context_observation *observations,
    size_t observation_count,
    const char *source,
    double *mass_out,
    size_t *target_count_out
) {
    split_candidate all = {"position", "", "+"};
    target_mass *targets = 0;
    size_t target_count = 0;
    size_t target_cap = 0;
    size_t i;
    double total = 0.0;
    double cost = 0.0;
    (void)all;
    for (i = 0; i < observation_count; i++) {
        if (strcmp(observations[i].source, source) != 0) {
            continue;
        }
        if (add_target_mass(&targets, &target_count, &target_cap, observations[i].target, observations[i].weight) != RG_OK) {
            free(targets);
            *mass_out = 0.0;
            *target_count_out = 0;
            return INFINITY;
        }
        total += observations[i].weight;
    }
    for (i = 0; i < target_count; i++) {
        double p = total <= 0.0 ? 0.0 : targets[i].mass / total;
        if (p > 0.0) {
            cost += -targets[i].mass * log(p);
        }
    }
    free(targets);
    *mass_out = total;
    *target_count_out = target_count;
    return cost;
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

static rg_status commit_candidate_yes_counts(
    rg_pairwise_model *model,
    const context_observation *observations,
    size_t observation_count,
    const char *source,
    const split_candidate *candidate
) {
    rg_context_spec context;
    target_mass *targets = 0;
    size_t target_count = 0;
    size_t target_cap = 0;
    size_t conditioned_cap = model->conditioned_segment_count_count;
    size_t i;
    double source_total = source_total_for_rows(model->segment_counts, model->segment_count_count, source);
    rg_status status;
    status = context_from_candidate(candidate, &context);
    if (status != RG_OK) {
        return status;
    }
    for (i = 0; i < observation_count; i++) {
        if (strcmp(observations[i].source, source) == 0 && predicate_holds(&observations[i].context, candidate)) {
            status = add_target_mass(&targets, &target_count, &target_cap, observations[i].target, observations[i].weight);
            if (status != RG_OK) {
                rg_context_spec_clear_internal(&context);
                free(targets);
                return status;
            }
        }
    }
    for (i = 0; i < target_count; i++) {
        status = add_conditioned_segment_count(
            &model->conditioned_segment_counts,
            &model->conditioned_segment_count_count,
            &conditioned_cap,
            source,
            targets[i].target,
            &context,
            targets[i].mass,
            source_total
        );
        if (status != RG_OK) {
            break;
        }
    }
    rg_context_spec_clear_internal(&context);
    free(targets);
    return status;
}

static rg_status discover_immediate_context_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model
) {
    context_observation *observations = 0;
    size_t observation_count = 0;
    size_t observation_cap = 0;
    const char **sources = 0;
    size_t source_count = 0;
    size_t source_cap = 0;
    size_t i;
    rg_status status = RG_OK;
    int max_chunk_size = RG_DEFAULT_MAX_CHUNK_SIZE;
    double n_total = 0.0;
    double min_obs = 2.0;
    double threshold = 0.0;
    if (ctx == 0 || model == 0 || (pair_count > 0 && pairs == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    if (options != 0) {
        if (options->max_chunk_size > 0) {
            max_chunk_size = options->max_chunk_size;
        }
        if (options->bic.min_split_observations > 0) {
            min_obs = (double)options->bic.min_split_observations;
        }
        threshold = options->bic.delta_bic_threshold;
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
        for (j = 0; j < rg_alignment_link_count(alignment); j++) {
            const rg_link *link = rg_alignment_link_at(alignment, j);
            if (link->source_count == 1 && link->target_count == 1) {
                status = append_context_observation(&observations, &observation_count, &observation_cap, link, weight);
                if (status != RG_OK) {
                    break;
                }
                status = append_unique_source(&sources, &source_count, &source_cap, observations[observation_count - 1].source);
                if (status != RG_OK) {
                    break;
                }
                n_total += weight;
            }
        }
        rg_alignment_free(alignment);
    }
    if (status == RG_OK && observation_count > 0) {
        qsort(sources, source_count, sizeof(*sources), string_ptr_cmp);
        for (i = 0; i < source_count && status == RG_OK; i++) {
            size_t target_count = 0;
            size_t c;
            double mass = 0.0;
            double baseline = baseline_group_cost(observations, observation_count, sources[i], &mass, &target_count);
            const split_candidate *best = 0;
            double best_delta = threshold;
            if (target_count < 2 || mass < min_obs || n_total <= 0.0 || isinf(baseline)) {
                continue;
            }
            for (c = 0; c < sizeof(immediate_split_candidates) / sizeof(immediate_split_candidates[0]); c++) {
                double yes_mass = 0.0;
                double no_mass = 0.0;
                size_t yes_targets = 0;
                size_t no_targets = 0;
                double yes_cost = group_cost_for_candidate(observations, observation_count, sources[i], &immediate_split_candidates[c], 1, &yes_mass, &yes_targets);
                double no_cost = group_cost_for_candidate(observations, observation_count, sources[i], &immediate_split_candidates[c], 0, &no_mass, &no_targets);
                double reduction;
                double delta_bic;
                (void)yes_targets;
                (void)no_targets;
                if (yes_mass < min_obs || no_mass < min_obs || isinf(yes_cost) || isinf(no_cost)) {
                    continue;
                }
                reduction = baseline - (yes_cost + no_cost);
                delta_bic = -2.0 * reduction + log(n_total);
                if (delta_bic < best_delta) {
                    best_delta = delta_bic;
                    best = &immediate_split_candidates[c];
                }
            }
            if (best != 0) {
                status = commit_candidate_yes_counts(model, observations, observation_count, sources[i], best);
            }
        }
    }
    for (i = 0; i < observation_count; i++) {
        context_observation_clear(&observations[i]);
    }
    free(observations);
    free(sources);
    if (status == RG_OK && model->conditioned_segment_count_count > 1) {
        qsort(model->conditioned_segment_counts, model->conditioned_segment_count_count, sizeof(*model->conditioned_segment_counts), conditioned_count_row_cmp);
    }
    return status;
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

static rg_status model_link_cost(
    const rg_context *ctx,
    const rg_train_options *options,
    const rg_pairwise_model *model,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    double *out
) {
    return rg_score_link_with_context_model_internal(ctx, model, options, source, source_count, target, target_count, 0, out);
}

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
    size_t i;
    size_t promoted_cap = 0;
    rg_status status = RG_OK;
    int max_chunk_size = RG_DEFAULT_MAX_CHUNK_SIZE;
    double min_obs = 2.0;
    double n_observations = 0.0;
    if (ctx == 0 || model == 0 || (pair_count > 0 && pairs == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    if (options != 0) {
        if (options->max_chunk_size > 0) {
            max_chunk_size = options->max_chunk_size;
        }
        if (options->bic.min_chunk_observations > 0) {
            min_obs = (double)options->bic.min_chunk_observations;
        }
    }
    for (i = 0; i < pair_count && status == RG_OK; i++) {
        rg_alignment *alignment = 0;
        size_t link_count;
        size_t *src_starts = 0;
        size_t *src_ends = 0;
        size_t *tgt_starts = 0;
        size_t *tgt_ends = 0;
        size_t j;
        size_t s_pos = 0;
        size_t t_pos = 0;
        double weight = pairs[i].weight == 0.0 ? 1.0 : pairs[i].weight;
        if (weight <= 0.0) {
            continue;
        }
        status = rg_align_forms_with_model(ctx, model, options, &pairs[i].source, &pairs[i].target, max_chunk_size, &alignment);
        if (status != RG_OK) {
            break;
        }
        link_count = rg_alignment_link_count(alignment);
        src_starts = (size_t *)calloc(link_count, sizeof(*src_starts));
        src_ends = (size_t *)calloc(link_count, sizeof(*src_ends));
        tgt_starts = (size_t *)calloc(link_count, sizeof(*tgt_starts));
        tgt_ends = (size_t *)calloc(link_count, sizeof(*tgt_ends));
        if ((src_starts == 0 || src_ends == 0 || tgt_starts == 0 || tgt_ends == 0) && link_count > 0) {
            status = RG_ERR_OOM;
        }
        for (j = 0; j < link_count && status == RG_OK; j++) {
            const rg_link *link = rg_alignment_link_at(alignment, j);
            src_starts[j] = s_pos;
            tgt_starts[j] = t_pos;
            s_pos += link->source_count;
            t_pos += link->target_count;
            src_ends[j] = s_pos;
            tgt_ends[j] = t_pos;
            if (link->source_count == 1 && link->target_count == 1) {
                n_observations += weight;
            }
        }
        for (j = 0; j < link_count && status == RG_OK; j++) {
            size_t k;
            for (k = j; k < link_count; k++) {
                const rg_segment *source_chunk = 0;
                const rg_segment *target_chunk = 0;
                size_t source_count = src_ends[k] - src_starts[j];
                size_t target_count = tgt_ends[k] - tgt_starts[j];
                if (source_count > (size_t)max_chunk_size || target_count > (size_t)max_chunk_size) {
                    break;
                }
                if (source_count == 0 || target_count == 0 || source_count + target_count < 3) {
                    continue;
                }
                if (spans_break(src_starts[j], src_ends[k], pairs[i].source.morpheme_breaks, pairs[i].source.morpheme_break_count) ||
                    spans_break(tgt_starts[j], tgt_ends[k], pairs[i].target.morpheme_breaks, pairs[i].target.morpheme_break_count)) {
                    continue;
                }
                status = append_segments_from_link_span(alignment, j, k, 1, &source_chunk, &source_count);
                if (status == RG_OK) {
                    status = append_segments_from_link_span(alignment, j, k, 0, &target_chunk, &target_count);
                }
                if (status == RG_OK) {
                    status = add_chunk_candidate(&candidates, &candidate_count, &candidate_cap, source_chunk, source_count, target_chunk, target_count, weight);
                }
                segment_array_clear(source_chunk, source_count);
                segment_array_clear(target_chunk, target_count);
                if (status != RG_OK) {
                    break;
                }
            }
        }
        free(src_starts);
        free(src_ends);
        free(tgt_starts);
        free(tgt_ends);
        rg_alignment_free(alignment);
    }
    if (n_observations <= 0.0) {
        n_observations = 1.0;
    }
    for (i = 0; i < candidate_count && status == RG_OK; i++) {
        size_t variants = 0;
        double source_total = chunk_source_total(candidates, candidate_count, &candidates[i], &variants);
        double comp_cost = 0.0;
        double promoted_prob;
        double promoted_cost;
        double reduction;
        double k_params;
        double delta_bic;
        rg_chunk_row *next;
        if (candidates[i].count < min_obs || source_total <= 0.0 || variants == 0) {
            continue;
        }
        status = model_link_cost(ctx, options, model, candidates[i].source, candidates[i].source_count, candidates[i].target, candidates[i].target_count, &comp_cost);
        if (status != RG_OK) {
            break;
        }
        promoted_prob = (candidates[i].count + 1.0) / (source_total + (double)variants);
        if (promoted_prob <= 0.0) {
            continue;
        }
        promoted_cost = -log(promoted_prob);
        reduction = candidates[i].count * (comp_cost - promoted_cost);
        k_params = candidates[i].source_count > candidates[i].target_count ? (double)candidates[i].source_count : (double)candidates[i].target_count;
        delta_bic = -2.0 * reduction + k_params * log(n_observations);
        if (delta_bic >= 0.0) {
            continue;
        }
        if (model->chunk_count == promoted_cap) {
            size_t next_cap = promoted_cap == 0 ? 8 : promoted_cap * 2;
            next = (rg_chunk_row *)realloc(model->chunks, next_cap * sizeof(*model->chunks));
            if (next == 0) {
                status = RG_ERR_OOM;
                break;
            }
            model->chunks = next;
            promoted_cap = next_cap;
        }
        memset(&model->chunks[model->chunk_count], 0, sizeof(model->chunks[model->chunk_count]));
        status = segment_array_copy(candidates[i].source, candidates[i].source_count, &model->chunks[model->chunk_count].source);
        if (status == RG_OK) {
            status = segment_array_copy(candidates[i].target, candidates[i].target_count, &model->chunks[model->chunk_count].target);
        }
        if (status != RG_OK) {
            chunk_row_clear(&model->chunks[model->chunk_count]);
            break;
        }
        model->chunks[model->chunk_count].source_count = candidates[i].source_count;
        model->chunks[model->chunk_count].target_count = candidates[i].target_count;
        model->chunks[model->chunk_count].cost = promoted_cost;
        model->chunks[model->chunk_count].count = candidates[i].count;
        model->chunks[model->chunk_count].uncertainty = wilson_interval(candidates[i].count, n_observations);
        model->chunk_count++;
    }
    for (i = 0; i < candidate_count; i++) {
        chunk_candidate_clear(&candidates[i]);
    }
    free(candidates);
    if (status == RG_OK && model->chunk_count > 1) {
        qsort(model->chunks, model->chunk_count, sizeof(*model->chunks), chunk_row_cmp);
    }
    return status;
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

static int segment_has_context_feature(const rg_context *ctx, const rg_form *form, int index, const char *feature) {
    rg_feature_set *features = 0;
    rg_status status;
    int found = 0;
    if (ctx == 0 || form == 0 || feature == 0 || index < 0 || (size_t)index >= form->segment_count) {
        return 0;
    }
    if (form->segments[index].grapheme == 0) {
        return 0;
    }
    status = rg_context_grapheme_features(ctx, form->segments[index].grapheme, &features);
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
    rg_feature_set_free(features);
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
    double source_count
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
    (*rows)[*count].uncertainty = wilson_interval(rule_count, source_count);
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
    size_t feature_i;
    rg_status status = RG_OK;
    double min_count = 3.0;
    double min_confidence = 0.5;
    int max_chunk_size = RG_DEFAULT_MAX_CHUNK_SIZE;
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
        if (options->max_chunk_size > 0) {
            max_chunk_size = options->max_chunk_size;
        }
    }
    for (feature_i = 0; feature_i < sizeof(cross_dimensional_feature_names) / sizeof(cross_dimensional_feature_names[0]) && status == RG_OK; feature_i++) {
        const char *feature = cross_dimensional_feature_names[feature_i];
        int src_offset;
        for (src_offset = -1; src_offset <= 1 && status == RG_OK; src_offset++) {
            tone_mass *tones = 0;
            size_t tone_count = 0;
            size_t tone_cap = 0;
            double src_count = 0.0;
            size_t pair_i;
            for (pair_i = 0; pair_i < pair_count && status == RG_OK; pair_i++) {
                rg_alignment *alignment = 0;
                size_t link_i;
                size_t src_pos = 0;
                size_t tgt_pos = 0;
                double weight = pairs[pair_i].weight == 0.0 ? 1.0 : pairs[pair_i].weight;
                if (weight <= 0.0) {
                    continue;
                }
                status = rg_align_forms_with_model(ctx, model, options, &pairs[pair_i].source, &pairs[pair_i].target, max_chunk_size, &alignment);
                if (status != RG_OK) {
                    break;
                }
                for (link_i = 0; link_i < rg_alignment_link_count(alignment) && status == RG_OK; link_i++) {
                    const rg_link *link = rg_alignment_link_at(alignment, link_i);
                    if (link->source_count == 1 && link->target_count == 1 &&
                        segment_has_context_feature(ctx, &pairs[pair_i].source, (int)src_pos + src_offset, feature)) {
                        const char *tone = pairs[pair_i].target.segments[tgt_pos].tone;
                        if (tone != 0 && tone[0] != '\0') {
                            src_count += weight;
                            status = add_tone_mass(&tones, &tone_count, &tone_cap, tone, weight);
                        }
                    }
                    src_pos += link->source_count;
                    tgt_pos += link->target_count;
                }
                rg_alignment_free(alignment);
            }
            if (status == RG_OK && src_count >= min_count) {
                size_t tone_i;
                for (tone_i = 0; tone_i < tone_count; tone_i++) {
                    double confidence = tones[tone_i].count / src_count;
                    if (tones[tone_i].count >= min_count && confidence >= min_confidence) {
                        status = append_cross_dimensional_row(
                            &rows,
                            &row_count,
                            &row_cap,
                            feature,
                            "+",
                            relative_position_name(src_offset),
                            "tone",
                            tones[tone_i].tone,
                            0,
                            tones[tone_i].count,
                            src_count
                        );
                        if (status != RG_OK) {
                            break;
                        }
                    }
                }
            }
            tone_masses_clear(tones, tone_count);
        }
    }
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

static rg_status collect_pairwise_counts(
    const rg_context *ctx,
    const rg_pairwise_model *scoring_model,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model **out
) {
    rg_pairwise_model *model;
    rg_segment_count_row *rows = 0;
    rg_displacement_count_row *disp_rows = 0;
    rg_tonal_count_row *tonal_rows = 0;
    size_t row_count = 0;
    size_t row_cap = 0;
    size_t disp_count = 0;
    size_t disp_cap = 0;
    size_t tonal_count = 0;
    size_t tonal_cap = 0;
    size_t i;
    rg_status status = RG_OK;
    int max_chunk_size = RG_DEFAULT_MAX_CHUNK_SIZE;

    if (ctx == 0 || out == 0 || (pair_count > 0 && pairs == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
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
        if (scoring_model != 0) {
            status = rg_align_forms_with_model(ctx, scoring_model, options, &pairs[i].source, &pairs[i].target, max_chunk_size, &alignment);
        } else {
            status = rg_align_forms(ctx, &pairs[i].source, &pairs[i].target, max_chunk_size, &alignment);
        }
        if (status != RG_OK) {
            break;
        }
        for (j = 0; j < rg_alignment_link_count(alignment); j++) {
            const rg_link *link = rg_alignment_link_at(alignment, j);
            if (link->source_count == 1 && link->target_count == 1) {
                size_t d;
                status = add_segment_count(&rows, &row_count, &row_cap, link->source[0].grapheme, link->target[0].grapheme, weight);
                if (status != RG_OK) {
                    break;
                }
                for (d = 0; d < link->feature_displacement_count; d++) {
                    status = add_displacement_count(
                        &disp_rows,
                        &disp_count,
                        &disp_cap,
                        link->feature_displacement[d].feature,
                        link->feature_displacement[d].from_value,
                        link->feature_displacement[d].to_value,
                        weight
                    );
                    if (status != RG_OK) {
                        break;
                    }
                }
                if (status != RG_OK) {
                    break;
                }
                status = add_tonal_count(&tonal_rows, &tonal_count, &tonal_cap, link->source[0].tone, link->target[0].tone, weight);
                if (status != RG_OK) {
                    break;
                }
            }
        }
        rg_alignment_free(alignment);
    }
    if (status != RG_OK) {
        for (i = 0; i < row_count; i++) {
            segment_count_row_clear(&rows[i]);
        }
        free(rows);
        for (i = 0; i < disp_count; i++) {
            free((char *)disp_rows[i].feature);
            free((char *)disp_rows[i].from_value);
            free((char *)disp_rows[i].to_value);
        }
        free(disp_rows);
        for (i = 0; i < tonal_count; i++) {
            free((char *)tonal_rows[i].source_tone);
            free((char *)tonal_rows[i].target_tone);
        }
        free(tonal_rows);
        return status;
    }
    fill_source_totals(rows, row_count);
    fill_displacement_total(disp_rows, disp_count);
    fill_tonal_source_totals(tonal_rows, tonal_count);
    qsort(rows, row_count, sizeof(*rows), count_row_cmp);
    qsort(disp_rows, disp_count, sizeof(*disp_rows), displacement_row_cmp);
    qsort(tonal_rows, tonal_count, sizeof(*tonal_rows), tonal_row_cmp);
    model = (rg_pairwise_model *)calloc(1, sizeof(*model));
    if (model == 0) {
        for (i = 0; i < row_count; i++) {
            segment_count_row_clear(&rows[i]);
        }
        free(rows);
        for (i = 0; i < disp_count; i++) {
            free((char *)disp_rows[i].feature);
            free((char *)disp_rows[i].from_value);
            free((char *)disp_rows[i].to_value);
        }
        free(disp_rows);
        for (i = 0; i < tonal_count; i++) {
            free((char *)tonal_rows[i].source_tone);
            free((char *)tonal_rows[i].target_tone);
        }
        free(tonal_rows);
        return RG_ERR_OOM;
    }
    model->segment_counts = rows;
    model->segment_count_count = row_count;
    model->displacement_counts = disp_rows;
    model->displacement_count_count = disp_count;
    model->tonal_counts = tonal_rows;
    model->tonal_count_count = tonal_count;
    *out = model;
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

rg_status rg_train_pairwise_segment_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model **out
) {
    rg_pairwise_model *model = 0;
    rg_pairwise_model *next = 0;
    rg_train_options defaults;
    const rg_train_options *opts = options;
    int max_iter;
    int iter;
    double prev_cost = INFINITY;
    double eps;
    rg_status status;

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

    status = collect_pairwise_counts(ctx, 0, pairs, pair_count, opts, &model);
    if (status != RG_OK) {
        return status;
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
        status = collect_pairwise_counts(ctx, model, pairs, pair_count, opts, &next);
        if (status != RG_OK) {
            rg_pairwise_model_free(model);
            return status;
        }
        rg_pairwise_model_free(model);
        model = next;
        next = 0;
    }
    status = discover_immediate_context_counts(ctx, pairs, pair_count, opts, model);
    if (status != RG_OK) {
        rg_pairwise_model_free(model);
        return status;
    }
    status = promote_chunk_rows(ctx, pairs, pair_count, opts, model);
    if (status != RG_OK) {
        rg_pairwise_model_free(model);
        return status;
    }
    status = discover_cross_dimensional_rows(ctx, pairs, pair_count, opts, model);
    if (status != RG_OK) {
        rg_pairwise_model_free(model);
        return status;
    }
    *out = model;
    return RG_OK;
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

size_t rg_pairwise_model_displacement_count_row_count(const rg_pairwise_model *model) {
    if (model == 0) {
        return 0;
    }
    return model->displacement_count_count;
}

const rg_displacement_count_row *rg_pairwise_model_displacement_count_row_at(
    const rg_pairwise_model *model,
    size_t index
) {
    if (model == 0 || index >= model->displacement_count_count) {
        return 0;
    }
    return &model->displacement_counts[index];
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
