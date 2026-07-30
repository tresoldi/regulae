#include "internal.h"

#include <stdlib.h>
#include <string.h>

static int feature_cmp(const void *a, const void *b) {
    const char *const *sa = (const char *const *)a;
    const char *const *sb = (const char *const *)b;
    return strcmp(*sa, *sb);
}

static int feature_set_contains_name(const rg_feature_set *features, const char *name) {
    size_t i;
    for (i = 0; i < rg_feature_set_size(features); i++) {
        const char *item = rg_feature_set_get(features, i);
        if (item != 0 && strcmp(item, name) == 0) {
            return 1;
        }
    }
    return 0;
}

static rg_status append_displacement(
    rg_feature_displacement **items,
    size_t *count,
    size_t *cap,
    const char *feature,
    const char *from_value,
    const char *to_value
) {
    rg_feature_displacement *next;
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 8 : *cap * 2;
        next = (rg_feature_displacement *)realloc(*items, next_cap * sizeof(**items));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *items = next;
        *cap = next_cap;
    }
    (*items)[*count].feature = rg_strdup_internal(feature);
    (*items)[*count].from_value = rg_strdup_internal(from_value);
    (*items)[*count].to_value = rg_strdup_internal(to_value);
    if ((*items)[*count].feature == 0 || (*items)[*count].from_value == 0 || (*items)[*count].to_value == 0) {
        free((char *)(*items)[*count].feature);
        free((char *)(*items)[*count].from_value);
        free((char *)(*items)[*count].to_value);
        (*items)[*count].feature = 0;
        (*items)[*count].from_value = 0;
        (*items)[*count].to_value = 0;
        return RG_ERR_OOM;
    }
    (*count)++;
    return RG_OK;
}

static rg_status append_difference(
    const rg_feature_set *left,
    const rg_feature_set *right,
    const char *from_value,
    const char *to_value,
    rg_feature_displacement **items,
    size_t *count,
    size_t *cap
) {
    const char **names = 0;
    size_t name_count = 0;
    size_t i;
    rg_status status;
    names = (const char **)calloc(rg_feature_set_size(left), sizeof(*names));
    if (names == 0 && rg_feature_set_size(left) > 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < rg_feature_set_size(left); i++) {
        const char *name = rg_feature_set_get(left, i);
        if (name != 0 && !feature_set_contains_name(right, name)) {
            names[name_count++] = name;
        }
    }
    qsort(names, name_count, sizeof(*names), feature_cmp);
    for (i = 0; i < name_count; i++) {
        status = append_displacement(items, count, cap, names[i], from_value, to_value);
        if (status != RG_OK) {
            free(names);
            return status;
        }
    }
    free(names);
    return RG_OK;
}

rg_status rg_compute_displacement(
    const rg_context *ctx,
    rg_segment source,
    rg_segment target,
    rg_feature_displacement **out,
    size_t *out_count
) {
    rg_feature_set *source_features = 0;
    rg_feature_set *target_features = 0;
    rg_feature_displacement *items = 0;
    size_t count = 0;
    size_t cap = 0;
    rg_status status;

    if (ctx == 0 || source.grapheme == 0 || target.grapheme == 0 || out == 0 || out_count == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    *out_count = 0;
    status = rg_context_grapheme_features(ctx, source.grapheme, &source_features);
    if (status != RG_OK) {
        return status;
    }
    status = rg_context_grapheme_features(ctx, target.grapheme, &target_features);
    if (status != RG_OK) {
        rg_feature_set_free(source_features);
        return status;
    }
    status = append_difference(source_features, target_features, "present", "absent", &items, &count, &cap);
    if (status == RG_OK) {
        status = append_difference(target_features, source_features, "absent", "present", &items, &count, &cap);
    }
    rg_feature_set_free(source_features);
    rg_feature_set_free(target_features);
    if (status != RG_OK) {
        rg_feature_displacement_free(items, count);
        return status;
    }
    *out = items;
    *out_count = count;
    return RG_OK;
}

void rg_feature_displacement_free(rg_feature_displacement *items, size_t count) {
    size_t i;
    if (items == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        free((char *)items[i].feature);
        free((char *)items[i].from_value);
        free((char *)items[i].to_value);
    }
    free(items);
}

static rg_status segment_distance_checked(
    const rg_context *ctx,
    rg_segment source,
    rg_segment target,
    double *out
) {
    int is_segment = 0;
    rg_status status;
    status = rg_context_is_segment(ctx, source.grapheme, &is_segment);
    if (status != RG_OK) {
        return status;
    }
    if (!is_segment) {
        return RG_ERR_UNKNOWN_GRAPHEME;
    }
    status = rg_context_is_segment(ctx, target.grapheme, &is_segment);
    if (status != RG_OK) {
        return status;
    }
    if (!is_segment) {
        return RG_ERR_UNKNOWN_GRAPHEME;
    }
    return rg_context_segment_distance(ctx, source.grapheme, target.grapheme, out);
}

rg_status rg_score_link(
    const rg_context *ctx,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    double *out
) {
    size_t paired;
    size_t i;
    double pair_cost = 0.0;
    int asymmetry;
    rg_status status;
    if (ctx == 0 || out == 0 || (source_count > 0 && source == 0) || (target_count > 0 && target == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0.0;
    if (source_count == 0 && target_count == 0) {
        return RG_OK;
    }
    if (source_count == 0) {
        *out = RG_DEFAULT_GAP_COST * (double)target_count;
        return RG_OK;
    }
    if (target_count == 0) {
        *out = RG_DEFAULT_GAP_COST * (double)source_count;
        return RG_OK;
    }
    if (source_count == 1 && target_count == 1) {
        return segment_distance_checked(ctx, source[0], target[0], out);
    }
    paired = source_count < target_count ? source_count : target_count;
    for (i = 0; i < paired; i++) {
        double distance = 0.0;
        status = segment_distance_checked(ctx, source[i], target[i], &distance);
        if (status != RG_OK) {
            return status;
        }
        pair_cost += distance;
    }
    asymmetry = (int)source_count - (int)target_count;
    if (asymmetry < 0) {
        asymmetry = -asymmetry;
    }
    *out = pair_cost + RG_DEFAULT_GAP_COST * (double)asymmetry + RG_DEFAULT_CHUNK_PENALTY * (double)asymmetry;
    return RG_OK;
}
