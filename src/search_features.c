#include "search_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int feature_set_contains(const rg_feature_set *features, const char *feature) {
    size_t i;
    size_t count;
    if (features == 0 || feature == 0) {
        return 0;
    }
    count = rg_feature_set_size(features);
    for (i = 0; i < count; i++) {
        const char *item = rg_feature_set_get(features, i);
        if (item != 0 && strcmp(item, feature) == 0) {
            return 1;
        }
    }
    return 0;
}

rg_status context_features_for_segment(
    const rg_context *ctx,
    const rg_segment *segment,
    const rg_feature_constraint **out,
    size_t *out_count
) {
    if (ctx == 0 || segment == 0 || segment->grapheme == 0 || out == 0 || out_count == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    return rg_context_constraints_internal(ctx, segment->grapheme, out, out_count);
}

/* The rows are borrowed from the context's per-grapheme cache; only the index
 * arrays belong to the matrix. */
void feature_matrix_clear(
    const rg_feature_constraint **features,
    const size_t *feature_counts,
    size_t count
) {
    (void)count;
    free(features);
    rg_free_owned_internal(feature_counts);
}

rg_status feature_matrix_build(
    const rg_context *ctx,
    const rg_form *form,
    const rg_feature_constraint ***out_features,
    size_t **out_counts
) {
    const rg_feature_constraint **features;
    size_t *counts;
    size_t i;
    rg_status status = RG_OK;
    if (ctx == 0 || form == 0 || out_features == 0 || out_counts == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out_features = 0;
    *out_counts = 0;
    if (form->segment_count == 0) {
        return RG_OK;
    }
    features = (const rg_feature_constraint **)calloc(form->segment_count, sizeof(*features));
    counts = (size_t *)calloc(form->segment_count, sizeof(*counts));
    if (features == 0 || counts == 0) {
        free(features);
        free(counts);
        return RG_ERR_OOM;
    }
    for (i = 0; i < form->segment_count; i++) {
        status = context_features_for_segment(ctx, &form->segments[i], &features[i], &counts[i]);
        if (status != RG_OK) {
            /* NOLINTNEXTLINE(clang-analyzer-unix.Malloc): feature_matrix_clear
             * frees both arrays. It releases counts through
             * rg_free_owned_internal, whose pointer copy the analyzer does not
             * follow, so it reports the free it cannot see as a leak. */
            feature_matrix_clear(features, counts, form->segment_count);
            /* NOLINTNEXTLINE(clang-analyzer-unix.Malloc) */
            return status;
        }
    }
    *out_features = features;
    *out_counts = counts;
    return RG_OK;
}

rg_status context_copy_constraints(
    const rg_feature_constraint *src,
    size_t count,
    const rg_feature_constraint **out,
    size_t *out_count
) {
    rg_status status;
    if (out == 0 || out_count == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    *out_count = 0;
    status = rg_feature_constraint_array_copy_internal(src, count, out);
    if (status != RG_OK) {
        return status;
    }
    *out_count = count;
    return RG_OK;
}

rg_status context_copy_stress(
    const char *stress,
    const rg_feature_constraint **out,
    size_t *out_count
) {
    rg_feature_constraint constraint;
    if (stress == 0 || stress[0] == '\0') {
        *out = 0;
        *out_count = 0;
        return RG_OK;
    }
    constraint.feature = "stress";
    constraint.value = stress;
    return context_copy_constraints(&constraint, 1, out, out_count);
}

rg_status distance_context_copy_two(
    const rg_feature_constraint *features_a,
    size_t feature_count_a,
    int offset_a,
    const rg_feature_constraint *features_b,
    size_t feature_count_b,
    int offset_b,
    const rg_distance_constraint **out,
    size_t *out_count
) {
    rg_distance_constraint *copy;
    size_t i;
    size_t pos = 0;
    rg_status status;
    if (out == 0 || out_count == 0 ||
        (feature_count_a > 0 && features_a == 0) ||
        (feature_count_b > 0 && features_b == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    *out_count = 0;
    if (feature_count_a + feature_count_b == 0) {
        return RG_OK;
    }
    copy = (rg_distance_constraint *)calloc(feature_count_a + feature_count_b, sizeof(*copy));
    if (copy == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < feature_count_a; i++) {
        copy[pos].offset = offset_a;
        status = rg_feature_constraint_copy_internal(&features_a[i], &copy[pos].constraint);
        if (status != RG_OK) {
            while (pos > 0) {
                pos--;
                rg_free_owned_internal(copy[pos].constraint.feature);
                rg_free_owned_internal(copy[pos].constraint.value);
            }
            free(copy);
            return status;
        }
        pos++;
    }
    for (i = 0; i < feature_count_b; i++) {
        copy[pos].offset = offset_b;
        status = rg_feature_constraint_copy_internal(&features_b[i], &copy[pos].constraint);
        if (status != RG_OK) {
            while (pos > 0) {
                pos--;
                rg_free_owned_internal(copy[pos].constraint.feature);
                rg_free_owned_internal(copy[pos].constraint.value);
            }
            free(copy);
            return status;
        }
        pos++;
    }
    *out = copy;
    *out_count = feature_count_a + feature_count_b;
    return RG_OK;
}


/* Union of the context features over [start, end), optionally skipping one
 * index. Feature names are emitted in the fixed alphabetical order of
 * context_feature_names. */
rg_status context_feature_union_copy_excluding(
    const rg_feature_constraint *const *source_features,
    const size_t *source_feature_counts,
    size_t start,
    size_t end,
    size_t exclude,
    int has_exclude,
    const rg_feature_constraint **out,
    size_t *out_count
) {
    rg_feature_constraint *constraints;
    size_t capacity = 0;
    size_t i;
    size_t count = 0;
    if (out == 0 || out_count == 0 || (end > start && (source_features == 0 || source_feature_counts == 0))) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    *out_count = 0;
    /* The union of what the span carries, deduplicated. Built from the
     * segments rather than from a list of feature names, because there is no
     * such list: what a grapheme carries is whatever the merkmal system in use
     * reports for it. */
    for (i = start; i < end; i++) {
        if (has_exclude && i == exclude) {
            continue;
        }
        capacity += source_feature_counts[i];
    }
    if (capacity == 0) {
        return RG_OK;
    }
    constraints = (rg_feature_constraint *)calloc(capacity, sizeof(*constraints));
    if (constraints == 0) {
        return RG_ERR_OOM;
    }
    for (i = start; i < end; i++) {
        size_t c;
        if (has_exclude && i == exclude) {
            continue;
        }
        for (c = 0; c < source_feature_counts[i]; c++) {
            const rg_feature_constraint *item = &source_features[i][c];
            size_t k;
            int seen = 0;
            for (k = 0; k < count && !seen; k++) {
                seen = strcmp(constraints[k].feature, item->feature) == 0 &&
                       strcmp(constraints[k].value, item->value) == 0;
            }
            if (!seen) {
                constraints[count++] = *item;
            }
        }
    }
    if (count == 0) {
        free(constraints);
        return RG_OK;
    }
    /* Sorted so the published environment does not depend on segment order. */
    for (i = 1; i < count; i++) {
        rg_feature_constraint key = constraints[i];
        size_t j = i;
        while (j > 0) {
            int c = strcmp(constraints[j - 1].feature, key.feature);
            if (c == 0) {
                c = strcmp(constraints[j - 1].value, key.value);
            }
            if (c <= 0) {
                break;
            }
            constraints[j] = constraints[j - 1];
            j--;
        }
        constraints[j] = key;
    }
    {
        rg_status status = rg_feature_constraint_array_copy_internal(constraints, count, out);
        free(constraints);
        if (status != RG_OK) {
            return status;
        }
    }
    *out_count = count;
    return RG_OK;
}

rg_status context_feature_union_copy(
    const rg_feature_constraint *const *source_features,
    const size_t *source_feature_counts,
    size_t start,
    size_t end,
    const rg_feature_constraint **out,
    size_t *out_count
) {
    return context_feature_union_copy_excluding(
        source_features,
        source_feature_counts,
        start,
        end,
        0,
        0,
        out,
        out_count
    );
}

