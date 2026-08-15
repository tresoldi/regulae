#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct rg_dp_step {
    size_t prev_i;
    size_t prev_j;
    size_t source_count;
    size_t target_count;
    int has_step;
} rg_dp_step;

struct rg_alignment {
    rg_form source_form;
    rg_form target_form;
    rg_link *links;
    size_t link_count;
};

static void form_clear(rg_form *form) {
    size_t i;
    if (form == 0) {
        return;
    }
    free((char *)form->lect_id);
    for (i = 0; i < form->segment_count; i++) {
        rg_segment_clear_internal((rg_segment *)&form->segments[i]);
    }
    free((rg_segment *)form->segments);
    free((int *)form->syllable_breaks);
    free((int *)form->morpheme_breaks);
    memset(form, 0, sizeof(*form));
}

static rg_status copy_ints(const int *items, size_t count, const int **out) {
    int *copy;
    if (count == 0) {
        *out = 0;
        return RG_OK;
    }
    if (items == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    copy = (int *)calloc(count, sizeof(*copy));
    if (copy == 0) {
        return RG_ERR_OOM;
    }
    memcpy(copy, items, count * sizeof(*copy));
    *out = copy;
    return RG_OK;
}

static rg_status form_copy(const rg_form *src, rg_form *out) {
    rg_segment *segments = 0;
    size_t i;
    rg_status status;
    if (src == 0 || out == 0 || (src->segment_count > 0 && src->segments == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    if (src->lect_id != 0) {
        out->lect_id = rg_strdup_internal(src->lect_id);
        if (out->lect_id == 0) {
            return RG_ERR_OOM;
        }
    }
    if (src->segment_count > 0) {
        segments = (rg_segment *)calloc(src->segment_count, sizeof(*segments));
        if (segments == 0) {
            form_clear(out);
            return RG_ERR_OOM;
        }
        for (i = 0; i < src->segment_count; i++) {
            status = rg_segment_copy_internal(&src->segments[i], &segments[i]);
            if (status != RG_OK) {
                while (i > 0) {
                    i--;
                    rg_segment_clear_internal(&segments[i]);
                }
                free(segments);
                form_clear(out);
                return status;
            }
        }
        out->segments = segments;
        out->segment_count = src->segment_count;
    }
    status = copy_ints(src->syllable_breaks, src->syllable_break_count, &out->syllable_breaks);
    if (status != RG_OK) {
        form_clear(out);
        return status;
    }
    out->syllable_break_count = src->syllable_break_count;
    status = copy_ints(src->morpheme_breaks, src->morpheme_break_count, &out->morpheme_breaks);
    if (status != RG_OK) {
        form_clear(out);
        return status;
    }
    out->morpheme_break_count = src->morpheme_break_count;
    return RG_OK;
}

static int feature_set_contains(const rg_feature_set *features, const char *feature) {
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

static rg_status context_features_for_segment(
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
static void feature_matrix_clear(
    const rg_feature_constraint **features,
    const size_t *feature_counts,
    size_t count
) {
    (void)count;
    free(features);
    free((size_t *)feature_counts);
}

static rg_status feature_matrix_build(
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
            feature_matrix_clear(features, counts, form->segment_count);
            return status;
        }
    }
    *out_features = features;
    *out_counts = counts;
    return RG_OK;
}

static rg_status context_copy_constraints(
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

static rg_status context_copy_stress(
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

static rg_status distance_context_copy_two(
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
                free((char *)copy[pos].constraint.feature);
                free((char *)copy[pos].constraint.value);
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
                free((char *)copy[pos].constraint.feature);
                free((char *)copy[pos].constraint.value);
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
 * context_feature_names, which is what the Go reference produces by sorting. */
static rg_status context_feature_union_copy_excluding(
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

static rg_status context_feature_union_copy(
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

/* syllable_data holds the syllable-structural arrays a form's link contexts
 * need: which syllable each segment belongs to, the feature union of each
 * syllable, and the feature union of each segment's own syllable excluding
 * itself. Built once per alignment, mirroring Go's computeLongRangeData. */
typedef struct syllable_data {
    size_t segment_count;
    size_t syllable_count;
    size_t *syllable_of;
    const rg_feature_constraint **syllable_features;
    size_t *syllable_feature_counts;
    const rg_feature_constraint **same_syllable_excluding;
    size_t *same_syllable_excluding_counts;
    /* Cumulative feature unions over [0, i) and [i, n), so the existential
     * slots of a link context are a lookup rather than a rescan. Recomputing
     * them per DP cell dominated training time. */
    const rg_feature_constraint **left_cumulative;
    size_t *left_cumulative_counts;
    const rg_feature_constraint **right_cumulative;
    size_t *right_cumulative_counts;
    /* Distance-bounded slots at offsets 2 and 3, indexed by span start and
     * span end, and the one-element stress constraint of each position. With
     * these every slot of a link context is a precomputed array, which is what
     * lets the DP score against a borrowed context instead of building an
     * owned one per cell. */
    const rg_distance_constraint **preceding_at_distance;
    size_t *preceding_at_distance_counts;
    const rg_distance_constraint **following_at_distance;
    size_t *following_at_distance_counts;
    const rg_feature_constraint **stress;
    size_t *stress_counts;
} syllable_data;

static void syllable_data_clear(syllable_data *data) {
    size_t i;
    if (data == 0) {
        return;
    }
    if (data->syllable_features != 0) {
        for (i = 0; i < data->syllable_count; i++) {
            rg_feature_constraint_array_clear_internal(
                data->syllable_features[i],
                data->syllable_feature_counts == 0 ? 0 : data->syllable_feature_counts[i]
            );
        }
    }
    if (data->same_syllable_excluding != 0) {
        for (i = 0; i < data->segment_count; i++) {
            rg_feature_constraint_array_clear_internal(
                data->same_syllable_excluding[i],
                data->same_syllable_excluding_counts == 0 ? 0 : data->same_syllable_excluding_counts[i]
            );
        }
    }
    if (data->left_cumulative != 0) {
        for (i = 0; i <= data->segment_count; i++) {
            rg_feature_constraint_array_clear_internal(
                data->left_cumulative[i],
                data->left_cumulative_counts == 0 ? 0 : data->left_cumulative_counts[i]
            );
        }
    }
    if (data->right_cumulative != 0) {
        for (i = 0; i <= data->segment_count; i++) {
            rg_feature_constraint_array_clear_internal(
                data->right_cumulative[i],
                data->right_cumulative_counts == 0 ? 0 : data->right_cumulative_counts[i]
            );
        }
    }
    free(data->syllable_of);
    free(data->syllable_features);
    free(data->syllable_feature_counts);
    free(data->same_syllable_excluding);
    free(data->same_syllable_excluding_counts);
    if (data->preceding_at_distance != 0) {
        for (i = 0; i <= data->segment_count; i++) {
            rg_distance_constraint_array_clear_internal(
                data->preceding_at_distance[i],
                data->preceding_at_distance_counts == 0 ? 0 : data->preceding_at_distance_counts[i]
            );
        }
    }
    if (data->following_at_distance != 0) {
        for (i = 0; i <= data->segment_count; i++) {
            rg_distance_constraint_array_clear_internal(
                data->following_at_distance[i],
                data->following_at_distance_counts == 0 ? 0 : data->following_at_distance_counts[i]
            );
        }
    }
    if (data->stress != 0) {
        for (i = 0; i < data->segment_count; i++) {
            rg_feature_constraint_array_clear_internal(
                data->stress[i],
                data->stress_counts == 0 ? 0 : data->stress_counts[i]
            );
        }
    }
    free(data->left_cumulative);
    free(data->left_cumulative_counts);
    free(data->right_cumulative);
    free(data->right_cumulative_counts);
    free(data->preceding_at_distance);
    free(data->preceding_at_distance_counts);
    free(data->following_at_distance);
    free(data->following_at_distance_counts);
    free(data->stress);
    free(data->stress_counts);
    memset(data, 0, sizeof(*data));
}

static rg_status syllable_data_build(
    const rg_context *ctx,
    const rg_form *form,
    const rg_feature_constraint *const *source_features,
    const size_t *source_feature_counts,
    syllable_data *out
) {
    size_t *breaks = 0;
    size_t break_count = 0;
    size_t *starts = 0;
    size_t syllable_count;
    size_t n;
    size_t i;
    size_t s;
    rg_status status;

    if (out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    if (form == 0 || form->segment_count == 0) {
        return RG_OK;
    }
    n = form->segment_count;

    status = rg_compute_syllable_breaks_internal(ctx, form, &breaks, &break_count);
    if (status != RG_OK) {
        return status;
    }

    starts = (size_t *)calloc(break_count + 2, sizeof(*starts));
    if (starts == 0) {
        free(breaks);
        return RG_ERR_OOM;
    }
    starts[0] = 0;
    for (i = 0; i < break_count; i++) {
        starts[i + 1] = breaks[i] > n ? n : breaks[i];
    }
    starts[break_count + 1] = n;
    free(breaks);
    syllable_count = break_count + 1;

    out->segment_count = n;
    out->syllable_count = syllable_count;
    out->syllable_of = (size_t *)calloc(n, sizeof(*out->syllable_of));
    out->syllable_features = (const rg_feature_constraint **)calloc(syllable_count, sizeof(*out->syllable_features));
    out->syllable_feature_counts = (size_t *)calloc(syllable_count, sizeof(*out->syllable_feature_counts));
    out->same_syllable_excluding = (const rg_feature_constraint **)calloc(n, sizeof(*out->same_syllable_excluding));
    out->same_syllable_excluding_counts = (size_t *)calloc(n, sizeof(*out->same_syllable_excluding_counts));
    if (out->syllable_of == 0 || out->syllable_features == 0 || out->syllable_feature_counts == 0 ||
        out->same_syllable_excluding == 0 || out->same_syllable_excluding_counts == 0) {
        free(starts);
        syllable_data_clear(out);
        return RG_ERR_OOM;
    }

    s = 0;
    for (i = 0; i < n; i++) {
        while (s + 1 < syllable_count && i >= starts[s + 1]) {
            s++;
        }
        out->syllable_of[i] = s;
    }

    for (s = 0; s < syllable_count; s++) {
        status = context_feature_union_copy(
            source_features,
            source_feature_counts,
            starts[s],
            starts[s + 1],
            &out->syllable_features[s],
            &out->syllable_feature_counts[s]
        );
        if (status != RG_OK) {
            free(starts);
            syllable_data_clear(out);
            return status;
        }
    }

    for (i = 0; i < n; i++) {
        size_t idx = out->syllable_of[i];
        status = context_feature_union_copy_excluding(
            source_features,
            source_feature_counts,
            starts[idx],
            starts[idx + 1],
            i,
            1,
            &out->same_syllable_excluding[i],
            &out->same_syllable_excluding_counts[i]
        );
        if (status != RG_OK) {
            free(starts);
            syllable_data_clear(out);
            return status;
        }
    }

    out->left_cumulative = (const rg_feature_constraint **)calloc(n + 1, sizeof(*out->left_cumulative));
    out->left_cumulative_counts = (size_t *)calloc(n + 1, sizeof(*out->left_cumulative_counts));
    out->right_cumulative = (const rg_feature_constraint **)calloc(n + 1, sizeof(*out->right_cumulative));
    out->right_cumulative_counts = (size_t *)calloc(n + 1, sizeof(*out->right_cumulative_counts));
    if (out->left_cumulative == 0 || out->left_cumulative_counts == 0 ||
        out->right_cumulative == 0 || out->right_cumulative_counts == 0) {
        free(starts);
        syllable_data_clear(out);
        return RG_ERR_OOM;
    }
    for (i = 0; i <= n; i++) {
        status = context_feature_union_copy(
            source_features, source_feature_counts, 0, i,
            &out->left_cumulative[i], &out->left_cumulative_counts[i]
        );
        if (status == RG_OK) {
            status = context_feature_union_copy(
                source_features, source_feature_counts, i, n,
                &out->right_cumulative[i], &out->right_cumulative_counts[i]
            );
        }
        if (status != RG_OK) {
            free(starts);
            syllable_data_clear(out);
            return status;
        }
    }

    out->preceding_at_distance = (const rg_distance_constraint **)calloc(n + 1, sizeof(*out->preceding_at_distance));
    out->preceding_at_distance_counts = (size_t *)calloc(n + 1, sizeof(*out->preceding_at_distance_counts));
    out->following_at_distance = (const rg_distance_constraint **)calloc(n + 1, sizeof(*out->following_at_distance));
    out->following_at_distance_counts = (size_t *)calloc(n + 1, sizeof(*out->following_at_distance_counts));
    out->stress = (const rg_feature_constraint **)calloc(n, sizeof(*out->stress));
    out->stress_counts = (size_t *)calloc(n, sizeof(*out->stress_counts));
    if (out->preceding_at_distance == 0 || out->preceding_at_distance_counts == 0 ||
        out->following_at_distance == 0 || out->following_at_distance_counts == 0 ||
        out->stress == 0 || out->stress_counts == 0) {
        free(starts);
        syllable_data_clear(out);
        return RG_ERR_OOM;
    }
    for (i = 0; i <= n; i++) {
        const rg_feature_constraint *pre2 = 0;
        const rg_feature_constraint *pre3 = 0;
        const rg_feature_constraint *fol2 = 0;
        const rg_feature_constraint *fol3 = 0;
        size_t pre2_count = 0;
        size_t pre3_count = 0;
        size_t fol2_count = 0;
        size_t fol3_count = 0;
        if (i >= 2) {
            pre2 = source_features[i - 2];
            pre2_count = source_feature_counts[i - 2];
        }
        if (i >= 3) {
            pre3 = source_features[i - 3];
            pre3_count = source_feature_counts[i - 3];
        }
        if (i + 1 < n) {
            fol2 = source_features[i + 1];
            fol2_count = source_feature_counts[i + 1];
        }
        if (i + 2 < n) {
            fol3 = source_features[i + 2];
            fol3_count = source_feature_counts[i + 2];
        }
        status = distance_context_copy_two(
            pre2, pre2_count, 2, pre3, pre3_count, 3,
            &out->preceding_at_distance[i], &out->preceding_at_distance_counts[i]
        );
        if (status == RG_OK) {
            status = distance_context_copy_two(
                fol2, fol2_count, 2, fol3, fol3_count, 3,
                &out->following_at_distance[i], &out->following_at_distance_counts[i]
            );
        }
        if (status != RG_OK) {
            free(starts);
            syllable_data_clear(out);
            return status;
        }
    }
    for (i = 0; i < n; i++) {
        status = context_copy_stress(form->segments[i].stress, &out->stress[i], &out->stress_counts[i]);
        if (status != RG_OK) {
            free(starts);
            syllable_data_clear(out);
            return status;
        }
    }

    free(starts);
    return RG_OK;
}

/* Fills a context whose every slot points into the form's precomputed arrays.
 * Nothing is allocated and nothing must be cleared: the result is valid only
 * while the syllable_data and the feature matrix live, and only for reading.
 * The DP scores millions of these, so building an owned copy per cell was the
 * single largest cost in training. */
/* Where a segment sits in its own morpheme, and which morpheme that is.
 *
 * Boundaries are indices into the segment sequence: a break at i means a new
 * morpheme starts at i. A form with no boundaries reports neither value, so
 * the axis simply does not exist for corpora that do not carry them -- which
 * is most of them -- and the contrastive filter drops it.
 *
 * The names are borrowed from static storage and the index from a small table,
 * so a borrowed context owns nothing here. Words longer than the table are
 * reported as being in its last slot rather than not at all: a rule about the
 * ninth morpheme of a word is not one this is going to find. */
static const char *const morpheme_index_names[] = {
    "0", "1", "2", "3", "4", "5", "6", "7"
};

static void morpheme_placement(
    const rg_form *form,
    size_t start,
    size_t end,
    const char **out_position,
    const char **out_index
) {
    size_t index = 0;
    size_t morpheme_start = 0;
    size_t morpheme_end = form->segment_count;
    size_t i;

    *out_position = 0;
    *out_index = 0;
    if (form->morpheme_break_count == 0 || form->morpheme_breaks == 0) {
        return;
    }
    for (i = 0; i < form->morpheme_break_count; i++) {
        size_t at;
        if (form->morpheme_breaks[i] < 0) {
            continue;
        }
        at = (size_t)form->morpheme_breaks[i];
        if (at <= start) {
            if (at > morpheme_start) {
                morpheme_start = at;
            }
            if (at > 0) {
                index++;
            }
        } else if (at < morpheme_end) {
            morpheme_end = at;
        }
    }
    if (index >= sizeof(morpheme_index_names) / sizeof(morpheme_index_names[0])) {
        index = sizeof(morpheme_index_names) / sizeof(morpheme_index_names[0]) - 1;
    }
    *out_index = morpheme_index_names[index];
    if (morpheme_start == start && end == morpheme_end) {
        *out_position = "only";
    } else if (morpheme_start == start) {
        *out_position = "initial";
    } else if (end == morpheme_end) {
        *out_position = "final";
    } else {
        *out_position = "internal";
    }
}

static void build_link_context_borrowed(
    const rg_form *source,
    const rg_feature_constraint *const *source_features,
    const size_t *source_feature_counts,
    const syllable_data *syllables,
    size_t source_start,
    size_t source_count,
    size_t target_count,
    rg_context_spec *out
) {
    size_t source_end = source_start + source_count;
    size_t n = source->segment_count;

    rg_context_spec_init_empty(out);
    if (source_start == 0) {
        out->position = "initial";
    } else if (source_end == n) {
        out->position = "final";
    } else {
        out->position = "medial";
    }
    morpheme_placement(source, source_start, source_end, &out->morphological, &out->morpheme_index);
    if (source_start > 0) {
        out->preceding = source_features[source_start - 1];
        out->preceding_count = source_feature_counts[source_start - 1];
    }
    if (source_end < n) {
        out->following = source_features[source_end];
        out->following_count = source_feature_counts[source_end];
    }
    if (syllables == 0 || syllables->segment_count == 0) {
        return;
    }
    out->preceding_at_distance = syllables->preceding_at_distance[source_start];
    out->preceding_at_distance_count = syllables->preceding_at_distance_counts[source_start];
    out->following_at_distance = syllables->following_at_distance[source_end];
    out->following_at_distance_count = syllables->following_at_distance_counts[source_end];
    out->somewhere_preceding = syllables->left_cumulative[source_start];
    out->somewhere_preceding_count = syllables->left_cumulative_counts[source_start];
    out->somewhere_following = syllables->right_cumulative[source_end];
    out->somewhere_following_count = syllables->right_cumulative_counts[source_end];

    /* Syllable and stress slots mirror the DP: only 1-to-1 links carry them. */
    if (source_count != 1 || target_count != 1 || source_start >= n) {
        return;
    }
    {
        size_t syllable_index = syllables->syllable_of[source_start];
        out->same_syllable = syllables->same_syllable_excluding[source_start];
        out->same_syllable_count = syllables->same_syllable_excluding_counts[source_start];
        if (syllable_index + 1 < syllables->syllable_count) {
            out->next_syllable = syllables->syllable_features[syllable_index + 1];
            out->next_syllable_count = syllables->syllable_feature_counts[syllable_index + 1];
        }
        if (syllable_index > 0) {
            out->previous_syllable = syllables->syllable_features[syllable_index - 1];
            out->previous_syllable_count = syllables->syllable_feature_counts[syllable_index - 1];
        }
    }
    out->self_stress = syllables->stress[source_start];
    out->self_stress_count = syllables->stress_counts[source_start];
    if (source_start > 0) {
        out->preceding_stress = syllables->stress[source_start - 1];
        out->preceding_stress_count = syllables->stress_counts[source_start - 1];
    }
    if (source_end < n) {
        out->following_stress = syllables->stress[source_end];
        out->following_stress_count = syllables->stress_counts[source_end];
    }
}

static rg_status build_link_context(
    const rg_form *source,
    const rg_feature_constraint *const *source_features,
    const size_t *source_feature_counts,
    const syllable_data *syllables,
    size_t source_start,
    size_t source_count,
    size_t target_count,
    rg_context_spec *out
) {
    size_t source_end = source_start + source_count;
    rg_status status;
    rg_context_spec_init_empty(out);
    if (source_start == 0) {
        out->position = rg_strdup_internal("initial");
    } else if (source_end == source->segment_count) {
        out->position = rg_strdup_internal("final");
    } else {
        out->position = rg_strdup_internal("medial");
    }
    if (out->position == 0) {
        return RG_ERR_OOM;
    }
    {
        const char *placement = 0;
        const char *index = 0;
        morpheme_placement(source, source_start, source_end, &placement, &index);
        if (placement != 0) {
            out->morphological = rg_strdup_internal(placement);
            out->morpheme_index = rg_strdup_internal(index);
            if (out->morphological == 0 || out->morpheme_index == 0) {
                return RG_ERR_OOM;
            }
        }
    }
    if (source_start > 0) {
        status = context_copy_constraints(
            source_features[source_start - 1],
            source_feature_counts[source_start - 1],
            &out->preceding,
            &out->preceding_count
        );
        if (status != RG_OK) {
            rg_context_spec_clear_internal(out);
            return status;
        }
    }
    if (source_end < source->segment_count) {
        status = context_copy_constraints(
            source_features[source_end],
            source_feature_counts[source_end],
            &out->following,
            &out->following_count
        );
        if (status != RG_OK) {
            rg_context_spec_clear_internal(out);
            return status;
        }
    }
    {
        const rg_feature_constraint *pre2 = 0;
        const rg_feature_constraint *pre3 = 0;
        size_t pre2_count = 0;
        size_t pre3_count = 0;
        if (source_start >= 2) {
            pre2 = source_features[source_start - 2];
            pre2_count = source_feature_counts[source_start - 2];
        }
        if (source_start >= 3) {
            pre3 = source_features[source_start - 3];
            pre3_count = source_feature_counts[source_start - 3];
        }
        status = distance_context_copy_two(
            pre2,
            pre2_count,
            2,
            pre3,
            pre3_count,
            3,
            &out->preceding_at_distance,
            &out->preceding_at_distance_count
        );
        if (status != RG_OK) {
            rg_context_spec_clear_internal(out);
            return status;
        }
    }
    {
        const rg_feature_constraint *fol2 = 0;
        const rg_feature_constraint *fol3 = 0;
        size_t fol2_count = 0;
        size_t fol3_count = 0;
        if (source_end + 1 < source->segment_count) {
            fol2 = source_features[source_end + 1];
            fol2_count = source_feature_counts[source_end + 1];
        }
        if (source_end + 2 < source->segment_count) {
            fol3 = source_features[source_end + 2];
            fol3_count = source_feature_counts[source_end + 2];
        }
        status = distance_context_copy_two(
            fol2,
            fol2_count,
            2,
            fol3,
            fol3_count,
            3,
            &out->following_at_distance,
            &out->following_at_distance_count
        );
        if (status != RG_OK) {
            rg_context_spec_clear_internal(out);
            return status;
        }
    }
    if (syllables != 0 && syllables->left_cumulative != 0 && source_start <= syllables->segment_count) {
        status = context_copy_constraints(
            syllables->left_cumulative[source_start],
            syllables->left_cumulative_counts[source_start],
            &out->somewhere_preceding,
            &out->somewhere_preceding_count
        );
    } else {
        status = context_feature_union_copy(
            source_features,
            source_feature_counts,
            0,
            source_start,
            &out->somewhere_preceding,
            &out->somewhere_preceding_count
        );
    }
    if (status != RG_OK) {
        rg_context_spec_clear_internal(out);
        return status;
    }
    if (syllables != 0 && syllables->right_cumulative != 0 && source_end <= syllables->segment_count) {
        status = context_copy_constraints(
            syllables->right_cumulative[source_end],
            syllables->right_cumulative_counts[source_end],
            &out->somewhere_following,
            &out->somewhere_following_count
        );
    } else {
        status = context_feature_union_copy(
            source_features,
            source_feature_counts,
            source_end,
            source->segment_count,
            &out->somewhere_following,
            &out->somewhere_following_count
        );
    }
    if (status != RG_OK) {
        rg_context_spec_clear_internal(out);
        return status;
    }
    /* Syllable-structural slots mirror the DP: only 1-to-1 links carry them. */
    if (source_count == 1 && target_count == 1 && syllables != 0 && syllables->segment_count > 0 &&
        source_start < syllables->segment_count) {
        size_t syllable_index = syllables->syllable_of[source_start];
        status = context_copy_constraints(
            syllables->same_syllable_excluding[source_start],
            syllables->same_syllable_excluding_counts[source_start],
            &out->same_syllable,
            &out->same_syllable_count
        );
        if (status != RG_OK) {
            rg_context_spec_clear_internal(out);
            return status;
        }
        if (syllable_index + 1 < syllables->syllable_count) {
            status = context_copy_constraints(
                syllables->syllable_features[syllable_index + 1],
                syllables->syllable_feature_counts[syllable_index + 1],
                &out->next_syllable,
                &out->next_syllable_count
            );
            if (status != RG_OK) {
                rg_context_spec_clear_internal(out);
                return status;
            }
        }
        if (syllable_index > 0) {
            status = context_copy_constraints(
                syllables->syllable_features[syllable_index - 1],
                syllables->syllable_feature_counts[syllable_index - 1],
                &out->previous_syllable,
                &out->previous_syllable_count
            );
            if (status != RG_OK) {
                rg_context_spec_clear_internal(out);
                return status;
            }
        }
    }
    if (source_count == 1 && target_count == 1) {
        status = context_copy_stress(source->segments[source_start].stress, &out->self_stress, &out->self_stress_count);
        if (status != RG_OK) {
            rg_context_spec_clear_internal(out);
            return status;
        }
        if (source_start > 0) {
            status = context_copy_stress(source->segments[source_start - 1].stress, &out->preceding_stress, &out->preceding_stress_count);
            if (status != RG_OK) {
                rg_context_spec_clear_internal(out);
                return status;
            }
        }
        if (source_end < source->segment_count) {
            status = context_copy_stress(source->segments[source_end].stress, &out->following_stress, &out->following_stress_count);
            if (status != RG_OK) {
                rg_context_spec_clear_internal(out);
                return status;
            }
        }
    }
    return RG_OK;
}

void rg_context_spec_array_free_internal(rg_context_spec *contexts, size_t count) {
    size_t i;
    if (contexts == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        rg_context_spec_clear_internal(&contexts[i]);
    }
    free(contexts);
}

rg_status rg_form_position_contexts_internal(
    const rg_context *ctx,
    const rg_form *form,
    rg_context_spec **out,
    size_t *out_count
) {
    const rg_feature_constraint **features = 0;
    size_t *feature_counts = 0;
    syllable_data syllables;
    rg_context_spec *contexts = 0;
    size_t i;
    rg_status status;

    if (ctx == 0 || form == 0 || out == 0 || out_count == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    *out_count = 0;
    if (form->segment_count == 0) {
        return RG_OK;
    }
    memset(&syllables, 0, sizeof(syllables));
    status = feature_matrix_build(ctx, form, &features, &feature_counts);
    if (status != RG_OK) {
        return status;
    }
    status = syllable_data_build(ctx, form, features, feature_counts, &syllables);
    if (status != RG_OK) {
        feature_matrix_clear(features, feature_counts, form->segment_count);
        return status;
    }
    contexts = (rg_context_spec *)calloc(form->segment_count, sizeof(*contexts));
    if (contexts == 0) {
        feature_matrix_clear(features, feature_counts, form->segment_count);
        syllable_data_clear(&syllables);
        return RG_ERR_OOM;
    }
    for (i = 0; i < form->segment_count; i++) {
        status = build_link_context(form, features, feature_counts, &syllables, i, 1, 1, &contexts[i]);
        if (status != RG_OK) {
            rg_context_spec_array_free_internal(contexts, i);
            feature_matrix_clear(features, feature_counts, form->segment_count);
            syllable_data_clear(&syllables);
            return status;
        }
    }
    feature_matrix_clear(features, feature_counts, form->segment_count);
    syllable_data_clear(&syllables);
    *out = contexts;
    *out_count = form->segment_count;
    return RG_OK;
}

static rg_status link_from_slice(
    const rg_context *ctx,
    const rg_form *source,
    size_t source_start,
    size_t source_count,
    const rg_form *target,
    size_t target_start,
    size_t target_count,
    const rg_feature_constraint *const *source_features,
    const size_t *source_feature_counts,
    const syllable_data *syllables,
    rg_link *out
) {
    rg_status status;
    memset(out, 0, sizeof(*out));
    status = rg_link_copy_chunks_internal(
        out,
        source->segments + source_start,
        source_count,
        target->segments + target_start,
        target_count
    );
    if (status != RG_OK) {
        return status;
    }
    if (source_features != 0 && source_feature_counts != 0) {
        status = build_link_context(
            source,
            source_features,
            source_feature_counts,
            syllables,
            source_start,
            source_count,
            target_count,
            &out->context
        );
        if (status != RG_OK) {
            rg_link_clear_internal(out);
            return status;
        }
    }
    if (source_count == 1 && target_count == 1) {
        rg_feature_displacement *disp = 0;
        size_t disp_count = 0;
        status = rg_compute_displacement(ctx, source->segments[source_start], target->segments[target_start], &disp, &disp_count);
        if (status != RG_OK) {
            rg_link_clear_internal(out);
            return status;
        }
        out->feature_displacement = disp;
        out->feature_displacement_count = disp_count;
    }
    return RG_OK;
}

static rg_status align_forms_internal(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    const rg_train_options *options,
    const rg_form *source,
    const rg_form *target,
    int max_chunk_size,
    rg_alignment **out
) {
    size_t n;
    size_t m;
    size_t width;
    double *cost = 0;
    rg_dp_step *back = 0;
    size_t i;
    size_t j;
    rg_alignment *alignment = 0;
    rg_link *rev_links = 0;
    const rg_feature_constraint **source_features = 0;
    size_t *source_feature_counts = 0;
    syllable_data syllables;
    /* The target form's own view of each position. A conditioned rule may name
     * either form's environment, and the one it names is the one it has to be
     * matched against. */
    const rg_feature_constraint **target_features = 0;
    size_t *target_feature_counts = 0;
    syllable_data target_syllables;
    size_t rev_count = 0;
    size_t rev_cap = 0;
    rg_status status = RG_OK;

    if (ctx == 0 || source == 0 || target == 0 || out == 0 || (source->segment_count > 0 && source->segments == 0) || (target->segment_count > 0 && target->segments == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    if (max_chunk_size == 0) {
        max_chunk_size = RG_DEFAULT_MAX_CHUNK_SIZE;
    }
    if (max_chunk_size < 1) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    n = source->segment_count;
    m = target->segment_count;
    memset(&syllables, 0, sizeof(syllables));
    memset(&target_syllables, 0, sizeof(target_syllables));
    if (model != 0) {
        status = feature_matrix_build(ctx, source, &source_features, &source_feature_counts);
        if (status != RG_OK) {
            return status;
        }
        if (model->has_target_conditioned) {
            status = feature_matrix_build(ctx, target, &target_features, &target_feature_counts);
            if (status == RG_OK) {
                status = syllable_data_build(ctx, target, target_features, target_feature_counts,
                                             &target_syllables);
            }
        }
        if (status != RG_OK) {
            feature_matrix_clear(source_features, source_feature_counts, source->segment_count);
            feature_matrix_clear(target_features, target_feature_counts, target->segment_count);
            syllable_data_clear(&target_syllables);
            return status;
        }
        status = syllable_data_build(ctx, source, source_features, source_feature_counts, &syllables);
        if (status != RG_OK) {
            feature_matrix_clear(source_features, source_feature_counts, n);
            syllable_data_clear(&syllables);
            feature_matrix_clear(target_features, target_feature_counts, m);
            syllable_data_clear(&target_syllables);
            return status;
        }
    }
    width = m + 1;
    cost = (double *)calloc((n + 1) * (m + 1), sizeof(*cost));
    back = (rg_dp_step *)calloc((n + 1) * (m + 1), sizeof(*back));
    if (cost == 0 || back == 0) {
        feature_matrix_clear(source_features, source_feature_counts, n);
        syllable_data_clear(&syllables);
        feature_matrix_clear(target_features, target_feature_counts, m);
        syllable_data_clear(&target_syllables);
        free(cost);
        free(back);
        return RG_ERR_OOM;
    }
    for (i = 0; i <= n; i++) {
        for (j = 0; j <= m; j++) {
            cost[i * width + j] = INFINITY;
        }
    }
    cost[0] = 0.0;
    for (i = 0; i <= n && status == RG_OK; i++) {
        for (j = 0; j <= m && status == RG_OK; j++) {
            size_t k_max;
            size_t l_max;
            size_t base_k_max;
            size_t base_l_max;
            size_t reorder_span;
            size_t k;
            size_t l;
            if (i == 0 && j == 0) {
                continue;
            }
            k_max = (size_t)max_chunk_size < i ? (size_t)max_chunk_size : i;
            l_max = (size_t)max_chunk_size < j ? (size_t)max_chunk_size : j;
            base_k_max = k_max;
            base_l_max = l_max;
            reorder_span = 0;
            /* A span whose target is its own segments in another order is
             * allowed past the chunk limit, up to RG_MAX_REORDER_SPAN.
             *
             * The search is monotone, so a transposition can only be expressed
             * as one link covering everything between the two segments that
             * moved -- and for anything but an adjacent swap that is wider
             * than a chunk is allowed to be. Spanish milagro against Latin
             * miraculo needs five. The widening is not free, so it applies
             * only where the span really is a permutation, which is a cheap
             * test that fails on the first grapheme the two sides do not
             * share. */
            /* One extra candidate: the widest equal-length span ending here
             * whose two sides are the same segments in a different order.
             *
             * The search is monotone, so a transposition can only be expressed
             * as a single link covering everything between the two segments
             * that moved, and for anything but an adjacent swap that is wider
             * than a chunk may be. Spanish milagro against Latin miraculo
             * needs five. It enters as one candidate rather than by raising
             * the chunk limit, which would admit every ragged span up to that
             * width as well. The span is taken from whichever side is shorter,
             * so exchanging the lects cannot change what the search sees. */
            {
                size_t widest = i < j ? i : j;
                size_t span;
                if (widest > RG_MAX_REORDER_SPAN) {
                    widest = RG_MAX_REORDER_SPAN;
                }
                for (span = widest; span > base_k_max || span > base_l_max; span--) {
                    size_t pairing[RG_MAX_REORDER_SPAN];
                    if (span <= 1) {
                        break;
                    }
                    if (rg_link_is_reordering_internal(source->segments + (i - span), span,
                                                       target->segments + (j - span), span, pairing)) {
                        reorder_span = span;
                        break;
                    }
                }
                if (reorder_span > k_max) {
                    k_max = reorder_span;
                }
                if (reorder_span > l_max) {
                    l_max = reorder_span;
                }
            }
            for (k = 0; k <= k_max && status == RG_OK; k++) {
                for (l = 0; l <= l_max; l++) {
                    double prev;
                    double link_cost = 0.0;
                    double total;
                    if (k == 0 && l == 0) {
                        continue;
                    }
                    if ((k > base_k_max || l > base_l_max) &&
                        !(k == reorder_span && l == reorder_span)) {
                        continue;
                    }
                    prev = cost[(i - k) * width + (j - l)];
                    if (isinf(prev)) {
                        continue;
                    }
                    if (model != 0) {
                        /* Borrowed: read-only, points into the precomputed
                         * per-form arrays, and must not be cleared. */
                        rg_context_spec link_context;
                        rg_context_spec target_link_context;
                        int have_target = model->has_target_conditioned;
                        build_link_context_borrowed(
                            source,
                            source_features,
                            source_feature_counts,
                            &syllables,
                            i - k,
                            k,
                            l,
                            &link_context
                        );
                        if (have_target) {
                            build_link_context_borrowed(
                                target,
                                target_features,
                                target_feature_counts,
                                &target_syllables,
                                j - l,
                                l,
                                k,
                                &target_link_context
                            );
                        }
                        status = rg_score_link_with_context_model_internal(
                            ctx,
                            model,
                            options,
                            source->segments + (i - k),
                            k,
                            target->segments + (j - l),
                            l,
                            &link_context,
                            have_target ? &target_link_context : 0,
                            &link_cost
                        );
                    } else {
                        status = rg_score_link(ctx, source->segments + (i - k), k, target->segments + (j - l), l, &link_cost);
                    }
                    if (status != RG_OK) {
                        break;
                    }
                    total = prev + link_cost + RG_CHUNK_COMPLEXITY_PENALTY * (double)((int)k + (int)l - 2);
                    if (total < cost[i * width + j] - RG_TIE_EPSILON) {
                        cost[i * width + j] = total;
                        back[i * width + j].prev_i = i - k;
                        back[i * width + j].prev_j = j - l;
                        back[i * width + j].source_count = k;
                        back[i * width + j].target_count = l;
                        back[i * width + j].has_step = 1;
                    }
                }
            }
        }
    }
    if (status != RG_OK) {
        feature_matrix_clear(source_features, source_feature_counts, n);
        syllable_data_clear(&syllables);
        feature_matrix_clear(target_features, target_feature_counts, m);
        syllable_data_clear(&target_syllables);
        free(cost);
        free(back);
        return status;
    }
    alignment = (rg_alignment *)calloc(1, sizeof(*alignment));
    if (alignment == 0) {
        free(cost);
        free(back);
        feature_matrix_clear(source_features, source_feature_counts, n);
        syllable_data_clear(&syllables);
        feature_matrix_clear(target_features, target_feature_counts, m);
        syllable_data_clear(&target_syllables);
        return RG_ERR_OOM;
    }
    status = form_copy(source, &alignment->source_form);
    if (status == RG_OK) {
        status = form_copy(target, &alignment->target_form);
    }
    if (status != RG_OK) {
        rg_alignment_free(alignment);
        free(cost);
        free(back);
        feature_matrix_clear(source_features, source_feature_counts, n);
        syllable_data_clear(&syllables);
        feature_matrix_clear(target_features, target_feature_counts, m);
        syllable_data_clear(&target_syllables);
        return status;
    }
    i = n;
    j = m;
    while ((i > 0 || j > 0) && status == RG_OK) {
        rg_dp_step step = back[i * width + j];
        rg_link link;
        rg_link *next;
        if (!step.has_step) {
            status = RG_ERR_PARSE;
            break;
        }
        if (rev_count == rev_cap) {
            size_t next_cap = rev_cap == 0 ? 8 : rev_cap * 2;
            next = (rg_link *)realloc(rev_links, next_cap * sizeof(*rev_links));
            if (next == 0) {
                status = RG_ERR_OOM;
                break;
            }
            rev_links = next;
            rev_cap = next_cap;
        }
        status = link_from_slice(
            ctx,
            source,
            step.prev_i,
            step.source_count,
            target,
            step.prev_j,
            step.target_count,
            source_features,
            source_feature_counts,
            &syllables,
            &link
        );
        if (status != RG_OK) {
            break;
        }
        rev_links[rev_count++] = link;
        i = step.prev_i;
        j = step.prev_j;
    }
    if (status == RG_OK && rev_count > 0) {
        alignment->links = (rg_link *)calloc(rev_count, sizeof(*alignment->links));
        if (alignment->links == 0) {
            status = RG_ERR_OOM;
        } else {
            size_t a;
            for (a = 0; a < rev_count; a++) {
                alignment->links[a] = rev_links[rev_count - 1 - a];
            }
            alignment->link_count = rev_count;
        }
    }
    if (status != RG_OK) {
        for (i = 0; i < rev_count; i++) {
            rg_link_clear_internal(&rev_links[i]);
        }
        free(rev_links);
        rg_alignment_free(alignment);
        free(cost);
        free(back);
        feature_matrix_clear(source_features, source_feature_counts, n);
        syllable_data_clear(&syllables);
        feature_matrix_clear(target_features, target_feature_counts, m);
        syllable_data_clear(&target_syllables);
        return status;
    }
    free(rev_links);
    free(cost);
    free(back);
    feature_matrix_clear(source_features, source_feature_counts, n);
    syllable_data_clear(&syllables);
    feature_matrix_clear(target_features, target_feature_counts, m);
    syllable_data_clear(&target_syllables);
    *out = alignment;
    return RG_OK;
}

rg_status rg_align_forms(
    const rg_context *ctx,
    const rg_form *source,
    const rg_form *target,
    int max_chunk_size,
    rg_alignment **out
) {
    return align_forms_internal(ctx, 0, 0, source, target, max_chunk_size, out);
}

rg_status rg_align_forms_with_model(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    const rg_train_options *options,
    const rg_form *source,
    const rg_form *target,
    int max_chunk_size,
    rg_alignment **out
) {
    if (model == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    return align_forms_internal(ctx, model, options, source, target, max_chunk_size, out);
}

void rg_alignment_free(rg_alignment *alignment) {
    size_t i;
    if (alignment == 0) {
        return;
    }
    form_clear(&alignment->source_form);
    form_clear(&alignment->target_form);
    for (i = 0; i < alignment->link_count; i++) {
        rg_link_clear_internal(&alignment->links[i]);
    }
    free(alignment->links);
    free(alignment);
}

size_t rg_alignment_link_count(const rg_alignment *alignment) {
    if (alignment == 0) {
        return 0;
    }
    return alignment->link_count;
}

const rg_link *rg_alignment_link_at(const rg_alignment *alignment, size_t index) {
    if (alignment == 0 || index >= alignment->link_count) {
        return 0;
    }
    return &alignment->links[index];
}

static int parse_cross_dimensional_offset(const char *position) {
    if (position == 0) {
        return 0;
    }
    if (strcmp(position, "relative_-1") == 0) {
        return -1;
    }
    if (strcmp(position, "relative_+1") == 0) {
        return 1;
    }
    return 0;
}

static int cross_dimensional_source_holds(
    const rg_context *ctx,
    const rg_form *form,
    size_t src_pos,
    const rg_cross_dimensional_row *row
) {
    int offset;
    int index;
    const rg_feature_set *features = 0;
    rg_status status;
    if (ctx == 0 || form == 0 || row == 0 || row->source_feature == 0) {
        return 0;
    }
    offset = parse_cross_dimensional_offset(row->source_position);
    index = (int)src_pos + offset;
    if (index < 0 || (size_t)index >= form->segment_count) {
        return 0;
    }
    if (strcmp(row->source_feature, "tone") == 0) {
        const char *tone = form->segments[index].tone == 0 ? "" : form->segments[index].tone;
        return strcmp(tone, row->source_value == 0 ? "" : row->source_value) == 0;
    }
    if (form->segments[index].grapheme == 0) {
        return 0;
    }
    status = rg_context_features_internal(ctx, form->segments[index].grapheme, &features);
    if (status != RG_OK) {
        return 0;
    }
    {
        /* source_value "-" is the complementary environment, which is half of
         * every conditioned split. Reading the feature alone and ignoring the
         * value fires a rule about voiceless onsets on a voiced one. */
        int holds = feature_set_contains(features, row->source_feature);
        const char *value = row->source_value == 0 ? "+" : row->source_value;
        return strcmp(value, "-") == 0 ? !holds : holds;
    }
}

static double cross_dimensional_adjustment_for_row(
    const rg_pairwise_model *model,
    const rg_cross_dimensional_row *row,
    const char *actual
) {
    size_t i;
    double total_all = 0.0;
    double total_for_value = 0.0;
    double distinct = 0.0;
    double p_cond;
    double p_base;
    double alpha = 1.0;
    if (model == 0 || row == 0 || actual == 0 || actual[0] == '\0') {
        return 0.0;
    }
    for (i = 0; i < model->tonal_count_count; i++) {
        int first_for_tone = 1;
        size_t j;
        total_all += model->tonal_counts[i].count;
        if (strcmp(model->tonal_counts[i].target_tone, row->target_value) == 0) {
            total_for_value += model->tonal_counts[i].count;
        }
        if (model->tonal_counts[i].target_tone[0] == '\0') {
            first_for_tone = 0;
        }
        for (j = 0; j < i; j++) {
            if (strcmp(model->tonal_counts[i].target_tone, model->tonal_counts[j].target_tone) == 0) {
                first_for_tone = 0;
                break;
            }
        }
        if (first_for_tone) {
            distinct += 1.0;
        }
    }
    if (distinct < 2.0) {
        distinct = 2.0;
    }
    p_cond = (row->count + alpha) / (row->source_count + alpha * distinct);
    if (p_cond < 1e-12) {
        p_cond = 1e-12;
    }
    if (p_cond > 1.0 - 1e-12) {
        p_cond = 1.0 - 1e-12;
    }
    if (total_all <= 0.0) {
        p_base = 1.0 / distinct;
    } else {
        p_base = (total_for_value + alpha) / (total_all + alpha * distinct);
    }
    if (p_base < 1e-12) {
        p_base = 1e-12;
    }
    if (p_base > 1.0 - 1e-12) {
        p_base = 1.0 - 1e-12;
    }
    if (strcmp(actual, row->target_value) == 0) {
        return -(log(p_cond) - log(p_base));
    }
    return -(log(1.0 - p_cond) - log(1.0 - p_base));
}

static double cross_dimensional_alignment_adjustment(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    const rg_alignment *alignment
) {
    size_t link_i;
    size_t src_pos = 0;
    size_t tgt_pos = 0;
    double total = 0.0;
    if (ctx == 0 || model == 0 || alignment == 0 || model->cross_dimensional_count == 0) {
        return 0.0;
    }
    for (link_i = 0; link_i < alignment->link_count; link_i++) {
        const rg_link *link = &alignment->links[link_i];
        if (link->source_count == 1 && link->target_count == 1) {
            size_t row_i;
            for (row_i = 0; row_i < model->cross_dimensional_count; row_i++) {
                const rg_cross_dimensional_row *row = &model->cross_dimensional_rows[row_i];
                int tgt_index = (int)tgt_pos + row->target_position_offset;
                const char *actual = "";
                if (!cross_dimensional_source_holds(ctx, &alignment->source_form, src_pos, row)) {
                    continue;
                }
                if (tgt_index < 0 || (size_t)tgt_index >= alignment->target_form.segment_count) {
                    continue;
                }
                if (strcmp(row->target_dimension, "tone") == 0) {
                    actual = alignment->target_form.segments[tgt_index].tone == 0 ? "" : alignment->target_form.segments[tgt_index].tone;
                } else if (strcmp(row->target_dimension, "length") == 0) {
                    actual = alignment->target_form.segments[tgt_index].length == 0 ? "" : alignment->target_form.segments[tgt_index].length;
                } else if (strcmp(row->target_dimension, "stress") == 0) {
                    actual = alignment->target_form.segments[tgt_index].stress == 0 ? "" : alignment->target_form.segments[tgt_index].stress;
                }
                total += cross_dimensional_adjustment_for_row(model, row, actual);
            }
        }
        src_pos += link->source_count;
        tgt_pos += link->target_count;
    }
    return total;
}

rg_status rg_alignment_cost(const rg_context *ctx, const rg_alignment *alignment, double *out) {
    size_t i;
    double total = 0.0;
    rg_status status;
    if (ctx == 0 || alignment == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0.0;
    for (i = 0; i < alignment->link_count; i++) {
        double link_cost = 0.0;
        const rg_link *link = &alignment->links[i];
        status = rg_score_link(ctx, link->source, link->source_count, link->target, link->target_count, &link_cost);
        if (status != RG_OK) {
            return status;
        }
        total += link_cost;
    }
    *out = total;
    return RG_OK;
}

rg_status rg_alignment_cost_with_model(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    const rg_train_options *options,
    const rg_alignment *alignment,
    double *out
) {
    size_t i;
    size_t target_position = 0;
    double total = 0.0;
    rg_context_spec *target_contexts = 0;
    size_t target_context_count = 0;
    rg_status status;
    if (ctx == 0 || model == 0 || alignment == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0.0;
    /* A link carries the source form's context only, so the target's is
     * rebuilt here. A rule naming the target's environment would otherwise be
     * silently inert whenever a cost is taken of an alignment already made. */
    if (model->has_target_conditioned) {
        status = rg_form_position_contexts_internal(ctx, &alignment->target_form,
                                                    &target_contexts, &target_context_count);
        if (status != RG_OK) {
            return status;
        }
    }
    for (i = 0; i < alignment->link_count; i++) {
        double link_cost = 0.0;
        const rg_link *link = &alignment->links[i];
        const rg_context_spec *target_context =
            (link->target_count == 1 && target_position < target_context_count)
                ? &target_contexts[target_position]
                : 0;
        target_position += link->target_count;
        status = rg_score_link_with_context_model_internal(ctx, model, options, link->source, link->source_count, link->target, link->target_count, &link->context, target_context, &link_cost);
        if (status != RG_OK) {
            rg_context_spec_array_free_internal(target_contexts, target_context_count);
            return status;
        }
        total += link_cost;
    }
    rg_context_spec_array_free_internal(target_contexts, target_context_count);
    total += cross_dimensional_alignment_adjustment(ctx, model, alignment);
    *out = total;
    return RG_OK;
}
