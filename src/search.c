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

static const char *const context_feature_names[] = {
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
    rg_feature_set *features = 0;
    rg_feature_constraint constraints[sizeof(context_feature_names) / sizeof(context_feature_names[0])];
    size_t i;
    size_t count = 0;
    rg_status status;
    if (ctx == 0 || segment == 0 || segment->grapheme == 0 || out == 0 || out_count == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    *out_count = 0;
    status = rg_context_grapheme_features(ctx, segment->grapheme, &features);
    if (status != RG_OK) {
        return status;
    }
    for (i = 0; i < sizeof(context_feature_names) / sizeof(context_feature_names[0]); i++) {
        if (feature_set_contains(features, context_feature_names[i])) {
            constraints[count].feature = context_feature_names[i];
            constraints[count].value = "+";
            count++;
        }
    }
    rg_feature_set_free(features);
    status = rg_feature_constraint_array_copy_internal(constraints, count, out);
    if (status != RG_OK) {
        return status;
    }
    *out_count = count;
    return RG_OK;
}

static void feature_matrix_clear(
    const rg_feature_constraint **features,
    const size_t *feature_counts,
    size_t count
) {
    size_t i;
    if (features == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        rg_feature_constraint_array_clear_internal(features[i], feature_counts == 0 ? 0 : feature_counts[i]);
    }
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

static int constraint_array_has(const rg_feature_constraint *items, size_t count, const char *feature, const char *value) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (strcmp(items[i].feature, feature) == 0 && strcmp(items[i].value, value) == 0) {
            return 1;
        }
    }
    return 0;
}

static rg_status context_feature_union_copy(
    const rg_feature_constraint *const *source_features,
    const size_t *source_feature_counts,
    size_t start,
    size_t end,
    const rg_feature_constraint **out,
    size_t *out_count
) {
    rg_feature_constraint constraints[sizeof(context_feature_names) / sizeof(context_feature_names[0])];
    size_t i;
    size_t f;
    size_t count = 0;
    if (out == 0 || out_count == 0 || (end > start && (source_features == 0 || source_feature_counts == 0))) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    *out_count = 0;
    for (f = 0; f < sizeof(context_feature_names) / sizeof(context_feature_names[0]); f++) {
        int found = 0;
        for (i = start; i < end && !found; i++) {
            found = constraint_array_has(source_features[i], source_feature_counts[i], context_feature_names[f], "+");
        }
        if (found) {
            constraints[count].feature = context_feature_names[f];
            constraints[count].value = "+";
            count++;
        }
    }
    if (count == 0) {
        return RG_OK;
    }
    {
        rg_status status = rg_feature_constraint_array_copy_internal(constraints, count, out);
        if (status != RG_OK) {
            return status;
        }
    }
    *out_count = count;
    return RG_OK;
}

static rg_status build_link_context(
    const rg_form *source,
    const rg_feature_constraint *const *source_features,
    const size_t *source_feature_counts,
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
    status = context_feature_union_copy(
        source_features,
        source_feature_counts,
        0,
        source_start,
        &out->somewhere_preceding,
        &out->somewhere_preceding_count
    );
    if (status != RG_OK) {
        rg_context_spec_clear_internal(out);
        return status;
    }
    status = context_feature_union_copy(
        source_features,
        source_feature_counts,
        source_end,
        source->segment_count,
        &out->somewhere_following,
        &out->somewhere_following_count
    );
    if (status != RG_OK) {
        rg_context_spec_clear_internal(out);
        return status;
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
    if (model != 0) {
        status = feature_matrix_build(ctx, source, &source_features, &source_feature_counts);
        if (status != RG_OK) {
            return status;
        }
    }
    width = m + 1;
    cost = (double *)calloc((n + 1) * (m + 1), sizeof(*cost));
    back = (rg_dp_step *)calloc((n + 1) * (m + 1), sizeof(*back));
    if (cost == 0 || back == 0) {
        feature_matrix_clear(source_features, source_feature_counts, n);
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
            size_t k;
            size_t l;
            if (i == 0 && j == 0) {
                continue;
            }
            k_max = (size_t)max_chunk_size < i ? (size_t)max_chunk_size : i;
            l_max = (size_t)max_chunk_size < j ? (size_t)max_chunk_size : j;
            for (k = 0; k <= k_max && status == RG_OK; k++) {
                for (l = 0; l <= l_max; l++) {
                    double prev;
                    double link_cost = 0.0;
                    double total;
                    if (k == 0 && l == 0) {
                        continue;
                    }
                    prev = cost[(i - k) * width + (j - l)];
                    if (isinf(prev)) {
                        continue;
                    }
                    if (model != 0) {
                        rg_context_spec link_context;
                        rg_context_spec_init_empty(&link_context);
                        status = build_link_context(
                            source,
                            source_features,
                            source_feature_counts,
                            i - k,
                            k,
                            l,
                            &link_context
                        );
                        if (status == RG_OK) {
                            status = rg_score_link_with_context_model_internal(
                                ctx,
                                model,
                                options,
                                source->segments + (i - k),
                                k,
                                target->segments + (j - l),
                                l,
                                &link_context,
                                &link_cost
                            );
                        }
                        rg_context_spec_clear_internal(&link_context);
                    } else {
                        status = rg_score_link(ctx, source->segments + (i - k), k, target->segments + (j - l), l, &link_cost);
                    }
                    if (status != RG_OK) {
                        break;
                    }
                    total = prev + link_cost + RG_CHUNK_COMPLEXITY_PENALTY * (double)((int)k + (int)l - 2);
                    if (total < cost[i * width + j]) {
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
        free(cost);
        free(back);
        return status;
    }
    alignment = (rg_alignment *)calloc(1, sizeof(*alignment));
    if (alignment == 0) {
        free(cost);
        free(back);
        feature_matrix_clear(source_features, source_feature_counts, n);
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
        return status;
    }
    free(rev_links);
    free(cost);
    free(back);
    feature_matrix_clear(source_features, source_feature_counts, n);
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
    rg_feature_set *features = 0;
    rg_status status;
    int found = 0;
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
    status = rg_context_grapheme_features(ctx, form->segments[index].grapheme, &features);
    if (status != RG_OK) {
        return 0;
    }
    found = feature_set_contains(features, row->source_feature);
    rg_feature_set_free(features);
    return found;
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
    double total = 0.0;
    rg_status status;
    if (ctx == 0 || model == 0 || alignment == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0.0;
    for (i = 0; i < alignment->link_count; i++) {
        double link_cost = 0.0;
        const rg_link *link = &alignment->links[i];
        status = rg_score_link_with_context_model_internal(ctx, model, options, link->source, link->source_count, link->target, link->target_count, &link->context, &link_cost);
        if (status != RG_OK) {
            return status;
        }
        total += link_cost;
    }
    total += cross_dimensional_alignment_adjustment(ctx, model, alignment);
    *out = total;
    return RG_OK;
}
