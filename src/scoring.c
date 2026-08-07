#include "internal.h"

#include <math.h>
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
    const rg_feature_set *source_features = 0;
    const rg_feature_set *target_features = 0;
    rg_feature_displacement *items = 0;
    size_t count = 0;
    size_t cap = 0;
    rg_status status;

    if (ctx == 0 || source.grapheme == 0 || target.grapheme == 0 || out == 0 || out_count == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    *out_count = 0;
    status = rg_context_features_internal(ctx, source.grapheme, &source_features);
    if (status != RG_OK) {
        return status;
    }
    status = rg_context_features_internal(ctx, target.grapheme, &target_features);
    if (status != RG_OK) {

        return status;
    }
    status = append_difference(source_features, target_features, "present", "absent", &items, &count, &cap);
    if (status == RG_OK) {
        status = append_difference(target_features, source_features, "absent", "present", &items, &count, &cap);
    }

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
        rg_context_note_unknown_grapheme_internal(ctx, source.grapheme);
        return RG_ERR_UNKNOWN_GRAPHEME;
    }
    status = rg_context_is_segment(ctx, target.grapheme, &is_segment);
    if (status != RG_OK) {
        return status;
    }
    if (!is_segment) {
        rg_context_note_unknown_grapheme_internal(ctx, target.grapheme);
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

static int scoring_segment_equal(const rg_segment *a, const rg_segment *b) {
    return strcmp(a->grapheme == 0 ? "" : a->grapheme, b->grapheme == 0 ? "" : b->grapheme) == 0 &&
        strcmp(a->tone == 0 ? "" : a->tone, b->tone == 0 ? "" : b->tone) == 0 &&
        strcmp(a->length == 0 ? "" : a->length, b->length == 0 ? "" : b->length) == 0 &&
        strcmp(a->stress == 0 ? "" : a->stress, b->stress == 0 ? "" : b->stress) == 0;
}

static int scoring_segment_array_equal(const rg_segment *a, size_t a_count, const rg_segment *b, size_t b_count) {
    size_t i;
    if (a_count != b_count) {
        return 0;
    }
    for (i = 0; i < a_count; i++) {
        if (!scoring_segment_equal(&a[i], &b[i])) {
            return 0;
        }
    }
    return 1;
}

static const rg_chunk_row *find_chunk_row(
    const rg_pairwise_model *model,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count
) {
    size_t i;
    if (model == 0 || source_count == 0 || target_count == 0) {
        return 0;
    }
    for (i = 0; i < model->chunk_count; i++) {
        if (scoring_segment_array_equal(model->chunks[i].source, model->chunks[i].source_count, source, source_count) &&
            scoring_segment_array_equal(model->chunks[i].target, model->chunks[i].target_count, target, target_count)) {
            return &model->chunks[i];
        }
    }
    return 0;
}

/* The learned tables are published sorted by (source, target), so every hot
 * lookup below is a binary search. These run inside the alignment DP's inner
 * loop; linear scans here dominate training time. */
static const rg_segment_count_row *find_segment_count(
    const rg_pairwise_model *model,
    const char *source,
    const char *target
) {
    size_t low = 0;
    size_t high;
    if (model == 0 || source == 0 || target == 0) {
        return 0;
    }
    high = model->segment_count_count;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        int c = strcmp(model->segment_counts[mid].source, source);
        if (c == 0) {
            c = strcmp(model->segment_counts[mid].target, target);
        }
        if (c == 0) {
            return &model->segment_counts[mid];
        }
        if (c < 0) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return 0;
}

static const rg_conditioned_segment_count_row *find_conditioned_segment_count(
    const rg_pairwise_model *model,
    const char *source,
    const char *target,
    const rg_context_spec *link_context
) {
    const rg_conditioned_segment_count_row *best = 0;
    size_t best_specificity = 0;
    size_t low = 0;
    size_t high;
    size_t i;
    if (model == 0 || source == 0 || target == 0 || link_context == 0) {
        return 0;
    }
    high = model->conditioned_segment_count_count;
    /* Lower bound of the (source, target) block; entries within it differ only
     * by context, so the most specific match is chosen by a short linear scan. */
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        int c = strcmp(model->conditioned_segment_counts[mid].source, source);
        if (c == 0) {
            c = strcmp(model->conditioned_segment_counts[mid].target, target);
        }
        if (c < 0) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    for (i = low; i < model->conditioned_segment_count_count; i++) {
        const rg_conditioned_segment_count_row *row = &model->conditioned_segment_counts[i];
        int subset = 0;
        size_t specificity;
        if (strcmp(row->source, source) != 0 || strcmp(row->target, target) != 0) {
            break;
        }
        if (rg_context_spec_is_subset(&row->context, link_context, &subset) != RG_OK || !subset) {
            continue;
        }
        specificity = rg_context_spec_constraint_count(&row->context);
        if (best == 0 || specificity > best_specificity) {
            best = row;
            best_specificity = specificity;
        }
    }
    return best;
}

/* -log P(displacement vector) - log V, with a Dirichlet-smoothed estimate over
 * the observed vectors. Returns zero when no displacement table exists yet. */
static rg_status displacement_model_cost(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    rg_segment source,
    rg_segment target,
    double *out
) {
    rg_feature_displacement *disp = 0;
    size_t disp_count = 0;
    size_t i;
    double n = 0.0;
    double total = 0.0;
    double alpha = 1.0;
    double v;
    double p;
    rg_status status;

    *out = 0.0;
    if (model == 0 || model->displacement_row_count == 0) {
        return RG_OK;
    }
    status = rg_compute_displacement(ctx, source, target, &disp, &disp_count);
    if (status != RG_OK) {
        return status;
    }
    total = model->displacement_rows[0].total;
    for (i = 0; i < model->displacement_row_count; i++) {
        const rg_displacement_row *row = &model->displacement_rows[i];
        size_t j;
        if (row->item_count != disp_count) {
            continue;
        }
        for (j = 0; j < disp_count; j++) {
            if (strcmp(row->items[j].feature, disp[j].feature) != 0 ||
                strcmp(row->items[j].from_value, disp[j].from_value) != 0 ||
                strcmp(row->items[j].to_value, disp[j].to_value) != 0) {
                break;
            }
        }
        if (j == disp_count) {
            n = row->count;
            break;
        }
    }
    rg_feature_displacement_free(disp, disp_count);
    v = (double)model->displacement_row_count;
    if (v < 1.0) {
        v = 1.0;
    }
    if (alpha * v + total <= 0.0) {
        p = 1.0;
    } else {
        p = (alpha + n) / (alpha * v + total);
    }
    *out = (p <= 0.0 ? INFINITY : -log(p)) - log(v);
    return RG_OK;
}

/* -log P(tone correspondence) - log V. Every observed correspondence carries a
 * pseudo-count of one, so the denominator is the source total plus the number
 * of observed correspondences. Silent for untoned data. */
static double tonal_model_cost(const rg_pairwise_model *model, const char *source_tone, const char *target_tone) {
    size_t i;
    const char *src = source_tone == 0 ? "" : source_tone;
    const char *tgt = target_tone == 0 ? "" : target_tone;
    double n = 0.0;
    double alpha = 0.0;
    double total_for_source = 0.0;
    double prior_mass;
    double denominator;
    double p;
    double v;

    if ((src[0] == '\0' && tgt[0] == '\0') || model == 0 || model->tonal_count_count == 0) {
        return 0.0;
    }
    for (i = 0; i < model->tonal_count_count; i++) {
        if (strcmp(model->tonal_counts[i].source_tone, src) == 0) {
            total_for_source = model->tonal_counts[i].source_total;
            if (strcmp(model->tonal_counts[i].target_tone, tgt) == 0) {
                n = model->tonal_counts[i].count;
                alpha = 1.0;
            }
        }
    }
    prior_mass = (double)model->tonal_count_count;
    if (prior_mass == 0.0) {
        prior_mass = 1.0;
    }
    denominator = total_for_source + prior_mass;
    if (denominator <= 0.0) {
        return 0.0;
    }
    p = (alpha + n) / denominator;
    if (p <= 0.0) {
        return 0.0;
    }
    v = (double)(model->tonal_count_count < 2 ? 2 : model->tonal_count_count);
    return -log(p) - log(v);
}

/* alpha(t|s) from the prior softmax. Returns 0 when the pair is absent, which
 * is how the caller learns the prior never saw it. */
static int segment_prior_lookup(
    const rg_pairwise_model *model,
    const char *source,
    const char *target,
    double *alpha
) {
    size_t low = 0;
    size_t high = model->segment_prior_count;
    *alpha = 0.0;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        int c = strcmp(model->segment_priors[mid].source, source);
        if (c == 0) {
            c = strcmp(model->segment_priors[mid].target, target);
        }
        if (c == 0) {
            *alpha = model->segment_priors[mid].alpha;
            return 1;
        }
        if (c < 0) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return 0;
}

static double segment_log_normalizer(const rg_pairwise_model *model, const char *source) {
    size_t low = 0;
    size_t high = model->log_normalizer_count;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        int c = strcmp(model->log_normalizers[mid].source, source);
        if (c == 0) {
            return model->log_normalizers[mid].value;
        }
        if (c < 0) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return 0.0;
}

/* N(source) over the unconditioned counts; conditioned entries share this
 * denominator so their probabilities stay comparable at scoring time. */
static double segment_source_total(const rg_pairwise_model *model, const char *source) {
    size_t low = 0;
    size_t high = model->segment_count_count;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        int c = strcmp(model->segment_counts[mid].source, source);
        if (c == 0) {
            return model->segment_counts[mid].source_total;
        }
        if (c < 0) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return 0.0;
}

/* P(target | source, context) under the most specific matching correspondence:
 * (alpha + n) / (beta + N(source)). Returns 0 when the pair is unknown to both
 * the counts and the prior, which is the caller's cue to fall back to the bare
 * merkmal distance. */
int rg_segment_posterior_internal(
    const rg_pairwise_model *model,
    const char *source,
    const char *target,
    const rg_context_spec *link_context,
    double *out
) {
    const rg_conditioned_segment_count_row *conditioned;
    const rg_segment_count_row *unconditioned;
    double alpha;
    double n;
    double denominator;

    *out = 0.0;
    if (model == 0 || source == 0 || target == 0) {
        return 0;
    }
    conditioned = find_conditioned_segment_count(model, source, target, link_context);
    unconditioned = find_segment_count(model, source, target);
    if (conditioned != 0) {
        /* Conditioned keys carry no prior mass of their own. */
        alpha = 0.0;
        n = conditioned->count;
    } else if (segment_prior_lookup(model, source, target, &alpha)) {
        n = unconditioned == 0 ? 0.0 : unconditioned->count;
    } else if (unconditioned != 0) {
        alpha = 0.0;
        n = unconditioned->count;
    } else {
        return 0;
    }
    denominator = model->concentration + segment_source_total(model, source);
    if (denominator <= 0.0) {
        return 0;
    }
    *out = (alpha + n) / denominator;
    return 1;
}

double rg_segment_log_normalizer_internal(const rg_pairwise_model *model, const char *source) {
    return segment_log_normalizer(model, source);
}

size_t rg_segment_vocab_size_internal(const rg_pairwise_model *model) {
    return model == 0 ? 0 : model->log_normalizer_count;
}

rg_status rg_score_link_with_model(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    const rg_train_options *options,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    double *out
) {
    return rg_score_link_with_context_model_internal(
        ctx,
        model,
        options,
        source,
        source_count,
        target,
        target_count,
        0,
        out
    );
}

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
) {
    rg_status status;
    if (ctx == 0 || out == 0 || (source_count > 0 && source == 0) || (target_count > 0 && target == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    if (model == 0) {
        return rg_score_link(ctx, source, source_count, target, target_count, out);
    }
    {
        const rg_chunk_row *chunk = find_chunk_row(model, source, source_count, target, target_count);
        if (chunk != 0) {
            *out = chunk->cost;
            return RG_OK;
        }
    }
    if (source_count == 0 || target_count == 0) {
        return rg_score_link(ctx, source, source_count, target, target_count, out);
    }
    if (source_count != 1 || target_count != 1) {
        size_t paired = source_count < target_count ? source_count : target_count;
        size_t i;
        int asymmetry = (int)source_count - (int)target_count;
        double total = 0.0;
        if (asymmetry < 0) {
            asymmetry = -asymmetry;
        }
        for (i = 0; i < paired; i++) {
            double pair_cost = 0.0;
            status = rg_score_link_with_context_model_internal(
                ctx,
                model,
                options,
                &source[i],
                1,
                &target[i],
                1,
                link_context,
                &pair_cost
            );
            if (status != RG_OK) {
                return status;
            }
            total += pair_cost;
        }
        *out = total + RG_DEFAULT_GAP_COST * (double)asymmetry + RG_DEFAULT_CHUNK_PENALTY * (double)asymmetry;
        return RG_OK;
    }
    {
        /* Posterior over the most specific matching correspondence, shifted by
         * the prior's log partition function so costs remain comparable across
         * sources. Pairs the prior never saw fall back to merkmal distance. */
        const char *src = source[0].grapheme;
        const char *tgt = target[0].grapheme;
        double posterior = 0.0;
        double seg_cost;
        double layered_cost;
        double disp_cost = 0.0;

        if (!rg_segment_posterior_internal(model, src, tgt, link_context, &posterior)) {
            /* The prior never saw this pair, so fall back to the bare merkmal
             * distance. Computed here rather than up front: the model path is
             * the common case and does not need it. */
            return rg_score_link(ctx, source, source_count, target, target_count, out);
        }
        seg_cost = (posterior <= 0.0 ? INFINITY : -log(posterior)) - segment_log_normalizer(model, src);
        if (model->displacement_row_count > 0) {
            status = displacement_model_cost(ctx, model, source[0], target[0], &disp_cost);
            if (status == RG_ERR_UNKNOWN_GRAPHEME) {
                return rg_score_link(ctx, source, source_count, target, target_count, out);
            }
            if (status != RG_OK) {
                return status;
            }
            layered_cost = model->segment_weight * seg_cost + model->displacement_weight * disp_cost;
        } else {
            layered_cost = seg_cost;
        }
        *out = layered_cost + model->tone_weight * tonal_model_cost(model, source[0].tone, target[0].tone);
    }
    return RG_OK;
}
