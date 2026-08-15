#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

rg_status rg_compute_displacement(
    const rg_context *ctx,
    rg_segment source,
    rg_segment target,
    rg_feature_displacement **out,
    size_t *out_count
) {
    const rg_feature_displacement *shared = 0;
    size_t shared_count = 0;
    rg_feature_displacement *items;
    size_t i;
    rg_status status;

    if (ctx == 0 || out == 0 || out_count == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    *out_count = 0;
    if (source.grapheme == 0 || target.grapheme == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    status = rg_context_displacement_internal(ctx, source.grapheme, target.grapheme,
                                              &shared, &shared_count);
    if (status != RG_OK) {
        return status;
    }
    if (shared_count == 0) {
        return RG_OK;
    }
    /* The shared derivation borrows its strings from the feature cache; a
     * caller-owned result cannot, because it outlives nothing in particular. */
    items = (rg_feature_displacement *)calloc(shared_count, sizeof(*items));
    if (items == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < shared_count; i++) {
        items[i].feature = rg_strdup_internal(shared[i].feature);
        items[i].from_value = rg_strdup_internal(shared[i].from_value);
        items[i].to_value = rg_strdup_internal(shared[i].to_value);
        if (items[i].feature == 0 || items[i].from_value == 0 || items[i].to_value == 0) {
            rg_feature_displacement_free(items, i + 1);
            return RG_ERR_OOM;
        }
    }
    *out = items;
    *out_count = shared_count;
    return RG_OK;
}

void rg_feature_displacement_free(rg_feature_displacement *items, size_t count) {
    size_t i;
    if (items == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        rg_free_owned_internal(items[i].feature);
        rg_free_owned_internal(items[i].from_value);
        rg_free_owned_internal(items[i].to_value);
    }
    free(items);
}

static rg_status segment_distance_checked(
    const rg_context *ctx,
    rg_segment source,
    rg_segment target,
    double *out
) {
    bool is_segment = false;
    rg_status status;
    status = rg_context_is_segment(ctx, source.grapheme, &is_segment);
    if (status != RG_OK) {
        return status;
    }
    if (!is_segment) {
        return rg_context_refusal_status_internal(ctx, source.grapheme);
    }
    status = rg_context_is_segment(ctx, target.grapheme, &is_segment);
    if (status != RG_OK) {
        return status;
    }
    if (!is_segment) {
        return rg_context_refusal_status_internal(ctx, target.grapheme);
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

/* Orders a candidate span against a published chunk row on the same key the
 * table was sorted by: graphemes of the shorter source first, then source
 * length, then the same for the target. Tone, length and stress are not part of
 * the key, and a row differing only in those would be unreachable -- chunk
 * promotion keys on graphemes, so no such row exists. */
static int chunk_key_cmp(
    const rg_chunk_row *row,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count
) {
    size_t i;
    size_t n = row->source_count < source_count ? row->source_count : source_count;
    for (i = 0; i < n; i++) {
        int c = strcmp(row->source[i].grapheme == 0 ? "" : row->source[i].grapheme,
                       source[i].grapheme == 0 ? "" : source[i].grapheme);
        if (c != 0) {
            return c;
        }
    }
    if (row->source_count != source_count) {
        return row->source_count < source_count ? -1 : 1;
    }
    n = row->target_count < target_count ? row->target_count : target_count;
    for (i = 0; i < n; i++) {
        int c = strcmp(row->target[i].grapheme == 0 ? "" : row->target[i].grapheme,
                       target[i].grapheme == 0 ? "" : target[i].grapheme);
        if (c != 0) {
            return c;
        }
    }
    if (row->target_count != target_count) {
        return row->target_count < target_count ? -1 : 1;
    }
    return 0;
}

static const rg_chunk_row *find_chunk_row(
    const rg_pairwise_model *model,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count
) {
    size_t low = 0;
    size_t high;
    size_t i;
    if (model == 0 || source_count == 0 || target_count == 0) {
        return 0;
    }
    high = model->chunk_count;
    /* Lower bound of the grapheme-key block. The table is sorted on graphemes
     * alone, so the block is where the search can end; a match must also agree
     * on tone, length and stress, which the ordering does not distinguish, so
     * those are checked by scanning the block. It is one row wide in practice.
     * Returning the first full match keeps the linear scan's answer, which is
     * the one every published model was built against. */
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        if (chunk_key_cmp(&model->chunks[mid], source, source_count, target, target_count) < 0) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    for (i = low; i < model->chunk_count; i++) {
        const rg_chunk_row *row = &model->chunks[i];
        if (chunk_key_cmp(row, source, source_count, target, target_count) != 0) {
            break;
        }
        if (scoring_segment_array_equal(row->source, row->source_count, source, source_count) &&
            scoring_segment_array_equal(row->target, row->target_count, target, target_count)) {
            return row;
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

/* A rule names one form's environment, and must be matched against that form's
 * context. `target_context` may be null where the caller has no target-side
 * context to offer, in which case target-side rules simply do not fire --
 * which is the safe direction to be wrong in. */
static const rg_conditioned_segment_count_row *find_conditioned_segment_count(
    const rg_pairwise_model *model,
    const char *source,
    const char *target,
    const rg_context_spec *link_context,
    const rg_context_spec *target_context
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
        bool subset = false;
        size_t specificity;
        if (strcmp(row->source, source) != 0 || strcmp(row->target, target) != 0) {
            break;
        }
        {
            const rg_context_spec *against = row->context_is_target ? target_context : link_context;
            if (against == 0) {
                continue;
            }
            if (rg_context_spec_is_subset(&row->context, against, &subset) != RG_OK || !subset) {
                continue;
            }
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
    const rg_feature_displacement *disp = 0;
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
    status = rg_context_displacement_internal(ctx, source.grapheme, target.grapheme,
                                              &disp, &disp_count);
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
    /* Borrowed from the context's displacement cache; nothing to free. */
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
    const rg_context_spec *target_context,
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
    conditioned = find_conditioned_segment_count(model, source, target, link_context, target_context);
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

static double segment_target_total(const rg_pairwise_model *model, const char *target) {
    size_t low = 0;
    size_t high;
    if (model == 0 || target == 0) {
        return 0.0;
    }
    high = model->log_normalizer_count;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        int c = strcmp(model->log_normalizers[mid].source, target);
        if (c == 0) {
            return model->log_normalizers[mid].target_total;
        }
        if (c < 0) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return 0.0;
}

/* The cost of pairing two segments, scored the same in either direction.
 *
 * P(b|a) and P(a|b) are different models. Scoring with one of them made the
 * whole analysis depend on which lect the corpus happened to name first: the
 * denominators differ, so aligning A against B and B against A cost different
 * amounts, and every class reconciled from those alignments inherited the
 * difference. regulae takes no view on which lect is ancestral, and a score
 * that does is making the claim by accident.
 *
 * The cost is the geometric mean of the two directions -- half the surprisal of
 * seeing b given a plus half of seeing a given b -- which keeps each direction
 * a proper conditional and is exactly symmetric in the pair. The observed count
 * is shared between them; only the denominators and the prior differ.
 *
 * The reverse prior needs no table of its own. alpha(a,b) is the concentration
 * times a softmax over merkmal distances from a, distance is symmetric, so
 * alpha(b,a) is alpha(a,b) scaled by the ratio of the two partition functions,
 * both of which are already stored.
 *
 * Returns 0 when the pair is unknown to the model, and the caller falls back to
 * the bare merkmal distance. */
static int segment_symmetric_cost(
    const rg_pairwise_model *model,
    const char *source,
    const char *target,
    const rg_context_spec *link_context,
    const rg_context_spec *target_context,
    double *out
) {
    const rg_conditioned_segment_count_row *conditioned;
    const rg_segment_count_row *unconditioned;
    double alpha = 0.0;
    double n = 0.0;
    double z_source;
    double z_target;
    double alpha_reverse;
    double forward;
    double reverse;
    double denominator_source;
    double denominator_target;

    *out = 0.0;
    if (model == 0 || source == 0 || target == 0) {
        return 0;
    }
    conditioned = find_conditioned_segment_count(model, source, target, link_context, target_context);
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
    denominator_source = model->concentration + segment_source_total(model, source);
    denominator_target = model->concentration + segment_target_total(model, target);
    if (denominator_source <= 0.0 || denominator_target <= 0.0) {
        return 0;
    }
    z_source = segment_log_normalizer(model, source);
    z_target = segment_log_normalizer(model, target);
    /* alpha(t,s) could be derived as alpha(s,t) * exp(z(s) - z(t)) -- merkmal's
     * distance is symmetric, so the two differ only by their partition
     * functions. Reading it out of the table instead costs one binary search
     * and avoids an exp of a difference of logs, which is the one step in this
     * function whose last bit moves between libm implementations. The tables
     * have to agree bit for bit across platforms; the published models are
     * compared that way. */
    if (alpha <= 0.0 || !segment_prior_lookup(model, target, source, &alpha_reverse)) {
        alpha_reverse = 0.0;
    }
    forward = alpha + n;
    reverse = alpha_reverse + n;
    if (forward <= 0.0 || reverse <= 0.0 || !isfinite(reverse)) {
        return 0;
    }
    /* Half of -log P(t|s) plus half of -log P(s|t), written as one logarithm of
     * the combined ratio. Four separate logs say the same thing in exact
     * arithmetic and disagree in the last bit between libm implementations,
     * which is enough to flip a tied alignment and make the wasm build publish
     * a different model from the native one.
     *
     * The trailing -z is the reference's own term: the one-sided cost was
     * -log P(t|s) - z(s), which discounts a source whose distance distribution
     * is diffuse. Symmetrising it means averaging the two directions' z as
     * well, not dropping it. */
    *out = 0.5 * log((denominator_source * denominator_target) / (forward * reverse))
         - 0.5 * (z_source + z_target);
    return 1;
}

/* The same quantity without the log-Z offset: half of -log P(t|s) plus half of
 * -log P(s|t) and nothing else.
 *
 * Chunk promotion needs this. It weighs a chunk's promoted cost against the
 * cost of building it from segment draws, and both sides of that comparison
 * have to be the same kind of number or the winner depends on which lect the
 * corpus named first. The compositional side used to be the bare forward
 * posterior: on place_dissimilation the p-lect's /t/ answers only to /t/, so
 * building "at" from segments looked free from that side and cost 0.48 nats
 * from the other, and chunks promoted in one direction that did not promote in
 * the other. */
int rg_segment_symmetric_raw_cost_internal(
    const rg_pairwise_model *model,
    const char *source,
    const char *target,
    double *out
) {
    double cost;
    if (!segment_symmetric_cost(model, source, target, 0, 0, &cost)) {
        return 0;
    }
    *out = cost + 0.5 * (segment_log_normalizer(model, source) +
                         segment_log_normalizer(model, target));
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
    const rg_context_spec *target_context,
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
        size_t pairing[RG_MAX_REORDER_SPAN];
        int reordering = rg_link_is_reordering_internal(source, source_count, target, target_count, pairing);
        size_t i;
        int asymmetry = (int)source_count - (int)target_count;
        double total = 0.0;
        if (asymmetry < 0) {
            asymmetry = -asymmetry;
        }
        for (i = 0; i < paired; i++) {
            double pair_cost = 0.0;
            /* A span whose target is its own segments in another order costs
             * what the reordering costs, not what pretending each position
             * substituted for the one below it would cost. Scoring it
             * positionally makes metathesis look like a pile of unrelated
             * substitutions and prices it out of the search. */
            size_t partner = reordering ? pairing[i] : i;
            status = rg_score_link_with_context_model_internal(
                ctx,
                model,
                options,
                &source[i],
                1,
                &target[partner],
                1,
                link_context,
                target_context,
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

        if (!segment_symmetric_cost(model, src, tgt, link_context, target_context, &seg_cost)) {
            /* The prior never saw this pair, so fall back to the bare merkmal
             * distance. Computed here rather than up front: the model path is
             * the common case and does not need it. */
            return rg_score_link(ctx, source, source_count, target, target_count, out);
        }
        (void)posterior;
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
