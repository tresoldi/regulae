#include "search_internal.h"

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

void form_clear(rg_form *form) {
    size_t i;
    if (form == 0) {
        return;
    }
    rg_free_owned_internal(form->lect_id);
    for (i = 0; i < form->segment_count; i++) {
        rg_segment_clear_internal(rg_owned_internal(&form->segments[i]));
    }
    rg_free_owned_internal(form->segments);
    rg_free_owned_internal(form->syllable_breaks);
    rg_free_owned_internal(form->morpheme_breaks);
    memset(form, 0, sizeof(*form));
}

rg_status copy_ints(const int *items, size_t count, const int **out) {
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

rg_status form_copy(const rg_form *src, rg_form *out) {
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
                    /* What the committed cross-dimensional rules make of this
                     * position, charged here rather than added to the finished
                     * alignment afterwards.
                     *
                     * The adjustment is local to the transition -- its source
                     * predicate reads the source form at i-k and its target
                     * value the target form at (j-l)+offset, and both forms are
                     * fixed input -- so the DP can price it. Until it did, the
                     * search chose an alignment without knowing the rules would
                     * fire and the rules re-scored what it had already chosen,
                     * which docs/correspondence_discovery.md called provisional
                     * and more principled to integrate.
                     *
                     * Zero until cross-dimensional discovery has committed
                     * something, which is every stage before it. */
                    total = prev + link_cost
                          + cross_dimensional_link_adjustment(ctx, model, source, i - k,
                                                              target, j - l, k, l)
                          + RG_CHUNK_COMPLEXITY_PENALTY * (double)((int)k + (int)l - 2);
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

