#include "model_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

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

/* The same, counting from the target chunk: how much mass answers to it, and
 * from how many distinct sources. */
static double chunk_target_total(const chunk_candidate *items, size_t count, const chunk_candidate *candidate, size_t *source_variants) {
    size_t i;
    double total = 0.0;
    size_t variants = 0;
    for (i = 0; i < count; i++) {
        if (segment_array_equal(items[i].target, items[i].target_count, candidate->target, candidate->target_count)) {
            total += items[i].count;
            variants++;
        }
    }
    *source_variants = variants;
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
            double segment_cost = 0.0;
            if (!rg_segment_symmetric_raw_cost_internal(
                    model, link->source[0].grapheme, link->target[0].grapheme, &segment_cost)) {
                rg_alignment_free(sub);
                return RG_OK;
            }
            cost += segment_cost;
        } else {
            size_t span = link->source_count > link->target_count ? link->source_count : link->target_count;
            cost += gap_cost_per_segment * (double)span;
        }
    }
    rg_alignment_free(sub);
    *out = cost;
    return RG_OK;
}

/* Laplace-smoothed MLE cost of the chunk under the candidate counts, as the
 * geometric mean of the two conditionals.
 *
 * P(target chunk | source chunk) alone is a claim about one direction, and it
 * prices the same chunk differently depending on which lect the corpus happens
 * to name first. On place_dissimilation the p-lect's "pa" is ambiguous -- it
 * answers to both "ta" and "pa" -- while the t-lect's "ta" answers to "pa" and
 * nothing else, so read from the t-lect the chunk was free and always beat its
 * compositional cost, and read from the p-lect it did not. Chunk promotion then
 * swallowed the evidence for p > t / _ [labial ...] in one direction only.
 *
 * The segment scorer was symmetrized for the same reason; this is the same
 * quantity one level up. One log call, not two: four of them in the segment
 * cost were enough to make native and wasm disagree in the last bit. */
static double promoted_chunk_cost(
    const chunk_candidate *items,
    size_t count,
    const chunk_candidate *candidate,
    double alpha
) {
    size_t target_variants = 0;
    size_t source_variants = 0;
    double source_total = chunk_source_total(items, count, candidate, &target_variants);
    double target_total = chunk_target_total(items, count, candidate, &source_variants);
    double numerator;
    double denominator;
    if (target_variants == 0 || source_variants == 0) {
        return INFINITY;
    }
    numerator = (source_total + alpha * (double)target_variants) *
                (target_total + alpha * (double)source_variants);
    denominator = (candidate->count + alpha) * (candidate->count + alpha);
    if (!(numerator > 0.0) || !(denominator > 0.0)) {
        return INFINITY;
    }
    return 0.5 * log(numerator / denominator);
}

/* Enumerates contiguous sub-alignments as candidate chunks and promotes the
 * ones whose BIC improves. Each candidate is judged independently against the
 * unchanged segment table, so promotion order cannot matter. */
rg_status promote_chunk_rows(
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
    double min_transparency = 0.0;
    size_t i;
    size_t kept;
    int max_chunk_size = RG_DEFAULT_MAX_CHUNK_SIZE;
    rg_status status = RG_OK;

    if (options != 0) {
        if (options->max_chunk_size > 0) {
            max_chunk_size = options->max_chunk_size;
        }
        if (options->bic.min_chunk_observations > 0) {
            min_chunk_obs = (double)options->bic.min_chunk_observations;
        }
        min_transparency = options->chunk_min_transparency;
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
        /* NOLINTNEXTLINE(clang-analyzer-unix.Malloc) */
        for (a = 0; a < link_count && status == RG_OK; a++) {
            size_t b;
            /* NOLINTNEXTLINE(clang-analyzer-unix.Malloc) */
            for (b = a; b < link_count && status == RG_OK; b++) {
                /* NOLINTNEXTLINE(clang-analyzer-unix.Malloc): both chunks
                 * are released by segment_array_clear on every path out. It
                 * frees through rg_free_owned_internal, which copies the
                 * pointer value to drop the const, and the analyzer loses the
                 * allocation's identity across that copy. */
                const rg_segment *source_chunk = 0;
                const rg_segment *target_chunk = 0;
                size_t source_chunk_count = 0;
                size_t target_chunk_count = 0;
                /* NOLINTNEXTLINE(clang-analyzer-unix.Malloc) */
                status = append_segments_from_link_span(alignment, a, b, 1, &source_chunk, &source_chunk_count);
                if (status != RG_OK) {
                    break;
                }
                /* NOLINTNEXTLINE(clang-analyzer-unix.Malloc) */
                status = append_segments_from_link_span(alignment, a, b, 0, &target_chunk, &target_chunk_count);
                if (status != RG_OK) {
                    segment_array_clear(source_chunk, source_chunk_count);
                    break;
                }
                if (source_chunk_count > (size_t)max_chunk_size || target_chunk_count > (size_t)max_chunk_size) {
                    /* A reordering is allowed past the chunk limit, for the
                     * same reason the alignment search allows the span: a
                     * transposition over any distance is one link covering
                     * everything between the two segments that moved, and
                     * capping it at the chunk width would let the search find
                     * the reordering and then leave the model with no row
                     * saying so. */
                    size_t pairing[RG_MAX_REORDER_SPAN];
                    if (!rg_link_is_reordering_internal(source_chunk, source_chunk_count,
                                                        target_chunk, target_chunk_count, pairing)) {
                        segment_array_clear(source_chunk, source_chunk_count);
                        segment_array_clear(target_chunk, target_chunk_count);
                        break;
                    }
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
        double prior_penalty;
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
        prior_penalty = rg_chunk_prior_penalty_internal(
            ctx,
            candidates[i].source, candidates[i].source_count,
            candidates[i].target, candidates[i].target_count);
        delta_bic = -2.0 * reduction + ((double)k_params + prior_penalty) * log(n_observations);
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
        {
            size_t pairing[RG_MAX_REORDER_SPAN];
            rows[row_count].reordering = rg_link_is_reordering_internal(
                candidates[i].source, candidates[i].source_count,
                candidates[i].target, candidates[i].target_count, pairing);
        }
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

    /* Transparency is scored after the whole promoted set is known, because
     * part of the score is whether a chunk merely wraps a smaller chunk that
     * was also promoted, and that cannot be asked of a row in isolation. */
    for (i = 0; i < row_count && status == RG_OK; i++) {
        status = rg_chunk_transparency_internal(
            ctx, options, model,
            rows[i].source, rows[i].source_count,
            rows[i].target, rows[i].target_count,
            rows, row_count, &rows[i].transparency);
    }
    if (status != RG_OK) {
        for (i = 0; i < row_count; i++) {
            chunk_row_clear(&rows[i]);
        }
        free(rows);
        return status;
    }
    kept = 0;
    for (i = 0; i < row_count; i++) {
        if (rows[i].transparency < min_transparency) {
            chunk_row_clear(&rows[i]);
            continue;
        }
        if (kept != i) {
            rows[kept] = rows[i];
        }
        kept++;
    }
    row_count = kept;

    for (i = 0; i < model->chunk_count; i++) {
        chunk_row_clear(&model->chunks[i]);
    }
    free(model->chunks);
    model->chunks = rows;
    model->chunk_count = row_count;
    return RG_OK;
}
