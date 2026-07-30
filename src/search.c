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

static rg_status link_from_slice(
    const rg_context *ctx,
    const rg_form *source,
    size_t source_start,
    size_t source_count,
    const rg_form *target,
    size_t target_start,
    size_t target_count,
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

rg_status rg_align_forms(
    const rg_context *ctx,
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
    width = m + 1;
    cost = (double *)calloc((n + 1) * (m + 1), sizeof(*cost));
    back = (rg_dp_step *)calloc((n + 1) * (m + 1), sizeof(*back));
    if (cost == 0 || back == 0) {
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
                    status = rg_score_link(ctx, source->segments + (i - k), k, target->segments + (j - l), l, &link_cost);
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
        free(cost);
        free(back);
        return status;
    }
    alignment = (rg_alignment *)calloc(1, sizeof(*alignment));
    if (alignment == 0) {
        free(cost);
        free(back);
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
        status = link_from_slice(ctx, source, step.prev_i, step.source_count, target, step.prev_j, step.target_count, &link);
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
        return status;
    }
    free(rev_links);
    free(cost);
    free(back);
    *out = alignment;
    return RG_OK;
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
