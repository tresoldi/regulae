#include "regulae.h"

#include <assert.h>
#include <math.h>
#include <string.h>

static rg_form form(const char *lect, const rg_segment *segments, size_t count) {
    rg_form f;
    f.lect_id = lect;
    f.segments = segments;
    f.segment_count = count;
    f.syllable_breaks = 0;
    f.syllable_break_count = 0;
    f.morpheme_breaks = 0;
    f.morpheme_break_count = 0;
    return f;
}

static rg_alignment *must_align(rg_context *ctx, const rg_form *src, const rg_form *tgt, int max_chunk) {
    rg_alignment *alignment = 0;
    assert(rg_align_forms(ctx, src, tgt, max_chunk, &alignment) == RG_OK);
    assert(alignment != 0);
    return alignment;
}

static double must_cost(rg_context *ctx, const rg_alignment *alignment) {
    double cost = -1.0;
    assert(rg_alignment_cost(ctx, alignment, &cost) == RG_OK);
    return cost;
}

static void assert_coverage(const rg_alignment *alignment, const rg_segment *src, size_t src_n, const rg_segment *tgt, size_t tgt_n) {
    size_t i;
    size_t src_pos = 0;
    size_t tgt_pos = 0;
    for (i = 0; i < rg_alignment_link_count(alignment); i++) {
        const rg_link *link = rg_alignment_link_at(alignment, i);
        size_t j;
        assert(link != 0);
        for (j = 0; j < link->source_count; j++) {
            assert(src_pos < src_n);
            assert(strcmp(link->source[j].grapheme, src[src_pos].grapheme) == 0);
            src_pos++;
        }
        for (j = 0; j < link->target_count; j++) {
            assert(tgt_pos < tgt_n);
            assert(strcmp(link->target[j].grapheme, tgt[tgt_pos].grapheme) == 0);
            tgt_pos++;
        }
    }
    assert(src_pos == src_n);
    assert(tgt_pos == tgt_n);
}

int main(void) {
    rg_context *ctx = 0;
    rg_segment empty_segments[] = {{"p", 0, 0, 0}};
    rg_form empty = form("A", empty_segments, 0);
    rg_segment pater[] = {{"p", 0, 0, 0}, {"a", 0, 0, 0}, {"t", 0, 0, 0}, {"e", 0, 0, 0}, {"r", 0, 0, 0}};
    rg_segment fadar[] = {{"f", 0, 0, 0}, {"a", 0, 0, 0}, {"d", 0, 0, 0}, {"a", 0, 0, 0}, {"r", 0, 0, 0}};
    rg_segment nokt[] = {{"n", 0, 0, 0}, {"o", 0, 0, 0}, {"k", 0, 0, 0}, {"t", 0, 0, 0}};
    rg_segment nott[] = {{"n", 0, 0, 0}, {"o", 0, 0, 0}, {"t", 0, 0, 0}, {"t", 0, 0, 0}};
    rg_segment qq[] = {{"QQZZ", 0, 0, 0}};
    rg_form fpater = form("latin", pater, 5);
    rg_form ffadar = form("gothic", fadar, 5);
    rg_form fnokt = form("A", nokt, 4);
    rg_form fnott = form("B", nott, 4);
    rg_form fqq = form("A", qq, 1);
    rg_alignment *alignment;
    rg_alignment *alignment2;
    double c1;
    double c2;
    double c3;
    size_t i;

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    alignment = must_align(ctx, &empty, &empty, 0);
    assert(rg_alignment_link_count(alignment) == 0);
    assert(fabs(must_cost(ctx, alignment)) < 1e-12);
    rg_alignment_free(alignment);

    alignment = must_align(ctx, &fpater, &fpater, 0);
    assert(rg_alignment_link_count(alignment) == 5);
    assert(fabs(must_cost(ctx, alignment)) < 1e-12);
    assert_coverage(alignment, pater, 5, pater, 5);
    for (i = 0; i < rg_alignment_link_count(alignment); i++) {
        const rg_link *link = rg_alignment_link_at(alignment, i);
        assert(link->source_count == 1);
        assert(link->target_count == 1);
        assert(link->feature_displacement_count == 0);
    }
    rg_alignment_free(alignment);

    alignment = must_align(ctx, &fpater, &ffadar, 0);
    assert(rg_alignment_link_count(alignment) == 5);
    assert_coverage(alignment, pater, 5, fadar, 5);
    assert(strcmp(rg_alignment_link_at(alignment, 0)->source[0].grapheme, "p") == 0);
    assert(strcmp(rg_alignment_link_at(alignment, 0)->target[0].grapheme, "f") == 0);
    assert(rg_alignment_link_at(alignment, 0)->feature_displacement_count > 0);
    alignment2 = must_align(ctx, &fpater, &ffadar, 0);
    assert(rg_alignment_link_count(alignment) == rg_alignment_link_count(alignment2));
    for (i = 0; i < rg_alignment_link_count(alignment); i++) {
        const rg_link *a = rg_alignment_link_at(alignment, i);
        const rg_link *b = rg_alignment_link_at(alignment2, i);
        assert(a->source_count == b->source_count);
        assert(a->target_count == b->target_count);
        assert(strcmp(a->source[0].grapheme, b->source[0].grapheme) == 0);
        assert(strcmp(a->target[0].grapheme, b->target[0].grapheme) == 0);
    }
    rg_alignment_free(alignment2);
    rg_alignment_free(alignment);

    alignment = must_align(ctx, &fnokt, &fnott, 1);
    c1 = must_cost(ctx, alignment);
    rg_alignment_free(alignment);
    alignment = must_align(ctx, &fnokt, &fnott, 2);
    c2 = must_cost(ctx, alignment);
    rg_alignment_free(alignment);
    alignment = must_align(ctx, &fnokt, &fnott, 3);
    c3 = must_cost(ctx, alignment);
    rg_alignment_free(alignment);
    assert(c2 <= c1 + 1e-9);
    assert(c3 <= c2 + 1e-9);

    assert(rg_align_forms(ctx, &fqq, &fpater, 0, &alignment) == RG_ERR_UNKNOWN_GRAPHEME);
    assert(rg_align_forms(ctx, &fpater, &ffadar, -1, &alignment) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_alignment_link_at(0, 0) == 0);
    assert(rg_alignment_link_count(0) == 0);
    assert(rg_alignment_cost(ctx, 0, &c1) == RG_ERR_INVALID_ARGUMENT);
    rg_alignment_free(0);
    rg_context_free(ctx);
    return 0;
}
