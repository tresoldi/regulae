#include "regulae.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Expected break positions are the Go reference output, produced by
 * `go run ./tools/goref syllables`. */

typedef struct syllable_case {
    const char *graphemes[8];
    size_t segment_count;
    int breaks[4];
    size_t break_count;
} syllable_case;

static const syllable_case cases[] = {
    {{"p", "a", "t", "e", "r"}, 5, {2}, 1},
    {{"k", "a", "s", "a"}, 4, {2}, 1},
    {{"a", "m", "i", "k", "o"}, 5, {1, 3}, 2},
    {{"s", "t", "r", "a", "t", "a"}, 6, {4}, 1},
    {{"k", "a", "r", "t", "l", "i"}, 6, {3}, 1},
    {{"t", "a"}, 2, {0}, 0},
    {{"a"}, 1, {0}, 0},
    {{"m", "b", "u", "n", "d", "u"}, 6, {4}, 1},
    {{"f", "i", "l", "i", "u", "s"}, 6, {2, 4}, 2},
    {{"w", "a", "t", "e", "r"}, 5, {2}, 1},
    {{"j", "a", "m"}, 3, {0}, 0},
    {{"l", "i", "k", "e"}, 4, {2}, 1},
    {{"n", "g", "a", "t", "a"}, 5, {3}, 1}
};

static void test_breaks_match_go_reference(rg_context *ctx) {
    size_t c;
    for (c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        rg_segment segments[8];
        rg_form form;
        int *breaks = 0;
        size_t break_count = 0;
        size_t i;

        memset(segments, 0, sizeof(segments));
        for (i = 0; i < cases[c].segment_count; i++) {
            segments[i].grapheme = cases[c].graphemes[i];
        }
        memset(&form, 0, sizeof(form));
        form.lect_id = "x";
        form.segments = segments;
        form.segment_count = cases[c].segment_count;

        assert(rg_compute_syllable_breaks(ctx, &form, &breaks, &break_count) == RG_OK);
        if (break_count != cases[c].break_count) {
            fprintf(stderr, "case %zu: got %zu breaks, want %zu\n", c, break_count, cases[c].break_count);
            abort();
        }
        for (i = 0; i < break_count; i++) {
            if (breaks[i] != cases[c].breaks[i]) {
                fprintf(stderr, "case %zu: break %zu is %d, want %d\n", c, i, breaks[i], cases[c].breaks[i]);
                abort();
            }
        }
        rg_syllable_breaks_free(breaks);
    }
}

/* Caller-supplied breaks are the escape hatch for language-specific
 * phonotactics and must be returned unchanged. */
static void test_supplied_breaks_are_respected(rg_context *ctx) {
    rg_segment segments[5];
    rg_form form;
    int supplied[2] = {1, 4};
    int *breaks = 0;
    size_t break_count = 0;
    size_t i;
    static const char *const graphemes[5] = {"p", "a", "t", "e", "r"};

    memset(segments, 0, sizeof(segments));
    for (i = 0; i < 5; i++) {
        segments[i].grapheme = graphemes[i];
    }
    memset(&form, 0, sizeof(form));
    form.lect_id = "x";
    form.segments = segments;
    form.segment_count = 5;
    form.syllable_breaks = supplied;
    form.syllable_break_count = 2;

    assert(rg_compute_syllable_breaks(ctx, &form, &breaks, &break_count) == RG_OK);
    assert(break_count == 2);
    assert(breaks[0] == 1);
    assert(breaks[1] == 4);
    rg_syllable_breaks_free(breaks);
}

/* An unknown grapheme scores neutral sonority rather than failing, matching the
 * Go bridge, which maps an unresolved grapheme to an absent feature set. */
static void test_unknown_grapheme_does_not_fail(rg_context *ctx) {
    rg_segment segments[3];
    rg_form form;
    int *breaks = 0;
    size_t break_count = 0;

    memset(segments, 0, sizeof(segments));
    segments[0].grapheme = "\xc7\x9d\xc7\x9d\xc7\x9d";
    segments[1].grapheme = "a";
    segments[2].grapheme = "t";
    memset(&form, 0, sizeof(form));
    form.lect_id = "x";
    form.segments = segments;
    form.segment_count = 3;

    assert(rg_compute_syllable_breaks(ctx, &form, &breaks, &break_count) == RG_OK);
    rg_syllable_breaks_free(breaks);
}

int main(void) {
    rg_context *ctx = 0;
    assert(rg_context_new_builtin(&ctx) == RG_OK);
    test_breaks_match_go_reference(ctx);
    test_supplied_breaks_are_respected(ctx);
    test_unknown_grapheme_does_not_fail(ctx);
    rg_context_free(ctx);
    printf("syllabification tests passed\n");
    return 0;
}
