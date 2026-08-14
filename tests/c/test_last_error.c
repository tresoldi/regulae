#include "regulae.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

/* The Go reference's UnknownGraphemeError names the offending grapheme and the
 * feature system. A C status code cannot, so the context carries it: "unknown
 * grapheme" without saying which one is unactionable on a corpus of any size. */

static rg_segment seg(const char *g) {
    rg_segment s = {g, 0, 0, 0};
    return s;
}

static void test_names_the_grapheme(rg_context *ctx) {
    rg_segment source[2];
    rg_segment target[2];
    rg_form a;
    rg_form b;
    rg_alignment *alignment = 0;
    const char *grapheme = 0;
    const char *system = 0;

    /* Nothing has failed yet. */
    rg_context_last_error(ctx, &grapheme, &system);
    assert(grapheme == 0);
    assert(system != 0 && strcmp(system, RG_DEFAULT_FEATURE_SYSTEM) == 0);

    source[0] = seg("p");
    source[1] = seg("\xe2\x80\xa1");
    target[0] = seg("p");
    target[1] = seg("a");
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    a.lect_id = "A";
    a.segments = source;
    a.segment_count = 2;
    b.lect_id = "B";
    b.segments = target;
    b.segment_count = 2;

    assert(rg_align_forms(ctx, &a, &b, 0, &alignment) == RG_ERR_UNKNOWN_GRAPHEME);
    rg_context_last_error(ctx, &grapheme, &system);
    assert(grapheme != 0);
    assert(strcmp(grapheme, "\xe2\x80\xa1") == 0);
    assert(system != 0 && strcmp(system, RG_DEFAULT_FEATURE_SYSTEM) == 0);
}

/* A CLDF boundary marker is a gap in the source data, not a sound the feature
 * system is missing, and the two ask different things of the user. */
static void test_source_markup_is_told_apart(rg_context *ctx) {
    rg_segment source[1];
    rg_segment target[1];
    rg_form a;
    rg_form b;
    rg_alignment *alignment = 0;
    const char *grapheme = 0;
    rg_grapheme_diagnosis diagnosis;

    source[0] = seg("+");
    target[0] = seg("a");
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    a.lect_id = "A";
    a.segments = source;
    a.segment_count = 1;
    b.lect_id = "B";
    b.segments = target;
    b.segment_count = 1;

    assert(rg_align_forms(ctx, &a, &b, 0, &alignment) == RG_ERR_SOURCE_MARKER);
    rg_context_last_error(ctx, &grapheme, 0);
    assert(grapheme != 0 && strcmp(grapheme, "+") == 0);
    assert(rg_context_last_diagnosis(ctx, &diagnosis) == 1);
    assert(diagnosis.status == RG_ERR_SOURCE_MARKER);

    /* The cached verdict repeats the reason rather than flattening it. */
    assert(rg_align_forms(ctx, &a, &b, 0, &alignment) == RG_ERR_SOURCE_MARKER);
}

/* A refusal localises itself: the prefix that does resolve, and the character
 * where it stops. */
static void test_diagnosis_localises_the_break(rg_context *ctx) {
    rg_grapheme_diagnosis diagnosis;

    assert(rg_context_diagnose(ctx, "p", &diagnosis) == RG_OK);
    assert(diagnosis.status == RG_OK);
    assert(diagnosis.valid_prefix_bytes == 1);

    assert(rg_context_diagnose(ctx, "p\xe2\x80\xa1", &diagnosis) == RG_OK);
    assert(diagnosis.status == RG_ERR_UNKNOWN_GRAPHEME);
    assert(diagnosis.valid_prefix_bytes == 1);
    assert(strcmp(diagnosis.offending, "\xe2\x80\xa1") == 0);

    assert(rg_context_diagnose(0, "p", &diagnosis) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_context_diagnose(ctx, 0, &diagnosis) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_context_diagnose(ctx, "p", 0) == RG_ERR_INVALID_ARGUMENT);
}

/* A second failure replaces the first, so a caller always reads the current
 * one rather than whichever grapheme failed first. */
static void test_latest_failure_wins(rg_context *ctx) {
    rg_segment source[1];
    rg_segment target[1];
    rg_form a;
    rg_form b;
    rg_alignment *alignment = 0;
    const char *grapheme = 0;

    target[0] = seg("a");
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    a.lect_id = "A";
    a.segments = source;
    a.segment_count = 1;
    b.lect_id = "B";
    b.segments = target;
    b.segment_count = 1;

    source[0] = seg("+");
    assert(rg_align_forms(ctx, &a, &b, 0, &alignment) == RG_ERR_SOURCE_MARKER);
    rg_context_last_error(ctx, &grapheme, 0);
    assert(grapheme != 0 && strcmp(grapheme, "+") == 0);

    source[0] = seg("=");
    assert(rg_align_forms(ctx, &a, &b, 0, &alignment) == RG_ERR_UNKNOWN_GRAPHEME);
    rg_context_last_error(ctx, &grapheme, 0);
    assert(grapheme != 0 && strcmp(grapheme, "=") == 0);
}

/* A cached negative verdict must still name the grapheme: the second lookup of
 * the same bad grapheme takes a different path through the cache. */
static void test_cached_failure_still_names_it(rg_context *ctx) {
    rg_segment source[1];
    rg_segment target[1];
    rg_form a;
    rg_form b;
    rg_alignment *alignment = 0;
    const char *grapheme = 0;
    int i;

    source[0] = seg("\xe2\x80\xa1");
    target[0] = seg("a");
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    a.lect_id = "A";
    a.segments = source;
    a.segment_count = 1;
    b.lect_id = "B";
    b.segments = target;
    b.segment_count = 1;

    for (i = 0; i < 2; i++) {
        assert(rg_align_forms(ctx, &a, &b, 0, &alignment) == RG_ERR_UNKNOWN_GRAPHEME);
        rg_context_last_error(ctx, &grapheme, 0);
        assert(grapheme != 0);
        assert(strcmp(grapheme, "\xe2\x80\xa1") == 0);
    }
}

static void test_null_arguments(rg_context *ctx) {
    const char *grapheme = 0;
    const char *system = 0;
    rg_context_last_error(0, &grapheme, &system);
    assert(grapheme == 0);
    assert(system == 0);
    /* Either output may be omitted. */
    rg_context_last_error(ctx, 0, 0);
    rg_context_last_error(ctx, &grapheme, 0);
}

int main(void) {
    rg_context *ctx = 0;
    assert(rg_context_new_builtin(&ctx) == RG_OK);
    test_names_the_grapheme(ctx);
    test_source_markup_is_told_apart(ctx);
    test_diagnosis_localises_the_break(ctx);
    test_latest_failure_wins(ctx);
    test_cached_failure_still_names_it(ctx);
    test_null_arguments(ctx);
    rg_context_free(ctx);
    printf("last-error tests passed\n");
    return 0;
}
