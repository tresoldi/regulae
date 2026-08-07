#include "regulae.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Progress reporting and cancellation. Cancellation exists because a browser
 * needs an escape hatch that isn't the tab's stop button, but it is a library
 * feature: the CLI and any future wrapper get it too. */

#define MAX_STAGES 512

typedef struct recorder {
    char stages[MAX_STAGES][64];
    size_t completed[MAX_STAGES];
    size_t totals[MAX_STAGES];
    size_t count;
    size_t cancel_after;
} recorder;

static int record(const char *stage, size_t completed, size_t total, void *user_data) {
    recorder *r = (recorder *)user_data;
    if (r->count < MAX_STAGES) {
        snprintf(r->stages[r->count], sizeof(r->stages[0]), "%s", stage);
        r->completed[r->count] = completed;
        r->totals[r->count] = total;
        r->count++;
    }
    if (r->cancel_after > 0 && r->count >= r->cancel_after) {
        return 1;
    }
    return 0;
}

static rg_corpus *load(const char *name) {
    rg_corpus *corpus = 0;
    char path[1024];
    snprintf(path, sizeof(path), "%s/testdata/parity/%s", REGULAE_SOURCE_DIR, name);
    assert(rg_corpus_load_tsv(path, 0, &corpus) == RG_OK);
    return corpus;
}

static void test_progress_is_monotonic_and_bounded(rg_context *ctx) {
    rg_corpus *corpus = load("three_lect_basic.tsv");
    rg_multi_model *model = 0;
    rg_train_options options;
    recorder r;
    size_t i;
    int saw_pairwise_stage = 0;
    int saw_multilect_stage = 0;

    memset(&r, 0, sizeof(r));
    rg_train_options_init_defaults(&options);
    options.progress = record;
    options.progress_user_data = &r;

    assert(rg_train_model(ctx, rg_corpus_cognates(corpus), rg_corpus_cognate_count(corpus),
                          &options, &model) == RG_OK);
    assert(r.count > 0);

    for (i = 0; i < r.count; i++) {
        /* A fraction a caller can render: never decreasing, never over one. */
        assert(r.completed[i] <= r.totals[i]);
        assert(r.totals[i] == r.totals[0]);
        if (i > 0) {
            assert(r.completed[i] >= r.completed[i - 1]);
        }
        if (strcmp(r.stages[i], "chunk promotion") == 0) {
            saw_pairwise_stage = 1;
        }
        if (strcmp(r.stages[i], "class discovery") == 0) {
            saw_multilect_stage = 1;
        }
    }
    /* Three lects means three pairs, each running the eight pairwise stages,
     * plus the three multi-lect stages. */
    assert(r.totals[0] == 3 * 8 + 3);
    assert(r.count == r.totals[0]);
    assert(saw_pairwise_stage);
    assert(saw_multilect_stage);

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

static void test_cancellation(rg_context *ctx) {
    rg_corpus *corpus = load("real_ppn_hawaiian.tsv");
    rg_multi_model *model = 0;
    rg_train_options options;
    recorder r;

    memset(&r, 0, sizeof(r));
    r.cancel_after = 2;
    rg_train_options_init_defaults(&options);
    options.progress = record;
    options.progress_user_data = &r;

    assert(rg_train_model(ctx, rg_corpus_cognates(corpus), rg_corpus_cognate_count(corpus),
                          &options, &model) == RG_ERR_CANCELLED);
    /* Nothing is handed back on cancellation, so there is nothing to leak or
     * to mistake for a finished model. */
    assert(model == 0);
    /* It stopped near where it was asked to, not at the end. */
    assert(r.count >= 2);
    assert(r.count < 8);

    rg_corpus_free(corpus);
}

/* Cancelling on the very first callback must still unwind cleanly. */
static void test_cancel_immediately(rg_context *ctx) {
    rg_corpus *corpus = load("three_lect_basic.tsv");
    rg_multi_model *model = 0;
    rg_train_options options;
    recorder r;

    memset(&r, 0, sizeof(r));
    r.cancel_after = 1;
    rg_train_options_init_defaults(&options);
    options.progress = record;
    options.progress_user_data = &r;

    assert(rg_train_model(ctx, rg_corpus_cognates(corpus), rg_corpus_cognate_count(corpus),
                          &options, &model) == RG_ERR_CANCELLED);
    assert(model == 0);
    assert(r.count == 1);
    rg_corpus_free(corpus);
}

/* Pairwise training reports on its own, for callers not going through the
 * multi-lect entry point. */
static void test_pairwise_progress(rg_context *ctx) {
    rg_segment source[2];
    rg_segment target[2];
    rg_form_pair pairs[4];
    rg_pairwise_model *model = 0;
    rg_train_options options;
    recorder r;
    size_t i;

    memset(source, 0, sizeof(source));
    memset(target, 0, sizeof(target));
    source[0].grapheme = "p";
    source[1].grapheme = "a";
    target[0].grapheme = "f";
    target[1].grapheme = "a";
    for (i = 0; i < 4; i++) {
        memset(&pairs[i], 0, sizeof(pairs[i]));
        pairs[i].source.lect_id = "A";
        pairs[i].source.segments = source;
        pairs[i].source.segment_count = 2;
        pairs[i].target.lect_id = "B";
        pairs[i].target.segments = target;
        pairs[i].target.segment_count = 2;
        pairs[i].weight = 1.0;
    }

    memset(&r, 0, sizeof(r));
    rg_train_options_init_defaults(&options);
    options.progress = record;
    options.progress_user_data = &r;
    assert(rg_train_pairwise(ctx, pairs, 4, &options, &model) == RG_OK);
    assert(r.count == 8);
    assert(strcmp(r.stages[0], "initial prior") == 0);
    assert(strcmp(r.stages[7], "long-range discovery") == 0);
    rg_pairwise_model_free(model);

    memset(&r, 0, sizeof(r));
    r.cancel_after = 3;
    model = 0;
    assert(rg_train_pairwise(ctx, pairs, 4, &options, &model) == RG_ERR_CANCELLED);
    assert(model == 0);
    rg_pairwise_model_free(model);
}

/* A null callback is the normal case and must cost nothing. */
static void test_no_callback(rg_context *ctx) {
    rg_corpus *corpus = load("three_lect_basic.tsv");
    rg_multi_model *model = 0;
    rg_train_options options;

    rg_train_options_init_defaults(&options);
    assert(options.progress == 0);
    assert(rg_train_model(ctx, rg_corpus_cognates(corpus), rg_corpus_cognate_count(corpus),
                          &options, &model) == RG_OK);
    assert(model != 0);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

int main(void) {
    rg_context *ctx = 0;
    assert(rg_context_new_builtin(&ctx) == RG_OK);
    test_progress_is_monotonic_and_bounded(ctx);
    test_cancellation(ctx);
    test_cancel_immediately(ctx);
    test_pairwise_progress(ctx);
    test_no_callback(ctx);
    rg_context_free(ctx);
    printf("progress tests passed\n");
    return 0;
}
