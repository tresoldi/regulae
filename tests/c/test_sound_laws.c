#include "regulae.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Sound changes regulae has to be able to find.
 *
 * The corpora under testdata/soundlaws/ are small curated illustrations of
 * changes the field has agreed on for a century, not field data: each one is
 * built so that exactly one answer is right, and so that the wrong answers are
 * available. A synthetic fixture where the only possible split is the intended
 * one proves nothing, so each corpus carries the contrast environments too --
 * the /s/ that does not rhotacise is as much a part of the test as the /s/
 * that does.
 *
 * A test here failing means the method has stopped being able to find a
 * relationship that is not in doubt. */

static rg_corpus *load(const char *name) {
    rg_corpus *corpus = 0;
    char path[512];
    snprintf(path, sizeof(path), "%s/testdata/soundlaws/%s.tsv", REGULAE_SOURCE_DIR, name);
    assert(rg_corpus_load_tsv(path, 0, &corpus) == RG_OK);
    assert(corpus != 0);
    return corpus;
}

static rg_multi_model *train(rg_context *ctx, rg_corpus *corpus) {
    rg_multi_model *model = 0;
    rg_train_options options;
    size_t count = rg_corpus_cognate_count(corpus);
    const rg_cognate_set *sets = rg_corpus_cognate_at(corpus, 0);
    rg_train_options_init_defaults(&options);
    assert(rg_train_model(ctx, sets, count, &options, &model) == RG_OK);
    return model;
}

/* Whether some unconditioned class pairs these two graphemes, in either lect
 * order: which lect a class lists first is a labelling detail. */
static int has_correspondence(const rg_multi_model *model, const char *a, const char *b) {
    size_t i;
    for (i = 0; i < rg_multi_model_unconditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_unconditioned_class_at(model, i);
        size_t j;
        int seen_a = 0;
        int seen_b = 0;
        for (j = 0; j < row->segment_count; j++) {
            if (strcmp(row->graphemes[j], a) == 0) {
                seen_a = 1;
            }
            if (strcmp(row->graphemes[j], b) == 0) {
                seen_b = 1;
            }
        }
        if (seen_a && seen_b) {
            return 1;
        }
    }
    return 0;
}

/* Whether a conditioned class pairs the two graphemes and names `feature` in
 * some lect's environment. */
static int has_conditioned(
    const rg_multi_model *model,
    const char *a,
    const char *b,
    const char *feature
) {
    size_t i;
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
        size_t j;
        int seen_a = 0;
        int seen_b = 0;
        int seen_feature = 0;
        for (j = 0; j < row->segment_count; j++) {
            if (strcmp(row->graphemes[j], a) == 0) {
                seen_a = 1;
            }
            if (strcmp(row->graphemes[j], b) == 0) {
                seen_b = 1;
            }
        }
        for (j = 0; j < row->segment_count; j++) {
            const rg_context_spec *context = &row->contexts[j];
            size_t k;
            for (k = 0; k < context->preceding_count; k++) {
                if (strcmp(context->preceding[k].feature, feature) == 0) {
                    seen_feature = 1;
                }
            }
            for (k = 0; k < context->following_count; k++) {
                if (strcmp(context->following[k].feature, feature) == 0) {
                    seen_feature = 1;
                }
            }
        }
        if (seen_a && seen_b && seen_feature) {
            return 1;
        }
    }
    return 0;
}

/* Grimm's Law: three shifts at once, none of them conditioned. The test is not
 * that some correspondence is found but that all nine are, and as
 * unconditioned classes -- a tool that split them on environment would be
 * inventing conditioning that the change does not have. */
static void test_grimm(rg_context *ctx) {
    rg_corpus *corpus = load("grimm");
    rg_multi_model *model = train(ctx, corpus);

    assert(has_correspondence(model, "p", "f"));
    assert(has_correspondence(model, "t", "\xce\xb8"));
    assert(has_correspondence(model, "k", "x"));

    assert(has_correspondence(model, "d", "t"));
    assert(has_correspondence(model, "g", "k"));

    assert(has_correspondence(model, "b\xca\xb0", "b"));
    assert(has_correspondence(model, "d\xca\xb0", "d"));
    assert(has_correspondence(model, "g\xca\xb0", "g"));

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* Latin rhotacism: /s/ becomes /r/ between vowels and stays /s/ everywhere
 * else. The corpus carries initial, final and preconsonantal /s/ so that
 * "medial" does not predict the change on its own -- without those the
 * position predicate answers perfectly and the environment doing the work is
 * never tested. */
static void test_rhotacism(rg_context *ctx) {
    rg_corpus *corpus = load("rhotacism");
    rg_multi_model *model = train(ctx, corpus);

    assert(has_correspondence(model, "s", "r"));
    assert(has_conditioned(model, "s", "r", "vowel"));

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* A lect id is metadata. Listing a corpus's rows in a different order, which
 * changes nothing about the data, must not change the model.
 *
 * It used to. Lects were held in the order the corpus first mentioned them,
 * while reconciliation, class discovery and the outlier ranking all walk pairs
 * in ascending id order, so a corpus that did not happen to list its lects
 * alphabetically had those stages align each pair in the opposite direction
 * from the one its model was trained in. Reading a model of P(b|a) as P(a|b)
 * misses nearly every lookup and falls back to the untrained prior. */
static void test_row_order_does_not_change_the_model(rg_context *ctx) {
    rg_corpus *forward = load("rhotacism");
    rg_corpus *reversed = 0;
    rg_multi_model *a;
    rg_multi_model *b;
    char path[512];
    size_t i;

    snprintf(path, sizeof(path), "%s/testdata/soundlaws/rhotacism_reordered.tsv",
             REGULAE_SOURCE_DIR);
    assert(rg_corpus_load_tsv(path, 0, &reversed) == RG_OK);

    a = train(ctx, forward);
    b = train(ctx, reversed);

    assert(rg_multi_model_lect_count(a) == rg_multi_model_lect_count(b));
    for (i = 0; i < rg_multi_model_lect_count(a); i++) {
        assert(strcmp(rg_multi_model_lect_at(a, i), rg_multi_model_lect_at(b, i)) == 0);
    }
    assert(rg_multi_model_unconditioned_class_count(a) ==
           rg_multi_model_unconditioned_class_count(b));
    assert(rg_multi_model_conditioned_class_count(a) ==
           rg_multi_model_conditioned_class_count(b));

    rg_multi_model_free(a);
    rg_multi_model_free(b);
    rg_corpus_free(forward);
    rg_corpus_free(reversed);
}

int main(void) {
    rg_context *ctx = 0;
    assert(rg_context_new_builtin(&ctx) == RG_OK);
    test_grimm(ctx);
    test_rhotacism(ctx);
    test_row_order_does_not_change_the_model(ctx);
    rg_context_free(ctx);
    printf("sound-law tests passed\n");
    return 0;
}
