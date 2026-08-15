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

/* Every constraint slot of a context, so a test can ask "is this conditioned on
 * nasality anywhere" without caring which slot carried it. */
static int context_names(const rg_context_spec *c, const char *feature) {
    size_t i;
    const rg_feature_constraint *slots[] = {
        c->preceding, c->following, c->somewhere_preceding, c->somewhere_following,
        c->same_syllable, c->next_syllable, c->previous_syllable,
        c->self_stress, c->preceding_stress, c->following_stress
    };
    const size_t counts[] = {
        c->preceding_count, c->following_count, c->somewhere_preceding_count,
        c->somewhere_following_count, c->same_syllable_count, c->next_syllable_count,
        c->previous_syllable_count, c->self_stress_count, c->preceding_stress_count,
        c->following_stress_count
    };
    for (i = 0; i < sizeof(slots) / sizeof(slots[0]); i++) {
        size_t j;
        for (j = 0; j < counts[i]; j++) {
            if (strcmp(slots[i][j].feature, feature) == 0) {
                return 1;
            }
        }
    }
    for (i = 0; i < c->preceding_at_distance_count; i++) {
        if (strcmp(c->preceding_at_distance[i].constraint.feature, feature) == 0) {
            return 1;
        }
    }
    for (i = 0; i < c->following_at_distance_count; i++) {
        if (strcmp(c->following_at_distance[i].constraint.feature, feature) == 0) {
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
            if (context_names(&row->contexts[j], feature)) {
                seen_feature = 1;
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

/* Western Romance lenition: voiceless stops voice between vowels. The corpus
 * carries the same stops after a consonant, where they do not lenite. */
static void test_lenition(rg_context *ctx) {
    rg_corpus *corpus = load("lenition");
    rg_multi_model *model = train(ctx, corpus);

    assert(has_correspondence(model, "p", "b"));
    assert(has_correspondence(model, "t", "d"));
    assert(has_conditioned(model, "p", "b", "vowel"));
    assert(has_conditioned(model, "t", "d", "vowel"));

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* Grassmann's Law: of two aspirates in a word the first loses its aspiration.
 *
 * This one documents a limit rather than a success, and is kept because a
 * limit nobody can point at is a limit nobody fixes. The correspondence is
 * found -- pie tʰ answers to greek t as well as to greek tʰ -- but the
 * environment is not: what conditions it is an aspirate later in the word,
 * which is neither adjacent nor at a fixed distance, and the only predicate
 * that can express it is an existential one. Two things stand in the way, both
 * recorded in c_conversion_roadmap.md:
 *
 *   - Conditioning is discovered from the alphabetically first lect of a pair,
 *     and a change is only visible from the side that has the split. Here
 *     "greek" sorts before "pie", and every Greek segment has exactly one PIE
 *     source, so there is nothing on that side to split.
 *   - Existential predicates are searched only at the top of their own stage,
 *     never as a refinement of a positional split, so "word-initial *and* an
 *     aspirate somewhere after" cannot be reached.
 *
 * The test asserts what is true today. When either limit is lifted it should
 * be tightened to assert the environment. */
static void test_grassmann_finds_the_correspondence(rg_context *ctx) {
    rg_corpus *corpus = load("grassmann");
    rg_multi_model *model = train(ctx, corpus);

    assert(has_correspondence(model, "t\xca\xb0", "t"));
    assert(has_correspondence(model, "k\xca\xb0", "k"));
    assert(has_correspondence(model, "p\xca\xb0", "p"));

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* Verner's Law: Proto-Germanic voiceless fricatives voice unless the accent
 * fell on the immediately preceding syllable. The canonical stress-conditioned
 * change, and until 2026-08-15 not expressible at all -- the model has carried
 * a stress field and stress split candidates since the port, and no loader ever
 * filled the field, so the whole apparatus was reachable only from C.
 *
 * Each stem appears twice, in the two accent placements, so the fricative is
 * the only thing that can vary and the accent is the only thing that can
 * explain it. */
static void test_verner(rg_context *ctx) {
    rg_corpus *corpus = load("verner");
    rg_multi_model *model = train(ctx, corpus);

    assert(has_correspondence(model, "s", "z"));
    assert(has_correspondence(model, "f", "b"));
    assert(has_conditioned(model, "s", "z", "stress"));
    assert(has_conditioned(model, "\xce\xb8", "\xce\xb8", "stress"));

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* The graded ladder: one change, seven kinds of conditioning, so a failure
 * says which *kind* the search cannot reach rather than only that something is
 * wrong. Asserts what is true today; a rung that starts passing more should
 * have its assertion tightened rather than left loose. */
static void test_conditioning_ladder(rg_context *ctx) {
    struct { const char *name; const char *feature; int reachable; } rungs[] = {
        /* Unconditioned: the change must NOT acquire an environment. */
        {"graded_0_unconditioned", 0, 0},
        {"graded_1_adjacent", "front", 1},
        /* Position is found, though by way of a correlate rather than the
         * position predicate itself; the test asks only that it is conditioned
         * at all, because which of two equally good predicates wins is a
         * tie-break. */
        {"graded_2_position", 0, 1},
        {"graded_3_stress", "stress", 1},
        /* Two predicates at once. Only one of them is ever committed: the
         * search is greedy and refinement does not reach the second here. The
         * change is conditioned, but under-described. */
        {"graded_4_conjunction", "front", 1},
        {"graded_5_distance_two", "nasal", 1},
        {"graded_6_existential", "nasal", 1}
    };
    size_t i;
    for (i = 0; i < sizeof(rungs) / sizeof(rungs[0]); i++) {
        rg_corpus *corpus = load(rungs[i].name);
        rg_multi_model *model = train(ctx, corpus);
        int conditioned = 0;
        size_t j;
        assert(has_correspondence(model, "p", "f"));
        for (j = 0; j < rg_multi_model_conditioned_class_count(model); j++) {
            const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, j);
            size_t k;
            int seen_p = 0;
            int seen_f = 0;
            for (k = 0; k < row->segment_count; k++) {
                if (strcmp(row->graphemes[k], "p") == 0) {
                    seen_p = 1;
                }
                if (strcmp(row->graphemes[k], "f") == 0) {
                    seen_f = 1;
                }
            }
            if (seen_p && seen_f) {
                conditioned = 1;
            }
        }
        assert(conditioned == rungs[i].reachable);
        if (rungs[i].feature != 0) {
            assert(has_conditioned(model, "p", "f", rungs[i].feature));
        }
        rg_multi_model_free(model);
        rg_corpus_free(corpus);
    }
}

int main(void) {
    rg_context *ctx = 0;
    assert(rg_context_new_builtin(&ctx) == RG_OK);
    test_grimm(ctx);
    test_rhotacism(ctx);
    test_lenition(ctx);
    test_grassmann_finds_the_correspondence(ctx);
    test_verner(ctx);
    test_conditioning_ladder(ctx);
    test_row_order_does_not_change_the_model(ctx);
    rg_context_free(ctx);
    printf("sound-law tests passed\n");
    return 0;
}
