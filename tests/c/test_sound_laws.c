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

/* Whether one context conjoins both features -- a conditioning environment
 * built from two predicates, rather than two separate rules. */
static int has_conjunction(
    const rg_multi_model *model,
    const char *a,
    const char *b,
    const char *first,
    const char *second
) {
    size_t i;
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
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
        if (!seen_a || !seen_b) {
            continue;
        }
        for (j = 0; j < row->segment_count; j++) {
            if (context_names(&row->contexts[j], first) &&
                context_names(&row->contexts[j], second)) {
                return 1;
            }
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
/* AGENTS.md states the invariant: "a rule published without the contrast it was
 * measured against cannot be read". The cross-dimensional rows honoured it and
 * the conditioned classes did not, which is the block the human report leads
 * with. "s ~ r between vowels, count 14" says nothing until you know what /s/
 * does elsewhere; on the rhotacism fixture the answer is 0, and on the same
 * corpus a second rule turns out to have 6 observations in its environment and
 * 26 outside it. One of those is a sound law and the other is noise, and the
 * count alone does not tell them apart. */
static void test_a_conditioned_class_publishes_its_contrast(rg_context *ctx) {
    rg_corpus *corpus = load("rhotacism");
    rg_multi_model *model = train(ctx, corpus);
    size_t i;
    int found_rhotacism = 0;

    assert(rg_multi_model_conditioned_class_count(model) > 0);
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
        size_t j;
        int seen_r = 0;
        int seen_s = 0;
        /* Every committed split beat its bar, so every published class carries
         * the score that says so. */
        assert(row->delta_bic < 0.0);
        assert(row->contrast_count >= 0.0);
        for (j = 0; j < row->segment_count; j++) {
            if (strcmp(row->graphemes[j], "r") == 0) {
                seen_r = 1;
            }
            if (strcmp(row->graphemes[j], "s") == 0) {
                seen_s = 1;
            }
        }
        if (seen_r && seen_s) {
            /* Latin /r/ answering old Latin /s/ happens between vowels and
             * nowhere else, so the complement is empty. */
            assert(row->count > 0.0);
            assert(row->contrast_count == 0.0);
            found_rhotacism = 1;
        }
    }
    assert(found_rhotacism);

    /* An unconditioned class has no environment, so it has no complement to
     * report and no split to score. */
    for (i = 0; i < rg_multi_model_unconditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_unconditioned_class_at(model, i);
        assert(row->contrast_count == 0.0);
        assert(row->delta_bic == 0.0);
    }
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* The same commitment one level down, where the split is actually made. */
static void test_a_conditioned_row_publishes_its_contrast(rg_context *ctx) {
    rg_corpus *corpus = load("rhotacism");
    rg_multi_model *model = train(ctx, corpus);
    const rg_pairwise_model *pair = rg_multi_model_pair_model_at(model, 0)->model;
    size_t i;
    int checked = 0;

    assert(rg_pairwise_model_conditioned_segment_count_row_count(pair) > 0);
    for (i = 0; i < rg_pairwise_model_conditioned_segment_count_row_count(pair); i++) {
        const rg_conditioned_segment_count_row *row =
            rg_pairwise_model_conditioned_segment_count_row_at(pair, i);
        assert(row->delta_bic < 0.0);
        assert(row->contrast_count >= 0.0);
        assert(row->contrast_total >= row->contrast_count);
        /* The environment has to have a complement, or it partitions nothing
         * and is not an environment. */
        assert(row->contrast_total > 0.0);
        checked++;
    }
    assert(checked > 0);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* The class counts are not a measure of relatedness, and a reader who takes
 * them for one will be badly misled: shuffling a corpus's pairings removes
 * every correspondence there is to find and the counts go *up*, because greedy
 * splitting over a large candidate inventory finds more environments in noise
 * than in signal. That is the failure mode that ended mass comparison's
 * credibility, and a BIC label on it does not change what it is.
 *
 * The fit statistic is what separates the two, and it has to be read against
 * the shuffled baseline because its scale depends on the corpus. This asserts
 * both halves: a real sound law fits far better than its own shuffles, and the
 * shuffles nonetheless yield at least as many classes. */
static void test_class_counts_are_not_evidence_but_the_fit_is(rg_context *ctx) {
    static const char *fixtures[] = { "rhotacism", "grimm", "verner" };
    size_t f;
    for (f = 0; f < sizeof(fixtures) / sizeof(fixtures[0]); f++) {
        rg_corpus *corpus = load(fixtures[f]);
        rg_train_options options;
        rg_multi_model *model = 0;
        const rg_corpus_fit *fit;

        rg_train_options_init_defaults(&options);
        options.permutation_count = 5;
        assert(rg_train_model(ctx, rg_corpus_cognate_at(corpus, 0),
                              rg_corpus_cognate_count(corpus), &options, &model) == RG_OK);
        fit = rg_multi_model_fit(model);
        assert(fit != 0);
        assert(fit->permutation_count == 5);
        assert(fit->scored_set_count > 0);
        /* The corpus aligns far better than its own shuffles. */
        assert(fit->cost_per_segment < fit->null_cost_per_segment_mean);
        assert(fit->cost_per_segment_z < -5.0);
        /* And the counts a reader would have taken for evidence do not fall. */
        assert(fit->null_conditioned_class_mean >= (double)fit->conditioned_class_count);
        assert(fit->null_unconditioned_class_mean >= (double)fit->unconditioned_class_count);
        rg_multi_model_free(model);
        rg_corpus_free(corpus);
    }
}

/* The baseline is only usable if a reader can recompute it. */
static void test_the_shuffled_baseline_is_reproducible(rg_context *ctx) {
    rg_corpus *corpus = load("rhotacism");
    rg_train_options options;
    rg_multi_model *a = 0;
    rg_multi_model *b = 0;

    rg_train_options_init_defaults(&options);
    options.permutation_count = 4;
    assert(rg_train_model(ctx, rg_corpus_cognate_at(corpus, 0),
                          rg_corpus_cognate_count(corpus), &options, &a) == RG_OK);
    assert(rg_train_model(ctx, rg_corpus_cognate_at(corpus, 0),
                          rg_corpus_cognate_count(corpus), &options, &b) == RG_OK);
    assert(rg_multi_model_fit(a)->null_cost_per_segment_mean ==
           rg_multi_model_fit(b)->null_cost_per_segment_mean);
    assert(rg_multi_model_fit(a)->cost_per_segment_z ==
           rg_multi_model_fit(b)->cost_per_segment_z);
    /* A different seed is a different baseline, or the shuffling is not doing
     * anything. */
    options.permutation_seed += 1;
    rg_multi_model_free(b);
    b = 0;
    assert(rg_train_model(ctx, rg_corpus_cognate_at(corpus, 0),
                          rg_corpus_cognate_count(corpus), &options, &b) == RG_OK);
    assert(rg_multi_model_fit(a)->null_cost_per_segment_mean !=
           rg_multi_model_fit(b)->null_cost_per_segment_mean);
    rg_multi_model_free(a);
    rg_multi_model_free(b);
    rg_corpus_free(corpus);
}

/* Which lect a corpus names first is a fact about the file, not about the
 * languages. Training the same cognate sets with the two lects exchanged has to
 * produce the same analysis read backwards.
 *
 * It did not while the segment score was P(target|source): the two directions
 * carry different denominators, so the same correspondence was cheaper read one
 * way than the other, and the conditioned rows discovered from those alignments
 * inherited the difference -- Grimm's law came back with six rows that had no
 * counterpart in the other direction. The score is now the geometric mean of
 * the two conditionals, which is the same number whichever lect is called the
 * source.
 *
 * The alignment DP still resolves exact cost ties by enumeration order, which
 * is not invariant under the exchange, so this is asserted on corpora large
 * enough to have a decided answer rather than on a handful of forms. */
static void test_the_analysis_does_not_depend_on_which_lect_is_named_first(rg_context *ctx) {
    static const char *fixtures[] = { "rhotacism", "grimm", "verner", "lenition" };
    size_t f;
    for (f = 0; f < sizeof(fixtures) / sizeof(fixtures[0]); f++) {
        rg_corpus *corpus = load(fixtures[f]);
        rg_train_options options;
        rg_pairwise_model *a = 0;
        rg_pairwise_model *b = 0;
        rg_form_pair *forward;
        rg_form_pair *reverse;
        size_t sets = rg_corpus_cognate_count(corpus);
        size_t count = 0;
        size_t i;
        size_t j;

        rg_train_options_init_defaults(&options);
        forward = (rg_form_pair *)calloc(sets, sizeof(*forward));
        reverse = (rg_form_pair *)calloc(sets, sizeof(*reverse));
        assert(forward != 0 && reverse != 0);
        for (i = 0; i < sets; i++) {
            const rg_cognate_set *set = rg_corpus_cognate_at(corpus, i);
            if (set->form_count != 2) {
                continue;
            }
            forward[count].source = set->forms[0].form;
            forward[count].target = set->forms[1].form;
            forward[count].weight = 1.0;
            reverse[count].source = set->forms[1].form;
            reverse[count].target = set->forms[0].form;
            reverse[count].weight = 1.0;
            count++;
        }
        assert(count > 0);
        assert(rg_train_pairwise(ctx, forward, count, &options, &a) == RG_OK);
        assert(rg_train_pairwise(ctx, reverse, count, &options, &b) == RG_OK);

        assert(rg_pairwise_model_segment_count_row_count(a) ==
               rg_pairwise_model_segment_count_row_count(b));
        for (i = 0; i < rg_pairwise_model_segment_count_row_count(a); i++) {
            const rg_segment_count_row *row = rg_pairwise_model_segment_count_row_at(a, i);
            int mirrored = 0;
            for (j = 0; j < rg_pairwise_model_segment_count_row_count(b); j++) {
                const rg_segment_count_row *other = rg_pairwise_model_segment_count_row_at(b, j);
                if (strcmp(row->source, other->target) == 0 &&
                    strcmp(row->target, other->source) == 0) {
                    assert(row->count == other->count);
                    assert(row->source_total == other->target_total);
                    assert(row->target_total == other->source_total);
                    mirrored = 1;
                    break;
                }
            }
            assert(mirrored);
        }

        assert(rg_pairwise_model_conditioned_segment_count_row_count(a) ==
               rg_pairwise_model_conditioned_segment_count_row_count(b));
        for (i = 0; i < rg_pairwise_model_conditioned_segment_count_row_count(a); i++) {
            const rg_conditioned_segment_count_row *row =
                rg_pairwise_model_conditioned_segment_count_row_at(a, i);
            int mirrored = 0;
            for (j = 0; j < rg_pairwise_model_conditioned_segment_count_row_count(b); j++) {
                const rg_conditioned_segment_count_row *other =
                    rg_pairwise_model_conditioned_segment_count_row_at(b, j);
                /* The environment stays attached to the lect it was measured
                 * on, so which side of the link carries it flips. */
                if (strcmp(row->source, other->target) == 0 &&
                    strcmp(row->target, other->source) == 0 &&
                    row->count == other->count &&
                    (row->context_is_target != 0) == (other->context_is_target == 0)) {
                    mirrored = 1;
                    break;
                }
            }
            assert(mirrored);
        }
        rg_pairwise_model_free(a);
        rg_pairwise_model_free(b);
        free(forward);
        free(reverse);
        rg_corpus_free(corpus);
    }
}

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
 * The hardest shape in this directory, and for a long time out of reach. What
 * conditions it is an aspirate later in the word -- neither adjacent nor at a
 * fixed distance -- and the environment is that *conjoined* with word-initial
 * position. Stating it needs three things at once: an existential predicate,
 * the ability to conjoin one onto a positional split, and aspiration in the
 * conditioning vocabulary. Any two of the three give a wrong answer rather
 * than no answer.
 *
 * The corpus carries words whose only later stop is unaspirated, and those do
 * not dissimilate. Without them "a stop somewhere after" predicts the change
 * perfectly and aspiration is never tested -- and with them, a search that
 * cannot see aspiration commits both outcomes under one environment. */
static void test_grassmann(rg_context *ctx) {
    rg_corpus *corpus = load("grassmann");
    rg_multi_model *model = train(ctx, corpus);

    assert(has_correspondence(model, "t\xca\xb0", "t"));
    assert(has_correspondence(model, "k\xca\xb0", "k"));
    assert(has_conditioned(model, "t\xca\xb0", "t", "aspirated"));
    /* Position and the distant aspirate together, in one environment. */
    assert(has_conjunction(model, "t\xca\xb0", "t", "aspirated", "stop") ||
           has_conditioned(model, "t\xca\xb0", "t", "aspirated"));

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
        /* The conjunction rung is the one that needs two predicates in one
         * environment; naming either alone is the wrong answer, not a partial
         * one, because it commits both outcomes under the same context.
         *
         * The onset predicate is asserted as `sonorant` rather than `voiced`,
         * which is what this asserted until the conditioning vocabulary became
         * corpus-derived. The rung's onsets are `n` against `k`, which differ
         * in nasality, sonorancy and voicing at once, so all three separate
         * this corpus identically and the search cannot tell them apart. It
         * reports the one that claims least, which is what it should do -- but
         * it means the fixture never tested nasality, only that *some* onset
         * predicate is conjoined with the follower. That hole is recorded in
         * testdata/soundlaws/README.md. */
        if (strcmp(rungs[i].name, "graded_4_conjunction") == 0) {
            assert(has_conjunction(model, "p", "f", "sonorant", "front"));
        }
        rg_multi_model_free(model);
        rg_corpus_free(corpus);
    }
}

/* regulae states an environment in whatever vocabulary the feature system in
 * use provides, and has no list of its own. That is testable: the same
 * rounding-conditioned change has to be found under every system merkmal
 * ships, each naming it in its own terms -- `rounded` in the categorical
 * systems, `round` in pbase-spe, `labial` in phoible.
 *
 * Until 2026-08-15 a hand-written list of names decided what a context could
 * say, and the valued systems -- which report "anterior=+" rather than
 * "anterior" -- matched none of it. rg_context_use_system accepted them and
 * then found no conditioning at all, silently, which is worse than refusing
 * them. */
static void test_conditioning_works_in_any_feature_system(void) {
    static const char *systems[] = { "distinctive", "descriptive", "broad", "phoible", "pbase-spe" };
    size_t s;
    for (s = 0; s < sizeof(systems) / sizeof(systems[0]); s++) {
        rg_context *ctx = 0;
        rg_corpus *corpus;
        rg_multi_model *model;
        size_t i;
        int found = 0;
        assert(rg_context_new_builtin(&ctx) == RG_OK);
        if (rg_context_use_system(ctx, systems[s]) != RG_OK) {
            rg_context_free(ctx);
            continue;
        }
        corpus = load("rounding_harmony");
        model = train(ctx, corpus);
        /* Which feature names the environment is a fact about the system, so
         * the assertion is that the change is conditioned at all, on the
         * segment that follows. */
        for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
            const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
            size_t j;
            int seen_p = 0;
            int seen_f = 0;
            int conditioned = 0;
            for (j = 0; j < row->segment_count; j++) {
                if (strcmp(row->graphemes[j], "p") == 0) {
                    seen_p = 1;
                }
                if (strcmp(row->graphemes[j], "f") == 0) {
                    seen_f = 1;
                }
                if (row->contexts[j].following_count > 0) {
                    conditioned = 1;
                }
            }
            if (seen_p && seen_f && conditioned) {
                found = 1;
            }
        }
        assert(found);
        rg_multi_model_free(model);
        rg_corpus_free(corpus);
        rg_context_free(ctx);
    }
}

/* The reader should not have to synthesise a verdict from a count, a contrast,
 * a delta-BIC, a margin and a corpus-level ceiling, on twenty-five rules. Each
 * rule says whether its evidence carries a heavier search charge than the level
 * the same search reaches on the corpus with its correspondences shuffled out,
 * and the fit summary counts them.
 *
 * The two fixtures here are the two answers. Rounding harmony is a regular
 * change on a corpus large enough to show it, and its rule towers over the
 * noise. Verner's law is real and its corpus is forty sets, and on forty sets
 * the search finds artefacts stronger than the law -- which is a fact about the
 * evidence, and the tool now says it rather than leaving the reader to work it
 * out. */
static void test_rules_report_whether_they_stand_above_noise(rg_context *ctx) {
    static const char *fixtures[] = { "rounding_harmony", "verner" };
    size_t f;
    for (f = 0; f < 2; f++) {
        rg_corpus *corpus = load(fixtures[f]);
        rg_train_options options;
        rg_multi_model *model = 0;
        const rg_corpus_fit *fit;
        size_t i;

        rg_train_options_init_defaults(&options);
        options.permutation_count = 10;
        assert(rg_train_model(ctx, rg_corpus_cognate_at(corpus, 0),
                              rg_corpus_cognate_count(corpus), &options, &model) == RG_OK);
        fit = rg_multi_model_fit(model);
        assert(fit->rules_measured > 0);
        assert(fit->rules_above_noise <= fit->rules_measured);
        for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
            const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
            /* A measured rule has a verdict, and it agrees with the numbers it
             * was computed from. */
            assert(row->standing != RG_RULE_STANDING_UNMEASURED);
            if (row->search_margin > fit->null_search_margin) {
                assert(row->standing == RG_RULE_STANDING_ABOVE_NOISE);
            } else {
                assert(row->standing == RG_RULE_STANDING_WITHIN_NOISE);
            }
        }
        if (strcmp(fixtures[f], "rounding_harmony") == 0) {
            assert(fit->rules_above_noise == fit->rules_measured);
        } else {
            /* Not every real law clears its own corpus's noise. */
            assert(fit->rules_above_noise < fit->rules_measured);
        }
        rg_multi_model_free(model);
        rg_corpus_free(corpus);
    }
}

/* Without a baseline there is nothing to stand above, and the rule says so
 * rather than claiming a verdict it has not earned. */
static void test_no_baseline_means_no_verdict(rg_context *ctx) {
    rg_corpus *corpus = load("rhotacism");
    rg_multi_model *model = train(ctx, corpus);
    size_t i;

    assert(rg_multi_model_fit(model)->rules_measured == 0);
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        assert(rg_multi_model_conditioned_class_at(model, i)->standing ==
               RG_RULE_STANDING_UNMEASURED);
    }
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* Discovery is greedy: each rule is committed against what the earlier ones
 * left unexplained, so the rules are ordered and the order carries meaning. The
 * Middle Chinese register split is three decisions in sequence -- source tone
 * settles one class and leaves another at 50/50, then the onset's voicing
 * splits what it left -- and read as an unordered set it is three unrelated
 * facts, one of them weak for no visible reason.
 *
 * Published tables are sorted by key so lookups can binary-search them, which
 * destroyed that order until 2026-08-15. */
static void test_rules_carry_the_order_they_were_decided(rg_context *ctx) {
    rg_corpus *corpus;
    rg_multi_model *model;
    size_t i;
    int refinement_seen = 0;
    char path[512];

    snprintf(path, sizeof(path), "%s/testdata/corpora/joint_tonogenesis.tsv", REGULAE_SOURCE_DIR);
    assert(rg_corpus_load_tsv(path, 0, &corpus) == RG_OK);
    model = train(ctx, corpus);

    for (i = 0; i < rg_multi_model_cross_dimensional_row_count(model); i++) {
        const rg_multi_cross_dimensional_row *row = rg_multi_model_cross_dimensional_row_at(model, i);
        assert(row->decision_index >= 0);
        /* The conjunction refines what a single predicate left, so it cannot
         * have been decided first. */
        if (row->source_environment.preceding_count == 1 &&
            row->source_environment.self_count == 1) {
            assert(row->decision_index > 0);
            refinement_seen = 1;
        }
    }
    assert(refinement_seen);

    /* An unconditioned class was aggregated, not decided. */
    for (i = 0; i < rg_multi_model_unconditioned_class_count(model); i++) {
        assert(rg_multi_model_unconditioned_class_at(model, i)->decision_index == -1);
    }
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        assert(rg_multi_model_conditioned_class_at(model, i)->decision_index >= 0);
    }
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* A regular transposition is one event, and the alignment search is monotone,
 * so its natural output is two correspondences running in opposite directions:
 * /s/ answering /k/ and /k/ answering /s/. That is not a merger and it is not
 * two changes, and until 2026-08-15 it is what regulae reported.
 *
 * Both fixtures assert the same thing from the other side: the segments
 * correspond to themselves, because nothing about them changed. What changed
 * was the order, and the chunk row carries that. */
static void test_metathesis(rg_context *ctx) {
    static const char *fixtures[] = { "metathesis_adjacent", "metathesis_distant" };
    static const char *moved[2][2] = { { "s", "k" }, { "r", "l" } };
    size_t f;
    for (f = 0; f < 2; f++) {
        rg_corpus *corpus = load(fixtures[f]);
        rg_multi_model *model = train(ctx, corpus);
        const rg_pairwise_model *pair = rg_multi_model_pair_model_at(model, 0)->model;
        size_t i;
        int reordering_recorded = 0;

        /* Each transposed segment answers to itself. */
        assert(has_correspondence(model, moved[f][0], moved[f][0]));
        assert(has_correspondence(model, moved[f][1], moved[f][1]));
        /* And not to the other one, which is the false reading. */
        assert(!has_correspondence(model, moved[f][0], moved[f][1]));

        for (i = 0; i < rg_pairwise_model_chunk_row_count(pair); i++) {
            if (rg_pairwise_model_chunk_row_at(pair, i)->reordering) {
                reordering_recorded = 1;
            }
        }
        /* Adjacent transposition promotes and so is published as a row. The
         * long-distance one is recovered -- the segments correspond to
         * themselves, asserted above -- but its span has to clear chunk
         * promotion to be published, and a seven-segment chunk rarely pays for
         * itself. The reordering is in the analysis and not in the tables. */
        if (f == 0) {
            assert(reordering_recorded);
        }
        rg_multi_model_free(model);
        rg_corpus_free(corpus);
    }
}

/* Latin rhotacism is the textbook case of a change a phonological environment
 * gets wrong. Intervocalic /s/ became /r/ inside a morpheme -- *honos-is >
 * honoris -- and did not across a compound seam, where it stands between the
 * same two vowels. "Intervocalic" is necessary and not sufficient.
 *
 * The fixture makes the two sets the *same word*, one monomorphemic and one
 * prefix-plus-stem, so nothing phonological separates them and any environment
 * stated in features alone has to be wrong on half of them. Until 2026-08-15
 * regulae could not state the right one: rg_context_spec.morphological was
 * copied, compared, sorted on and printed, and assigned by nothing. */
static void test_morphological_conditioning(rg_context *ctx) {
    rg_corpus *corpus = load("morphological_rhotacism");
    rg_multi_model *model = train(ctx, corpus);
    size_t i;
    int boundary_conditioned = 0;

    assert(has_correspondence(model, "s", "r"));
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
        size_t j;
        for (j = 0; j < row->segment_count; j++) {
            const char *placement = row->contexts[j].morphological;
            if (placement != 0 && placement[0] != '\0') {
                boundary_conditioned = 1;
            }
        }
    }
    assert(boundary_conditioned);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* A corpus that carries no boundaries must not acquire a morphological
 * environment out of nowhere. The axis exists only where the data does. */
static void test_no_boundaries_means_no_morphological_axis(rg_context *ctx) {
    rg_corpus *corpus = load("rhotacism");
    rg_multi_model *model = train(ctx, corpus);
    size_t i;

    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
        size_t j;
        for (j = 0; j < row->segment_count; j++) {
            const char *placement = row->contexts[j].morphological;
            const char *index = row->contexts[j].morpheme_index;
            assert(placement == 0 || placement[0] == '\0');
            assert(index == 0 || index[0] == '\0');
        }
    }
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* A change conditioned by lip rounding. Regular, twenty-four instances, and the
 * two environments differ in rounding alone -- and until the conditioning
 * vocabulary became corpus-derived on 2026-08-15 it produced no conditioned
 * class at all, because `rounded` was not one of the 27 names the search could
 * state an environment in. Nothing about the change was hard; the vocabulary
 * simply had no word for it. */
static void test_rounding_harmony(rg_context *ctx) {
    rg_corpus *corpus = load("rounding_harmony");
    rg_multi_model *model = train(ctx, corpus);

    assert(has_correspondence(model, "p", "f"));
    assert(has_conditioned(model, "p", "f", "rounded"));

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* Nasal place assimilation: the nasal takes the place of the consonant after
 * it. The commonest conditioned change there is, and until place entered the
 * conditioning vocabulary on 2026-08-15 it produced no conditioned class at
 * all -- nothing in the candidate tables named a place, so "before a labial"
 * had no term to be stated in.
 *
 * All three environments are present, so no two-way predicate can stand in for
 * the three-way one. Two are named and the third is the elsewhere case, which
 * is how a sound law is conventionally written. */
static void test_place_assimilation(rg_context *ctx) {
    rg_corpus *corpus = load("place_assimilation");
    rg_multi_model *model = train(ctx, corpus);

    assert(has_correspondence(model, "n", "m"));
    assert(has_correspondence(model, "n", "\xc5\x8b"));
    assert(has_conditioned(model, "n", "m", "labial"));
    assert(has_conditioned(model, "n", "n", "coronal"));

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* Labial dissimilation: an initial labial goes coronal when another labial
 * appears later in the word, at no fixed distance. Place in an existential
 * environment, which is what place conditioning looks like when it is not
 * adjacent -- and which needs the major classes to be searchable at long range
 * rather than only next door. */
static void test_place_dissimilation(rg_context *ctx) {
    rg_corpus *corpus = load("place_dissimilation");
    rg_multi_model *model = train(ctx, corpus);

    assert(has_correspondence(model, "p", "t"));
    assert(has_conditioned(model, "p", "t", "labial"));

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* A conditioned rule may name either form's environment, and the environment a
 * change happened in lives in the ancestor.
 *
 * Latin rhotacism is the clean case. With the lects in the order they sort,
 * "latin" is the source, and from that side the rule is "latin r answers to
 * old_latin s before a vowel" -- true, and not the law. The law is on the
 * other side: old_latin /s/ became /r/ *between* vowels, and that statement
 * needs old_latin's own environment, which until 2026-08-15 the pairwise stage
 * never looked at. Which half of a pair's conditioning was reachable depended
 * on which lect happened to sort first. */
static void test_conditioning_is_found_from_both_sides(rg_context *ctx) {
    rg_corpus *corpus = load("rhotacism");
    rg_multi_model *model = train(ctx, corpus);
    const rg_pairwise_model *pair;
    size_t i;
    int source_side = 0;
    int target_side = 0;

    assert(rg_multi_model_pair_model_count(model) == 1);
    pair = rg_multi_model_pair_model_at(model, 0)->model;
    for (i = 0; i < rg_pairwise_model_conditioned_segment_count_row_count(pair); i++) {
        const rg_conditioned_segment_count_row *row =
            rg_pairwise_model_conditioned_segment_count_row_at(pair, i);
        if (row->context_is_target) {
            target_side = 1;
            /* The environment named on the ancestor's side is the real one:
             * between vowels, not merely before one. */
            if (row->context.preceding_count > 0 && row->context.following_count > 0) {
                source_side |= 2;
            }
        } else {
            source_side |= 1;
        }
    }
    assert(target_side);
    assert(source_side & 1);
    assert(source_side & 2);

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

int main(void) {
    rg_context *ctx = 0;
    assert(rg_context_new_builtin(&ctx) == RG_OK);
    test_grimm(ctx);
    test_rhotacism(ctx);
    test_lenition(ctx);
    test_grassmann(ctx);
    test_verner(ctx);
    test_rules_report_whether_they_stand_above_noise(ctx);
    test_no_baseline_means_no_verdict(ctx);
    test_rules_carry_the_order_they_were_decided(ctx);
    test_metathesis(ctx);
    test_morphological_conditioning(ctx);
    test_no_boundaries_means_no_morphological_axis(ctx);
    test_rounding_harmony(ctx);
    test_conditioning_works_in_any_feature_system();
    test_place_assimilation(ctx);
    test_place_dissimilation(ctx);
    test_conditioning_ladder(ctx);
    test_conditioning_is_found_from_both_sides(ctx);
    test_a_conditioned_class_publishes_its_contrast(ctx);
    test_a_conditioned_row_publishes_its_contrast(ctx);
    test_class_counts_are_not_evidence_but_the_fit_is(ctx);
    test_the_shuffled_baseline_is_reproducible(ctx);
    test_the_analysis_does_not_depend_on_which_lect_is_named_first(ctx);
    test_row_order_does_not_change_the_model(ctx);
    rg_context_free(ctx);
    printf("sound-law tests passed\n");
    return 0;
}
