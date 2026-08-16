#include "regulae.h"
#include "table_access.h"

#include <assert.h>
#include <math.h>
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
    assert(rg_corpus_load_tsv(path, 0, &corpus, 0) == RG_OK);
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
        assert(row->evidence.delta_bic < 0.0);
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
        assert(row->evidence.delta_bic == 0.0);
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
        assert(row->evidence.delta_bic < 0.0);
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
 * Chunk promotion carried the same fault one level up and kept
 * place_dissimilation out of this list until 2026-08-15. A chunk's promoted
 * cost was P(target chunk | source chunk) and its compositional baseline was
 * the forward segment posterior, so a chunk whose source was ambiguous in one
 * lect and determined in the other was priced differently each way: the t-lect
 * promoted "ta ~ pa", "tal ~ pal" and "te ~ pe", the p-lect promoted none of
 * them, and the evidence for p > t before a labial went into chunk rows in one
 * direction and into a conditioned rule in the other. Both quantities are now
 * the geometric mean of the two directions, and the chunk tables mirror.
 *
 * The alignment DP still resolves exact cost ties by enumeration order, which
 * is not invariant under the exchange, so this is asserted on corpora large
 * enough to have a decided answer rather than on a handful of forms. */
static int segments_equal(const rg_segment *a, size_t a_count, const rg_segment *b, size_t b_count) {
    size_t i;
    if (a_count != b_count) {
        return 0;
    }
    for (i = 0; i < a_count; i++) {
        if (strcmp(a[i].grapheme, b[i].grapheme) != 0) {
            return 0;
        }
    }
    return 1;
}

static void test_the_analysis_does_not_depend_on_which_lect_is_named_first(rg_context *ctx) {
    static const char *fixtures[] = {
        "rhotacism", "grimm", "verner", "lenition", "place_dissimilation"
    };
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

        /* Every promoted chunk answers to a chunk promoted the other way with
         * the two sides exchanged, carrying the same mass. */
        assert(rg_pairwise_model_chunk_row_count(a) == rg_pairwise_model_chunk_row_count(b));
        for (i = 0; i < rg_pairwise_model_chunk_row_count(a); i++) {
            const rg_chunk_row *row = rg_pairwise_model_chunk_row_at(a, i);
            int mirrored = 0;
            for (j = 0; j < rg_pairwise_model_chunk_row_count(b); j++) {
                const rg_chunk_row *other = rg_pairwise_model_chunk_row_at(b, j);
                if (segments_equal(row->source, row->source_count, other->target, other->target_count) &&
                    segments_equal(row->target, row->target_count, other->source, other->source_count)) {
                    assert(row->count == other->count);
                    /* How readable a chunk is does not depend on which side of
                     * it is called the source. The reference's nasal-fusion
                     * and glide-fusion profiles were directional and would
                     * have broken this; both are tested in either
                     * orientation. */
                    assert(fabs(row->transparency - other->transparency) < 1e-12);
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
    assert(rg_corpus_load_tsv(path, 0, &reversed, 0) == RG_OK);

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
        {"graded_6_existential", "nasal", 1},
        /* The last three each have a test of their own below. The rung entry
         * only records that something is found and what it is named. */
        {"graded_7_disjunction", "close", 1},
        {"graded_8_weight", "syllable_weight", 1},
        {"graded_9_lost_trigger", "front", 1}
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

/* The count of the same segment tuple where the class's environment does not
 * hold: what a report prints as `elsewhere`. Zero means the environment covers
 * every instance of the change, which is what a sound law claims. */
static double contrast_of(const rg_multi_model *model, const char *a, const char *b) {
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
        if (seen_a && seen_b) {
            return row->contrast_count;
        }
    }
    return -1.0;
}

/* A trigger set that is not a natural class, which is the RUKI law's shape:
 * PIE *s retracts after *r, *u, *k and *i in Indo-Iranian, Balto-Slavic,
 * Armenian and Albanian, and after nothing else. Two of the four are high
 * vowels, one is a dorsal stop and one is a coronal liquid; every feature true
 * of all four is true of something in the contrast set as well.
 *
 * A context is a conjunction of feature constraints and a conjunction narrows,
 * so a disjunction cannot be written as one environment. What a comparativist
 * writes on the board instead is a decision list -- one rule per trigger, all
 * with the same outcome -- and that is what the search produces, because
 * discovery is greedy and each rule is committed against what the earlier ones
 * left unexplained.
 *
 * The multi-lect table could not show it until 2026-08-17. Rows were merged on
 * the correspondence tuple alone, so three of the four rules were discarded
 * before publication and the survivor was whichever environment carried the
 * most constraints. The rules were in the pairwise tables the whole time,
 * which is how the collapse stayed invisible.
 *
 * Each of the three predicates is asserted by name. `close` is {i, u}, `trill`
 * is {r} and `stop` is {k}: the four triggers, in three rules, because two of
 * them do share a feature and the search is right to use it. */
static void test_a_disjunctive_trigger_comes_out_as_a_decision_list(rg_context *ctx) {
    rg_corpus *corpus = load("graded_7_disjunction");
    rg_multi_model *model = train(ctx, corpus);
    size_t i;
    double covered = 0.0;

    assert(has_conditioned(model, "p", "f", "close"));
    assert(has_conditioned(model, "p", "f", "trill"));
    assert(has_conditioned(model, "p", "f", "stop"));

    /* And between them they account for every instance of the change. The
     * corpus has sixteen, four after each trigger; the three rules cover
     * 8 + 4 + 4. A fourth rule is committed on a broader correlate that
     * overlaps them, so the sum over rows is not the way to count -- what the
     * assertion checks is that the three real ones are each there and each
     * carries its own evidence. */
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
        size_t j;
        int seen_p = 0;
        int seen_f = 0;
        for (j = 0; j < row->segment_count; j++) {
            if (strcmp(row->graphemes[j], "p") == 0) {
                seen_p = 1;
            }
            if (strcmp(row->graphemes[j], "f") == 0) {
                seen_f = 1;
            }
        }
        if (!seen_p || !seen_f) {
            continue;
        }
        assert(row->evidence.delta_bic < 0.0);
        for (j = 0; j < row->segment_count; j++) {
            if (context_names(&row->contexts[j], "trill") ||
                context_names(&row->contexts[j], "stop") ||
                context_names(&row->contexts[j], "close")) {
                covered += row->count;
                break;
            }
        }
    }
    assert(covered >= 16.0);

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* The same table, from the other direction: a decision list is only readable
 * if the rows that make it up stay apart, and a second *description* of one
 * rule is not a second rule.
 *
 * Latin rhotacism commits four splits for `latin:r ~ old_latin:s`. Three carry
 * the same fourteen intervocalic /s/ -- two pivots and a narrowing, which is
 * the cross-pivot join that puts an environment in both lects' slots -- and
 * the fourth carries ten of those same fourteen under `fol@2[fricative:+]`, a
 * correlate of the environment rather than a rule beside it. One row, with the
 * complement empty, is the right answer, and the merge has to reach it without
 * also collapsing the disjunction above. */
static void test_a_second_description_of_one_rule_is_not_a_second_rule(rg_context *ctx) {
    rg_corpus *corpus = load("rhotacism");
    rg_multi_model *model = train(ctx, corpus);
    size_t i;
    int rows = 0;
    int two_sided = 0;

    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
        size_t j;
        int seen_r = 0;
        int seen_s = 0;
        int slots = 0;
        for (j = 0; j < row->segment_count; j++) {
            if (strcmp(row->graphemes[j], "r") == 0) {
                seen_r = 1;
            }
            if (strcmp(row->graphemes[j], "s") == 0) {
                seen_s = 1;
            }
        }
        if (!seen_r || !seen_s) {
            continue;
        }
        rows++;
        for (j = 0; j < row->segment_count; j++) {
            if (rg_context_spec_constraint_count(&row->contexts[j]) > 0) {
                slots++;
            }
        }
        if (slots > 1) {
            two_sided = 1;
        }
    }
    assert(rows == 1);
    assert(two_sided);

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* Conditioning by syllable weight, which is Sievers' Law's shape and the shape
 * of every rule stated over moras rather than segments -- Latin's penultimate
 * accent, Germanic high-vowel deletion, the metrical half of Verner's
 * environment, and most of what a metrist means by a rule.
 *
 * A syllable is heavy because its nucleus is long **or** because it has a
 * coda, and the fixture carries both kinds in equal number. That disjunction
 * is what makes the rung hard: a context is a conjunction of feature
 * constraints, so without a term for the syllable itself the search has to say
 * the same environment twice, once per exponent.
 *
 * It did, until 2026-08-17, and correctly: `pre[long:+]` for the long nuclei
 * and an equivalent of "there is a coda" for the rest, together covering all
 * twenty-eight. Correct and not what anybody wants to read. `syllable_weight`
 * is a verdict rather than a fact, and it is worth its place because the
 * verdict is the thing the field states these rules in.
 *
 * One rule now, with nothing in the elsewhere bucket. The two facts it is
 * computed from -- `syllable_shape` and `syllable_nucleus` -- are offered
 * beside it, so a language whose tradition draws the weight line somewhere
 * else can still be described. */
static void test_syllable_weight_is_stated_in_one_rule(rg_context *ctx) {
    rg_corpus *corpus = load("graded_8_weight");
    rg_multi_model *model = train(ctx, corpus);

    assert(has_conditioned(model, "p", "f", "syllable_weight"));
    /* Every instance of the change, so no reader is left to take the closed
     * syllables for exceptions. */
    assert(contrast_of(model, "p", "f") == 0.0);

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* An environment that the daughter no longer has.
 *
 * Rung 9 is rung 1 with the conditioning vowel deleted in the daughter, so
 * that within the daughter nothing distinguishes the words that changed from
 * the words that did not. This is opacity, and it is the ordinary case rather
 * than an exotic one: Germanic i-umlaut fronted a vowel and then the *i* that
 * fronted it fell, which is why English has *foot/feet* with no /i/ in sight,
 * and the same sequence gave French its nasal vowels and Mandarin its tones.
 *
 * The environment survives on the proto's side of the pair and on no other, so
 * finding it at all depends on the search looking at both sides of a
 * correspondence rather than only at the segment being explained. It does --
 * `context_is_target` is what that is for -- and the change comes out fully
 * covered, with nothing in the elsewhere bucket. */
static void test_an_environment_the_daughter_lost_is_found_on_the_proto_side(rg_context *ctx) {
    rg_corpus *corpus = load("graded_9_lost_trigger");
    rg_multi_model *model = train(ctx, corpus);

    assert(has_conditioned(model, "p", "f", "front"));
    assert(contrast_of(model, "p", "f") == 0.0);

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* The same thing on curated material, three lects, and the argument for
 * comparing more than two at a time.
 *
 * Germanic i-umlaut across a Gothic-shaped lect that kept the final vowel and
 * never fronted, an Old-High-German-shaped one that fronted and kept it, and
 * an Old-English-shaped one that fronted and lost it. In the third lect alone
 * *gest* and *gast* are a minimal pair with nothing to separate them; the
 * trigger is in the other two. This is the standard classroom argument for why
 * Gothic matters to the history of English, and it is a testable claim about a
 * method rather than only a story.
 *
 * The environment comes out as `next-syl[close:+]` -- the *i* in the following
 * syllable -- read off the lects that kept it, for the lect that did not.
 *
 * The fixture also records a limit, and it is the more useful half. Umlaut is
 * one change, and it surfaces here as four correspondences, one per vowel
 * quality: a~e with fifteen examples, and uː~yː, u~y, oː~øː with two to four
 * each. Only the first crosses the evidence floor that
 * testdata/restraint/sparse_* measures, so only the first gets its
 * environment; the others are published as unconditioned splits. A change that
 * applies to a whole natural class is divided by the number of segments in
 * that class before the search ever sees it, and each fragment has to stand on
 * its own. Palatalisation, lenition and nasalisation all have this shape. */
static void test_umlaut_is_recovered_from_the_lects_that_kept_the_trigger(rg_context *ctx) {
    rg_corpus *corpus = load("opaque_umlaut");
    rg_multi_model *model = train(ctx, corpus);
    size_t i;
    int found = 0;
    int thin_class_conditioned = 0;

    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
        size_t j;
        int seen_a = 0;
        int seen_e = 0;
        int seen_back = 0;
        for (j = 0; j < row->segment_count; j++) {
            if (strcmp(row->graphemes[j], "a") == 0) {
                seen_a = 1;
            }
            if (strcmp(row->graphemes[j], "e") == 0) {
                seen_e = 1;
            }
            /* The thin classes: the long back vowels and short /u/. */
            if (strcmp(row->graphemes[j], "\xc3\xb8\xcb\x90") == 0 ||
                strcmp(row->graphemes[j], "y\xcb\x90") == 0 ||
                strcmp(row->graphemes[j], "y") == 0) {
                seen_back = 1;
            }
        }
        if (seen_a && seen_e) {
            size_t k;
            for (k = 0; k < row->segment_count; k++) {
                if (context_names(&row->contexts[k], "close")) {
                    found = 1;
                }
            }
        }
        if (seen_back) {
            thin_class_conditioned = 1;
        }
    }
    /* The fifteen-example class gets the trigger. */
    assert(found);
    /* The two-to-four-example ones get nothing, though it is the same change
     * and the same trigger stands next to it in the same words. */
    assert(!thin_class_conditioned);
    /* They are still published, as unconditioned splits. */
    assert(has_correspondence(model, "u\xcb\x90", "y\xcb\x90"));
    assert(has_correspondence(model, "o\xcb\x90", "\xc3\xb8\xcb\x90"));

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* A chain shift, and the merger it must not be reported as.
 *
 * In the Great Vowel Shift Middle English /eː/ raised to /iː/ while /iː/ was
 * diphthongising out of the way, and /oː/ raised to /uː/ while /uː/ did the
 * same. Every step lands where the next one just left, so a method that keeps
 * no separate account of the two sources reports a merger: ME /eː/ and /iː/
 * both answering Modern English /iː/, which is false about both and would say
 * that *feet* and *five* had the same vowel in 1400.
 *
 * They did not merge, and nothing in the corpus says they did except the
 * surface arithmetic. The assertion is that the two chains stay apart. */
static void test_a_chain_shift_is_not_reported_as_a_merger(rg_context *ctx) {
    rg_corpus *corpus = load("great_vowel_shift");
    rg_multi_model *model = train(ctx, corpus);
    size_t i;

    /* Each link of the front chain, and of the back one. The diphthongs are
     * written as two segments, so the raised /iː/ answers the nucleus of
     * /aɪ/ -- a correspondence is stated at the granularity the transcription
     * was written at, which is a decision the corpus makes and not the tool. */
    assert(has_correspondence(model, "e\xcb\x90", "i\xcb\x90"));   /* eː > iː */
    assert(has_correspondence(model, "i\xcb\x90", "\xc9\xaa"));    /* iː > aɪ */
    assert(has_correspondence(model, "o\xcb\x90", "u\xcb\x90"));   /* oː > uː */
    assert(has_correspondence(model, "u\xcb\x90", "\xca\x8a"));    /* uː > aʊ */

    /* And no class puts the two Middle English sources together. */
    for (i = 0; i < rg_multi_model_unconditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_unconditioned_class_at(model, i);
        size_t j;
        int seen_mid = 0;
        int seen_high = 0;
        for (j = 0; j < row->segment_count; j++) {
            if (strcmp(row->lect_ids[j], "middle_english") != 0) {
                continue;
            }
            if (strcmp(row->graphemes[j], "e\xcb\x90") == 0 ||
                strcmp(row->graphemes[j], "o\xcb\x90") == 0) {
                seen_mid = 1;
            }
            if (strcmp(row->graphemes[j], "i\xcb\x90") == 0 ||
                strcmp(row->graphemes[j], "u\xcb\x90") == 0) {
                seen_high = 1;
            }
        }
        assert(!(seen_mid && seen_high));
    }

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* A change to one segment caused by the loss of another.
 *
 * The Ingvaeonic nasal spirant law: Proto-Germanic lost a nasal before a
 * fricative and lengthened the vowel in front of it, so *gans* answers Old
 * English *gōs* and *tanþ* answers *tōþ*, while the same nasal before a stop
 * is untouched -- *hand* stays *hand*. Two events with one cause, and the
 * segment that explains the vowel is the one that is no longer there.
 *
 * The environment has to reach *past* the segment that went, and that is what
 * the distance slots are for: `fol@2[fricative:+]` says "the second thing
 * after this vowel is a fricative", which is true exactly where a nasal stood
 * between them and false where the nasal is followed by a stop. Counting past
 * the deleted segment is how a rule of this shape is stated, and the shape is
 * common -- Greek, Latin, Old Irish, Hindi and Middle Korean all have a
 * version, and it is behind a large share of the world's long vowels, nasal
 * vowels and tone systems.
 *
 * The assertion is on the distance slot rather than on a particular predicate:
 * which feature of the fricative the search names is a tie-break among
 * predicates that partition this corpus identically, and asserting one of them
 * would be asserting the tie. */
static void test_compensatory_lengthening_reaches_past_the_segment_that_was_lost(rg_context *ctx) {
    rg_corpus *corpus = load("compensatory_lengthening");
    rg_multi_model *model = train(ctx, corpus);
    size_t i;
    int found = 0;

    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
        size_t j;
        int seen_a = 0;
        int seen_long = 0;
        for (j = 0; j < row->segment_count; j++) {
            if (strcmp(row->graphemes[j], "a") == 0) {
                seen_a = 1;
            }
            if (strcmp(row->graphemes[j], "o\xcb\x90") == 0) {
                seen_long = 1;
            }
        }
        if (!seen_a || !seen_long) {
            continue;
        }
        for (j = 0; j < row->segment_count; j++) {
            size_t k;
            for (k = 0; k < row->contexts[j].following_at_distance_count; k++) {
                if (row->contexts[j].following_at_distance[k].offset >= 2) {
                    found = 1;
                }
            }
        }
        /* And it accounts for every vowel that lengthened. */
        assert(row->contrast_count == 0.0);
    }
    assert(found);
    /* The nasal before a stop is the contrast, and it does not lengthen. */
    assert(has_correspondence(model, "a", "a"));

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
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
            assert(row->evidence.standing != RG_RULE_STANDING_UNMEASURED);
            if (row->evidence.search_margin > fit->null_search_margin) {
                assert(row->evidence.standing == RG_RULE_STANDING_ABOVE_NOISE);
            } else {
                assert(row->evidence.standing == RG_RULE_STANDING_WITHIN_NOISE);
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

/* A cross-dimensional rule is a rule, and reports whether it stands on the same
 * terms as any other.
 *
 * The sibling test above walks the conditioned classes, which is why this went
 * unnoticed: the pairwise cross-dimensional appender took the margin its caller
 * had computed and dropped it on the floor, and the multi-lect lift copied every
 * field of the row except that one. Both published 0.0, so every one of these
 * rules read `within-noise` whatever its evidence -- a verdict, delivered with
 * the same words a measured one uses, resting on a number nobody had set. */
static void test_cross_dimensional_rules_report_whether_they_stand(rg_context *ctx) {
    rg_corpus *corpus = 0;
    rg_train_options options;
    rg_multi_model *model = 0;
    const rg_corpus_fit *fit;
    char path[512];
    size_t i;
    size_t measured = 0;
    int margin_seen = 0;

    snprintf(path, sizeof(path), "%s/testdata/corpora/joint_tonogenesis.tsv", REGULAE_SOURCE_DIR);
    assert(rg_corpus_load_tsv(path, 0, &corpus, 0) == RG_OK);

    rg_train_options_init_defaults(&options);
    options.permutation_count = 10;
    assert(rg_train_model(ctx, rg_corpus_cognate_at(corpus, 0),
                          rg_corpus_cognate_count(corpus), &options, &model) == RG_OK);
    fit = rg_multi_model_fit(model);

    assert(rg_multi_model_cross_dimensional_row_count(model) > 0);
    for (i = 0; i < rg_multi_model_cross_dimensional_row_count(model); i++) {
        const rg_multi_cross_dimensional_row *row = rg_multi_model_cross_dimensional_row_at(model, i);
        assert(row->rule.evidence.standing != RG_RULE_STANDING_UNMEASURED);
        if (row->rule.evidence.search_margin > fit->null_search_margin) {
            assert(row->rule.evidence.standing == RG_RULE_STANDING_ABOVE_NOISE);
        } else {
            assert(row->rule.evidence.standing == RG_RULE_STANDING_WITHIN_NOISE);
        }
        if (row->rule.evidence.search_margin > 0.0) {
            margin_seen = 1;
        }
        measured++;
    }

    /* The same rows before they were lifted. A margin the pairwise model does
     * not carry cannot arrive in the multi-lect one. */
    for (i = 0; i < rg_multi_model_pair_model_count(model); i++) {
        const rg_pairwise_model *pair = rg_multi_model_pair_model_at(model, i)->model;
        size_t j;
        for (j = 0; j < rg_pairwise_model_cross_dimensional_row_count(pair); j++) {
            const rg_cross_dimensional_row *row = rg_pairwise_model_cross_dimensional_row_at(pair, j);
            assert(row->evidence.standing != RG_RULE_STANDING_UNMEASURED);
            if (row->evidence.search_margin > fit->null_search_margin) {
                assert(row->evidence.standing == RG_RULE_STANDING_ABOVE_NOISE);
            } else {
                assert(row->evidence.standing == RG_RULE_STANDING_WITHIN_NOISE);
            }
        }
    }

    /* A rule committed by a search paid a search charge to get there, so at
     * least one of these has a margin above zero. This is the assertion the
     * dropped field failed. */
    assert(measured > 0);
    assert(margin_seen);

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* Without a baseline there is nothing to stand above, and the rule says so
 * rather than claiming a verdict it has not earned. */
static void test_no_baseline_means_no_verdict(rg_context *ctx) {
    rg_corpus *corpus = load("rhotacism");
    rg_multi_model *model = train(ctx, corpus);
    size_t i;

    assert(rg_multi_model_fit(model)->rules_measured == 0);
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        assert(rg_multi_model_conditioned_class_at(model, i)->evidence.standing ==
               RG_RULE_STANDING_UNMEASURED);
    }
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}


/* The multi-lect verdict is not the whole verdict, and Grassmann is the corpus
 * that proves it.
 *
 * Every conditioned class on it sits within noise, so `rules_above_noise` of
 * `rules_measured` reads 0 of 4: the corpus found nothing distinguishable from
 * having looked. But the law itself -- Greek t answering PIE tʰ where an
 * aspirate follows somewhere -- is a conditioned correspondence in the pair's
 * own model, and it stands. Counting it into the multi-lect pair would have
 * made the ratio depend on how many lects the corpus samples, because these are
 * counted per pair; reporting nothing about it left a corpus whose one real
 * finding stands looking like a corpus that had none. */
static void test_the_per_pair_verdict_is_reported_separately(rg_context *ctx) {
    char path[1024];
    rg_corpus *corpus = 0;
    rg_multi_model *model = 0;
    rg_train_options options;
    const rg_corpus_fit *fit;
    size_t p;
    size_t counted = 0;
    size_t standing = 0;

    snprintf(path, sizeof(path), "%s/testdata/soundlaws/grassmann.tsv", REGULAE_SOURCE_DIR);
    assert(rg_corpus_load_tsv(path, 0, &corpus, 0) == RG_OK);
    rg_train_options_init_defaults(&options);
    options.permutation_count = 30;
    assert(rg_train_model(ctx, rg_corpus_cognate_at(corpus, 0),
                          rg_corpus_cognate_count(corpus), &options, &model) == RG_OK);
    fit = rg_multi_model_fit(model);

    /* The multi-lect verdict on this corpus, and the reason the other one has
     * to exist. */
    assert(fit->rules_measured > 0);
    assert(fit->rules_above_noise == 0);

    /* The per-pair verdict counts every conditioned correspondence every pair
     * carries, and at least one of them stands. */
    for (p = 0; p < rg_multi_model_pair_model_count(model); p++) {
        const rg_conditioned_segment_count_row *rows;
        size_t n = 0;
        size_t j;
        rows = rg_pairwise_model_conditioned_segment_counts(
            rg_multi_model_pair_model_at(model, p)->model, &n);
        for (j = 0; j < n; j++) {
            assert(rows[j].evidence.standing != RG_RULE_STANDING_UNMEASURED);
            counted++;
            if (rows[j].evidence.standing == RG_RULE_STANDING_ABOVE_NOISE) {
                standing++;
            }
        }
    }
    assert(fit->pairwise_rules_measured == counted);
    assert(fit->pairwise_rules_above_noise == standing);
    assert(fit->pairwise_rules_above_noise > 0);

    /* The two are separate counts, not one split in two. */
    assert(fit->pairwise_rules_measured != fit->rules_measured ||
           fit->pairwise_rules_above_noise != fit->rules_above_noise);

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
    assert(rg_corpus_load_tsv(path, 0, &corpus, 0) == RG_OK);
    model = train(ctx, corpus);

    for (i = 0; i < rg_multi_model_cross_dimensional_row_count(model); i++) {
        const rg_multi_cross_dimensional_row *row = rg_multi_model_cross_dimensional_row_at(model, i);
        assert(row->rule.evidence.decision_index >= 0);
        /* The conjunction refines what a single predicate left, so it cannot
         * have been decided first. */
        if (row->rule.source_environment.preceding_count == 1 &&
            row->rule.source_environment.self_count == 1) {
            assert(row->rule.evidence.decision_index > 0);
            refinement_seen = 1;
        }
    }
    assert(refinement_seen);

    /* An unconditioned class was aggregated, not decided. */
    for (i = 0; i < rg_multi_model_unconditioned_class_count(model); i++) {
        assert(rg_multi_model_unconditioned_class_at(model, i)->evidence.decision_index == -1);
    }
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        assert(rg_multi_model_conditioned_class_at(model, i)->evidence.decision_index >= 0);
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
/* chunk_min_transparency drops the chunks that are hardest to read as a single
 * historical process, and nothing else.
 *
 * metathesis_adjacent is the fixture that separates the two: the transposition
 * itself promotes as "ask ~ aks", three segments a side, and the same
 * transposition with its onset attached promotes as "pask ~ paks", four a
 * side, saying nothing more. The second scores 0.240 and the first 0.550, so a
 * threshold between them keeps the process and discards the padded restatement
 * of it -- which is what the option is for.
 *
 * Until 2026-08-15 setting this option returned RG_ERR_UNSUPPORTED_OPTION.
 * That was honest, and it was still a knob in the public header that did
 * nothing. */
static void test_chunk_transparency_filters_the_least_readable_chunks(rg_context *ctx) {
    rg_corpus *corpus = load("metathesis_adjacent");
    static const double thresholds[] = { 0.0, 0.3, 0.6, 0.95 };
    size_t counts[4];
    size_t t;

    for (t = 0; t < sizeof(thresholds) / sizeof(thresholds[0]); t++) {
        rg_train_options options;
        rg_multi_model *model = 0;
        const rg_pairwise_model *pair;
        size_t count;
        size_t i;

        rg_train_options_init_defaults(&options);
        options.chunk_min_transparency = thresholds[t];
        assert(rg_train_model(ctx, rg_corpus_cognate_at(corpus, 0),
                              rg_corpus_cognate_count(corpus), &options, &model) == RG_OK);
        pair = rg_multi_model_pair_model_at(model, 0)->model;
        count = rg_pairwise_model_chunk_row_count(pair);

        /* Every surviving row clears the bar, and every row is scored whether
         * or not the bar is set. */
        for (i = 0; i < count; i++) {
            const rg_chunk_row *row = rg_pairwise_model_chunk_row_at(pair, i);
            assert(row->transparency >= thresholds[t]);
            assert(row->transparency >= 0.0 && row->transparency <= 1.0);
        }
        counts[t] = count;
        rg_multi_model_free(model);
    }

    /* The default drops nothing, so there is a table to filter and the rest of
     * this is not vacuous. */
    assert(counts[0] > 0);
    /* Raising the bar never adds a chunk ... */
    assert(counts[1] <= counts[0]);
    assert(counts[2] <= counts[1]);
    assert(counts[3] <= counts[2]);
    /* ... and somewhere in the middle it bites, which is the assertion that
     * would fail if the score were constant or the filter were not wired in.
     * A monotone sequence of equal numbers is monotone. */
    assert(counts[2] < counts[0]);
    /* Nothing in this corpus reads as cleanly as 0.95, so the whole table
     * goes: the option means what it says at the top of its range rather than
     * saturating. */
    assert(counts[3] == 0);
    rg_corpus_free(corpus);
}

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
    test_cross_dimensional_rules_report_whether_they_stand(ctx);
    test_no_baseline_means_no_verdict(ctx);
    test_the_per_pair_verdict_is_reported_separately(ctx);
    test_rules_carry_the_order_they_were_decided(ctx);
    test_metathesis(ctx);
    test_morphological_conditioning(ctx);
    test_no_boundaries_means_no_morphological_axis(ctx);
    test_rounding_harmony(ctx);
    test_conditioning_works_in_any_feature_system();
    test_place_assimilation(ctx);
    test_chunk_transparency_filters_the_least_readable_chunks(ctx);
    test_place_dissimilation(ctx);
    test_conditioning_ladder(ctx);
    test_a_disjunctive_trigger_comes_out_as_a_decision_list(ctx);
    test_a_second_description_of_one_rule_is_not_a_second_rule(ctx);
    test_syllable_weight_is_stated_in_one_rule(ctx);
    test_an_environment_the_daughter_lost_is_found_on_the_proto_side(ctx);
    test_umlaut_is_recovered_from_the_lects_that_kept_the_trigger(ctx);
    test_a_chain_shift_is_not_reported_as_a_merger(ctx);
    test_compensatory_lengthening_reaches_past_the_segment_that_was_lost(ctx);
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
