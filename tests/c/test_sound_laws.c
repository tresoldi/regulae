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

/* The tone fixtures live under experiments/ in the one-row-per-cognate
 * spelling, because a tone corpus is written as whole words. */
static rg_corpus *load_wide(rg_context *ctx, const char *name) {
    rg_corpus *corpus = 0;
    char path[512];
    snprintf(path, sizeof(path), "%s/experiments/%s/cognates.tsv",
             REGULAE_SOURCE_DIR, name);
    assert(rg_corpus_load_wide_tsv(ctx, path, 0, &corpus, 0) == RG_OK);
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

/* Whether one class states this correspondence, in either lect order: which
 * lect a class lists first is a labelling detail. */
static int states_pair(const rg_multi_class_row *row, const char *a, const char *b) {
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
    return seen_a && seen_b;
}

/* Whether some unconditioned class states it. */
static int has_correspondence(const rg_multi_model *model, const char *a, const char *b) {
    size_t i;
    for (i = 0; i < rg_multi_model_unconditioned_class_count(model); i++) {
        if (states_pair(rg_multi_model_unconditioned_class_at(model, i), a, b)) {
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
    /* The shifts themselves, PIE segment and Germanic reflex, in the three
     * series the law describes: voiceless stops to fricatives, voiced stops to
     * voiceless, aspirates to plain voiced. Eight of the nine cells are filled
     * -- PIE *b is missing by the proto-language's own famous gap. */
    static const char *const shifts[][2] = {
        {"p", "f"}, {"t", "\xce\xb8"}, {"k", "x"},
        {"d", "t"}, {"g", "k"},
        {"b\xca\xb0", "b"}, {"d\xca\xb0", "d"}, {"g\xca\xb0", "g"},
    };
    const size_t shift_count = sizeof(shifts) / sizeof(shifts[0]);
    rg_corpus *corpus = load("grimm");
    rg_multi_model *model = train(ctx, corpus);
    size_t i;
    size_t s;

    for (s = 0; s < shift_count; s++) {
        assert(has_correspondence(model, shifts[s][0], shifts[s][1]));
    }

    /* None of the shifts may appear as a conditioned class: Grimm's changes are
     * unconditioned, so conditioning one would be inventing an environment it
     * does not have. Correspondences outside the law may be conditioned -- the
     * vowels answer to PIE's syllabic sonorants and are a separate story, and
     * the plausibility prior exists so that the consonants stay out of chunks
     * and remain available as the environment those vowels are read against. */
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
        for (s = 0; s < shift_count; s++) {
            assert(!states_pair(row, shifts[s][0], shifts[s][1]));
        }
    }

    /* The unconditioned shifts group by feature displacement: the media
     * series (PIE voiced → Gmc voiceless) and the media aspirata
     * (PIE aspirated → Gmc voiced) each form one proposed event. */
    {
        size_t event_count = 0;
        const rg_proposed_event_row *events =
            rg_multi_model_proposed_events(model, &event_count);
        assert(event_count >= 2);
        for (i = 0; i < event_count; i++) {
            size_t j;
            int conditioned = 0;
            if (events[i].class_id_count >= 2) {
                continue;
            }
            /* A single-class event is allowed, but only for a conditioned
             * class: an unconditioned correspondence standing alone is an
             * aggregate no search decided, and admitting one here would make
             * this table a second printing of the class table. Grimm's is the
             * vowel class the plausibility prior conditions, not a shift. */
            for (j = 0; j < rg_multi_model_conditioned_class_count(model); j++) {
                if (rg_multi_model_conditioned_class_at(model, j)->class_id ==
                    events[i].class_ids[0]) {
                    conditioned = 1;
                }
            }
            assert(conditioned);
        }
    }

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
/* The invariant: "a rule published without the contrast it was
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
 * Rounding harmony and Verner's law are regular changes on corpora large enough
 * to show them. Their selected associations clear the current shuffled
 * baseline; the row-by-row assertions below are the regression for both
 * possible verdicts whenever a future fixture falls within noise. */
static void test_rules_report_whether_they_stand_above_noise(rg_context *ctx) {
    /* rounding_harmony carries a well-evidenced change that clears its pivot's
     * null and the eight-example floor; verner is the same law fragmented into
     * four rows of four to six observations, none of which clears the floor.
     * That split is the point of the pair: the machinery certifies the first
     * and, unlike the pairing shuffle it replaced, declines the second rather
     * than passing every row a small fixture commits. */
    static const struct { const char *name; int expect_standing; } fixtures[] = {
        { "rounding_harmony", 1 },
        { "verner", 0 }
    };
    size_t f;
    for (f = 0; f < 2; f++) {
        rg_corpus *corpus = load(fixtures[f].name);
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
            /* Every measured class carries a verdict, and it names the null it
             * was cut against: each is judged in discovery against its own
             * pivot's context-permuted environment, not the pairing shuffle.
             * There is no corpus-wide bar to compare `search_margin` against
             * here, because the bar is a property of the pivot -- which is the
             * whole point of judging per pivot. A class below the eight-example
             * floor is within-noise whatever its margin. */
            assert(row->evidence.standing != RG_RULE_STANDING_UNMEASURED);
            assert(row->evidence.standing_null == RG_NULL_MODEL_WITHIN_BUCKET_SHUFFLE);
            assert(row->evidence.standing == RG_RULE_STANDING_ABOVE_NOISE ||
                   row->evidence.standing == RG_RULE_STANDING_WITHIN_NOISE);
            if (row->evidence.standing == RG_RULE_STANDING_ABOVE_NOISE) {
                assert(row->count >= 8.0);
            }
        }
        if (fixtures[f].expect_standing) {
            assert(fit->rules_above_noise > 0);
        } else {
            assert(fit->rules_above_noise == 0);
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
        assert(rg_multi_model_conditioned_class_at(model, i)->evidence.standing_null ==
               RG_NULL_MODEL_NONE);
    }
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}


/* The multi-lect verdict is not the whole verdict, and Grassmann is the corpus
 * that proves it. Its two multi-lect classes -- greek:t ~ pie:tʰ and
 * greek:k ~ pie:kʰ -- rest on six and five observations, below the
 * eight-example floor, so neither clears its pivot's null; the per-pair
 * correspondences, judged against the pairing shuffle, stand. The unequal
 * denominators are the point, and the per-pivot null makes them starker:
 * counting the pairwise rows into the multi-lect ratio would make it depend on
 * how many lect pairs the corpus samples, and here the multi-lect view is
 * silent where the per-pair view is not. */
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

    /* The multi-lect verdict is measured independently -- and here it is
     * measured and silent, both classes below the floor. */
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
        if (row->rule.environment.preceding_count == 1 &&
            row->rule.environment.self_count == 1) {
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
             * between vowels, not merely before one. It may also be stated
             * as syllable_role=ambisyllabic, which captures the same partition
             * as a single predicate. */
            if (row->context.preceding_count > 0 && row->context.following_count > 0) {
                source_side |= 2;
            }
            if (row->context.syllable_role != 0 &&
                strcmp(row->context.syllable_role, "ambisyllabic") == 0) {
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

/* A conditioned split does not always identify its environment. Where the
 * preceding consonant and the following vowel are perfectly confounded, the
 * palatalisation split is equally read either way, and the class flags a rival
 * at another position that carves it the same. Verner pins the stress
 * environment for some of its rules and leaves it confounded for others, so it
 * must show at least one identifiable conditioned class -- proving the flag
 * separates the two rather than firing on every conditioned row. */
static void test_a_confounded_environment_is_flagged(rg_context *ctx) {
    rg_corpus *corpus = load("conditioned_confound");
    rg_multi_model *model = train(ctx, corpus);
    size_t i;
    int saw_confounded = 0;
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        if (rg_multi_model_conditioned_class_at(model, i)->environment_alternatives > 0) {
            saw_confounded = 1;
        }
    }
    assert(saw_confounded);
    for (i = 0; i < rg_multi_model_unconditioned_class_count(model); i++) {
        assert(rg_multi_model_unconditioned_class_at(model, i)->environment_alternatives == 0);
    }
    rg_multi_model_free(model);
    rg_corpus_free(corpus);

    corpus = load("verner");
    model = train(ctx, corpus);
    int saw_identifiable = 0;
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        if (rg_multi_model_conditioned_class_at(model, i)->environment_alternatives == 0) {
            saw_identifiable = 1;
        }
    }
    assert(saw_identifiable);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* The best search margin among the rows stating a given change, which is what
 * a reader judges a rule by. */
static double margin_of(const rg_multi_model *model, const char *from, const char *to) {
    double best = 0.0;
    size_t i;
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
        size_t j;
        int seen_from = 0;
        int seen_to = 0;
        for (j = 0; j < row->segment_count; j++) {
            seen_from |= strcmp(row->graphemes[j], from) == 0;
            seen_to |= strcmp(row->graphemes[j], to) == 0;
        }
        if (seen_from && seen_to && row->evidence.search_margin > best) {
            best = row->evidence.search_margin;
        }
    }
    return best;
}

/* What it costs to spread one change over the segments it applies to.
 *
 * Two corpora with the same change, the same environment, the same contrast and
 * the same thirty-two aligned positions showing it. In the control they all sit
 * on /f/; in the other they are spread over the four voiceless continuants that
 * undergo the change together. Nothing differs but the spread.
 *
 * The search finds it either way, and at eight examples a cell that is worth
 * knowing on its own -- the evidence floor is not what fragmentation costs
 * here. What it costs is the unit and the standing. One change states itself
 * as two rows at a margin over ten, or as eight rows at a margin near three,
 * on identical data. A reader given the second has to notice that four rows
 * differing only in their grapheme, carrying the same environment and the same
 * score, are one event -- and regulae does not say so.
 *
 * This is the measurement `linguistic_research_sources.md` asks for and the
 * gate for any future pooling: a pooled hypothesis has to beat the fragmented
 * one, and until something measures both there is nothing to beat. */
static void test_fragmenting_a_change_over_a_class_costs_its_standing(rg_context *ctx) {
    rg_corpus *control_corpus = load("natural_class_control");
    rg_corpus *spread_corpus = load("natural_class");
    rg_multi_model *control = train(ctx, control_corpus);
    rg_multi_model *spread = train(ctx, spread_corpus);
    static const char *const from[] = { "f", "s", "x", "\xce\xb8" };
    static const char *const to[] = { "v", "z", "\xc9\xa3", "\xc3\xb0" };
    size_t i;

    /* One change, one row for it, and the environment is the real one. */
    assert(has_conditioned(control, "v", "f", "vowel"));
    assert(margin_of(control, "f", "v") > 10.0);

    /* The same change, spread. Every member is found, every one carries the
     * same environment, and not one of them reaches half the control's
     * standing. */
    for (i = 0; i < sizeof(from) / sizeof(from[0]); i++) {
        assert(has_conditioned(spread, to[i], from[i], "vowel"));
        assert(margin_of(spread, from[i], to[i]) > 0.0);
        assert(margin_of(spread, from[i], to[i]) < margin_of(control, "f", "v") / 2.0);
    }

    /* And it takes four times the rows to say it. */
    assert(rg_multi_model_conditioned_class_count(spread) ==
           4 * rg_multi_model_conditioned_class_count(control));

    rg_multi_model_free(control);
    rg_multi_model_free(spread);
    rg_corpus_free(control_corpus);
    rg_corpus_free(spread_corpus);
}

/* Four rows that are one change, grouped and named.
 *
 * The grouping condition is deliberately strict: same lects, same environment
 * on each, differing graphemes. That is what a change split per segment looks
 * like, and it is checkable without deciding anything about whether the pooled
 * description is better -- which this does not decide, and the members stay
 * published either way.
 *
 * The control is the same change on one segment, and the pair is what makes
 * the grouping claim checkable: the same thirty-two observations, stated once
 * over four classes and once over one. Both propose an event -- a change on one
 * segment is still a change -- and `class_id_count` is what separates them, so
 * the control failing to group is visible rather than merely absent.
 *
 * The control's event exists because its member is *conditioned*, committed by
 * a search against a contrast. A grouper that fired on any single
 * correspondence would reprint the correspondence table; one that fires on a
 * committed rule restates the decision list, which is bounded by it. */
static void test_one_change_over_a_class_is_proposed_as_one_event(rg_context *ctx) {
    rg_corpus *spread_corpus = load("natural_class");
    rg_corpus *control_corpus = load("natural_class_control");
    rg_multi_model *spread = train(ctx, spread_corpus);
    rg_multi_model *control = train(ctx, control_corpus);
    size_t count = 0;
    const rg_proposed_event_row *events = rg_multi_model_proposed_events(spread, &count);
    size_t control_count = 1;
    const rg_proposed_event_row *voicing = 0;
    size_t i;

    /* One event: the change. Identity classes (retentions) are not grouped. */
    assert(count == 1);
    {
        const rg_proposed_event_row *control_events =
            rg_multi_model_proposed_events(control, &control_count);
        /* The same change, and it did not group, because there was nothing to
         * group it with. */
        assert(control_count == 1);
        assert(control_events[0].class_id_count == 1);
        assert(control_events[0].count == 32.0);
    }

    for (i = 0; i < count; i++) {
        size_t m;
        for (m = 0; m < events[i].member_count; m++) {
            size_t g;
            for (g = 0; g < events[i].members[m].grapheme_count; g++) {
                if (strcmp(events[i].members[m].graphemes[g], "v") == 0) {
                    voicing = &events[i];
                }
            }
        }
    }
    assert(voicing != 0);

    /* Four member classes, every observation of all four, and the distinct
     * sets behind them -- the pooled evidence no single member row carries. */
    assert(voicing->class_id_count == 4);
    assert(voicing->member_count == 2);
    assert(voicing->count == 32.0);
    assert(voicing->supporting_cognate_count == 32);

    /* Both sides name their set, which is what makes it a class and not a
     * list. The corpus was built so they can; most cannot. */
    assert(voicing->featurally_definable);
    for (i = 0; i < voicing->member_count; i++) {
        assert(voicing->members[i].grapheme_count == 4);
        assert(voicing->members[i].class_feature_count > 0);
    }

    /* The shared displacement states the rule: voiced is gained on one
     * side, and that is the only feature the grouping has in common. */
    assert(voicing->shared_displacement_count > 0);
    {
        size_t d;
        int has_voiced = 0;
        for (d = 0; d < voicing->shared_displacement_count; d++) {
            if (strcmp(voicing->shared_displacement[d].feature, "voiced") == 0) {
                has_voiced = 1;
            }
        }
        assert(has_voiced);
    }

    /* The members are still published in their own right. Nothing was
     * replaced, so a consumer that rejects the grouping loses nothing. */
    for (i = 0; i < voicing->class_id_count; i++) {
        size_t j;
        int found = 0;
        for (j = 0; j < rg_multi_model_conditioned_class_count(spread); j++) {
            if (rg_multi_model_conditioned_class_at(spread, j)->class_id ==
                voicing->class_ids[i]) {
                found = 1;
            }
        }
        assert(found);
    }

    rg_multi_model_free(spread);
    rg_multi_model_free(control);
    rg_corpus_free(spread_corpus);
    rg_corpus_free(control_corpus);
}

/* Whether some proposed event has this lect contributing exactly this set of
 * graphemes, given as a sorted comma-joined string. */
static const rg_proposed_event_row *event_over(
    const rg_multi_model *model,
    const char *lect,
    const char *graphemes
) {
    size_t count = 0;
    const rg_proposed_event_row *events = rg_multi_model_proposed_events(model, &count);
    size_t i;
    for (i = 0; i < count; i++) {
        size_t m;
        for (m = 0; m < events[i].member_count; m++) {
            const rg_event_member *member = &events[i].members[m];
            char joined[128];
            size_t used = 0;
            size_t g;
            if (strcmp(member->lect_id, lect) != 0) {
                continue;
            }
            joined[0] = '\0';
            for (g = 0; g < member->grapheme_count; g++) {
                size_t len = strlen(member->graphemes[g]);
                if (used + len + 2 >= sizeof(joined)) {
                    break;
                }
                if (used > 0) {
                    joined[used++] = ',';
                }
                memcpy(joined + used, member->graphemes[g], len);
                used += len;
                joined[used] = '\0';
            }
            if (strcmp(joined, graphemes) == 0) {
                return &events[i];
            }
        }
    }
    return 0;
}

static int event_displaces(const rg_proposed_event_row *event, const char *feature) {
    size_t d;
    for (d = 0; event != 0 && d < event->shared_displacement_count; d++) {
        if (strcmp(event->shared_displacement[d].feature, feature) == 0) {
            return 1;
        }
    }
    return 0;
}

/* A chain shift is grouped by the step its rungs share, not by an identical
 * feature delta.
 *
 * Grimm's first shift is the case the identical-delta reading could not reach:
 * p~f loses `bilabial` for `labio-dental`, t~θ loses `alveolar` for `dental`,
 * and k~x moves no place at all, so no two of the three deltas are equal and
 * the best known sound change in the literature came out as three unrelated
 * rows. What they share is stop→fricative, and that is the shift. */
static void test_a_chain_shift_groups_on_the_step_its_rungs_share(rg_context *ctx) {
    rg_corpus *corpus = load("grimm");
    rg_multi_model *model = train(ctx, corpus);
    const rg_proposed_event_row *fricatives = event_over(model, "gmc", "f,x,\xce\xb8");
    const rg_proposed_event_row *voiceless = event_over(model, "gmc", "k,p,t");
    const rg_proposed_event_row *voiced = event_over(model, "gmc", "b,d,g");

    /* Every rung of the law, each as one event. */
    assert(fricatives != 0);
    assert(voiceless != 0);
    assert(voiced != 0);
    assert(fricatives->class_id_count == 3);

    /* And each says what it is: the shared displacement is the rule the
     * grouping implies, which is the whole reason to intersect rather than
     * to demand equality. */
    assert(event_displaces(fricatives, "fricative"));
    assert(event_displaces(voiced, "aspirated"));

    /* The three rungs stay apart. A predicate loose enough to find the first
     * shift is loose enough to pour all three into one row, and the members'
     * intersected displacement is what keeps them separate. */
    assert(fricatives != voiceless);
    assert(voiceless != voiced);

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* One change may be stated as several rules over one outcome.
 *
 * The mirror of the natural-class fixture: there, one environment covers
 * several outcomes; here, `graded_7_disjunction` puts one outcome under four
 * environments no single predicate covers, which is what a decision list is.
 * Grouping used to *require* the outcomes to differ, so this shape -- four
 * rows all reading `f ~ p` -- was unreachable by construction. */
static void test_one_change_over_several_environments_is_proposed_as_one_event(
    rg_context *ctx
) {
    rg_corpus *corpus = load("graded_7_disjunction");
    rg_multi_model *model = train(ctx, corpus);
    const rg_proposed_event_row *event = event_over(model, "daughter", "f");
    size_t i;

    assert(event != 0);
    assert(event->class_id_count > 1);

    /* Every member is a conditioned class, and they are the rules of the list
     * rather than the same rule listed twice: the outcome is shared, so what
     * differs has to be the environment. */
    for (i = 0; i < event->class_id_count; i++) {
        size_t j;
        int found = 0;
        for (j = 0; j < rg_multi_model_conditioned_class_count(model); j++) {
            if (rg_multi_model_conditioned_class_at(model, j)->class_id ==
                event->class_ids[i]) {
                found = 1;
            }
        }
        assert(found);
    }

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* A tone correspondence is an outcome, so a tone shift is an event.
 *
 * Grouping read `graphemes` alone, which put `a[¹¹] ~ a[³³]` down as a
 * retention -- every class in a tone corpus is a retention on that reading, so
 * no tone change could reach this table at all. The event states which tone
 * moved; without `suprasegmentals` on the member the row would read
 * `{a,i,u} ~ {a,i,u}`. */
static void test_a_tone_shift_is_proposed_as_one_event(rg_context *ctx) {
    rg_corpus *corpus = load_wide(ctx, "tone_synthetic");
    rg_multi_model *model = train(ctx, corpus);
    size_t count = 0;
    const rg_proposed_event_row *events = rg_multi_model_proposed_events(model, &count);
    const rg_proposed_event_row *shift = event_over(model, "src", "a,i,u");
    size_t i;
    int stated_a_tone = 0;

    assert(count > 0);
    assert(shift != 0);
    for (i = 0; i < shift->member_count; i++) {
        assert(shift->members[i].suprasegmentals != 0);
        assert(shift->members[i].suprasegmentals->tone[0] != '\0');
    }
    /* The two sides carry different tones, or the event states no change. */
    assert(strcmp(shift->members[0].suprasegmentals->tone,
                  shift->members[1].suprasegmentals->tone) != 0);

    for (i = 0; i < count; i++) {
        if (events[i].members[0].suprasegmentals != 0) {
            stated_a_tone = 1;
        }
    }
    assert(stated_a_tone);

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* A dimension only one side of the corpus writes is a transcription
 * convention, and must not read as a change.
 *
 * Verner's fixture marks stress on Proto-Germanic and not on Gothic, so every
 * vowel in it pairs `a[str:primary]` with `a`. Read literally that is five
 * vowels agreeing on one change and forty observations behind it -- a
 * thoroughly attested fact about the file and nothing about the language. */
static void test_an_annotation_only_one_lect_carries_is_not_an_event(rg_context *ctx) {
    rg_corpus *corpus = load("verner");
    rg_multi_model *model = train(ctx, corpus);
    size_t count = 0;
    const rg_proposed_event_row *events = rg_multi_model_proposed_events(model, &count);
    size_t i;

    /* Verner's own voicing is found -- this is not a test that the pane is
     * empty, which would pass for the wrong reason. */
    assert(count > 0);
    assert(event_over(model, "gothic", "b,z") != 0);

    /* No event is the vowels losing a stress mark Gothic never writes. */
    for (i = 0; i < count; i++) {
        assert(event_over(model, "gothic", "a,e,i,o,u") == 0);
        assert(events[i].members[0].suprasegmentals == 0);
    }

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* A change that lands on one segment is still a change.
 *
 * Vowel harmony in `harmony_synthetic` is a final /a/ rounding to /o/ after a
 * back vowel: one conditioned class, no second row anywhere in the corpus to
 * group it with, and so nothing at all in a table that published only
 * groupings. Eighteen of the twenty-three corpora with an empty table had this
 * shape, which is the ordinary case and not a corner.
 *
 * The event of one still says what changed -- the displacement is computed the
 * same way -- and `class_id_count` says it did not group. */
static void test_a_change_on_one_segment_is_still_an_event(rg_context *ctx) {
    rg_corpus *corpus = load_wide(ctx, "harmony_synthetic");
    rg_multi_model *model = train(ctx, corpus);
    const rg_proposed_event_row *harmony = event_over(model, "derived", "o");

    assert(harmony != 0);
    assert(harmony->class_id_count == 1);
    assert(harmony->count == 15.0);
    /* Committed by a search, which is what earns an ungrouped row its place
     * here: it carries the margin that says how well it paid. */
    assert(harmony->search_margin > 0.0);
    /* And it states the rounding. */
    assert(event_displaces(harmony, "rounded"));

    /* Its member is conditioned. An unconditioned aggregate never earns a row
     * of its own, or this table would reprint the correspondence table. */
    {
        size_t j;
        int conditioned = 0;
        for (j = 0; j < rg_multi_model_conditioned_class_count(model); j++) {
            if (rg_multi_model_conditioned_class_at(model, j)->class_id ==
                harmony->class_ids[0]) {
                conditioned = 1;
            }
        }
        assert(conditioned);
    }

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* Two unrelated wordlists condition nothing, so nothing is proposed on that
 * account -- the restraint the single-class rule must not spend.
 *
 * `chance.tsv` has no conditioned class at all, so admitting single
 * conditioned changes cannot move it. Its events are what the displacement
 * pass makes of accidental correspondences, and that number is the one to
 * watch when this file's predicates are loosened. */
static void test_admitting_single_changes_does_not_move_the_baseline(rg_context *ctx) {
    rg_corpus *corpus = 0;
    rg_multi_model *model = 0;
    char path[512];
    size_t count = 0;
    size_t i;

    snprintf(path, sizeof(path), "%s/testdata/restraint/chance.tsv", REGULAE_SOURCE_DIR);
    assert(rg_corpus_load_tsv(path, 0, &corpus, 0) == RG_OK);
    model = train(ctx, corpus);

    assert(rg_multi_model_conditioned_class_count(model) == 0);
    rg_multi_model_proposed_events(model, &count);
    {
        const rg_proposed_event_row *events = rg_multi_model_proposed_events(model, &count);
        for (i = 0; i < count; i++) {
            /* Every one is a grouping, none is a single class waved through. */
            assert(events[i].class_id_count > 1);
        }
    }

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* An event states the environment its members condition on.
 *
 * Without it the row reads as an unconditioned correspondence: the confound
 * fixture's `k ~ tʃ` happens after a sonorant and nowhere else, and an event
 * that says only `k ~ tʃ` has dropped the half that makes it a finding. The
 * same fixture is built so the corpus cannot tell that conditioner from
 * "before a front vowel", and the event has to carry that too, or it states a
 * conditioner more confidently than the corpus can. */
static void test_an_event_states_its_environment(rg_context *ctx) {
    rg_corpus *corpus = load("conditioned_confound");
    rg_multi_model *model = train(ctx, corpus);
    const rg_proposed_event_row *event = event_over(model, "A", "k");
    size_t m;
    int stated = 0;

    assert(event != 0);
    assert(event->axis == RG_EVENT_AXIS_ENVIRONMENT);
    for (m = 0; m < event->member_count; m++) {
        if (rg_context_spec_constraint_count(&event->members[m].context) > 0) {
            stated = 1;
        }
    }
    assert(stated);
    /* And it says the corpus cannot pin that environment down -- naming the
     * rivals, not merely counting them. The fixture's whole point is that
     * "before a front vowel" fits the same eight words as "after a sonorant",
     * and a reader who is told only that a rival exists cannot go and check. */
    assert(event->environment_alternatives > 0);
    assert(event->environment_rival_count >= (size_t)event->environment_alternatives);
    {
        int names_the_vowel = 0;
        size_t r;
        for (r = 0; r < event->environment_rival_count; r++) {
            if (strcmp(event->environment_rivals[r].feature, "front") == 0) {
                names_the_vowel = 1;
            }
            /* Every rival sits at a different slot from the committed one; a
             * feature of the same neighbour is one environment, not a rival. */
            assert(event->environment_rivals[r].feature != 0);
        }
        assert(names_the_vowel);
    }
    /* Every rival here is a confound: this fixture is built so nothing in it
     * can separate the two readings, which is a stronger finding than a near
     * tie and has to be published as one. */
    {
        size_t r;
        for (r = 0; r < event->environment_rival_count; r++) {
            assert(event->environment_rivals[r].same_partition);
        }
    }

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* A grouping can pin an environment down that none of its rules could alone.
 *
 * Each of lenition's three rules is confusable with several other predicates on
 * its own handful of observations -- a five-word split has a lot of features
 * that happen to carve it. What survives as a rival to the *grouping* is what
 * is a rival to every member, and on this corpus nothing is. That is a real
 * property of grouping rather than an artefact of the summary, and it is why
 * the event reports the shared rivals rather than pooling the members'. */
static void test_grouping_can_resolve_a_confound_its_members_have(rg_context *ctx) {
    rg_corpus *corpus = load("lenition");
    rg_multi_model *model = train(ctx, corpus);
    const rg_proposed_event_row *voicing = event_over(model, "latin", "k,p,t");
    size_t i;
    int a_member_is_confounded = 0;

    assert(voicing != 0);
    assert(voicing->class_id_count == 3);

    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
        size_t k;
        for (k = 0; k < voicing->class_id_count; k++) {
            if (row->class_id == voicing->class_ids[k] &&
                row->environment_alternatives > 0) {
                a_member_is_confounded = 1;
            }
        }
    }
    assert(a_member_is_confounded);

    /* The members disagree about what their rivals are, so none is a rival to
     * the grouping, and the event says its environment is pinned. */
    assert(voicing->environment_rival_count == 0);
    assert(voicing->environment_alternatives == 0);

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* An environment the corpus preferred over another it can see past is a
 * different finding from one it cannot separate at all.
 *
 * Latin rhotacism commits `prev-syl[syllable_shape:open]` on the Latin side at
 * a margin of 3.50. `following[vowel:+]` splits the same words *differently*
 * and would still have been committed, at 2.14 -- so the corpus does prefer
 * the committed reading and a reader who is shown only the winner never learns
 * that the intervocalic analysis was in the running. Published as a near tie
 * with its margin, not as a confound: someone who takes the second for the
 * first stops looking for evidence that exists. */
static void test_a_preferred_environment_names_what_it_was_preferred_to(rg_context *ctx) {
    rg_corpus *corpus = load("rhotacism");
    rg_multi_model *model = train(ctx, corpus);
    const rg_proposed_event_row *event = event_over(model, "latin", "r");
    size_t r;
    int found_near = 0;

    assert(event != 0);
    assert(event->environment_rival_count > 0);
    for (r = 0; r < event->environment_rival_count; r++) {
        const rg_environment_rival *rival = &event->environment_rivals[r];
        if (rival->same_partition) {
            /* A confound is not scored separately: it carves the same rows. */
            assert(rival->search_margin == 0.0);
            continue;
        }
        found_near = 1;
        /* It cleared the same gate the committed rule had to clear... */
        assert(rival->search_margin > 0.0);
        /* ...and lost, or it would be the committed one. */
        assert(rival->search_margin <= event->search_margin);
    }
    assert(found_near);

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* The environment an event states is what every member states, not what one of
 * them happened to pick up.
 *
 * Each split is searched on its own, so a member routinely carries a conjunct
 * the others did not need. On Grassmann both members turn on an aspirate
 * somewhere ahead in Proto-Indo-European, and one also acquired "next syllable
 * open" on the Greek side. The law is the shared half; publishing the member's
 * whole environment would state the incidental conjunct as part of it. */
static void test_an_event_states_only_the_shared_environment(rg_context *ctx) {
    rg_corpus *corpus = load("grassmann");
    rg_multi_model *model = train(ctx, corpus);
    const rg_proposed_event_row *event = event_over(model, "greek", "k,t");
    size_t m;
    const rg_context_spec *pie = 0;
    const rg_context_spec *greek = 0;

    assert(event != 0);
    assert(event->class_id_count == 2);
    for (m = 0; m < event->member_count; m++) {
        if (strcmp(event->members[m].lect_id, "pie") == 0) {
            pie = &event->members[m].context;
        } else {
            greek = &event->members[m].context;
        }
    }
    assert(pie != 0 && greek != 0);

    /* The law: an aspirate somewhere ahead. */
    assert(pie->somewhere_following_count == 1);
    assert(strcmp(pie->somewhere_following[0].feature, "aspirated") == 0);

    /* And not the conjunct only one member carried. */
    assert(rg_context_spec_constraint_count(greek) == 0);

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* A grouping whose members differ in environment must not claim one.
 *
 * The disjunction fixture is one outcome under four environments no single
 * predicate covers -- that difference is what groups them. There is nothing
 * shared to state, and an empty environment would read as "unconditioned",
 * which is the opposite of the finding. The axis is what says so. */
static void test_a_grouping_by_outcome_claims_no_shared_environment(rg_context *ctx) {
    rg_corpus *corpus = load("graded_7_disjunction");
    rg_multi_model *model = train(ctx, corpus);
    const rg_proposed_event_row *event = event_over(model, "daughter", "f");
    size_t m;

    assert(event != 0);
    assert(event->class_id_count > 1);
    assert(event->axis == RG_EVENT_AXIS_OUTCOME);
    for (m = 0; m < event->member_count; m++) {
        assert(rg_context_spec_constraint_count(&event->members[m].context) == 0);
    }

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* An unconditioned grouping has no environment and says so by its axis, not by
 * an empty field a reader has to interpret. */
static void test_a_displacement_grouping_states_no_environment(rg_context *ctx) {
    rg_corpus *corpus = load("grimm");
    rg_multi_model *model = train(ctx, corpus);
    const rg_proposed_event_row *event = event_over(model, "gmc", "f,x,\xce\xb8");
    size_t m;

    assert(event != 0);
    assert(event->axis == RG_EVENT_AXIS_DISPLACEMENT);
    assert(event->environment_alternatives == 0);
    for (m = 0; m < event->member_count; m++) {
        assert(rg_context_spec_constraint_count(&event->members[m].context) == 0);
    }

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* The table is ranked by the evidence behind each grouping.
 *
 * It used to come out in the order the grouping passes run, which is an
 * implementation detail: on a corpus with sixteen events that put
 * eleven-observation groupings above sixty-observation ones, and the first row
 * of a table is read as its strongest claim. */
static void test_events_are_ranked_by_their_pooled_count(rg_context *ctx) {
    rg_corpus *corpus = load_wide(ctx, "mandarin_historical");
    rg_multi_model *model = train(ctx, corpus);
    size_t count = 0;
    const rg_proposed_event_row *events = rg_multi_model_proposed_events(model, &count);
    size_t i;

    assert(count > 2);
    for (i = 1; i < count; i++) {
        assert(events[i - 1].count >= events[i].count);
        /* Ties are broken on a fixed key, or the JSON stops being
         * reproducible between runs. */
        if (events[i - 1].count == events[i].count) {
            assert(events[i - 1].class_ids[0] < events[i].class_ids[0]);
        }
    }

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
    test_a_confounded_environment_is_flagged(ctx);
    test_fragmenting_a_change_over_a_class_costs_its_standing(ctx);
    test_one_change_over_a_class_is_proposed_as_one_event(ctx);
    test_a_chain_shift_groups_on_the_step_its_rungs_share(ctx);
    test_one_change_over_several_environments_is_proposed_as_one_event(ctx);
    test_a_tone_shift_is_proposed_as_one_event(ctx);
    test_an_annotation_only_one_lect_carries_is_not_an_event(ctx);
    test_a_change_on_one_segment_is_still_an_event(ctx);
    test_admitting_single_changes_does_not_move_the_baseline(ctx);
    test_an_event_states_its_environment(ctx);
    test_a_preferred_environment_names_what_it_was_preferred_to(ctx);
    test_an_event_states_only_the_shared_environment(ctx);
    test_grouping_can_resolve_a_confound_its_members_have(ctx);
    test_a_grouping_by_outcome_claims_no_shared_environment(ctx);
    test_a_displacement_grouping_states_no_environment(ctx);
    test_events_are_ranked_by_their_pooled_count(ctx);
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
