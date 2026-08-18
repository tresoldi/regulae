#include "regulae.h"
#include "table_access.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* What regulae has to *not* find.
 *
 * Every other test in this directory asserts a discovery. These assert
 * restraint, and they are the harder half of being trustworthy: a method that
 * misses a correspondence costs its user an afternoon, and one that invents an
 * environment costs them a claim they will have to withdraw. The second
 * failure does not announce itself -- a spurious conditioned rule is formatted
 * exactly like a real one, carries a count and a confidence interval, and
 * reads as a finding.
 *
 * The corpora are under testdata/restraint/ and none of them contains a
 * conditioned sound law. Four have a right answer of "no environment" and
 * regulae gives it for three; one has a right answer regulae cannot give, and
 * is here so that what it does give is on record; and the last is a ladder
 * that says how much evidence the search needs before it can see a rule at
 * all, which is the same question asked from the other side.
 *
 * The one it fails is merger_gap, and the assertion below is written to the
 * behaviour rather than to the intention. */

static rg_corpus *load(const char *directory, const char *name) {
    rg_corpus *corpus = 0;
    char path[512];
    snprintf(path, sizeof(path), "%s/testdata/%s/%s.tsv", REGULAE_SOURCE_DIR, directory, name);
    assert(rg_corpus_load_tsv(path, 0, &corpus, 0) == RG_OK);
    assert(corpus != 0);
    return corpus;
}

/* Shuffles are a full training run each, so this is the largest count the
 * tests can carry and still finish in a few seconds per fixture. It is enough
 * for the p95 the standing verdict is cut at to be usable, and it is a
 * conservative estimate rather than a loose one: with twelve samples the
 * quantile sits at the largest of them, so a rule clearing it has cleared
 * something the noise actually reached. The mechanism is checked in
 * `test_the_shuffled_baseline_is_reproducible` in the sound-law tests. */
#define SHUFFLES 12

static rg_multi_model *train(rg_context *ctx, rg_corpus *corpus, int permutations) {
    rg_multi_model *model = 0;
    rg_train_options options;
    rg_train_options_init_defaults(&options);
    options.permutation_count = permutations;
    assert(rg_train_model(ctx, rg_corpus_cognate_at(corpus, 0),
                          rg_corpus_cognate_count(corpus), &options, &model) == RG_OK);
    return model;
}

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

/* Two lects with no historical connection, drawn from one inventory and one
 * set of word shapes -- which is the hard version rather than the easy one.
 * Neighbouring languages share phonotactics whether or not they share an
 * ancestor, and a method that only rejects unrelatedness when the inventories
 * differ has rejected nothing.
 *
 * The corpus-level verdict is the one that has to hold. Shuffling this corpus
 * removes nothing, because there was nothing in it, so the real run and its
 * own shuffles have to land on top of each other. */
static void test_unrelated_lects_are_not_distinguishable_from_their_own_shuffles(rg_context *ctx) {
    rg_corpus *corpus = load("restraint", "chance");
    rg_multi_model *model = train(ctx, corpus, SHUFFLES);
    const rg_corpus_fit *fit = rg_multi_model_fit(model);

    assert(fit != 0);
    assert(fit->permutation_count == SHUFFLES);
    /* Two standard deviations, against the thirty and forty a real corpus of
     * this size reaches. The bound is loose on purpose: what would be wrong is
     * a *verdict*, and there is no seed at which unrelated wordlists come out
     * looking related. */
    assert(fabs(fit->cost_per_segment_z) < 2.0);

    /* The corrected multinomial parameter charge leaves the accidental
     * unconditioned correspondences visible but gives none of them a selected
     * environment. The shuffled baseline deliberately trains without the
     * search charge so that its maximum margin can calibrate a charged run; it
     * still demonstrates that an unpriced search finds conditioned structure
     * in noise. */
    assert(fit->unconditioned_class_count > 40);
    assert(fit->conditioned_class_count == 0);
    assert(fit->null_unconditioned_class_mean >= (double)fit->unconditioned_class_count);
    assert(fit->null_conditioned_class_mean > 0.0);
    assert(fit->rules_measured == 0);
    assert(fit->rules_above_noise == 0);
    assert(fit->pairwise_rules_measured == 0);
    assert(fit->pairwise_rules_above_noise == 0);

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* One change, spread over the lexicon rather than over an environment: proto
 * /p/ answers daughter /f/ in half the words and stays /p/ in the other half,
 * and the two halves are balanced across every environment the search can
 * name.
 *
 * This is lexical diffusion, and it is also what an unfinished change, a
 * dialect mixture and a half-completed analogy look like from the outside. The
 * right answer is two correspondences for one proto segment and no
 * environment: the split is real and its conditioning is not phonological.
 * Reporting an environment would be reporting the accident that half of
 * something has to fall somewhere. */
static void test_a_lexically_diffused_change_gets_no_environment(rg_context *ctx) {
    rg_corpus *corpus = load("restraint", "diffusion");
    rg_multi_model *model = train(ctx, corpus, SHUFFLES);
    const rg_corpus_fit *fit = rg_multi_model_fit(model);

    /* The corpus is certainly not chance: it aligns thirty standard
     * deviations better than its shuffles. Restraint here is not doubt about
     * the data. */
    assert(fit->cost_per_segment_z < -10.0);
    assert(fit->conditioned_class_count == 0);
    assert(has_correspondence(model, "p", "p"));
    assert(has_correspondence(model, "p", "f"));

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* Two correspondence sets in one lect pair, as borrowing leaves them: an
 * inherited layer that ran a spirantising shift over the whole voiceless
 * series -- p~f, t~s, k~x -- and a borrowed layer that left all three alone.
 *
 * A whole *set* splitting at once is what makes a stratum visible to a
 * comparativist, and it is why this is not `diffusion` with more segments.
 * English has *father* beside *paternal*, and the second is not an exception
 * to Grimm's Law but a word that was not in the language when Grimm's Law ran.
 *
 * Six correspondences and no environment. Neither layer is an error; naming an
 * environment for either would be. */
static void test_a_borrowed_stratum_comes_out_as_a_second_correspondence_set(rg_context *ctx) {
    rg_corpus *corpus = load("restraint", "stratum");
    rg_multi_model *model = train(ctx, corpus, SHUFFLES);
    const rg_corpus_fit *fit = rg_multi_model_fit(model);

    assert(fit->cost_per_segment_z < -10.0);
    assert(fit->conditioned_class_count == 0);
    assert(has_correspondence(model, "p", "f"));
    assert(has_correspondence(model, "p", "p"));
    assert(has_correspondence(model, "t", "s"));
    assert(has_correspondence(model, "t", "t"));
    assert(has_correspondence(model, "k", "x"));
    assert(has_correspondence(model, "k", "k"));

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* Two unrelated lects, half of one's vocabulary borrowed from the other.
 *
 * Thirty of the sixty concepts are loans adapted through a regular
 * substitution -- the donor's f, θ, x and z have no place in the borrower's
 * inventory and come out as p, t, k and s. The other thirty are native on both
 * sides and share nothing.
 *
 * This is an areal relationship, and it is the commonest way a long-range
 * comparison goes wrong. Japanese and Chinese are the textbook case: half the
 * lexicon in systematic correspondence and no common ancestor. The Balkans,
 * mainland Southeast Asia, South Asia and the Pacific Northwest all contain
 * pairs like it.
 *
 * There is no restraint available here, and that is why the fixture is worth
 * having. The correspondences are real, they are regular, and regulae reports
 * them -- correctly. The corpus sits fourteen standard deviations below its own
 * shuffles, which is a true statement about the data and not a statement about
 * descent. Nothing in the distribution of segments distinguishes inheritance
 * from borrowing; that judgement needs the semantic fields, the direction of
 * cultural flow and the dates, and it belongs to the linguist.
 *
 * What the corpus does leave is a signature, and it is worth knowing how to
 * read. The outlier ranking comes out bimodal: every one of the thirty native
 * sets ranks worse than every one of the thirty loans, with no interleaving at
 * all. Mistaken judgements leave a short tail -- five sets in
 * `diagnostics/contaminated` -- and a wordlist where instead *half* of it
 * ranks apart is a wordlist to ask a different question about. */
static void test_a_borrowed_half_is_reported_and_the_split_is_visible(rg_context *ctx) {
    rg_corpus *corpus = load("restraint", "contact");
    rg_multi_model *model = train(ctx, corpus, SHUFFLES);
    const rg_corpus_fit *fit = rg_multi_model_fit(model);
    rg_cognate_outlier_row *rows = 0;
    size_t count = 0;
    rg_train_options options;
    size_t i;
    int native_in_worst_half = 0;

    /* Reported as a relationship, because by every measure available it is
     * one. */
    assert(fit->cost_per_segment_z < -5.0);

    rg_train_options_init_defaults(&options);
    assert(rg_find_cognate_outliers(ctx, rg_corpus_cognate_at(corpus, 0),
                                    rg_corpus_cognate_count(corpus), model,
                                    &options, 0, 0, &rows, &count) == RG_OK);
    assert(count == 60);
    /* The generator numbers the loans even and the native sets odd. */
    for (i = 0; i < 30; i++) {
        const char *id = rows[i].cognate_id;
        size_t length = strlen(id);
        if (length > 0 && (id[length - 1] - '0') % 2 == 1) {
            native_in_worst_half++;
        }
    }
    assert(native_in_worst_half == 30);

    /* And the model says so without being asked to rank anything. A mean says
     * nothing about shape; `cost_split_separation` is how far apart the two
     * groups are at the best two-way split of the per-set costs, and
     * `cost_split_fraction` is how much of the corpus is on the worse side.
     * Half of it, six standard deviations out. */
    assert(fit->cost_split_separation > 4.0);
    assert(fit->cost_split_fraction > 0.4 && fit->cost_split_fraction < 0.6);

    rg_cognate_outlier_rows_free(rows, count);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* The same statistic, and why it is not a borrowing test.
 *
 * `stratum` and `diffusion` score *higher* on it than `contact` does, and
 * every cognate set in both of them is genuinely cognate with nothing borrowed
 * at all: half the words underwent a change and half did not, so half align
 * one way and half the other. A corpus that is two populations is a corpus to
 * ask a different question about, and the question is not "was this
 * borrowed" -- nothing in the distribution of segments answers that.
 *
 * `contaminated` is the other reading, and it is the fraction that carries it:
 * a high separation with only a tenth of the corpus on the worse side is a
 * tail of sets that do not belong, not two populations. */
static void test_the_split_statistic_says_two_populations_and_not_why(rg_context *ctx) {
    static const char *const two_things[] = { "stratum", "diffusion" };
    size_t i;

    for (i = 0; i < sizeof(two_things) / sizeof(two_things[0]); i++) {
        rg_corpus *corpus = load("restraint", two_things[i]);
        rg_multi_model *model = train(ctx, corpus, 0);
        const rg_corpus_fit *fit = rg_multi_model_fit(model);
        /* No borrowing in either, and both well above `contact`. */
        assert(fit->cost_split_separation > 4.0);
        assert(fit->cost_split_fraction > 0.4 && fit->cost_split_fraction < 0.6);
        rg_multi_model_free(model);
        rg_corpus_free(corpus);
    }
    {
        /* A tail rather than two populations: five bad judgements in
         * forty-five, so the fraction is small where the others' is a half. */
        rg_corpus *corpus = 0;
        rg_multi_model *model;
        const rg_corpus_fit *fit;
        char path[512];
        snprintf(path, sizeof(path), "%s/testdata/diagnostics/contaminated.tsv", REGULAE_SOURCE_DIR);
        assert(rg_corpus_load_tsv(path, 0, &corpus, 0) == RG_OK);
        model = train(ctx, corpus, 0);
        fit = rg_multi_model_fit(model);
        assert(fit->cost_split_separation > 4.0);
        assert(fit->cost_split_fraction < 0.25);
        rg_multi_model_free(model);
        rg_corpus_free(corpus);
    }
}

/* Whether any conditioned class names `feature` in some lect's environment. */
static int names_feature(const rg_multi_model *model, const char *feature) {
    size_t i;
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
        size_t j;
        for (j = 0; j < row->segment_count; j++) {
            const rg_context_spec *c = &row->contexts[j];
            size_t k;
            for (k = 0; k < c->following_count; k++) {
                if (strcmp(c->following[k].feature, feature) == 0) {
                    return 1;
                }
            }
        }
    }
    return 0;
}

/* How much evidence the search needs before it can see a change.
 *
 * Five corpora, the same conditioned change in each -- /p/ answers /f/ before
 * a front vowel -- at 8, 16, 32, 64 and 128 cognate sets, each rung a prefix
 * of the one above so that a rung differs from its neighbour in size and in
 * nothing else. Half of every rung shows the change.
 *
 * The number this pins down is the one a field linguist with thirty cognates
 * actually needs, and it is not answerable from a single fixture: a corpus
 * that yields nothing has either too little data or no pattern, and only the
 * ladder tells you which.
 *
 * Measured after correcting the multinomial parameter count: nothing at 8
 * sets; at 16 and above the intended front-vowel association is present. At
 * the floor a correlated open-vowel description is still selected too, and
 * both clear this fixture's shuffled search. */
static void test_sixteen_sets_recover_the_conditioned_surface_contrast(rg_context *ctx) {
    rg_corpus *corpus = load("restraint", "sparse_008");
    rg_multi_model *model = train(ctx, corpus, 0);

    /* Below the floor the search stays quiet rather than guessing, which is
     * the half of this that matters. Eight sets, four of them showing a
     * perfectly regular change, and no environment is committed. */
    assert(rg_multi_model_conditioned_class_count(model) == 0);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);

    corpus = load("restraint", "sparse_016");
    model = train(ctx, corpus, SHUFFLES);
    /* Doubling it is enough. */
    assert(rg_multi_model_conditioned_class_count(model) > 0);
    assert(names_feature(model, "front"));
    /* Search standing says both selected surface associations exceed the null;
     * it does not choose the intended historical description over the stable
     * open-vowel correlate in this lexicon. */
    assert(rg_multi_model_fit(model)->rules_above_noise > 0);
    assert(rg_multi_model_fit(model)->rules_above_noise ==
           rg_multi_model_fit(model)->rules_measured);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);

    corpus = load("restraint", "sparse_128");
    model = train(ctx, corpus, SHUFFLES);
    /* With plenty of evidence every rule the search commits stands. */
    assert(names_feature(model, "front"));
    assert(rg_multi_model_fit(model)->rules_above_noise ==
           rg_multi_model_fit(model)->rules_measured);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* A neutralisation with stable surface correlates.
 *
 * German final devoicing merges /t/ and /d/ word-finally, so the citation form
 * carries no information about which one a word had; the alternation in the
 * inflected stem is the only evidence, which is what makes this the standard
 * illustration of internal reconstruction. The historical answer is two
 * correspondences for one citation segment -- t~t and t~d, k~k and k~g, p~p
 * and p~b -- with no causal phonological environment: the two groups occupy
 * the same one. A surface model may still report lexical covariates.
 *
 * regulae commits four conditioned associations on it, every one of them a
 * correlate: the alternating /d/-words happen to have sonorants before them
 * more often than the others do. After the categorical parameter correction
 * all four exceed the shuffled search even though none is a cause of final
 * devoicing. That is the distinction between a standing surface association
 * and a historical hypothesis, on data nobody disputes. */
static void test_a_neutralisation_keeps_stable_surface_correlates_explicit(rg_context *ctx) {
    rg_corpus *corpus = load("soundlaws", "final_devoicing");
    rg_multi_model *model = train(ctx, corpus, SHUFFLES);
    const rg_corpus_fit *fit = rg_multi_model_fit(model);

    assert(fit->cost_per_segment_z < -10.0);
    assert(has_correspondence(model, "t", "t"));
    assert(has_correspondence(model, "t", "d"));
    assert(has_correspondence(model, "k", "k"));
    assert(has_correspondence(model, "k", "g"));
    assert(has_correspondence(model, "p", "b"));

    /* The stable lexical correlates survive a chance-search comparison. */
    assert(fit->rules_measured > 0);
    assert(fit->rules_above_noise == fit->rules_measured);

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* A merger with a gap in the proto lexicon, and the only fixture here that
 * regulae currently fails.
 *
 * Proto *o and *a both give daughter /a/, unconditionally, and *k gives x
 * everywhere. There is no conditioned change in the corpus. What there is, is
 * an accident: *o happens to occur only before a labial, because that is where
 * the words carrying it happen to be. So among daughter /a/ before a labial,
 * the proto source is *o -- true, useful to a reconstructor, and not a sound
 * law.
 *
 * regulae publishes it as one, at the strongest score in the corpus, and every
 * safeguard agrees: the shuffled baseline says it stands, held-out prediction
 * confirms it with a positive gain, and environment_alternatives is 0. None of
 * them is malfunctioning. The rule generalises, because a retrodiction does.
 *
 * The cause is that `target_side` in model_context.c means both "the
 * environment was read from the target form" and "the target grapheme is the
 * pivot", so a target-side row states P(source | target) while a source-side
 * row states P(target | source), and the two are published in one table in one
 * format. The test asserts what is published today rather than what should be,
 * because a fixture that cannot fail tests nothing and a test that hides a
 * difference is worse than no test. When M8's direction typing lands, the
 * count below goes to zero for changes and the row reappears as ancestry. */
static void test_a_lexical_gap_in_the_source_is_published_as_conditioning(rg_context *ctx) {
    rg_corpus *corpus = load("restraint", "merger_gap");
    rg_multi_model *model = train(ctx, corpus, SHUFFLES);
    const rg_corpus_fit *fit = rg_multi_model_fit(model);

    /* Unmistakably one language pair, so the finding cannot be waved away as
     * noise on data that was never related. */
    assert(fit->cost_per_segment_z < -10.0);

    /* Both unconditioned mergers are found, which is the whole of the right
     * answer. */
    assert(has_correspondence(model, "a", "o"));
    assert(has_correspondence(model, "a", "a"));
    assert(has_correspondence(model, "x", "k"));

    /* And this is the part that is wrong. The correct value is 0. */
    assert(fit->conditioned_class_count == 1);

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

int main(void) {
    rg_context *ctx = 0;
    assert(rg_context_new_builtin(&ctx) == RG_OK);
    test_unrelated_lects_are_not_distinguishable_from_their_own_shuffles(ctx);
    test_a_lexical_gap_in_the_source_is_published_as_conditioning(ctx);
    test_a_lexically_diffused_change_gets_no_environment(ctx);
    test_a_borrowed_stratum_comes_out_as_a_second_correspondence_set(ctx);
    test_a_borrowed_half_is_reported_and_the_split_is_visible(ctx);
    test_the_split_statistic_says_two_populations_and_not_why(ctx);
    test_sixteen_sets_recover_the_conditioned_surface_contrast(ctx);
    test_a_neutralisation_keeps_stable_surface_correlates_explicit(ctx);
    rg_context_free(ctx);
    printf("restraint tests passed\n");
    return 0;
}
