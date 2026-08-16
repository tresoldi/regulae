#include "regulae.h"
#include "table_access.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Whether the corpus is wrong, and whether regulae can say so.
 *
 * The sound-law fixtures ask whether a change can be found and the restraint
 * fixtures ask whether a non-change can be declined. Both assume the wordlist
 * is right. No wordlist is: cognate judgements are made by people and some are
 * mistaken, two sources transcribing one language do not agree with each other,
 * and a compound is cognate in one element and not the other. None of that is a
 * failure of the method, all of it changes what the method reports, and
 * noticing it is most of what a comparativist's week is spent on.
 *
 * What a tool owes its user is not immunity to bad data -- there is no such
 * thing -- but a way of finding out. These tests measure how much of that
 * regulae provides, including where the answer is none. */

static rg_corpus *load(const char *name) {
    rg_corpus *corpus = 0;
    char path[512];
    snprintf(path, sizeof(path), "%s/testdata/diagnostics/%s.tsv", REGULAE_SOURCE_DIR, name);
    assert(rg_corpus_load_tsv(path, 0, &corpus, 0) == RG_OK);
    assert(corpus != 0);
    return corpus;
}

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

/* Cognate sets ranked by how badly they align under the trained model, worst
 * first -- what `regulae outliers --model` prints. */
static rg_cognate_outlier_row *rank(
    rg_context *ctx,
    rg_corpus *corpus,
    const rg_multi_model *model,
    size_t *count
) {
    rg_cognate_outlier_row *rows = 0;
    rg_train_options options;
    rg_train_options_init_defaults(&options);
    assert(rg_find_cognate_outliers(ctx, rg_corpus_cognate_at(corpus, 0),
                                    rg_corpus_cognate_count(corpus), model,
                                    &options, 0, 0, &rows, count) == RG_OK);
    assert(rows != 0);
    return rows;
}

/* A wrong cognate judgement is two unrelated words filed under one identifier,
 * and it enters training at full weight. Forty regular sets with five of those
 * mixed in, and no confidence column -- a linguist who knew which five were
 * wrong would have taken them out.
 *
 * The claim is the one that makes `regulae outliers` a tool rather than a
 * statistic: ranking sets by how badly they align under the trained model puts
 * the bad ones at the top, so a wordlist can be re-read in the order most
 * likely to repay it. All five come out first, and the gap to the sixth is
 * larger than the whole spread of the forty good ones.
 *
 * Worth knowing what this does *not* test. The five bad daughters are drawn
 * from the daughter's inventory and share nothing with their protos; a wrong
 * judgement between two words that happen to resemble each other is not
 * findable this way and is not findable by any distributional method, which is
 * the reason cognacy is a judgement and not a measurement. An earlier draft of
 * the fixture built the bad daughters by displacing the proto's own reflexes,
 * and two of the five ranked below the median -- correctly, since a scrambled
 * right answer is mostly a right answer. */
static void test_the_outlier_ranking_finds_mistaken_cognate_judgements(rg_context *ctx) {
    rg_corpus *corpus = load("contaminated");
    rg_multi_model *model = train(ctx, corpus, 0);
    size_t count = 0;
    rg_cognate_outlier_row *rows = rank(ctx, corpus, model, &count);
    size_t i;
    int bad_in_top_five = 0;

    assert(count == 45);
    for (i = 0; i < 5; i++) {
        if (strncmp(rows[i].cognate_id, "bad", 3) == 0) {
            bad_in_top_five++;
        }
    }
    assert(bad_in_top_five == 5);
    /* And separated, not merely ordered: the last bad set scores well above
     * the first good one, so a reader stopping at the gap stops in the right
     * place. */
    assert(rows[4].z_score > rows[5].z_score + 1.0);

    rg_cognate_outlier_rows_free(rows, count);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* A compound cognate in its first element and not its second. Twenty of those
 * against twenty simplex sets that are cognate throughout.
 *
 * This is ordinary in a wordlist gathered by concept rather than by etymon:
 * the languages both name the thing with a compound and they do not use the
 * same second element. The set is not wrong -- half of it is evidence -- and
 * dropping it loses the half that is good.
 *
 * Both halves of what the corpus needs hold. The regular correspondences still
 * come out of the cognate elements, and every set carrying a non-cognate
 * element ranks in the worse half, so the ones to look at again are the ones
 * at the top. */
static void test_partial_cognacy_is_visible_in_the_ranking(rg_context *ctx) {
    rg_corpus *corpus = load("partial");
    rg_multi_model *model = train(ctx, corpus, 0);
    size_t count = 0;
    rg_cognate_outlier_row *rows = rank(ctx, corpus, model, &count);
    size_t i;
    int compounds_in_worse_half = 0;

    assert(count == 40);
    for (i = 0; i < 20; i++) {
        if (rows[i].cognate_id[0] == 'c') {
            compounds_in_worse_half++;
        }
    }
    assert(compounds_in_worse_half == 20);

    /* The cognate halves still carry the correspondence set. */
    assert(has_correspondence(model, "p", "f"));
    assert(has_correspondence(model, "t", "\xce\xb8"));
    assert(has_correspondence(model, "k", "x"));

    rg_cognate_outlier_rows_free(rows, count);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* One language, two transcription conventions, and what training makes of it.
 *
 * The `narrow` lect writes what `broad` writes as single graphemes the way a
 * different source would: the affricate as a stop plus a fricative, aspiration
 * as a following /h/, a long vowel as two vowels. Every form is the same word.
 * There is no sound change in this corpus at all.
 *
 * This is the commonest way a comparative dataset goes wrong and the least
 * visible one. Both transcriptions are valid IPA and every grapheme resolves,
 * so nothing refuses to load -- the two sources simply disagree about where a
 * segment ends. What comes out of *training* is a set
 * of clean, well-supported correspondences that a reader will take for three
 * well-attested sound changes: deaffrication, loss of aspiration, loss of
 * vowel length. None of them happened.
 *
 * The shuffled baseline does not catch it either, and cannot. A baseline
 * distinguishes a pattern from chance, and this pattern is not chance: it is
 * perfectly systematic, which is exactly what a sound law is. Both invented
 * conditioned rules stand above the noise floor with room to spare.
 *
 * So this asserts what training says about a corpus that is wrong, and it
 * still says it: the false correspondences are published, the fit is
 * excellent, and both invented conditioned rules stand above the noise floor.
 * That has not changed and should not -- given this corpus those are the right
 * answers to the questions training asks.
 *
 * What changed on 2026-08-17 is that a different question is now asked
 * somewhere else. `rg_find_transcription_drift`, asserted in the two tests
 * below, compares the lects' inventories and goes looking for the pieces; this
 * test is what it is protecting against. */
static void test_transcription_drift_is_reported_as_sound_change(rg_context *ctx) {
    rg_corpus *corpus = load("drift");
    rg_multi_model *model = train(ctx, corpus, 24);
    const rg_corpus_fit *fit = rg_multi_model_fit(model);

    /* The three "changes", every one of them an artefact of segmentation. */
    assert(has_correspondence(model, "t\xca\x83", "\xca\x83"));  /* tʃ ~ ʃ  */
    assert(has_correspondence(model, "t\xca\xb0", "t"));         /* tʰ ~ t  */
    assert(has_correspondence(model, "k\xca\xb0", "k"));         /* kʰ ~ k  */
    assert(has_correspondence(model, "i\xcb\x90", "i"));         /* iː ~ i  */
    assert(has_correspondence(model, "a\xcb\x90", "a"));         /* aː ~ a  */

    /* And the corpus looks like excellent data by every measure regulae has. */
    assert(fit->cost_per_segment_z < -10.0);
    assert(fit->rules_measured > 0);
    assert(fit->rules_above_noise == fit->rules_measured);

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* And now something catches it.
 *
 * `regulae check` reported nothing on this corpus until 2026-08-17, because
 * every grapheme in it resolves and that was the only question being asked.
 * The question that finds drift is about the writing rather than about the
 * sounds: take a grapheme one lect uses and the other never does, work out
 * what the other lect would have to write instead, and go and see whether it
 * writes it.
 *
 * That last step is what keeps this from being a resemblance heuristic. Two
 * different languages, one of which lost its affricates, have exactly the
 * inventory asymmetry that drift has. What they do not have is the *same
 * cognate sets* showing the pieces in the same order, which is why the row
 * carries a ratio and not a verdict. On this corpus every row is at 1.0
 * because every word is the same word; on Kessler's Latin/French wordlist the
 * one row it produces -- Latin `kʷ` against French `k w` -- sits at 1 of 5,
 * which is what a sound change looks like.
 *
 * All three kinds are asserted, because they are found three different ways
 * and only the first falls out of segmentation. */
static void test_transcription_drift_is_found_by_what_the_other_lect_writes(rg_context *ctx) {
    rg_corpus *corpus = load("drift");
    rg_transcription_drift_row *rows = 0;
    size_t count = 0;
    size_t i;
    int affricate = 0;
    int aspiration = 0;
    int length = 0;

    assert(rg_find_transcription_drift(ctx, rg_corpus_cognate_at(corpus, 0),
                                       rg_corpus_cognate_count(corpus),
                                       &rows, &count) == RG_OK);
    assert(count == 5);
    for (i = 0; i < count; i++) {
        /* Every one of them is the same word written two ways, so the other
         * lect writes the pieces in every set where this one writes the
         * whole. */
        assert(rows[i].corroborated == rows[i].forms);
        assert(rows[i].corroborated > 0);
        assert(strcmp(rows[i].lect, "broad") == 0);
        assert(strcmp(rows[i].other_lect, "narrow") == 0);
        if (rows[i].kind == RG_DRIFT_SEGMENTATION) {
            affricate = 1;
            assert(strcmp(rows[i].grapheme, "t\xca\x83") == 0);
            assert(strcmp(rows[i].written_as, "t \xca\x83") == 0);
        }
        if (rows[i].kind == RG_DRIFT_MODIFIER) {
            aspiration = 1;
        }
        if (rows[i].kind == RG_DRIFT_LENGTH) {
            length = 1;
        }
    }
    assert(affricate && aspiration && length);
    /* Strongest evidence first, so a reader with four hundred rows can read
     * the top of the list. */
    for (i = 1; i < count; i++) {
        assert(rows[i - 1].corroborated >= rows[i].corroborated);
    }
    rg_transcription_drift_rows_free(rows, count);
    rg_corpus_free(corpus);
}

/* And it stays quiet on corpora that are merely different.
 *
 * The whole value of a diagnostic is in what it does *not* say. Every corpus
 * in this repository other than `drift.tsv` reports nothing, and that includes
 * the ones with the most inventory asymmetry there is -- `chance.tsv`, whose
 * two lects have no history between them at all, and `contact.tsv`, where the
 * borrower's inventory is a strict subset of the donor's by construction.
 * Those are exactly the shapes a similarity heuristic would fire on. */
static void test_drift_detection_is_quiet_on_corpora_that_are_merely_different(rg_context *ctx) {
    static const char *const clean[] = {
        "contaminated", "partial"
    };
    static const char *const restraint[] = {
        "chance", "contact", "stratum"
    };
    size_t i;

    for (i = 0; i < sizeof(clean) / sizeof(clean[0]); i++) {
        rg_corpus *corpus = load(clean[i]);
        rg_transcription_drift_row *rows = 0;
        size_t count = 1;
        assert(rg_find_transcription_drift(ctx, rg_corpus_cognate_at(corpus, 0),
                                           rg_corpus_cognate_count(corpus),
                                           &rows, &count) == RG_OK);
        assert(count == 0);
        rg_transcription_drift_rows_free(rows, count);
        rg_corpus_free(corpus);
    }
    for (i = 0; i < sizeof(restraint) / sizeof(restraint[0]); i++) {
        rg_corpus *corpus = 0;
        rg_transcription_drift_row *rows = 0;
        size_t count = 1;
        char path[512];
        snprintf(path, sizeof(path), "%s/testdata/restraint/%s.tsv", REGULAE_SOURCE_DIR, restraint[i]);
        assert(rg_corpus_load_tsv(path, 0, &corpus, 0) == RG_OK);
        assert(rg_find_transcription_drift(ctx, rg_corpus_cognate_at(corpus, 0),
                                           rg_corpus_cognate_count(corpus),
                                           &rows, &count) == RG_OK);
        assert(count == 0);
        rg_transcription_drift_rows_free(rows, count);
        rg_corpus_free(corpus);
    }
}

int main(void) {
    rg_context *ctx = 0;
    assert(rg_context_new_builtin(&ctx) == RG_OK);
    test_the_outlier_ranking_finds_mistaken_cognate_judgements(ctx);
    test_partial_cognacy_is_visible_in_the_ranking(ctx);
    test_transcription_drift_is_reported_as_sound_change(ctx);
    test_transcription_drift_is_found_by_what_the_other_lect_writes(ctx);
    test_drift_detection_is_quiet_on_corpora_that_are_merely_different(ctx);
    rg_context_free(ctx);
    printf("diagnostics tests passed\n");
    return 0;
}
