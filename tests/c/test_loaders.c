#include "regulae.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Loader behaviour, checked against the corpora the parity harness runs. */

static const rg_cognate_set *find_cognate(const rg_corpus *corpus, const char *cognate_id) {
    size_t i;
    for (i = 0; i < rg_corpus_cognate_count(corpus); i++) {
        const rg_cognate_set *set = rg_corpus_cognate_at(corpus, i);
        if (strcmp(set->cognate_id, cognate_id) == 0) {
            return set;
        }
    }
    return 0;
}

static const rg_form *form_for(const rg_cognate_set *set, const char *lect_id) {
    size_t i;
    for (i = 0; i < set->form_count; i++) {
        if (strcmp(set->forms[i].lect_id, lect_id) == 0) {
            return &set->forms[i].form;
        }
    }
    return 0;
}

static void test_tsv_grouping_and_order(void) {
    rg_corpus *corpus = 0;
    rg_tsv_load_options options;
    const rg_cognate_set *first;

    memset(&options, 0, sizeof(options));
    options.confidence_column = "confidence";
    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/parity/three_lect_basic.tsv", &options, &corpus) == RG_OK);
    assert(rg_corpus_cognate_count(corpus) == 6);

    /* Cognate sets keep first-appearance order, and so do the forms inside
     * them. */
    first = rg_corpus_cognate_at(corpus, 0);
    assert(strcmp(first->cognate_id, "c1") == 0);
    assert(first->form_count == 3);
    assert(strcmp(first->forms[0].lect_id, "alpha") == 0);
    assert(strcmp(first->forms[1].lect_id, "beta") == 0);
    assert(strcmp(first->forms[2].lect_id, "gamma") == 0);
    assert(first->forms[0].form.segment_count == 4);
    assert(strcmp(first->forms[0].form.segments[0].grapheme, "p") == 0);
    assert(first->confidence == 1.0);

    rg_corpus_free(corpus);
}

/* A cognate set takes the lowest confidence any of its rows reports, and a
 * zero there means zero training weight rather than "unspecified". */
static void test_tsv_confidence_is_the_minimum(void) {
    rg_corpus *corpus = 0;
    rg_tsv_load_options options;
    const rg_cognate_set *low;
    const rg_cognate_set *zero;

    memset(&options, 0, sizeof(options));
    options.confidence_column = "confidence";
    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/parity/partial_coverage.tsv", &options, &corpus) == RG_OK);

    low = find_cognate(corpus, "q5");
    assert(low != 0);
    assert(low->confidence == 0.5);

    zero = find_cognate(corpus, "q7");
    assert(zero != 0);
    assert(zero->confidence == 0.0);

    rg_corpus_free(corpus);
}

/* Ignoring the confidence column leaves every set at full weight. */
static void test_tsv_without_confidence_column(void) {
    rg_corpus *corpus = 0;
    const rg_cognate_set *low;

    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/parity/partial_coverage.tsv", 0, &corpus) == RG_OK);
    low = find_cognate(corpus, "q5");
    assert(low != 0);
    assert(low->confidence == 1.0);
    rg_corpus_free(corpus);
}

static void test_arcaverborum_morpheme_boundaries(void) {
    rg_corpus *corpus = 0;
    const rg_cognate_set *set;
    const rg_form *form;

    assert(rg_corpus_load_arcaverborum(REGULAE_SOURCE_DIR "/testdata/parity/morph_boundary.csv", 0, &corpus) == RG_OK);
    assert(rg_corpus_cognate_count(corpus) == 6);

    set = find_cognate(corpus, "b1");
    assert(set != 0);
    form = form_for(set, "one");
    assert(form != 0);
    /* "p a + t a" is four segments with a boundary after the second; the "+"
     * is a marker, not a segment. */
    assert(form->segment_count == 4);
    assert(strcmp(form->segments[2].grapheme, "t") == 0);
    assert(form->morpheme_break_count == 1);
    assert(form->morpheme_breaks[0] == 2);

    rg_corpus_free(corpus);
}

static void test_missing_file_and_columns(void) {
    rg_corpus *corpus = 0;
    rg_tsv_load_options options;

    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/parity/does_not_exist.tsv", 0, &corpus) == RG_ERR_IO);
    assert(corpus == 0);

    memset(&options, 0, sizeof(options));
    options.segments_column = "not_a_column";
    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/parity/three_lect_basic.tsv", &options, &corpus) == RG_ERR_PARSE);
    assert(corpus == 0);
}

/* A directed pairwise corpus lifts into cognate sets so two-lect studies use
 * the same entry point as everything else. */
static void test_corpus_from_pairs(void) {
    rg_segment source[2];
    rg_segment target[2];
    rg_form_pair pairs[2];
    rg_corpus *corpus = 0;
    const rg_cognate_set *set;
    size_t i;

    memset(source, 0, sizeof(source));
    memset(target, 0, sizeof(target));
    source[0].grapheme = "p";
    source[1].grapheme = "a";
    target[0].grapheme = "f";
    target[1].grapheme = "a";
    for (i = 0; i < 2; i++) {
        memset(&pairs[i], 0, sizeof(pairs[i]));
        pairs[i].source.lect_id = "ignored";
        pairs[i].source.segments = source;
        pairs[i].source.segment_count = 2;
        pairs[i].target.lect_id = "ignored";
        pairs[i].target.segments = target;
        pairs[i].target.segment_count = 2;
        pairs[i].weight = 1.0;
    }

    assert(rg_corpus_from_pairs(pairs, 2, "A", "B", 0, &corpus) == RG_OK);
    assert(rg_corpus_cognate_count(corpus) == 2);
    set = rg_corpus_cognate_at(corpus, 0);
    assert(strcmp(set->cognate_id, "pair.00000") == 0);
    assert(set->form_count == 2);
    assert(strcmp(set->forms[0].lect_id, "A") == 0);
    assert(strcmp(set->forms[1].lect_id, "B") == 0);
    assert(set->confidence == 1.0);
    assert(strcmp(rg_corpus_cognate_at(corpus, 1)->cognate_id, "pair.00001") == 0);
    rg_corpus_free(corpus);

    assert(rg_corpus_from_pairs(pairs, 2, "A", "B", "romance", &corpus) == RG_OK);
    assert(strcmp(rg_corpus_cognate_at(corpus, 0)->cognate_id, "romance.00000") == 0);
    rg_corpus_free(corpus);

    /* Both lects must be named, and they must differ. */
    assert(rg_corpus_from_pairs(pairs, 2, "A", "A", 0, &corpus) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_corpus_from_pairs(pairs, 2, "", "B", 0, &corpus) == RG_ERR_INVALID_ARGUMENT);
}

int main(void) {
    test_tsv_grouping_and_order();
    test_tsv_confidence_is_the_minimum();
    test_tsv_without_confidence_column();
    test_arcaverborum_morpheme_boundaries();
    test_missing_file_and_columns();
    test_corpus_from_pairs();
    printf("loader tests passed\n");
    return 0;
}
