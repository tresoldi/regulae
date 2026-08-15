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
    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/corpora/three_lect_basic.tsv", &options, &corpus, 0) == RG_OK);
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
    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/corpora/partial_coverage.tsv", &options, &corpus, 0) == RG_OK);

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

    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/corpora/partial_coverage.tsv", 0, &corpus, 0) == RG_OK);
    low = find_cognate(corpus, "q5");
    assert(low != 0);
    assert(low->confidence == 1.0);
    rg_corpus_free(corpus);
}

static void test_arcaverborum_morpheme_boundaries(void) {
    rg_corpus *corpus = 0;
    const rg_cognate_set *set;
    const rg_form *form;

    assert(rg_corpus_load_arcaverborum(REGULAE_SOURCE_DIR "/testdata/corpora/morph_boundary.csv", 0, &corpus, 0) == RG_OK);
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

    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/corpora/does_not_exist.tsv", 0, &corpus, 0) == RG_ERR_IO);
    assert(corpus == 0);

    memset(&options, 0, sizeof(options));
    options.segments_column = "not_a_column";
    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/corpora/three_lect_basic.tsv", &options, &corpus, 0) == RG_ERR_PARSE);
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

/* The wide loader segments through merkmal; the long fixture beside it was
 * produced by splitting the same words on characters. That fixture is
 * parity-verified against the Go reference, so agreeing with it here checks the
 * segmentation bridge against a known-good result rather than against my own
 * expectations. */
static void test_wide_matches_the_parity_verified_corpus(rg_context *ctx) {
    rg_corpus *wide = 0;
    rg_corpus *long_form = 0;
    rg_tsv_load_options tsv_options;
    size_t i;

    memset(&tsv_options, 0, sizeof(tsv_options));
    tsv_options.confidence_column = "confidence";
    assert(rg_corpus_load_wide_tsv(ctx, REGULAE_SOURCE_DIR "/experiments/latin_spanish/cognates.tsv", 0, &wide, 0) == RG_OK);
    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/corpora/real_latin_spanish.tsv", &tsv_options, &long_form, 0) == RG_OK);
    assert(rg_corpus_cognate_count(wide) == rg_corpus_cognate_count(long_form));
    assert(rg_corpus_cognate_count(wide) > 90);

    for (i = 0; i < rg_corpus_cognate_count(wide); i++) {
        const rg_cognate_set *a = rg_corpus_cognate_at(wide, i);
        const rg_cognate_set *b = rg_corpus_cognate_at(long_form, i);
        size_t f;
        assert(a->form_count == b->form_count);
        for (f = 0; f < a->form_count; f++) {
            size_t g;
            assert(strcmp(a->forms[f].lect_id, b->forms[f].lect_id) == 0);
            assert(a->forms[f].form.segment_count == b->forms[f].form.segment_count);
            for (g = 0; g < a->forms[f].form.segment_count; g++) {
                assert(strcmp(a->forms[f].form.segments[g].grapheme,
                              b->forms[f].form.segments[g].grapheme) == 0);
            }
        }
    }
    rg_corpus_free(wide);
    rg_corpus_free(long_form);
}

/* Rows glossed the same are distinct cognate sets, not one set to merge. */
static void test_wide_repeated_gloss_is_not_merged(rg_context *ctx) {
    rg_corpus *corpus = 0;
    size_t i;
    size_t die_like = 0;

    assert(rg_corpus_load_wide_tsv(ctx, REGULAE_SOURCE_DIR "/experiments/latin_spanish/cognates.tsv", 0, &corpus, 0) == RG_OK);
    for (i = 0; i < rg_corpus_cognate_count(corpus); i++) {
        const char *id = rg_corpus_cognate_at(corpus, i)->cognate_id;
        if (strncmp(id, "die", 3) == 0) {
            die_like++;
        }
    }
    /* The corpus glosses two separate rows "die"; both must survive. */
    assert(die_like == 2);
    rg_corpus_free(corpus);
}

/* "<lect>_breaks" carries morpheme boundaries; "<lect>_tone" is recognised so
 * it is not mistaken for a lect, but tone is not carried through yet. */
static void test_wide_breaks_and_column_conventions(rg_context *ctx) {
    rg_corpus *corpus = 0;
    size_t i;
    size_t with_breaks = 0;

    assert(rg_corpus_load_wide_tsv(ctx, REGULAE_SOURCE_DIR "/experiments/latin_spanish/cognates.tsv", 0, &corpus, 0) == RG_OK);
    for (i = 0; i < rg_corpus_cognate_count(corpus); i++) {
        const rg_cognate_set *set = rg_corpus_cognate_at(corpus, i);
        size_t f;
        /* The "_breaks" columns must not have become lects of their own. */
        assert(set->form_count <= 2);
        for (f = 0; f < set->form_count; f++) {
            assert(strcmp(set->forms[f].lect_id, "latin") == 0 ||
                   strcmp(set->forms[f].lect_id, "spanish") == 0);
            if (set->forms[f].form.morpheme_break_count > 0) {
                with_breaks++;
                assert(set->forms[f].form.morpheme_breaks[0] > 0);
            }
        }
    }
    /* The corpus carries boundaries on 18 rows; the Go experiments never read
     * them, so this is the first consumer. */
    assert(with_breaks > 0);
    rg_corpus_free(corpus);

    assert(rg_corpus_load_wide_tsv(ctx, REGULAE_SOURCE_DIR "/experiments/mandarin_historical/cognates.tsv", 0, &corpus, 0) == RG_OK);
    for (i = 0; i < rg_corpus_cognate_count(corpus); i++) {
        const rg_cognate_set *set = rg_corpus_cognate_at(corpus, i);
        size_t f;
        for (f = 0; f < set->form_count; f++) {
            /* mc_tone and md_tone must not be treated as lects. */
            assert(strcmp(set->forms[f].lect_id, "middle_chinese") == 0 ||
                   strcmp(set->forms[f].lect_id, "mandarin") == 0);
        }
    }
    rg_corpus_free(corpus);
}

/* Tone reaches a form by either route: written on the word, or annotated in a
 * companion column when the corpus records tone categories rather than pitch.
 * The column wins where both are present, since it is the deliberate one. */
static void test_wide_carries_tone(rg_context *ctx) {
    rg_corpus *corpus = 0;
    const rg_cognate_set *set;
    size_t f;
    size_t toned = 0;

    assert(rg_corpus_load_wide_tsv(
        ctx, REGULAE_SOURCE_DIR "/experiments/tone_vietnamese_like/cognates.tsv",
        0, &corpus, 0) == RG_OK);
    assert(rg_corpus_cognate_count(corpus) > 0);
    set = rg_corpus_cognate_at(corpus, 0);
    for (f = 0; f < set->form_count; f++) {
        size_t g;
        /* The "_tone" columns must not have become lects of their own. */
        assert(strcmp(set->forms[f].lect_id, "hanoi") == 0 ||
               strcmp(set->forms[f].lect_id, "saigon") == 0);
        for (g = 0; g < set->forms[f].form.segment_count; g++) {
            if (set->forms[f].form.segments[g].tone != 0) {
                toned++;
            }
        }
    }
    assert(toned == 2);
    rg_corpus_free(corpus);

    /* The same corpus written with Chao superscripts on the word instead. */
    assert(rg_corpus_load_wide_tsv(
        ctx, REGULAE_SOURCE_DIR "/experiments/tone_synthetic/cognates.tsv",
        0, &corpus, 0) == RG_OK);
    set = rg_corpus_cognate_at(corpus, 0);
    for (f = 0; f < set->form_count; f++) {
        size_t g;
        int found = 0;
        for (g = 0; g < set->forms[f].form.segment_count; g++) {
            const rg_segment *segment = &set->forms[f].form.segments[g];
            /* Tone is its own dimension: it never stays in the grapheme. */
            assert(strstr(segment->grapheme, "\xe2\x81\xb5") == 0);
            if (segment->tone != 0) {
                found = 1;
            }
        }
        assert(found);
    }
    rg_corpus_free(corpus);
}

static void test_wide_confidence_and_bad_input(rg_context *ctx) {
    rg_corpus *corpus = 0;
    size_t i;
    int saw_low = 0;

    assert(rg_corpus_load_wide_tsv(ctx, REGULAE_SOURCE_DIR "/experiments/contaminated_cognates_synthetic/cognates.tsv", 0, &corpus, 0) == RG_OK);
    for (i = 0; i < rg_corpus_cognate_count(corpus); i++) {
        if (rg_corpus_cognate_at(corpus, i)->confidence < 1.0) {
            saw_low = 1;
        }
    }
    assert(saw_low);
    rg_corpus_free(corpus);

    assert(rg_corpus_load_wide_tsv(ctx, REGULAE_SOURCE_DIR "/no_such_file.tsv", 0, &corpus, 0) == RG_ERR_IO);
    assert(corpus == 0);
    assert(rg_corpus_load_wide_tsv(0, REGULAE_SOURCE_DIR "/experiments/latin_spanish/cognates.tsv", 0, &corpus, 0) == RG_ERR_INVALID_ARGUMENT);
}

/* Segmentation must respect multi-codepoint graphemes rather than splitting
 * on characters, which is the whole reason it goes through merkmal. */
static void test_segment_word(rg_context *ctx) {
    rg_segment *segments = 0;
    size_t count = 0;

    assert(rg_context_segment_word(ctx, "pater", &segments, &count) == RG_OK);
    assert(count == 5);
    assert(strcmp(segments[0].grapheme, "p") == 0);
    assert(strcmp(segments[4].grapheme, "r") == 0);
    rg_segments_free(segments, count);

    /* Multi-codepoint graphemes stay whole. Splitting on characters would
     * break every one of these, which is why segmentation goes through
     * merkmal rather than being done by the caller. */
    assert(rg_context_segment_word(ctx, "p\xca\xb0" "a", &segments, &count) == RG_OK);
    assert(count == 2);
    assert(strcmp(segments[0].grapheme, "p\xca\xb0") == 0);
    rg_segments_free(segments, count);

    /* Tie-bar affricate. */
    assert(rg_context_segment_word(ctx, "t\xcd\xa1\xca\x83" "a", &segments, &count) == RG_OK);
    assert(count == 2);
    rg_segments_free(segments, count);

    /* A combining diacritic attaches to its base. This is exactly the case
     * that had to be hand-corrected when the Romance parity corpus was built
     * by splitting on characters. */
    assert(rg_context_segment_word(ctx, "e\xcc\x83", &segments, &count) == RG_OK);
    assert(count == 1);
    rg_segments_free(segments, count);

    assert(rg_context_segment_word(ctx, "", &segments, &count) == RG_OK);
    assert(count == 0);
    rg_segments_free(segments, count);
}

/* The parse variants must produce exactly what the path variants do; the
 * WebAssembly build has no filesystem and reaches the library only through
 * them, so a divergence here would show up only in the browser. */
static void test_parse_matches_load(rg_context *ctx) {
    static const char *const wide_text =
        "gloss\tlatin\tspanish\tlatin_breaks\tspanish_breaks\n"
        "father\tpater\tpadre\t-\t-\n"
        "woman\tfemina\tember\t-\t3\n";
    static const char *const long_text =
        "cognate_id\tlect_id\tsegments\tconfidence\n"
        "c1\talpha\tp a t a\t1.0\n"
        "c1\tbeta\tf a t a\t0.5\n";
    rg_corpus *from_text = 0;
    rg_corpus *from_path = 0;
    rg_tsv_load_options tsv_options;
    size_t i;

    /* Wide: parsed text against the same file on disk. */
    assert(rg_corpus_parse_wide_tsv(ctx, wide_text, 0, &from_text, 0) == RG_OK);
    assert(rg_corpus_cognate_count(from_text) == 2);
    {
        const rg_cognate_set *woman = rg_corpus_cognate_at(from_text, 1);
        const rg_form *spanish = form_for(woman, "spanish");
        assert(spanish != 0);
        assert(spanish->morpheme_break_count == 1);
        assert(spanish->morpheme_breaks[0] == 3);
    }
    rg_corpus_free(from_text);

    assert(rg_corpus_load_wide_tsv(ctx, REGULAE_SOURCE_DIR "/experiments/latin_spanish/cognates.tsv", 0, &from_path, 0) == RG_OK);
    {
        char *text = 0;
        long size;
        FILE *fh = fopen(REGULAE_SOURCE_DIR "/experiments/latin_spanish/cognates.tsv", "rb");
        assert(fh != 0);
        fseek(fh, 0, SEEK_END);
        size = ftell(fh);
        fseek(fh, 0, SEEK_SET);
        text = (char *)malloc((size_t)size + 1);
        assert(text != 0);
        assert(fread(text, 1, (size_t)size, fh) == (size_t)size);
        text[size] = '\0';
        fclose(fh);
        assert(rg_corpus_parse_wide_tsv(ctx, text, 0, &from_text, 0) == RG_OK);
        free(text);
    }
    assert(rg_corpus_cognate_count(from_text) == rg_corpus_cognate_count(from_path));
    for (i = 0; i < rg_corpus_cognate_count(from_text); i++) {
        const rg_cognate_set *a = rg_corpus_cognate_at(from_text, i);
        const rg_cognate_set *b = rg_corpus_cognate_at(from_path, i);
        size_t f;
        assert(strcmp(a->cognate_id, b->cognate_id) == 0);
        assert(a->form_count == b->form_count);
        for (f = 0; f < a->form_count; f++) {
            size_t g;
            assert(a->forms[f].form.segment_count == b->forms[f].form.segment_count);
            for (g = 0; g < a->forms[f].form.segment_count; g++) {
                assert(strcmp(a->forms[f].form.segments[g].grapheme,
                              b->forms[f].form.segments[g].grapheme) == 0);
            }
        }
    }
    rg_corpus_free(from_text);
    rg_corpus_free(from_path);

    /* Long format, including the minimum-confidence rule. */
    memset(&tsv_options, 0, sizeof(tsv_options));
    tsv_options.confidence_column = "confidence";
    assert(rg_corpus_parse_tsv(long_text, &tsv_options, &from_text, 0) == RG_OK);
    assert(rg_corpus_cognate_count(from_text) == 1);
    assert(rg_corpus_cognate_at(from_text, 0)->confidence == 0.5);
    rg_corpus_free(from_text);

    /* Arcaverborum text has no extension to sniff, so it defaults to comma. */
    assert(rg_corpus_parse_arcaverborum("Language_ID,Segments,Cognacy\n"
                                        "one,p a + t a,b1\n"
                                        "two,f a + t a,b1\n", 0, &from_text, 0) == RG_OK);
    assert(rg_corpus_cognate_count(from_text) == 1);
    assert(rg_corpus_cognate_at(from_text, 0)->forms[0].form.morpheme_break_count == 1);
    rg_corpus_free(from_text);

    {
        rg_arcaverborum_load_options arca;
        memset(&arca, 0, sizeof(arca));
        arca.delimiter = '\t';
        assert(rg_corpus_parse_arcaverborum("Language_ID\tSegments\tCognacy\n"
                                            "one\tp a\tb1\n"
                                            "two\tf a\tb1\n", &arca, &from_text, 0) == RG_OK);
        assert(rg_corpus_cognate_count(from_text) == 1);
        rg_corpus_free(from_text);
    }

    /* Neither a path nor text is a caller error, not a crash. */
    assert(rg_corpus_parse_tsv(0, 0, &from_text, 0) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_corpus_parse_wide_tsv(ctx, 0, 0, &from_text, 0) == RG_ERR_INVALID_ARGUMENT);
}

/* Tone is a suprasegmental in its own column, positionally parallel to the
 * segments cell, because merkmal's segmenter cannot merge Chao digits back
 * onto their vowel. Without this column the tonal table and the whole
 * cross-dimensional stage are unreachable from a corpus file. */
static void test_tsv_tone_column_attaches_by_position(void) {
    static const char *const text =
        "cognate_id\tlect_id\tsegments\ttone\n"
        "c1\tsrc\tb a\t- 2\n"
        "c1\ttgt\tb a\t- 4\n"
        "c2\tsrc\tp a n\t- 1 -\n";
    rg_corpus *corpus = 0;
    const rg_cognate_set *set;
    const rg_form *form;

    assert(rg_corpus_parse_tsv(text, 0, &corpus, 0) == RG_OK);

    set = find_cognate(corpus, "c1");
    assert(set != 0);
    form = form_for(set, "src");
    assert(form != 0 && form->segment_count == 2);
    assert(form->segments[0].tone == 0 || form->segments[0].tone[0] == '\0');
    assert(form->segments[1].tone != 0 && strcmp(form->segments[1].tone, "2") == 0);
    form = form_for(set, "tgt");
    assert(form->segments[1].tone != 0 && strcmp(form->segments[1].tone, "4") == 0);

    set = find_cognate(corpus, "c2");
    form = form_for(set, "src");
    assert(form->segment_count == 3);
    assert(form->segments[1].tone != 0 && strcmp(form->segments[1].tone, "1") == 0);
    assert(form->segments[2].tone == 0 || form->segments[2].tone[0] == '\0');
    rg_corpus_free(corpus);
}

/* A corpus that annotates tone is doing so deliberately, so a row whose tone
 * count disagrees with its segment count is an error rather than a truncation
 * that would tone the wrong vowel. */
static void test_tsv_tone_length_mismatch_is_refused(void) {
    static const char *const text =
        "cognate_id\tlect_id\tsegments\ttone\n"
        "c1\tsrc\tb a\t- 2 3\n";
    rg_corpus *corpus = 0;
    assert(rg_corpus_parse_tsv(text, 0, &corpus, 0) == RG_ERR_PARSE);
    assert(corpus == 0);
}

/* A corpus with no tone column keeps every segment untoned, so adding the
 * column changes nothing for the corpora that predate it. */
static void test_tsv_without_tone_column_is_untoned(void) {
    static const char *const text =
        "cognate_id\tlect_id\tsegments\n"
        "c1\tsrc\tb a\n";
    rg_corpus *corpus = 0;
    const rg_form *form;

    assert(rg_corpus_parse_tsv(text, 0, &corpus, 0) == RG_OK);
    form = form_for(find_cognate(corpus, "c1"), "src");
    assert(form->segments[0].tone == 0 || form->segments[0].tone[0] == '\0');
    assert(form->segments[1].tone == 0 || form->segments[1].tone[0] == '\0');
    rg_corpus_free(corpus);
}

/* The design docs told users to supply their own syllable breaks where their
 * language's phonotactics differ from the sonority default. No loader could
 * read them until 2026-08-15, so the escape hatch was reachable only from C. */
static void test_syllable_breaks_column(void) {
    rg_context *ctx = 0;
    rg_corpus *corpus = 0;
    const rg_cognate_set *set;

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/corpora/syllable_breaks_column.tsv",
                              0, &corpus, 0) == RG_OK);
    set = rg_corpus_cognate_at(corpus, 0);
    assert(set != 0);
    assert(set->form_count == 2);
    assert(set->forms[0].form.syllable_break_count == 1);
    assert(set->forms[0].form.syllable_breaks[0] == 3);
    assert(set->forms[1].form.syllable_break_count == 1);
    assert(set->forms[1].form.syllable_breaks[0] == 2);
    rg_corpus_free(corpus);
    rg_context_free(ctx);
}

/* A lect with two reflexes in one cognate set is a doublet, and it is a fact
 * about the language rather than an error in the file: 3.2% of cognate-set
 * members across the Lexibank datasets with expert judgements. Until
 * 2026-08-15 regulae had four behaviours for it -- the generic loader errored,
 * GLED and arcaverborum kept the first, the wide loader skipped, and the C API
 * dropped the second silently.
 *
 * The set now reads as one set per combination of reflexes, each carrying its
 * share of the confidence, so both reflexes are counted and neither is a
 * second vote. */
static void test_doublets_expand_into_weighted_sets(void) {
    rg_corpus *corpus = 0;
    size_t i;
    double proto_total = 0.0;
    int saw_d = 0;
    int saw_l = 0;

    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/corpora/doublet.tsv", 0, &corpus, 0) == RG_OK);
    assert(rg_corpus_doublet_set_count(corpus) == 1);
    assert(rg_corpus_doublet_expansion_count(corpus) == 1);
    /* Two readings of w1 plus the plain w2. */
    assert(rg_corpus_cognate_count(corpus) == 3);
    for (i = 0; i < rg_corpus_cognate_count(corpus); i++) {
        const rg_cognate_set *set = rg_corpus_cognate_at(corpus, i);
        size_t f;
        assert(set->form_count == 2);
        if (strcmp(set->cognate_id, "w1") == 0) {
            /* Split in two, so each reading is worth half a set. */
            assert(set->confidence == 0.5);
            proto_total += set->confidence;
        }
        for (f = 0; f < set->form_count; f++) {
            if (strcmp(set->forms[f].lect_id, "daughter") != 0) {
                continue;
            }
            if (strcmp(set->forms[f].form.segments[2].grapheme, "d") == 0) {
                saw_d = 1;
            }
            if (strcmp(set->forms[f].form.segments[2].grapheme, "l") == 0) {
                saw_l = 1;
            }
        }
    }
    /* Both reflexes present, and between them worth one set. */
    assert(saw_d && saw_l);
    assert(proto_total == 1.0);
    rg_corpus_free(corpus);
}

/* "parse error" names neither the line nor the reason, on a file that may have
 * ten thousand rows. */
static void test_load_failure_names_the_reason(void) {
    rg_corpus *corpus = 0;
    rg_load_diagnosis diagnosis;

    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/corpora/missing_lect_column.tsv",
                              0, &corpus, &diagnosis) == RG_ERR_PARSE);
    assert(strstr(diagnosis.message, "lect_id") != 0);
    assert(diagnosis.line == 1);
}

/* The diagnosis belongs to the call, not to the process.
 *
 * While it was a file-scope buffer, two loads shared one, and only the load_*
 * entry points cleared it -- so a parse_* that succeeded after any earlier
 * failure still reported that failure to anyone who asked. Both are stated
 * here, because neither is visible from a single successful load. */
static void test_a_diagnosis_belongs_to_its_own_call(void) {
    rg_corpus *first = 0;
    rg_corpus *second = 0;
    rg_corpus *good = 0;
    rg_load_diagnosis a;
    rg_load_diagnosis b;
    rg_load_diagnosis after;
    rg_tsv_load_options options;

    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/corpora/missing_lect_column.tsv",
                              0, &first, &a) == RG_ERR_PARSE);
    assert(rg_corpus_parse_tsv("", 0, &second, &b) != RG_OK);
    /* The first call's diagnosis is still the first call's. */
    assert(strstr(a.message, "lect_id") != 0);

    /* And a success does not hand back the last failure. */
    memset(&options, 0, sizeof(options));
    assert(rg_corpus_parse_tsv("cognate_id\tlect_id\tsegments\n"
                               "c1\tone\tp a\n"
                               "c1\ttwo\tf a\n", &options, &good, &after) == RG_OK);
    assert(after.message[0] == '\0');
    assert(after.line == 0);
    rg_corpus_free(good);
}

int main(void) {
    rg_context *ctx = 0;
    assert(rg_context_new_builtin(&ctx) == RG_OK);
    test_segment_word(ctx);
    test_wide_matches_the_parity_verified_corpus(ctx);
    test_wide_repeated_gloss_is_not_merged(ctx);
    test_wide_breaks_and_column_conventions(ctx);
    test_wide_carries_tone(ctx);
    test_wide_confidence_and_bad_input(ctx);
    test_parse_matches_load(ctx);
    rg_context_free(ctx);
    test_tsv_grouping_and_order();
    test_tsv_tone_column_attaches_by_position();
    test_tsv_tone_length_mismatch_is_refused();
    test_tsv_without_tone_column_is_untoned();
    test_tsv_confidence_is_the_minimum();
    test_tsv_without_confidence_column();
    test_arcaverborum_morpheme_boundaries();
    test_missing_file_and_columns();
    test_corpus_from_pairs();
    test_syllable_breaks_column();
    test_doublets_expand_into_weighted_sets();
    test_load_failure_names_the_reason();
    test_a_diagnosis_belongs_to_its_own_call();
    printf("loader tests passed\n");
    return 0;
}
