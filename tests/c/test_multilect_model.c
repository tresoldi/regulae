#include "regulae.h"
#include "table_access.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The test allocated these strings and stored them in fields the API declares
 * `const char *`, because that is what they are to a reader of a form. Freeing
 * them means taking the qualifier back off; going through a copy of the pointer
 * value keeps that defined. The library does the same thing in one place, as
 * rg_free_owned_internal. */
static void free_owned(const void *owned) {
    void *value;
    memcpy(&value, &owned, sizeof(value));
    free(value);
}

static rg_form form(const char *lect, const rg_segment *segments, size_t count) {
    rg_form f;
    f.lect_id = lect;
    f.segments = segments;
    f.segment_count = count;
    f.syllable_breaks = 0;
    f.syllable_break_count = 0;
    f.morpheme_breaks = 0;
    f.morpheme_break_count = 0;
    return f;
}

static char *dup_string(const char *value) {
    size_t len = strlen(value);
    char *out = (char *)malloc(len + 1);
    assert(out != 0);
    memcpy(out, value, len + 1);
    return out;
}

static rg_segment *segments_from_ascii(const char *word, size_t *count) {
    size_t i;
    size_t n = strlen(word);
    rg_segment *segments = (rg_segment *)calloc(n, sizeof(*segments));
    assert(segments != 0);
    for (i = 0; i < n; i++) {
        char g[2];
        g[0] = word[i];
        g[1] = '\0';
        segments[i].grapheme = dup_string(g);
    }
    *count = n;
    return segments;
}

static void segments_free(rg_segment *segments, size_t count) {
    size_t i;
    for (i = 0; i < count; i++) {
        free_owned(segments[i].grapheme);
    }
    free(segments);
}

static int class_has(const rg_multi_class_row *row, const char *lect, const char *grapheme) {
    size_t i;
    for (i = 0; i < row->segment_count; i++) {
        if (strcmp(row->lect_ids[i], lect) == 0 && strcmp(row->graphemes[i], grapheme) == 0) {
            return 1;
        }
    }
    return 0;
}

static int no_duplicate_support(const rg_multi_class_row *row) {
    size_t i;
    size_t j;
    for (i = 0; i < row->supporting_cognate_count; i++) {
        for (j = i + 1; j < row->supporting_cognate_count; j++) {
            if (strcmp(row->supporting_cognates[i], row->supporting_cognates[j]) == 0) {
                return 0;
            }
        }
    }
    return 1;
}

static int model_has_class(
    const rg_multi_model *model,
    const char *lect_a,
    const char *graph_a,
    const char *lect_b,
    const char *graph_b,
    const char *lect_c,
    const char *graph_c,
    double count
) {
    size_t i;
    for (i = 0; i < rg_multi_model_unconditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_unconditioned_class_at(model, i);
        if (row->segment_count == 3 &&
            fabs(row->count - count) < 1e-12 &&
            class_has(row, lect_a, graph_a) &&
            class_has(row, lect_b, graph_b) &&
            class_has(row, lect_c, graph_c)) {
            assert(row->confidence == 1.0);
            /* Every set in this fixture realises the class once, so the two
             * numbers coincide here. They are not the same number -- see
             * test_supporting_sets_are_distinct, where they separate. */
            assert(row->supporting_cognate_count == (size_t)count);
            assert(row->supporting_cognates != 0);
            assert(no_duplicate_support(row));
            assert(row->uncertainty.lower <= row->uncertainty.estimate + 1e-12);
            assert(row->uncertainty.upper + 1e-12 >= row->uncertainty.estimate);
            return 1;
        }
    }
    return 0;
}

static void test_conditioned_palatalization(rg_context *ctx, const rg_train_options *options) {
    const char *source_words[16] = {
        "kita", "kite", "ketu", "keri", "kina", "keta", "kile", "kise",
        "kata", "koto", "kupa", "kala", "koma", "kuma", "kota", "kapa"
    };
    const char *target_words[16] = {
        "sita", "site", "setu", "seri", "sina", "seta", "sile", "sise",
        "kata", "koto", "kupa", "kala", "koma", "kuma", "kota", "kapa"
    };
    rg_segment *source_segments[16];
    rg_segment *target_segments[16];
    size_t source_counts[16];
    size_t target_counts[16];
    rg_cognate_form forms[16][2];
    rg_cognate_set cognates[16];
    rg_multi_model *model = 0;
    rg_train_options local_options = *options;
    size_t i;
    int found = 0;

    memset(cognates, 0, sizeof(cognates));
    local_options.max_chunk_size = 1;

    for (i = 0; i < 16; i++) {
        source_segments[i] = segments_from_ascii(source_words[i], &source_counts[i]);
        target_segments[i] = segments_from_ascii(target_words[i], &target_counts[i]);
        forms[i][0].lect_id = "A";
        forms[i][0].form = form("A", source_segments[i], source_counts[i]);
        forms[i][1].lect_id = "B";
        forms[i][1].form = form("B", target_segments[i], target_counts[i]);
        cognates[i].cognate_id = source_words[i];
        cognates[i].forms = forms[i];
        cognates[i].form_count = 2;
        cognates[i].confidence = 1.0;
    }

    assert(rg_train_model(ctx, cognates, 16, &local_options, &model) == RG_OK);
    assert(model != 0);
    assert(rg_multi_model_conditioned_class_count(model) > 0);
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
        assert(row != 0);
        assert(row->contexts != 0);
        assert(row->class_id >= (int)rg_multi_model_unconditioned_class_count(model));
        /* A committed rule has to name the sets it was committed on. Until ABI
         * 30 every conditioned class published an empty list, so the rows that
         * are decisions were the ones a reader could not check. */
        assert(row->supporting_cognate_count > 0);
        assert(row->supporting_cognates != 0);
        assert(no_duplicate_support(row));
        assert(row->supporting_cognate_count <= (size_t)row->count);
        if (row->segment_count == 2 &&
            strcmp(row->lect_ids[0], "A") == 0 &&
            strcmp(row->lect_ids[1], "B") == 0 &&
            strcmp(row->graphemes[0], "k") == 0 &&
            strcmp(row->graphemes[1], "s") == 0 &&
            row->contexts[0].following_count > 0 &&
            row->count >= 8.0 &&
            fabs(row->confidence - 0.5) < 1e-9) {
            /* k~s holds before a front vowel; where it does not, A:k takes B:k.
             * That contrast is what makes the split a split, and it must be
             * reachable -- a class id, not the ~0 same-reflex contrast_count. */
            const rg_multi_class_row *contrast;
            assert(row->contrast_class_id >= 0);
            assert((size_t)row->contrast_class_id < rg_multi_model_unconditioned_class_count(model));
            assert(row->contrast_alternative_count > 0.0);
            contrast = rg_multi_model_unconditioned_class_at(model, (size_t)row->contrast_class_id);
            assert(contrast != 0);
            assert(contrast->segment_count == 2);
            assert(strcmp(contrast->graphemes[0], "k") == 0);
            assert(strcmp(contrast->graphemes[1], "k") == 0);
            found = 1;
        }
    }
    assert(found);
    /* An unconditioned class has no environment, so no contrast row. */
    for (i = 0; i < rg_multi_model_unconditioned_class_count(model); i++) {
        assert(rg_multi_model_unconditioned_class_at(model, i)->contrast_class_id == -1);
    }
    assert(rg_multi_model_conditioned_class_at(model, 1000) == 0);
    rg_multi_model_free(model);
    for (i = 0; i < 16; i++) {
        segments_free(source_segments[i], source_counts[i]);
        segments_free(target_segments[i], target_counts[i]);
    }
}

/* The two numbers a reader can confuse, on a corpus built so they cannot
 * coincide. Every word realises p~f twice, so the class is worth twelve aligned
 * positions and rests on six cognate sets.
 *
 * `count` is easily misread as a count of words, and on
 * real corpora that reading inflates a row's support by however often a word
 * happens to repeat a segment. The distinct-set number is what is wanted, and
 * `supporting_cognates` publishes each id once, so its length reports the
 * distinct sets rather than agreeing with `count`. */
static void test_supporting_sets_are_distinct(rg_context *ctx, const rg_train_options *options) {
    const char *source_words[6] = {"papa", "pipi", "pupu", "pepe", "popo", "papi"};
    const char *target_words[6] = {"fafa", "fifi", "fufu", "fefe", "fofo", "fafi"};
    rg_segment *source_segments[6];
    rg_segment *target_segments[6];
    size_t source_counts[6];
    size_t target_counts[6];
    rg_cognate_form forms[6][2];
    rg_cognate_set cognates[6];
    rg_multi_model *model = 0;
    rg_train_options local_options = *options;
    size_t i;
    int found = 0;

    memset(cognates, 0, sizeof(cognates));
    local_options.max_chunk_size = 1;

    for (i = 0; i < 6; i++) {
        source_segments[i] = segments_from_ascii(source_words[i], &source_counts[i]);
        target_segments[i] = segments_from_ascii(target_words[i], &target_counts[i]);
        forms[i][0].lect_id = "A";
        forms[i][0].form = form("A", source_segments[i], source_counts[i]);
        forms[i][1].lect_id = "B";
        forms[i][1].form = form("B", target_segments[i], target_counts[i]);
        cognates[i].cognate_id = source_words[i];
        cognates[i].forms = forms[i];
        cognates[i].form_count = 2;
        cognates[i].confidence = 1.0;
    }

    assert(rg_train_model(ctx, cognates, 6, &local_options, &model) == RG_OK);
    for (i = 0; i < rg_multi_model_unconditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_unconditioned_class_at(model, i);
        assert(no_duplicate_support(row));
        assert(row->supporting_cognate_count <= (size_t)row->count);
        if (row->segment_count == 2 &&
            strcmp(row->graphemes[0], "p") == 0 &&
            strcmp(row->graphemes[1], "f") == 0) {
            assert(fabs(row->count - 12.0) < 1e-9);
            assert(row->supporting_cognate_count == 6);
            found = 1;
        }
    }
    assert(found);
    rg_multi_model_free(model);
    for (i = 0; i < 6; i++) {
        segments_free(source_segments[i], source_counts[i]);
        segments_free(target_segments[i], target_counts[i]);
    }
}

/* A loss has a class row now. Six words keep a final -n on A and drop it on B;
 * the class table must state {A:n, B:∅} rather than leave B silently absent, and
 * the gap must not leak into the conditioning search. */
static void test_multi_lect_deletion_class(rg_context *ctx, const rg_train_options *options) {
    const char *src[6] = {"apan", "atan", "akan", "aman", "asan", "alan"};
    const char *tgt[6] = {"apa", "ata", "aka", "ama", "asa", "ala"};
    rg_segment *src_seg[6];
    rg_segment *tgt_seg[6];
    size_t src_n[6];
    size_t tgt_n[6];
    rg_cognate_form forms[6][2];
    rg_cognate_set cognates[6];
    rg_multi_model *model = 0;
    rg_train_options local = *options;
    size_t i;
    int found_deletion = 0;

    memset(cognates, 0, sizeof(cognates));
    for (i = 0; i < 6; i++) {
        src_seg[i] = segments_from_ascii(src[i], &src_n[i]);
        tgt_seg[i] = segments_from_ascii(tgt[i], &tgt_n[i]);
        forms[i][0].lect_id = "A";
        forms[i][0].form = form("A", src_seg[i], src_n[i]);
        forms[i][1].lect_id = "B";
        forms[i][1].form = form("B", tgt_seg[i], tgt_n[i]);
        cognates[i].cognate_id = src[i];
        cognates[i].forms = forms[i];
        cognates[i].form_count = 2;
        cognates[i].confidence = 1.0;
    }
    assert(rg_train_model(ctx, cognates, 6, &local, &model) == RG_OK);
    for (i = 0; i < rg_multi_model_unconditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_unconditioned_class_at(model, i);
        if (row->segment_count == 2 &&
            strcmp(row->graphemes[0], "n") == 0 &&
            strcmp(row->graphemes[1], RG_GAP_GRAPHEME) == 0 &&
            strcmp(row->lect_ids[0], "A") == 0 &&
            strcmp(row->lect_ids[1], "B") == 0) {
            assert(row->count == 6.0);
            found_deletion = 1;
        }
    }
    assert(found_deletion);
    /* A gap conditions nothing: it must never reach the conditioned table. */
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
        size_t j;
        for (j = 0; j < row->segment_count; j++) {
            assert(strcmp(row->graphemes[j], RG_GAP_GRAPHEME) != 0);
        }
    }
    rg_multi_model_free(model);
    for (i = 0; i < 6; i++) {
        segments_free(src_seg[i], src_n[i]);
        segments_free(tgt_seg[i], tgt_n[i]);
    }
}

/* A tone correspondence set is a correspondence set, and gets a class row.
 *
 * `tone_three_lect.tsv` has one vowel /a/ in every set carrying a tone that
 * corresponds regularly but non-trivially across three lects: high in north and
 * central is mid in south (⁵⁵ ⁵⁵ ³³), and low in north is mid in central and
 * low in south (¹¹ ³³ ¹¹). The segments are identical across lects, so a
 * segment-only class table has only the trivial /a ~ a ~ a/ to publish and the
 * tone correspondence -- the whole point of a Sinitic or Hmong-Mien sample --
 * would be invisible. Because the suprasegmentals are part of the reconciled
 * outcome identity, the two tone patterns are two class rows, each
 * carrying its tone. */
static void test_tone_correspondence_is_a_class(rg_context *ctx, const rg_train_options *options) {
    char path[512];
    rg_corpus *corpus = 0;
    rg_multi_model *model = 0;
    size_t count = 0;
    const rg_multi_class_row *rows;
    size_t i;
    int saw_high = 0;
    int saw_low = 0;

    snprintf(path, sizeof(path), "%s/testdata/corpora/tone_three_lect.tsv", REGULAE_SOURCE_DIR);
    assert(rg_corpus_load_tsv(path, 0, &corpus, 0) == RG_OK);
    assert(rg_train_model(ctx, rg_corpus_cognate_at(corpus, 0),
                          rg_corpus_cognate_count(corpus), options, &model) == RG_OK);

    rows = rg_multi_model_unconditioned_classes(model, &count);
    for (i = 0; i < count; i++) {
        const rg_multi_class_row *row = &rows[i];
        int all_a = 1;
        size_t j;
        if (row->segment_count != 3 || row->suprasegmentals == 0) {
            continue;
        }
        for (j = 0; j < 3; j++) {
            if (strcmp(row->graphemes[j], "a") != 0) {
                all_a = 0;
            }
        }
        if (!all_a) {
            continue;
        }
        /* Segments are ordered by lect id: central, north, south. */
        if (strcmp(row->suprasegmentals[0].tone, "\xe2\x81\xb5\xe2\x81\xb5") == 0 &&
            strcmp(row->suprasegmentals[1].tone, "\xe2\x81\xb5\xe2\x81\xb5") == 0 &&
            strcmp(row->suprasegmentals[2].tone, "\xc2\xb3\xc2\xb3") == 0) {
            saw_high = 1;
        }
        if (strcmp(row->suprasegmentals[0].tone, "\xc2\xb3\xc2\xb3") == 0 &&
            strcmp(row->suprasegmentals[1].tone, "\xc2\xb9\xc2\xb9") == 0 &&
            strcmp(row->suprasegmentals[2].tone, "\xc2\xb9\xc2\xb9") == 0) {
            saw_low = 1;
        }
    }
    /* Both tone correspondences are published, each as its own class -- the
     * same graphemes under a different tone are not one class. */
    assert(saw_high);
    assert(saw_low);

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

int main(void) {
    rg_context *ctx = 0;
    rg_train_options options;
    rg_multi_model *model = 0;
    rg_cognate_outlier_row *outliers = 0;
    size_t outlier_count = 0;
    rg_segment a1[] = {{"p", 0, 0, 0}, {"a", 0, 0, 0}};
    rg_segment b1[] = {{"f", 0, 0, 0}, {"a", 0, 0, 0}};
    rg_segment c1[] = {{"b", 0, 0, 0}, {"a", 0, 0, 0}};
    rg_segment a2[] = {{"p", 0, 0, 0}, {"i", 0, 0, 0}};
    rg_segment b2[] = {{"f", 0, 0, 0}, {"i", 0, 0, 0}};
    rg_segment c2[] = {{"b", 0, 0, 0}, {"i", 0, 0, 0}};
    rg_cognate_form forms1[3];
    rg_cognate_form forms2[3];
    rg_cognate_set cognates[2];
    rg_cognate_set invalid;
    rg_cognate_form invalid_forms[1];
    size_t i;
    int saw_ab = 0;
    int saw_ac = 0;
    int saw_bc = 0;

    memset(cognates, 0, sizeof(cognates));
    memset(&invalid, 0, sizeof(invalid));
    assert(rg_context_new_builtin(&ctx) == RG_OK);
    rg_train_options_init_defaults(&options);
    test_conditioned_palatalization(ctx, &options);
    test_supporting_sets_are_distinct(ctx, &options);
    test_multi_lect_deletion_class(ctx, &options);
    test_tone_correspondence_is_a_class(ctx, &options);

    forms1[0].lect_id = "A";
    forms1[0].form = form("A", a1, 2);
    forms1[1].lect_id = "B";
    forms1[1].form = form("B", b1, 2);
    forms1[2].lect_id = "C";
    forms1[2].form = form("C", c1, 2);
    forms2[0].lect_id = "A";
    forms2[0].form = form("A", a2, 2);
    forms2[1].lect_id = "B";
    forms2[1].form = form("B", b2, 2);
    forms2[2].lect_id = "C";
    forms2[2].form = form("C", c2, 2);

    cognates[0].cognate_id = "one";
    cognates[0].forms = forms1;
    cognates[0].form_count = 3;
    cognates[0].confidence = 1.0;
    cognates[1].cognate_id = "two";
    cognates[1].forms = forms2;
    cognates[1].form_count = 3;
    cognates[1].confidence = 1.0;

    assert(rg_train_model(ctx, cognates, 2, &options, &model) == RG_OK);
    assert(model != 0);
    assert(rg_multi_model_lect_count(model) == 3);
    assert(strcmp(rg_multi_model_lect_at(model, 0), "A") == 0);
    assert(strcmp(rg_multi_model_lect_at(model, 1), "B") == 0);
    assert(strcmp(rg_multi_model_lect_at(model, 2), "C") == 0);
    assert(rg_multi_model_lect_at(model, 3) == 0);
    assert(rg_multi_model_pair_model_count(model) == 3);
    for (i = 0; i < rg_multi_model_pair_model_count(model); i++) {
        const rg_multi_pair_model_row *row = rg_multi_model_pair_model_at(model, i);
        assert(row != 0);
        assert(row->model != 0);
        if (strcmp(row->lect_a, "A") == 0 && strcmp(row->lect_b, "B") == 0) {
            saw_ab = 1;
        }
        if (strcmp(row->lect_a, "A") == 0 && strcmp(row->lect_b, "C") == 0) {
            saw_ac = 1;
        }
        if (strcmp(row->lect_a, "B") == 0 && strcmp(row->lect_b, "C") == 0) {
            saw_bc = 1;
        }
    }
    assert(saw_ab && saw_ac && saw_bc);
    assert(rg_multi_model_pair_model_at(model, 1000) == 0);
    assert(model_has_class(model, "A", "p", "B", "f", "C", "b", 2.0));
    assert(model_has_class(model, "A", "a", "B", "a", "C", "a", 1.0));
    assert(model_has_class(model, "A", "i", "B", "i", "C", "i", 1.0));
    assert(rg_multi_model_unconditioned_class_at(model, 1000) == 0);
    assert(rg_multi_model_cross_dimensional_row_count(model) == 0);
    assert(rg_multi_model_cross_dimensional_row_at(model, 0) == 0);
    assert(rg_find_cognate_outliers(ctx, cognates, 2, model, &options, 0, 0, &outliers, &outlier_count) == RG_OK);
    assert(outlier_count == 2);
    assert(outliers != 0);
    assert(outliers[0].pair_count == 3);
    assert(outliers[1].pair_count == 3);
    assert(outliers[0].z_score >= outliers[1].z_score);
    rg_cognate_outlier_rows_free(outliers, outlier_count);
    outliers = 0;
    outlier_count = 0;
    assert(rg_find_cognate_outliers(ctx, cognates, 2, model, &options, 1, 1, &outliers, &outlier_count) == RG_OK);
    assert(outlier_count == 1);
    rg_cognate_outlier_rows_free(outliers, outlier_count);
    outliers = 0;
    outlier_count = 0;
    rg_multi_model_free(model);
    model = 0;

    invalid_forms[0].lect_id = "A";
    invalid_forms[0].form = form("A", a1, 2);
    invalid.cognate_id = "bad";
    invalid.forms = invalid_forms;
    invalid.form_count = 1;
    invalid.confidence = 1.0;
    /* A one-form set is legitimate while the corpus only knows one lect: there
     * is nothing to reconcile it against. It becomes a structural error as soon
     * as a second lect appears, which the two-set corpus below checks. */
    assert(rg_train_model(ctx, &invalid, 1, &options, &model) == RG_OK);
    assert(rg_multi_model_lect_count(model) == 1);
    assert(rg_multi_model_unconditioned_class_count(model) == 0);
    rg_multi_model_free(model);
    model = 0;
    {
        rg_cognate_form paired_forms[2];
        rg_cognate_set mixed[2];
        memset(mixed, 0, sizeof(mixed));
        paired_forms[0].lect_id = "A";
        paired_forms[0].form = form("A", a1, 2);
        paired_forms[1].lect_id = "B";
        paired_forms[1].form = form("B", a1, 2);
        mixed[0].cognate_id = "ok";
        mixed[0].forms = paired_forms;
        mixed[0].form_count = 2;
        mixed[0].confidence = 1.0;
        mixed[1] = invalid;
        /* A set with one form has nothing to align against, so it contributes
         * no correspondence -- but every cognate-coded wordlist has some, and
         * refusing the corpus over one made regulae unable to read the field's
         * standard data. It is counted instead, and the count is published. */
        assert(rg_train_model(ctx, mixed, 2, &options, &model) == RG_OK);
        assert(rg_multi_model_unpaired_set_count(model) == 1);
        assert(rg_multi_model_lect_count(model) == 2);
        assert(rg_multi_model_unconditioned_class_count(model) > 0);
        rg_multi_model_free(model);
        model = 0;
    }
    invalid.form_count = 2;
    invalid.forms = 0;
    assert(rg_train_model(ctx, &invalid, 1, &options, &model) == RG_ERR_INVALID_ARGUMENT);
    invalid.forms = invalid_forms;
    invalid.form_count = 1;
    invalid.confidence = 1.5;
    assert(rg_train_model(ctx, &invalid, 1, &options, &model) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_train_model(ctx, 0, 1, &options, &model) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_train_model(0, cognates, 2, &options, &model) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_multi_model_lect_count(0) == 0);
    assert(rg_multi_model_pair_model_count(0) == 0);
    assert(rg_multi_model_unconditioned_class_count(0) == 0);
    assert(rg_multi_model_conditioned_class_count(0) == 0);
    assert(rg_multi_model_cross_dimensional_row_count(0) == 0);
    assert(rg_find_cognate_outliers(ctx, cognates, 2, 0, &options, 0, 0, &outliers, &outlier_count) == RG_ERR_INVALID_ARGUMENT);
    rg_cognate_outlier_rows_free(0, 0);
    rg_multi_model_free(0);

    rg_context_free(ctx);
    return 0;
}
