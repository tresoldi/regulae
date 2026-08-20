#include "regulae.h"
#include "table_access.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Formatting output is a diagnostic, not a machine contract, so these tests
 * check that the pieces a reader needs are present rather than pinning an
 * exact layout. */

static rg_segment seg(const char *g) {
    rg_segment s = {g, 0, 0, 0};
    return s;
}

static void test_segments_and_links(void) {
    rg_segment segments[2];
    rg_segment toned = {"a", "55", "long", "primary"};
    rg_link link;
    char *text;

    segments[0] = seg("p");
    segments[1] = seg("a");

    text = rg_format_segments(segments, 2);
    assert(text != 0);
    assert(strcmp(text, "p a") == 0);
    rg_string_free(text);

    text = rg_format_segments(0, 0);
    assert(text != 0);
    assert(strlen(text) > 0);
    rg_string_free(text);

    text = rg_format_segments(&toned, 1);
    assert(text != 0);
    assert(strstr(text, "T=55") != 0);
    assert(strstr(text, "L=long") != 0);
    assert(strstr(text, "S=primary") != 0);
    rg_string_free(text);

    memset(&link, 0, sizeof(link));
    link.source = segments;
    link.source_count = 1;
    link.target = segments + 1;
    link.target_count = 1;
    text = rg_format_link(&link);
    assert(text != 0);
    assert(strcmp(text, "p ~ a") == 0);
    rg_string_free(text);
}

static void test_model_rendering(rg_context *ctx) {
    rg_segment source[3];
    rg_segment target[3];
    rg_cognate_form forms[2];
    rg_cognate_set cognates[3];
    rg_multi_model *model = 0;
    rg_train_options options;
    rg_alignment *alignment = 0;
    char *text;
    size_t i;

    memset(cognates, 0, sizeof(cognates));
    source[0] = seg("p");
    source[1] = seg("a");
    source[2] = seg("t");
    target[0] = seg("f");
    target[1] = seg("a");
    target[2] = seg("t");

    for (i = 0; i < 3; i++) {
        cognates[i].cognate_id = i == 0 ? "c1" : (i == 1 ? "c2" : "c3");
        cognates[i].forms = forms;
        cognates[i].form_count = 2;
        cognates[i].confidence = 1.0;
    }
    memset(forms, 0, sizeof(forms));
    forms[0].lect_id = "A";
    forms[0].form.lect_id = "A";
    forms[0].form.segments = source;
    forms[0].form.segment_count = 3;
    forms[1].lect_id = "B";
    forms[1].form.lect_id = "B";
    forms[1].form.segments = target;
    forms[1].form.segment_count = 3;

    rg_train_options_init_defaults(&options);
    assert(rg_train_model(ctx, cognates, 3, &options, &model) == RG_OK);

    text = rg_format_multi_model(model, 0);
    assert(text != 0);
    assert(strstr(text, "MultiLectModel") != 0);
    assert(strstr(text, "lects (2): A, B") != 0);
    assert(strstr(text, "unconditioned classes") != 0 || strstr(text, "unconditioned cls") != 0);
    assert(strstr(text, "A:p ~ B:f") != 0);
    rg_string_free(text);

    text = rg_describe_multi_class(model, "A", "p");
    assert(text != 0);
    assert(strstr(text, "Classes with A:p") != 0);
    assert(strstr(text, "A:p ~ B:f") != 0);
    rg_string_free(text);

    /* An absent grapheme still renders, reporting nothing found. */
    text = rg_describe_multi_class(model, "A", "zzz");
    assert(text != 0);
    assert(strstr(text, "(none)") != 0);
    rg_string_free(text);

    assert(rg_multi_model_pair_model_count(model) == 1);
    text = rg_format_pairwise_model(rg_multi_model_pair_model_at(model, 0)->model, 0);
    assert(text != 0);
    assert(strstr(text, "PairwiseModel") != 0);
    assert(strstr(text, "p ~ f") != 0);
    rg_string_free(text);

    assert(rg_align_forms(ctx, &forms[0].form, &forms[1].form, 0, &alignment) == RG_OK);
    text = rg_format_alignment(alignment);
    assert(text != 0);
    assert(strstr(text, "p ~ f") != 0);
    rg_string_free(text);
    rg_alignment_free(alignment);

    rg_multi_model_free(model);
}

static void test_option_defaults(void) {
    rg_format_model_options options;
    memset(&options, 0, sizeof(options));
    rg_format_model_options_init_defaults(&options);
    assert(options.top_segments == 15);
    assert(options.top_displacements == 5);
    assert(options.min_count == 1.0);
    assert(options.top_classes == 20);
}

/* The default report leads with the change, not the retention, and the
 * elsewhere-heavy complement does not sit in the first conditioned block. */
static void test_conditioned_reading_order(rg_context *ctx) {
    rg_corpus *corpus = 0;
    rg_multi_model *model = 0;
    rg_train_options options;
    char *text;
    char *main_block;
    char *complements;
    char *change;
    char *weak_row;

    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/soundlaws/rhotacism.tsv",
                              0, &corpus, 0) == RG_OK);
    rg_train_options_init_defaults(&options);
    options.permutation_count = 0;
    assert(rg_train_model(ctx, rg_corpus_cognates(corpus), rg_corpus_cognate_count(corpus),
                          &options, &model) == RG_OK);
    text = rg_format_multi_model(model, 0);
    assert(text != 0);

    main_block = strstr(text, "--- Conditioned classes");
    complements = strstr(text, "--- Complements");
    assert(main_block != 0);
    assert(complements != 0);
    assert(complements > main_block);

    /* The change is in the main block; the retention is not what leads. */
    change = strstr(main_block, "latin:r ~ old_latin:s");
    assert(change != 0);
    assert(strncmp(main_block, "--- Complements", 15) != 0);
    assert(change < complements);
    weak_row = strstr(main_block, "elsewhere=26");
    assert(weak_row == 0 || weak_row > complements);

    rg_string_free(text);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* A transposition is one event, and the default report shows it: the chunk
 * carries [reordering] so the model does not read as silence. */
static void test_metathesis_chunk_is_visible(rg_context *ctx) {
    rg_corpus *corpus = 0;
    rg_multi_model *model = 0;
    rg_train_options options;
    char *text;

    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/soundlaws/metathesis_adjacent.tsv",
                              0, &corpus, 0) == RG_OK);
    rg_train_options_init_defaults(&options);
    options.permutation_count = 0;
    assert(rg_train_model(ctx, rg_corpus_cognates(corpus), rg_corpus_cognate_count(corpus),
                          &options, &model) == RG_OK);
    text = rg_format_multi_model(model, 0);
    assert(text != 0);
    assert(strstr(text, "Multi-segment correspondences") != 0);
    assert(strstr(text, "[reordering]") != 0);
    assert(strstr(text, "k s ~ s k") != 0);

    rg_string_free(text);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

/* Drift is a format-level result, not only a CLI regex: the formatter renders
 * it so a test can assert on the string without relying on CMake's stdout
 * match. */
static void test_drift_in_format(rg_context *ctx) {
    rg_corpus *corpus = 0;
    rg_transcription_drift_row *drift = 0;
    size_t drift_count = 0;
    char *text;

    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/diagnostics/drift.tsv",
                              0, &corpus, 0) == RG_OK);
    assert(rg_find_transcription_drift(ctx, rg_corpus_cognates(corpus),
                                       rg_corpus_cognate_count(corpus),
                                       &drift, &drift_count) == RG_OK);
    assert(drift_count > 0);
    text = rg_format_drift(drift, drift_count);
    assert(text != 0);
    assert(strstr(text, "DRIFT") != 0);
    assert(strstr(text, "broad") != 0);
    assert(strstr(text, "narrow") != 0);

    rg_string_free(text);
    rg_transcription_drift_rows_free(drift, drift_count);
    rg_corpus_free(corpus);
}

int main(void) {
    rg_context *ctx = 0;
    assert(rg_context_new_builtin(&ctx) == RG_OK);
    test_option_defaults();
    test_segments_and_links();
    test_model_rendering(ctx);
    test_conditioned_reading_order(ctx);
    test_metathesis_chunk_is_visible(ctx);
    test_drift_in_format(ctx);
    rg_context_free(ctx);
    printf("format tests passed\n");
    return 0;
}
