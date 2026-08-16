#include "regulae.h"
#include "table_access.h"
#include "internal.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The JSON layer is a transport format, so these check structure and the
 * refusal behaviour of the options reader rather than exact bytes. */

static char *train_json(rg_context *ctx, const char *corpus_path, int alignments, int outliers) {
    rg_corpus *corpus = 0;
    rg_multi_model *model = 0;
    rg_train_options options;
    char *text;

    assert(rg_corpus_load_tsv(corpus_path, 0, &corpus, 0) == RG_OK);
    rg_train_options_init_defaults(&options);
    assert(rg_train_model(ctx, rg_corpus_cognates(corpus), rg_corpus_cognate_count(corpus),
                          &options, &model) == RG_OK);
    text = rg_model_to_json(ctx, model, rg_corpus_cognates(corpus),
                            rg_corpus_cognate_count(corpus), &options, alignments, outliers);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
    return text;
}

static void test_model_json_shape(rg_context *ctx) {
    char *text = train_json(ctx, REGULAE_SOURCE_DIR "/testdata/corpora/conditioned_multilect.tsv", 1, 1);
    assert(text != 0);

    assert(strstr(text, "\"ok\":true") != 0);
    assert(strstr(text, "\"format_version\":1") != 0);
    /* Labelled so nothing downstream reads a single MAP system as
     * claim-capable interchange data. */
    assert(strstr(text, "\"export_kind\":\"debug_map_snapshot\"") != 0);
    assert(strstr(text, "\"lects\":[\"alpha\",\"beta\",\"gamma\"]") != 0);
    assert(strstr(text, "\"unconditioned\"") != 0);
    assert(strstr(text, "\"conditioned\"") != 0);
    assert(strstr(text, "\"pairwise\"") != 0);
    assert(strstr(text, "\"cross_dimensional\"") != 0);
    assert(strstr(text, "\"alignments\"") != 0);
    assert(strstr(text, "\"outliers\"") != 0);
    assert(strstr(text, "\"supporting_cognates\"") != 0);
    assert(strstr(text, "\"uncertainty\"") != 0);
    assert(strstr(text, "\"predictive\":{\"status\":\"unmeasured\"") != 0);
    /* A conditioned class carries its per-lect environment. */
    assert(strstr(text, "\"context\"") != 0);
    rg_string_free(text);
}

/* Alignments and outliers are opt-in; asking for neither must not emit them. */
static void test_optional_sections(rg_context *ctx) {
    char *text = train_json(ctx, REGULAE_SOURCE_DIR "/testdata/corpora/three_lect_basic.tsv", 0, 0);
    assert(text != 0);
    assert(strstr(text, "\"unconditioned\"") != 0);
    assert(strstr(text, "\"alignments\"") == 0);
    assert(strstr(text, "\"outliers\"") == 0);
    rg_string_free(text);
}

/* Repeated renders of the same model must agree byte for byte, or the demo and
 * the CLI could disagree without anything changing. */
static void test_determinism(rg_context *ctx) {
    char *first = train_json(ctx, REGULAE_SOURCE_DIR "/testdata/corpora/three_lect_basic.tsv", 1, 1);
    char *second = train_json(ctx, REGULAE_SOURCE_DIR "/testdata/corpora/three_lect_basic.tsv", 1, 1);
    assert(first != 0 && second != 0);
    assert(strcmp(first, second) == 0);
    rg_string_free(first);
    rg_string_free(second);
}

/* Each alignment link reports the classes it realises, taken from the
 * reconciliation. The point of publishing it is precision: a conditioned class
 * and the unconditioned one over the same segments are indistinguishable if a
 * consumer matches graphemes, and the environment is what the tool is for. */
static void test_links_carry_class_ids(rg_context *ctx) {
    char *text = train_json(ctx, REGULAE_SOURCE_DIR "/testdata/corpora/real_latin_spanish.tsv", 1, 1);
    const char *cursor;
    size_t links_with_classes = 0;

    assert(text != 0);
    assert(strstr(text, "\"classes\":[") != 0);

    for (cursor = text; (cursor = strstr(cursor, "\"classes\":[")) != 0; cursor++) {
        links_with_classes++;
    }
    assert(links_with_classes > 50);
    rg_string_free(text);
}

static void test_null_model(rg_context *ctx) {
    assert(rg_model_to_json(ctx, 0, 0, 0, 0, 0, 0) == 0);
}

/* The options reader rejects what it does not understand rather than quietly
 * falling back to defaults: a caller who misspells a knob in a browser would
 * otherwise get results that silently ignore their setting. */
static void test_options_reader(void) {
    rg_train_options options;
    char detail[256];

    assert(rg_json_read_train_options_internal(
                                               "{\"max_chunk_size\":2,\"temperature\":0.5,"
                                               "\"bootstrap_unit\":\"source_group\","
                                               "\"predictive_folds\":4,"
                                               "\"predictive_seed\":17,"
                                               "\"predictive_min_groups\":3,"
                                               "\"predictive_abstention_threshold\":0.6,"
                                               "\"predictive_top_k\":5,"
                                               "\"split_scorer\":\"dirichlet_marginal\","
                                               "\"split_prior_concentration\":2.0,"
                                               "\"search_penalty_gamma\":1.0}",
                                               &options, detail, sizeof(detail)) == RG_OK);
    assert(options.max_chunk_size == 2);
    assert(options.temperature == 0.5);
    assert(options.bootstrap_unit == RG_OBSERVATION_UNIT_SOURCE_GROUP);
    assert(options.predictive_folds == 4);
    assert(options.predictive_seed == 17);
    assert(options.predictive_min_groups == 3);
    assert(options.predictive_abstention_threshold == 0.6);
    assert(options.predictive_top_k == 5);
    assert(options.bic.split_scorer == RG_SPLIT_SCORER_DIRICHLET_MARGINAL);
    assert(options.bic.split_prior_concentration == 2.0);
    assert(options.bic.search_penalty_gamma == 1.0);
    /* Untouched fields keep their defaults. */
    assert(options.concentration == 5.0);

    /* Nested BIC knobs are flattened into the same object. */
    assert(rg_json_read_train_options_internal("{\"min_split_observations\":4}",
                                               &options, detail, sizeof(detail)) == RG_OK);
    assert(options.bic.min_split_observations == 4);

    /* Absent or empty options are the defaults, not an error. */
    assert(rg_json_read_train_options_internal(0, &options, detail, sizeof(detail)) == RG_OK);
    assert(options.max_chunk_size == RG_DEFAULT_MAX_CHUNK_SIZE);
    assert(rg_json_read_train_options_internal("", &options, detail, sizeof(detail)) == RG_OK);
    assert(options.max_chunk_size == RG_DEFAULT_MAX_CHUNK_SIZE);
    assert(rg_json_read_train_options_internal("{}", &options, detail, sizeof(detail)) == RG_OK);

    /* An unknown key is refused, and named. */
    assert(rg_json_read_train_options_internal("{\"max_chunk_sizze\":2}",
                                               &options, detail, sizeof(detail)) == RG_ERR_UNSUPPORTED_OPTION);
    assert(strstr(detail, "max_chunk_sizze") != 0);
    /* A rejected object leaves defaults behind, not a half-applied state. */
    assert(options.max_chunk_size == RG_DEFAULT_MAX_CHUNK_SIZE);

    /* Wrong types and malformed input are parse errors, not crashes. */
    assert(rg_json_read_train_options_internal("{\"max_chunk_size\":\"two\"}",
                                               &options, detail, sizeof(detail)) == RG_ERR_PARSE);
    assert(rg_json_read_train_options_internal("{\"max_chunk_size\":", &options, detail, sizeof(detail)) == RG_ERR_PARSE);
    assert(rg_json_read_train_options_internal("[1,2,3]", &options, detail, sizeof(detail)) == RG_ERR_PARSE);
    assert(rg_json_read_train_options_internal("not json at all", &options, detail, sizeof(detail)) == RG_ERR_PARSE);

    /* Deeply nested input must not blow the stack. */
    {
        char deep[4096];
        size_t i;
        for (i = 0; i < sizeof(deep) - 2; i++) {
            deep[i] = '[';
        }
        deep[sizeof(deep) - 2] = ']';
        deep[sizeof(deep) - 1] = '\0';
        assert(rg_json_read_train_options_internal(deep, &options, detail, sizeof(detail)) == RG_ERR_PARSE);
    }

    /* A null detail buffer is allowed. */
    assert(rg_json_read_train_options_internal("{\"nope\":1}", &options, 0, 0) == RG_ERR_UNSUPPORTED_OPTION);
}

static void test_error_payload(void) {
    char *text = rg_json_error_internal(RG_ERR_UNKNOWN_GRAPHEME, "grapheme \"q\" is unknown");
    assert(text != 0);
    assert(strstr(text, "\"ok\":false") != 0);
    assert(strstr(text, "\"status_code\"") != 0);
    assert(strstr(text, "grapheme") != 0);
    rg_string_free(text);

    text = rg_json_error_internal(RG_ERR_OOM, 0);
    assert(text != 0);
    assert(strstr(text, "\"ok\":false") != 0);
    rg_string_free(text);
}

int main(void) {
    rg_context *ctx = 0;
    assert(rg_context_new_builtin(&ctx) == RG_OK);
    test_model_json_shape(ctx);
    test_optional_sections(ctx);
    test_determinism(ctx);
    test_links_carry_class_ids(ctx);
    test_null_model(ctx);
    test_options_reader();
    test_error_payload();
    rg_context_free(ctx);
    printf("json tests passed\n");
    return 0;
}
