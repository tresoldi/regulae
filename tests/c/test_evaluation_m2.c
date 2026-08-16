#include "regulae.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static rg_multi_model *train(rg_context *ctx, const char *path, rg_corpus **out_corpus) {
    rg_train_options options;
    rg_multi_model *model = 0;
    assert(rg_corpus_load_tsv(path, 0, out_corpus, 0) == RG_OK);
    rg_train_options_init_defaults(&options);
    assert(rg_train_model(ctx, rg_corpus_cognates(*out_corpus),
                          rg_corpus_cognate_count(*out_corpus), &options, &model) == RG_OK);
    return model;
}

static double taxon_rule_margin(const rg_multi_model *model) {
    const rg_multi_class_row *rows;
    size_t count = 0;
    size_t i;
    rows = rg_multi_model_conditioned_classes(model, &count);
    for (i = 0; i < count; i++) {
        size_t j;
        int has_source = 0;
        int has_reflex = 0;
        for (j = 0; j < rows[i].segment_count; j++) {
            if (strcmp(rows[i].lect_ids[j], "ancestor") == 0 &&
                strcmp(rows[i].graphemes[j], "p") == 0) {
                has_source = 1;
            }
            if (strncmp(rows[i].lect_ids[j], "innovator", 9) == 0 &&
                strcmp(rows[i].graphemes[j], "f") == 0) {
                has_reflex = 1;
            }
        }
        if (has_source && has_reflex) {
            return rows[i].evidence.search_margin;
        }
    }
    return -1.0;
}

static void test_taxon_sampling_preserves_the_observed_rule(rg_context *ctx) {
    static const char *paths[] = {
        REGULAE_SOURCE_DIR "/testdata/linguistic/taxon_sampling_2lect.tsv",
        REGULAE_SOURCE_DIR "/testdata/linguistic/taxon_sampling_3lect.tsv",
        REGULAE_SOURCE_DIR "/testdata/linguistic/taxon_sampling_4lect.tsv"
    };
    double first = -1.0;
    size_t i;
    for (i = 0; i < sizeof(paths) / sizeof(paths[0]); i++) {
        rg_corpus *corpus = 0;
        rg_multi_model *model = train(ctx, paths[i], &corpus);
        double margin = taxon_rule_margin(model);
        assert(margin > 0.0);
        if (i == 0) {
            first = margin;
        } else {
            assert(fabs(margin - first) < 1e-12);
        }
        rg_multi_model_free(model);
        rg_corpus_free(corpus);
    }
}

static void test_repeated_cells_name_two_histories(rg_context *ctx) {
    rg_corpus *corpus = 0;
    rg_multi_model *model = train(
        ctx, REGULAE_SOURCE_DIR "/testdata/linguistic/repeated_etymon.tsv", &corpus);
    const rg_corpus_fit *fit = rg_multi_model_fit(model);
    assert(rg_corpus_cognate_count(corpus) == 16);
    assert(fit->etymon_group_count == 2);
    assert(fit->bootstrap_unit == RG_OBSERVATION_UNIT_ETYMON_GROUP);
    assert(fit->bootstrap_effective_unit_count == 2);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

int main(void) {
    rg_context *ctx = 0;
    assert(rg_context_new_builtin(&ctx) == RG_OK);
    test_taxon_sampling_preserves_the_observed_rule(ctx);
    test_repeated_cells_name_two_histories(ctx);
    rg_context_free(ctx);
    printf("M2 linguistic evaluation regressions passed\n");
    return 0;
}
