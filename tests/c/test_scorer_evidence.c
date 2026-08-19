#include "regulae.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static rg_corpus *load(const char *relative, int with_confidence) {
    char path[1024];
    rg_corpus *corpus = 0;
    rg_tsv_load_options options = {0};
    snprintf(path, sizeof(path), "%s/%s", REGULAE_SOURCE_DIR, relative);
    if (with_confidence) {
        options.confidence_column = "confidence";
    }
    assert(rg_corpus_load_tsv(path, with_confidence ? &options : 0, &corpus, 0) == RG_OK);
    return corpus;
}

static void test_every_scorer_publishes_what_selected_the_row(rg_context *ctx) {
    const rg_split_scorer scorers[] = {
        RG_SPLIT_SCORER_CORRECTED_BIC,
        RG_SPLIT_SCORER_MULTINOMIAL_NML,
        RG_SPLIT_SCORER_DIRICHLET_MARGINAL
    };
    rg_corpus *corpus = load("testdata/evaluation/m2/generated/sets_32.tsv", 0);
    size_t s;
    for (s = 0; s < sizeof(scorers) / sizeof(scorers[0]); s++) {
        rg_train_options options;
        rg_multi_model *model = 0;
        const rg_multi_class_row *rows;
        const rg_corpus_fit *fit;
        size_t count = 0;
        size_t i;
        rg_train_options_init_defaults(&options);
        options.bic.split_scorer = scorers[s];
        assert(rg_train_model(ctx, rg_corpus_cognates(corpus),
                              rg_corpus_cognate_count(corpus), &options, &model) == RG_OK);
        fit = rg_multi_model_fit(model);
        assert(fit->split_scorer == scorers[s]);
        rows = rg_multi_model_conditioned_classes(model, &count);
        assert(count > 0);
        for (i = 0; i < count; i++) {
            assert(rows[i].evidence.scorer == scorers[s]);
            assert(fabs(rows[i].evidence.delta_score - rows[i].evidence.delta_bic) < 1e-12);
        }
        rg_multi_model_free(model);
    }
    rg_corpus_free(corpus);
}

static void test_exact_nml_refuses_fractional_confidence(rg_context *ctx) {
    rg_corpus *corpus = load("testdata/evaluation/m2/generated/confidence_weights.tsv", 1);
    rg_train_options options;
    rg_multi_model *model = 0;
    rg_status status;
    rg_train_options_init_defaults(&options);
    options.bic.split_scorer = RG_SPLIT_SCORER_MULTINOMIAL_NML;
    status = rg_train_model(ctx, rg_corpus_cognates(corpus),
                            rg_corpus_cognate_count(corpus), &options, &model);
    assert(status == RG_ERR_UNSUPPORTED_OPTION);
    assert(model == 0);
    rg_corpus_free(corpus);
}

static void test_direct_pairwise_evidence_names_its_scorer(rg_context *ctx) {
    rg_corpus *corpus = load("testdata/evaluation/m2/generated/sets_32.tsv", 0);
    size_t pair_count = rg_corpus_cognate_count(corpus);
    rg_form_pair *pairs = (rg_form_pair *)calloc(pair_count, sizeof(*pairs));
    rg_train_options options;
    rg_pairwise_model *model = 0;
    const rg_conditioned_segment_count_row *rows;
    size_t row_count = 0;
    size_t i;
    assert(pairs != 0);
    for (i = 0; i < pair_count; i++) {
        const rg_cognate_set *set = rg_corpus_cognate_at(corpus, i);
        pairs[i].source = set->forms[0].form;
        pairs[i].target = set->forms[1].form;
        pairs[i].weight = set->confidence;
    }
    rg_train_options_init_defaults(&options);
    options.bic.split_scorer = RG_SPLIT_SCORER_DIRICHLET_MARGINAL;
    assert(rg_train_pairwise(ctx, pairs, pair_count, &options, &model) == RG_OK);
    rows = rg_pairwise_model_conditioned_segment_counts(model, &row_count);
    assert(row_count > 0);
    for (i = 0; i < row_count; i++) {
        assert(rows[i].evidence.scorer == RG_SPLIT_SCORER_DIRICHLET_MARGINAL);
        assert(fabs(rows[i].evidence.delta_score - rows[i].evidence.delta_bic) < 1e-12);
    }
    rg_pairwise_model_free(model);
    free(pairs);
    rg_corpus_free(corpus);
}

int main(void) {
    rg_context *ctx = 0;
    assert(rg_context_new_builtin(&ctx) == RG_OK);
    test_every_scorer_publishes_what_selected_the_row(ctx);
    test_exact_nml_refuses_fractional_confidence(ctx);
    test_direct_pairwise_evidence_names_its_scorer(ctx);
    rg_context_free(ctx);
    return 0;
}
