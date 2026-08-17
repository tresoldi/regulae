#include "regulae.h"

#include <assert.h>
#include <math.h>
#include <string.h>

static rg_multi_model *train_fixture(
    rg_context *ctx,
    const char *path,
    int folds,
    rg_corpus **out_corpus
) {
    rg_corpus *corpus = 0;
    rg_multi_model *model = 0;
    rg_load_diagnosis diagnosis;
    rg_train_options options;
    assert(rg_corpus_load_tsv(path, 0, &corpus, &diagnosis) == RG_OK);
    rg_train_options_init_defaults(&options);
    options.predictive_folds = folds;
    assert(rg_train_model(ctx, rg_corpus_cognates(corpus),
                          rg_corpus_cognate_count(corpus), &options, &model) == RG_OK);
    *out_corpus = corpus;
    return model;
}

static void test_grouped_prediction_confirms_a_recurrent_association(rg_context *ctx) {
    rg_corpus *corpus = 0;
    rg_multi_model *model = train_fixture(
        ctx, REGULAE_SOURCE_DIR "/testdata/evaluation/m4/conditioned_generalises.tsv",
        4, &corpus);
    const rg_corpus_fit *fit = rg_multi_model_fit(model);
    const rg_multi_class_row *rows;
    size_t row_count = 0;
    size_t i;
    int confirmed_pf = 0;

    /* Fifty cognate rows are only twenty-five independent histories. If rows
     * or aligned positions were split, this number would be fifty or larger. */
    assert(rg_corpus_cognate_count(corpus) == 50);
    assert(fit->predictive_group_count == 25);
    assert(fit->predictive.observation_unit == RG_OBSERVATION_UNIT_DEPENDENCY_COMPONENT);
    assert(fit->predictive.folds == 4);
    assert(fit->predictive_pair_orientations == 2);
    assert(fit->predictive.status == RG_PREDICTIVE_CONFIRMED);
    assert(fit->predictive.log_loss_gain > 0.04);
    assert(fit->predictive.conditioned.log_loss < fit->predictive.unconditioned.log_loss);
    assert(fit->predictive.conditioned.log_loss < fit->predictive_identity.log_loss);
    assert(fit->predictive.conditioned.log_loss < fit->predictive_inventory_frequency.log_loss);
    assert(fit->predictive.conditioned.log_loss < fit->predictive_feature_distance.log_loss);
    assert(fit->predictive.conditioned.top_k_coverage > 0.9);
    assert(fit->predictive.conditioned.unseen_reflex_count == 4);

    rows = rg_multi_model_conditioned_classes(model, &row_count);
    for (i = 0; i < row_count; i++) {
        if (rows[i].segment_count == 2 &&
            strcmp(rows[i].lect_ids[0], "A") == 0 &&
            strcmp(rows[i].graphemes[0], "p") == 0 &&
            strcmp(rows[i].lect_ids[1], "B") == 0 &&
            strcmp(rows[i].graphemes[1], "f") == 0) {
            assert(rows[i].evidence.predictive.status == RG_PREDICTIVE_CONFIRMED);
            assert(rows[i].evidence.predictive.folds >= 2);
            assert(rows[i].evidence.predictive.conditioned.observation_count >= 8);
            assert(rows[i].evidence.predictive.log_loss_gain > 0.0);
            confirmed_pf = 1;
        }
    }
    assert(confirmed_pf);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

static void test_real_negative_panel_does_not_gain_from_conditioning(rg_context *ctx) {
    rg_corpus *corpus = 0;
    rg_multi_model *model = train_fixture(
        ctx, REGULAE_SOURCE_DIR "/testdata/restraint/chance.tsv", 3, &corpus);
    const rg_corpus_fit *fit = rg_multi_model_fit(model);
    assert(fit->predictive.status == RG_PREDICTIVE_NOT_CONFIRMED);
    assert(fabs(fit->predictive.log_loss_gain) < 1e-12);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

static void test_cross_dimensional_rule_gets_its_own_confirmation(rg_context *ctx) {
    rg_corpus *corpus = 0;
    rg_multi_model *model = train_fixture(
        ctx, REGULAE_SOURCE_DIR "/testdata/evaluation/m4/cross_dimensional_generalises.tsv",
        4, &corpus);
    const rg_multi_cross_dimensional_row *rows;
    size_t row_count = 0;
    size_t i;
    int confirmed = 0;
    rows = rg_multi_model_cross_dimensional_rows(model, &row_count);
    for (i = 0; i < row_count; i++) {
        if (strcmp(rows[i].rule.dimension, "tone") == 0 &&
            strcmp(rows[i].rule.value, "1") == 0) {
            assert(rows[i].rule.evidence.predictive.status == RG_PREDICTIVE_CONFIRMED);
            assert(rows[i].rule.evidence.predictive.folds == 4);
            assert(rows[i].rule.evidence.predictive.conditioned.observation_count == 20);
            assert(rows[i].rule.evidence.predictive.log_loss_gain > 0.6);
            confirmed = 1;
        }
    }
    assert(confirmed);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

static void test_leave_one_lect_out_pools_other_lects(rg_context *ctx) {
    rg_corpus *corpus = 0;
    rg_multi_model *model = train_fixture(
        ctx, REGULAE_SOURCE_DIR "/testdata/evaluation/m4/leave_one_lect_out.tsv",
        3, &corpus);
    const rg_corpus_fit *fit = rg_multi_model_fit(model);
    assert(fit->predictive_pair_orientations == 6);
    assert(fit->predictive_leave_one_lect_out_cases == 108);
    assert(fit->predictive_leave_one_lect_out_conditioned.observation_count == 324);
    assert(fit->predictive_leave_one_lect_out_conditioned.log_loss <
           fit->predictive_leave_one_lect_out_unconditioned.log_loss);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

static void test_small_corpus_stays_descriptive(rg_context *ctx) {
    rg_corpus *corpus = 0;
    rg_multi_model *model = train_fixture(
        ctx, REGULAE_SOURCE_DIR "/testdata/evaluation/m4/descriptive_only.tsv", 5, &corpus);
    const rg_corpus_fit *fit = rg_multi_model_fit(model);
    assert(fit->predictive.status == RG_PREDICTIVE_DESCRIPTIVE_ONLY);
    assert(fit->predictive.conditioned.observation_count == 0);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
}

static void test_invalid_predictive_configuration_is_refused(rg_context *ctx) {
    rg_corpus *corpus = 0;
    rg_multi_model *model = 0;
    rg_train_options options;
    assert(rg_corpus_load_tsv(
        REGULAE_SOURCE_DIR "/testdata/evaluation/m4/descriptive_only.tsv",
        0, &corpus, 0) == RG_OK);
    rg_train_options_init_defaults(&options);
    options.predictive_folds = 1;
    assert(rg_train_model(ctx, rg_corpus_cognates(corpus),
                          rg_corpus_cognate_count(corpus), &options, &model) ==
           RG_ERR_INVALID_ARGUMENT);
    assert(model == 0);
    rg_train_options_init_defaults(&options);
    options.predictive_abstention_threshold = 1.1;
    assert(rg_train_model(ctx, rg_corpus_cognates(corpus),
                          rg_corpus_cognate_count(corpus), &options, &model) ==
           RG_ERR_INVALID_ARGUMENT);
    assert(model == 0);
    rg_corpus_free(corpus);
}

int main(void) {
    rg_context *ctx = 0;
    assert(rg_context_new_builtin(&ctx) == RG_OK);
    test_grouped_prediction_confirms_a_recurrent_association(ctx);
    test_real_negative_panel_does_not_gain_from_conditioning(ctx);
    test_cross_dimensional_rule_gets_its_own_confirmation(ctx);
    test_leave_one_lect_out_pools_other_lects(ctx);
    test_small_corpus_stays_descriptive(ctx);
    test_invalid_predictive_configuration_is_refused(ctx);
    rg_context_free(ctx);
    return 0;
}
