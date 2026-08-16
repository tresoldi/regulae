#include "split_score.h"

#include <assert.h>
#include <math.h>

static rg_split_score_config config(rg_split_scorer scorer) {
    rg_split_score_config out;
    out.scorer = scorer;
    out.dirichlet_concentration = 1.0;
    out.bic_log_sample_size = log(8.0);
    out.bic_extra_penalty = 0.0;
    out.candidate_count = 1;
    out.search_gamma = 0.0;
    return out;
}

static void test_corrected_bic_uses_multinomial_dimension(void) {
    const double pooled[] = {4.0, 4.0};
    const double inside[] = {4.0, 0.0};
    const double outside[] = {0.0, 4.0};
    rg_split_score_config cfg = config(RG_SPLIT_SCORER_CORRECTED_BIC);
    rg_split_score_result score;
    double expected = -16.0 * log(2.0) + log(8.0);

    assert(rg_categorical_split_score_internal(pooled, inside, outside, 2,
                                               &cfg, &score) == RG_OK);
    assert(fabs(score.delta - expected) < 1e-12);
    assert(fabs(score.complexity_charge - log(8.0)) < 1e-12);
}

static void test_search_charge_is_common_currency(void) {
    const double pooled[] = {4.0, 4.0};
    const double inside[] = {4.0, 0.0};
    const double outside[] = {0.0, 4.0};
    rg_split_score_config cfg = config(RG_SPLIT_SCORER_CORRECTED_BIC);
    rg_split_score_result score;

    cfg.candidate_count = 100;
    cfg.search_gamma = 0.5;
    assert(rg_categorical_split_score_internal(pooled, inside, outside, 2,
                                               &cfg, &score) == RG_OK);
    assert(fabs(score.search_charge - log(100.0)) < 1e-12);
}

static void test_exact_nml_normalizes_the_fixed_alphabet(void) {
    const double pooled2[] = {1.0, 1.0};
    const double inside2[] = {1.0, 0.0};
    const double outside2[] = {0.0, 1.0};
    const double pooled3[] = {1.0, 1.0, 0.0};
    const double inside3[] = {1.0, 0.0, 0.0};
    const double outside3[] = {0.0, 1.0, 0.0};
    rg_split_score_config cfg = config(RG_SPLIT_SCORER_MULTINOMIAL_NML);
    rg_split_score_result score;

    assert(rg_categorical_split_score_internal(pooled2, inside2, outside2, 2,
                                               &cfg, &score) == RG_OK);
    assert(fabs(score.delta - (-4.0 * log(2.0) + 2.0 * log(4.0 / 2.5))) < 1e-12);
    assert(rg_categorical_split_score_internal(pooled3, inside3, outside3, 3,
                                               &cfg, &score) == RG_OK);
    assert(fabs(score.delta - (-4.0 * log(2.0) + 2.0 * log(9.0 / 4.5))) < 1e-12);
}

static void test_exact_nml_refuses_fractional_pseudo_counts(void) {
    const double pooled[] = {0.5, 0.5};
    const double inside[] = {0.5, 0.0};
    const double outside[] = {0.0, 0.5};
    rg_split_score_config cfg = config(RG_SPLIT_SCORER_MULTINOMIAL_NML);
    rg_split_score_result score;

    assert(rg_categorical_split_score_internal(pooled, inside, outside, 2,
                                               &cfg, &score) == RG_ERR_UNSUPPORTED_OPTION);
}

static void test_dirichlet_marginal_accepts_fractional_mass(void) {
    const double pooled[] = {1.5, 0.5};
    const double inside[] = {1.0, 0.0};
    const double outside[] = {0.5, 0.5};
    rg_split_score_config cfg = config(RG_SPLIT_SCORER_DIRICHLET_MARGINAL);
    rg_split_score_result score;
    double alpha = 0.5;
    double baseline_log = lgamma(1.0) - lgamma(3.0) +
                          lgamma(1.5 + alpha) - lgamma(alpha) +
                          lgamma(0.5 + alpha) - lgamma(alpha);
    double inside_log = lgamma(1.0) - lgamma(2.0) +
                        lgamma(1.0 + alpha) - lgamma(alpha);
    double outside_log = lgamma(1.0) - lgamma(2.0) +
                         2.0 * (lgamma(0.5 + alpha) - lgamma(alpha));
    double expected = -2.0 * (inside_log + outside_log - baseline_log);

    assert(rg_categorical_split_score_internal(pooled, inside, outside, 2,
                                               &cfg, &score) == RG_OK);
    assert(fabs(score.delta - expected) < 1e-12);
}

int main(void) {
    test_corrected_bic_uses_multinomial_dimension();
    test_search_charge_is_common_currency();
    test_exact_nml_normalizes_the_fixed_alphabet();
    test_exact_nml_refuses_fractional_pseudo_counts();
    test_dirichlet_marginal_accepts_fractional_mass();
    return 0;
}
