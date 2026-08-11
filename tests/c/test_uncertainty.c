#include "regulae.h"

#include <assert.h>
#include <math.h>
#include <string.h>

static void test_wilson_brackets_the_rate(void) {
    rg_uncertainty_estimate e;

    assert(rg_wilson_interval(5.0, 10.0, RG_DEFAULT_ALPHA, &e) == RG_OK);
    assert(fabs(e.estimate - 0.5) < 1e-12);
    assert(e.lower < 0.5);
    assert(e.upper > 0.5);
    assert(e.lower >= 0.0);
    assert(e.upper <= 1.0);
    assert(fabs(e.n - 10.0) < 1e-12);
    assert(fabs(e.alpha - 0.05) < 1e-12);
    assert(e.method == RG_UNCERTAINTY_WILSON);
}

/* The interval narrows as evidence accumulates. This is the property the
 * whole apparatus exists for, so it is asserted directly rather than through
 * any single hard-coded bound. */
static void test_wilson_narrows_with_more_evidence(void) {
    rg_uncertainty_estimate small;
    rg_uncertainty_estimate large;

    assert(rg_wilson_interval(5.0, 10.0, RG_DEFAULT_ALPHA, &small) == RG_OK);
    assert(rg_wilson_interval(500.0, 1000.0, RG_DEFAULT_ALPHA, &large) == RG_OK);
    assert(large.upper - large.lower < small.upper - small.lower);
}

static void test_wilson_tightens_as_alpha_widens(void) {
    rg_uncertainty_estimate strict;
    rg_uncertainty_estimate loose;

    assert(rg_wilson_interval(5.0, 10.0, 0.01, &strict) == RG_OK);
    assert(rg_wilson_interval(5.0, 10.0, 0.10, &loose) == RG_OK);
    assert(loose.upper - loose.lower < strict.upper - strict.lower);
}

/* An unobserved rate is unconstrained, not zero. The earlier implementation
 * reported [0, 0] here, which reads as certainty that the rate is zero. */
static void test_no_observations_is_unconstrained_not_zero(void) {
    rg_uncertainty_estimate e;

    assert(rg_wilson_interval(0.0, 0.0, RG_DEFAULT_ALPHA, &e) == RG_OK);
    assert(fabs(e.lower - 0.0) < 1e-12);
    assert(fabs(e.upper - 1.0) < 1e-12);
    assert(e.method == RG_UNCERTAINTY_NONE);
}

/* Confidence weighting produces non-integer counts, so the interval has to
 * accept them rather than assume a whole number of observations. */
static void test_wilson_accepts_fractional_counts(void) {
    rg_uncertainty_estimate e;

    assert(rg_wilson_interval(2.5, 7.25, RG_DEFAULT_ALPHA, &e) == RG_OK);
    assert(e.method == RG_UNCERTAINTY_WILSON);
    assert(e.lower <= e.estimate);
    assert(e.upper >= e.estimate);
}

static void test_unsupported_alpha_is_refused(void) {
    rg_uncertainty_estimate e;

    assert(rg_wilson_interval(1.0, 2.0, 0.5, &e) == RG_ERR_UNSUPPORTED_OPTION);
    assert(rg_percentile_interval(0, 0, 0.0, 0.0, 0.5, &e) == RG_ERR_UNSUPPORTED_OPTION);
}

static void test_percentile_spans_the_sample_quantiles(void) {
    /* Eleven evenly spaced samples: the 2.5% and 97.5% quantiles land inside
     * the range, and the interval must contain the bulk of the mass. */
    double samples[11];
    rg_uncertainty_estimate e;
    int i;

    for (i = 0; i < 11; i++) {
        samples[i] = (double)i / 10.0;
    }
    assert(rg_percentile_interval(samples, 11, 0.5, 40.0, RG_DEFAULT_ALPHA, &e) == RG_OK);
    assert(fabs(e.estimate - 0.5) < 1e-12);
    assert(e.lower >= 0.0);
    assert(e.upper <= 1.0);
    assert(e.lower < e.upper);
    assert(e.lower <= 0.1);
    assert(e.upper >= 0.9);
    assert(fabs(e.n - 40.0) < 1e-12);
    assert(e.method == RG_UNCERTAINTY_BOOTSTRAP);
}

static void test_percentile_ignores_sample_order(void) {
    double ascending[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
    double shuffled[5] = {0.4, 0.1, 0.5, 0.3, 0.2};
    rg_uncertainty_estimate a;
    rg_uncertainty_estimate b;

    assert(rg_percentile_interval(ascending, 5, 0.3, 10.0, RG_DEFAULT_ALPHA, &a) == RG_OK);
    assert(rg_percentile_interval(shuffled, 5, 0.3, 10.0, RG_DEFAULT_ALPHA, &b) == RG_OK);
    assert(fabs(a.lower - b.lower) < 1e-12);
    assert(fabs(a.upper - b.upper) < 1e-12);
}

static void test_percentile_clamps_samples_to_a_rate(void) {
    double samples[4] = {-0.5, 0.25, 0.75, 1.5};
    rg_uncertainty_estimate e;

    assert(rg_percentile_interval(samples, 4, 0.5, 8.0, RG_DEFAULT_ALPHA, &e) == RG_OK);
    assert(e.lower >= 0.0);
    assert(e.upper <= 1.0);
}

static void test_no_samples_is_unconstrained(void) {
    rg_uncertainty_estimate e;

    assert(rg_percentile_interval(0, 0, 0.25, 12.0, RG_DEFAULT_ALPHA, &e) == RG_OK);
    assert(fabs(e.lower - 0.0) < 1e-12);
    assert(fabs(e.upper - 1.0) < 1e-12);
    assert(fabs(e.estimate - 0.25) < 1e-12);
    assert(e.method == RG_UNCERTAINTY_NONE);
}

static void test_method_names_are_stable(void) {
    assert(strcmp(rg_uncertainty_method_string(RG_UNCERTAINTY_NONE), "none") == 0);
    assert(strcmp(rg_uncertainty_method_string(RG_UNCERTAINTY_WILSON), "wilson") == 0);
    assert(strcmp(rg_uncertainty_method_string(RG_UNCERTAINTY_BOOTSTRAP), "bootstrap") == 0);
}

int main(void) {
    test_wilson_brackets_the_rate();
    test_wilson_narrows_with_more_evidence();
    test_wilson_tightens_as_alpha_widens();
    test_no_observations_is_unconstrained_not_zero();
    test_wilson_accepts_fractional_counts();
    test_unsupported_alpha_is_refused();
    test_percentile_spans_the_sample_quantiles();
    test_percentile_ignores_sample_order();
    test_percentile_clamps_samples_to_a_rate();
    test_no_samples_is_unconstrained();
    test_method_names_are_stable();
    return 0;
}
