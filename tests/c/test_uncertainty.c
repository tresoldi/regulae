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

/* bootstrap_n was accepted, JSON-round-tripped, and did nothing: the
 * percentile-interval code was written at the port and had no caller. What it
 * resamples matters -- whole cognate sets, not aligned positions. A position
 * is not an independent observation of anything; a Latin-Spanish corpus has
 * 413 of them over 97 sets. */
static void test_bootstrap_resamples_cognate_sets(void) {
    rg_context *ctx = 0;
    rg_corpus *corpus = 0;
    rg_multi_model *closed_form = 0;
    rg_multi_model *resampled = 0;
    rg_train_options options;
    size_t i;
    int any_differ = 0;

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/soundlaws/rhotacism.tsv", 0, &corpus) == RG_OK);
    rg_train_options_init_defaults(&options);
    assert(rg_train_model(ctx, rg_corpus_cognate_at(corpus, 0),
                          rg_corpus_cognate_count(corpus), &options, &closed_form) == RG_OK);
    options.bootstrap_n = 200;
    options.bootstrap_seed = 7;
    assert(rg_train_model(ctx, rg_corpus_cognate_at(corpus, 0),
                          rg_corpus_cognate_count(corpus), &options, &resampled) == RG_OK);

    assert(rg_multi_model_unconditioned_class_count(closed_form) ==
           rg_multi_model_unconditioned_class_count(resampled));
    for (i = 0; i < rg_multi_model_unconditioned_class_count(resampled); i++) {
        const rg_multi_class_row *plain = rg_multi_model_unconditioned_class_at(closed_form, i);
        const rg_multi_class_row *boot = rg_multi_model_unconditioned_class_at(resampled, i);
        assert(plain->uncertainty.method == RG_UNCERTAINTY_WILSON);
        assert(boot->uncertainty.method == RG_UNCERTAINTY_BOOTSTRAP);
        assert(boot->uncertainty.lower <= boot->uncertainty.estimate + 1e-9);
        assert(boot->uncertainty.upper + 1e-9 >= boot->uncertainty.estimate);
        if (boot->uncertainty.lower != plain->uncertainty.lower) {
            any_differ = 1;
        }
    }
    assert(any_differ);

    /* Same seed, same interval: an interval a reader cannot recompute is not
     * an interval. */
    {
        rg_multi_model *again = 0;
        assert(rg_train_model(ctx, rg_corpus_cognate_at(corpus, 0),
                              rg_corpus_cognate_count(corpus), &options, &again) == RG_OK);
        for (i = 0; i < rg_multi_model_unconditioned_class_count(again); i++) {
            assert(rg_multi_model_unconditioned_class_at(again, i)->uncertainty.lower ==
                   rg_multi_model_unconditioned_class_at(resampled, i)->uncertainty.lower);
        }
        rg_multi_model_free(again);
    }
    rg_multi_model_free(closed_form);
    rg_multi_model_free(resampled);
    rg_corpus_free(corpus);
    rg_context_free(ctx);
}

/* An interval on a bucket the search chose from the same data says how well
 * the rate is pinned given that environment, and not whether the environment
 * is real. It has to say so on the row. */
static void test_conditioned_intervals_say_they_are_post_selection(void) {
    rg_context *ctx = 0;
    rg_corpus *corpus = 0;
    rg_multi_model *model = 0;
    rg_train_options options;
    size_t i;

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/soundlaws/rhotacism.tsv", 0, &corpus) == RG_OK);
    rg_train_options_init_defaults(&options);
    assert(rg_train_model(ctx, rg_corpus_cognate_at(corpus, 0),
                          rg_corpus_cognate_count(corpus), &options, &model) == RG_OK);
    assert(rg_multi_model_conditioned_class_count(model) > 0);
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        assert(rg_multi_model_conditioned_class_at(model, i)->uncertainty.post_selection);
    }
    for (i = 0; i < rg_multi_model_unconditioned_class_count(model); i++) {
        assert(!rg_multi_model_unconditioned_class_at(model, i)->uncertainty.post_selection);
    }
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
    rg_context_free(ctx);
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
