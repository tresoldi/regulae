#include "split_score.h"

#include <float.h>
#include <math.h>
#include <string.h>

static double log_add(double a, double b) {
    double larger;
    double smaller;
    if (a == -INFINITY) {
        return b;
    }
    if (b == -INFINITY) {
        return a;
    }
    larger = a > b ? a : b;
    smaller = a > b ? b : a;
    return larger + log1p(exp(smaller - larger));
}

static double vector_total(const double *mass, size_t count) {
    double total = 0.0;
    size_t i;
    for (i = 0; i < count; i++) {
        total += mass[i];
    }
    return total;
}

static double multinomial_nll(const double *mass, size_t count) {
    double total = vector_total(mass, count);
    double cost = 0.0;
    size_t i;
    if (total <= 0.0) {
        return 0.0;
    }
    for (i = 0; i < count; i++) {
        if (mass[i] > 0.0) {
            cost -= mass[i] * log(mass[i] / total);
        }
    }
    return cost;
}

static int integer_sample_size(double mass, size_t *out) {
    double rounded;
    if (!isfinite(mass) || mass < 0.0 || mass > (double)SIZE_MAX) {
        return 0;
    }
    rounded = floor(mass + 0.5);
    if (fabs(mass - rounded) > 1e-9 * fmax(1.0, mass)) {
        return 0;
    }
    *out = (size_t)rounded;
    return 1;
}

static double binomial_log_normalizer(size_t n) {
    double log_sum = -INFINITY;
    double log_n_factorial = lgamma((double)n + 1.0);
    size_t r;
    if (n == 0) {
        return 0.0;
    }
    for (r = 0; r <= n; r++) {
        size_t other = n - r;
        double term = log_n_factorial - lgamma((double)r + 1.0) -
                      lgamma((double)other + 1.0);
        if (r > 0) {
            term += (double)r * log((double)r / (double)n);
        }
        if (other > 0) {
            term += (double)other * log((double)other / (double)n);
        }
        log_sum = log_add(log_sum, term);
    }
    return log_sum;
}

static double multinomial_log_normalizer(size_t n, size_t outcomes) {
    double previous_two = 0.0;
    double previous_one;
    size_t k;
    if (outcomes <= 1 || n == 0) {
        return 0.0;
    }
    previous_one = binomial_log_normalizer(n);
    for (k = 3; k <= outcomes; k++) {
        double current = log_add(previous_one,
                                 log((double)n / (double)(k - 2)) + previous_two);
        previous_two = previous_one;
        previous_one = current;
    }
    return previous_one;
}

static double dirichlet_nll(
    const double *mass,
    size_t count,
    double concentration
) {
    double alpha = concentration / (double)count;
    double total = vector_total(mass, count);
    double log_probability = lgamma(concentration) - lgamma(total + concentration);
    size_t i;
    for (i = 0; i < count; i++) {
        log_probability += lgamma(mass[i] + alpha) - lgamma(alpha);
    }
    return -log_probability;
}

rg_status rg_categorical_split_score_internal(
    const double *pooled,
    const double *inside,
    const double *outside,
    size_t outcome_count,
    const rg_split_score_config *config,
    rg_split_score_result *out
) {
    double baseline;
    double split;
    double search_charge;
    double complexity = 0.0;
    if (pooled == 0 || inside == 0 || outside == 0 || config == 0 || out == 0 ||
        outcome_count < 2 || config->search_gamma < 0.0 ||
        config->bic_extra_penalty < 0.0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    search_charge = config->candidate_count > 1
        ? config->search_gamma * 2.0 * log((double)config->candidate_count)
        : 0.0;
    baseline = multinomial_nll(pooled, outcome_count);
    split = multinomial_nll(inside, outcome_count) + multinomial_nll(outside, outcome_count);
    switch (config->scorer) {
        case RG_SPLIT_SCORER_CORRECTED_BIC:
            if (!isfinite(config->bic_log_sample_size) || config->bic_log_sample_size < 0.0) {
                return RG_ERR_INVALID_ARGUMENT;
            }
            complexity = (double)(outcome_count - 1) *
                         (config->bic_log_sample_size + config->bic_extra_penalty);
            break;
        case RG_SPLIT_SCORER_MULTINOMIAL_NML: {
            size_t pooled_n;
            size_t inside_n;
            size_t outside_n;
            if (!integer_sample_size(vector_total(pooled, outcome_count), &pooled_n) ||
                !integer_sample_size(vector_total(inside, outcome_count), &inside_n) ||
                !integer_sample_size(vector_total(outside, outcome_count), &outside_n) ||
                pooled_n != inside_n + outside_n) {
                return RG_ERR_UNSUPPORTED_OPTION;
            }
            complexity = 2.0 * (multinomial_log_normalizer(inside_n, outcome_count) +
                                multinomial_log_normalizer(outside_n, outcome_count) -
                                multinomial_log_normalizer(pooled_n, outcome_count));
            break;
        }
        case RG_SPLIT_SCORER_DIRICHLET_MARGINAL:
            if (!isfinite(config->dirichlet_concentration) ||
                config->dirichlet_concentration <= DBL_MIN) {
                return RG_ERR_INVALID_ARGUMENT;
            }
            baseline = dirichlet_nll(pooled, outcome_count, config->dirichlet_concentration);
            split = dirichlet_nll(inside, outcome_count, config->dirichlet_concentration) +
                    dirichlet_nll(outside, outcome_count, config->dirichlet_concentration);
            break;
        default:
            return RG_ERR_UNSUPPORTED_OPTION;
    }
    memset(out, 0, sizeof(*out));
    out->search_charge = search_charge;
    out->complexity_charge = complexity;
    out->delta = 2.0 * (split - baseline) + complexity + search_charge;
    return RG_OK;
}
