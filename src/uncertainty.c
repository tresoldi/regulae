/* Intervals on the counts a trained model publishes. Every count is an
 * estimate from finite observations, so every count-bearing row carries one.
 * The interval is always on a rate in [0, 1] — the conditional probability the
 * count represents — never on the raw count, so intervals stay comparable
 * across rows with different denominators. */

#include "internal.h"

#include <math.h>
#include <stdlib.h>

/* Two-sided z-scores, hard-coded so the library needs no statistics
 * dependency. Only these three alphas are accepted; anything else is a caller
 * error rather than a silently-wrong interval. */
static int z_for_alpha(double alpha, double *out) {
    if (alpha == 0.10) {
        *out = 1.6448536269514722;
        return 1;
    }
    if (alpha == 0.05) {
        *out = 1.959963984540054;
        return 1;
    }
    if (alpha == 0.01) {
        *out = 2.5758293035489004;
        return 1;
    }
    return 0;
}

static rg_uncertainty_estimate unconstrained(double estimate, double total, double alpha) {
    rg_uncertainty_estimate out;
    out.estimate = estimate;
    out.lower = 0.0;
    out.upper = 1.0;
    out.n = total > 0.0 ? total : 0.0;
    out.alpha = alpha;
    out.method = RG_UNCERTAINTY_NONE;
    out.post_selection = 0;
    return out;
}

rg_status rg_wilson_interval(
    double count,
    double total,
    double alpha,
    rg_uncertainty_estimate *out
) {
    double z = 0.0;
    if (out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    if (!z_for_alpha(alpha, &z)) {
        return RG_ERR_UNSUPPORTED_OPTION;
    }
    if (total <= 0.0) {
        *out = unconstrained(0.0, 0.0, alpha);
        return RG_OK;
    }
    {
        double p = count / total;
        double z2 = z * z;
        double denom = 1.0 + z2 / total;
        double center = (p + z2 / (2.0 * total)) / denom;
        double inner = (p * (1.0 - p) + z2 / (4.0 * total)) / total;
        double margin = z * sqrt(inner > 0.0 ? inner : 0.0) / denom;
        if (p < 0.0) {
            p = 0.0;
        }
        if (p > 1.0) {
            p = 1.0;
        }
        out->estimate = p;
        out->lower = center - margin < 0.0 ? 0.0 : center - margin;
        out->upper = center + margin > 1.0 ? 1.0 : center + margin;
        out->n = total;
        out->alpha = alpha;
        out->method = RG_UNCERTAINTY_WILSON;
        out->post_selection = 0;
    }
    return RG_OK;
}

static int compare_double(const void *a, const void *b) {
    double left = *(const double *)a;
    double right = *(const double *)b;
    if (left < right) {
        return -1;
    }
    if (left > right) {
        return 1;
    }
    return 0;
}

/* Linear-interpolated quantile on a pre-sorted array, q in [0, 1]. */
static double quantile(const double *ordered, size_t count, double q) {
    double pos;
    size_t lo_index;
    size_t hi_index;
    double frac;
    if (count == 0) {
        return 0.0;
    }
    if (count == 1 || q <= 0.0) {
        return ordered[0];
    }
    if (q >= 1.0) {
        return ordered[count - 1];
    }
    pos = q * (double)(count - 1);
    lo_index = (size_t)floor(pos);
    hi_index = (size_t)ceil(pos);
    if (lo_index == hi_index) {
        return ordered[lo_index];
    }
    frac = pos - (double)lo_index;
    return ordered[lo_index] * (1.0 - frac) + ordered[hi_index] * frac;
}

rg_status rg_percentile_interval(
    const double *samples,
    size_t sample_count,
    double estimate,
    double total,
    double alpha,
    rg_uncertainty_estimate *out
) {
    double z = 0.0;
    double *ordered;
    size_t i;
    if (out == 0 || (samples == 0 && sample_count > 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    if (!z_for_alpha(alpha, &z)) {
        return RG_ERR_UNSUPPORTED_OPTION;
    }
    if (sample_count == 0) {
        *out = unconstrained(estimate, total, alpha);
        return RG_OK;
    }
    ordered = (double *)malloc(sample_count * sizeof(*ordered));
    if (ordered == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < sample_count; i++) {
        double value = samples[i];
        if (value < 0.0) {
            value = 0.0;
        }
        if (value > 1.0) {
            value = 1.0;
        }
        ordered[i] = value;
    }
    qsort(ordered, sample_count, sizeof(*ordered), compare_double);
    out->estimate = estimate;
    out->lower = quantile(ordered, sample_count, alpha / 2.0);
    out->upper = quantile(ordered, sample_count, 1.0 - alpha / 2.0);
    out->n = total > 0.0 ? total : 0.0;
    out->alpha = alpha;
    out->method = RG_UNCERTAINTY_BOOTSTRAP;
    out->post_selection = 0;
    free(ordered);
    return RG_OK;
}

const char *rg_uncertainty_method_string(rg_uncertainty_method method) {
    switch (method) {
    case RG_UNCERTAINTY_NONE:
        return "none";
    case RG_UNCERTAINTY_WILSON:
        return "wilson";
    case RG_UNCERTAINTY_BOOTSTRAP:
        return "bootstrap";
    default:
        return "unknown";
    }
}

rg_uncertainty_estimate rg_wilson_default_internal(double count, double total) {
    rg_uncertainty_estimate out;
    if (rg_wilson_interval(count, total, RG_DEFAULT_ALPHA, &out) != RG_OK) {
        return unconstrained(0.0, total, RG_DEFAULT_ALPHA);
    }
    return out;
}

const char *rg_rule_standing_string(rg_rule_standing standing) {
    switch (standing) {
    case RG_RULE_STANDING_ABOVE_NOISE:
        return "above noise";
    case RG_RULE_STANDING_WITHIN_NOISE:
        return "within noise";
    case RG_RULE_STANDING_UNMEASURED:
        return "unmeasured";
    default:
        return "unknown";
    }
}

void rg_rule_evidence_judge_internal(rg_rule_evidence *evidence, double null_search_margin) {
    if (evidence == 0) {
        return;
    }
    evidence->standing = evidence->search_margin > null_search_margin
        ? RG_RULE_STANDING_ABOVE_NOISE : RG_RULE_STANDING_WITHIN_NOISE;
}
