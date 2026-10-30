#include "internal.h"
#include "multilect_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Ensembles (docs/ensembles.md): refit per member on Bayesian-bootstrap
 * weights with sampled alignments. */

#define RG_ENSEMBLE_DEFAULT_MEMBERS 50

static uint64_t ensemble_mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/* A uniform draw in (0, 1]: never 0, so -log(u) is finite. */
static double ensemble_uniform(uint64_t *state) {
    *state = ensemble_mix(*state);
    return ((double)(*state >> 11) + 1.0) * (1.0 / 9007199254740992.0);
}

void rg_ensemble_options_init_defaults(rg_ensemble_options *options) {
    if (options == 0) {
        return;
    }
    options->member_count = RG_ENSEMBLE_DEFAULT_MEMBERS;
    options->seed = 20261030;
    options->alignment_temperature = 1.0;
}

void rg_ensemble_free(rg_ensemble *ensemble) {
    size_t i;
    if (ensemble == 0) {
        return;
    }
    for (i = 0; i < ensemble->member_count; i++) {
        rg_multi_model_free(ensemble->members[i]);
        free(ensemble->set_weights[i]);
    }
    for (i = 0; i < ensemble->cognate_count; i++) {
        free(ensemble->cognate_ids[i]);
    }
    free(ensemble->members);
    free(ensemble->set_weights);
    free(ensemble->member_seeds);
    free(ensemble->cognate_ids);
    free(ensemble->feature_system);
    free(ensemble);
}

rg_status rg_train_ensemble(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    const rg_ensemble_options *ensemble_options,
    rg_ensemble **out
) {
    rg_ensemble *ensemble;
    rg_train_options base;
    rg_ensemble_options eopts;
    rg_cognate_set *weighted = 0;
    size_t *group_of = 0;
    double *group_weight = 0;
    size_t m;
    size_t i;
    rg_status status = RG_OK;
    const char *system_name = 0;

    if (ctx == 0 || out == 0 || (cognate_count > 0 && cognates == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    if (options == 0) {
        rg_train_options_init_defaults(&base);
    } else {
        base = *options;
    }
    if (ensemble_options == 0) {
        rg_ensemble_options_init_defaults(&eopts);
    } else {
        eopts = *ensemble_options;
    }
    if (eopts.member_count == 0 || !isfinite(eopts.alignment_temperature) ||
        eopts.alignment_temperature < 0.0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    /* NML scores counts, and Bayesian-bootstrap weights are not counts. */
    if (base.bic.split_scorer == RG_SPLIT_SCORER_MULTINOMIAL_NML) {
        return RG_ERR_UNSUPPORTED_OPTION;
    }
    for (i = 0; i < cognate_count; i++) {
        if (cognates[i].confidence < 0.0 || cognates[i].confidence > 1.0) {
            return RG_ERR_INVALID_ARGUMENT;
        }
    }
    /* What a member does not do: its spread across members is the uncertainty,
     * and the nested resampling procedures would multiply the cost by their
     * own counts. */
    base.permutation_count = 0;
    base.tune_search_penalty = 0;
    base.bootstrap_n = 0;
    base.predictive_folds = 0;
    base.progress = 0;
    base.progress_user_data = 0;
    base.alignment_sampling_temperature = eopts.alignment_temperature;

    ensemble = (rg_ensemble *)calloc(1, sizeof(*ensemble));
    if (ensemble == 0) {
        return RG_ERR_OOM;
    }
    ensemble->options = eopts;
    ensemble->cognate_count = cognate_count;
    ensemble->members = (rg_multi_model **)calloc(eopts.member_count, sizeof(*ensemble->members));
    ensemble->set_weights = (double **)calloc(eopts.member_count, sizeof(*ensemble->set_weights));
    ensemble->member_seeds = (int *)calloc(eopts.member_count, sizeof(*ensemble->member_seeds));
    ensemble->cognate_ids = (char **)calloc(cognate_count + 1, sizeof(*ensemble->cognate_ids));
    weighted = (rg_cognate_set *)calloc(cognate_count + 1, sizeof(*weighted));
    if (ensemble->members == 0 || ensemble->set_weights == 0 || ensemble->member_seeds == 0 ||
        ensemble->cognate_ids == 0 || weighted == 0) {
        free(weighted);
        rg_ensemble_free(ensemble);
        return RG_ERR_OOM;
    }
    for (i = 0; i < cognate_count; i++) {
        ensemble->cognate_ids[i] = rg_strdup_internal(cognates[i].cognate_id != 0 ? cognates[i].cognate_id : "");
        if (ensemble->cognate_ids[i] == 0) {
            free(weighted);
            rg_ensemble_free(ensemble);
            return RG_ERR_OOM;
        }
    }
    if (rg_context_system_name(ctx, &system_name) == RG_OK && system_name != 0) {
        ensemble->feature_system = rg_strdup_internal(system_name);
    }
    ensemble->unit = resolve_bootstrap_unit(cognates, cognate_count, base.bootstrap_unit);
    status = observation_groups(cognates, cognate_count, ensemble->unit, &group_of,
                                &ensemble->group_count);
    if (status == RG_OK) {
        group_weight = (double *)calloc(ensemble->group_count + 1, sizeof(*group_weight));
        if (group_weight == 0) {
            status = RG_ERR_OOM;
        }
    }

    for (m = 0; m < eopts.member_count && status == RG_OK; m++) {
        uint64_t rng = ensemble_mix((uint64_t)(unsigned int)eopts.seed * 0x100000001B3ULL + m);
        rg_train_options member_options = base;
        double total = 0.0;
        double *weights;
        size_t g;

        ensemble->member_seeds[m] = (int)(ensemble_mix(rng) & 0x7FFFFFFF);
        /* Dirichlet(1, ..., 1) over the groups: normalised exponentials. */
        for (g = 0; g < ensemble->group_count; g++) {
            group_weight[g] = -log(ensemble_uniform(&rng));
            total += group_weight[g];
        }
        weights = (double *)calloc(cognate_count + 1, sizeof(*weights));
        if (weights == 0) {
            status = RG_ERR_OOM;
            break;
        }
        ensemble->set_weights[m] = weights;
        for (i = 0; i < cognate_count; i++) {
            /* Scaled to mean 1 over the groups, so evidence floors and the BIC
             * N keep their scale on average. */
            weights[i] = total > 0.0
                ? group_weight[group_of[i]] * (double)ensemble->group_count / total
                : 1.0;
            weighted[i] = cognates[i];
            weighted[i].confidence = cognates[i].confidence * weights[i];
        }
        member_options.alignment_sampling_seed = ensemble->member_seeds[m];
        status = train_model_internal(ctx, weighted, cognate_count, &member_options, 1,
                                      &ensemble->members[m]);
        if (status == RG_OK) {
            ensemble->member_count = m + 1;
        }
    }
    free(group_of);
    free(group_weight);
    free(weighted);
    if (status != RG_OK) {
        /* Members trained so far are freed with the ensemble; set_weights of a
         * member that failed are freed here. */
        for (m = ensemble->member_count; m < eopts.member_count; m++) {
            free(ensemble->set_weights[m]);
            ensemble->set_weights[m] = 0;
        }
        rg_ensemble_free(ensemble);
        return status;
    }
    *out = ensemble;
    return RG_OK;
}

size_t rg_ensemble_member_count(const rg_ensemble *ensemble) {
    return ensemble == 0 ? 0 : ensemble->member_count;
}

const rg_multi_model *rg_ensemble_member_at(const rg_ensemble *ensemble, size_t index) {
    if (ensemble == 0 || index >= ensemble->member_count) {
        return 0;
    }
    return ensemble->members[index];
}

const double *rg_ensemble_member_set_weights(
    const rg_ensemble *ensemble,
    size_t index,
    size_t *count
) {
    if (count != 0) {
        *count = 0;
    }
    if (ensemble == 0 || index >= ensemble->member_count) {
        return 0;
    }
    if (count != 0) {
        *count = ensemble->cognate_count;
    }
    return ensemble->set_weights[index];
}
