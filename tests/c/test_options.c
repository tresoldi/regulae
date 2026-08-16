#include "regulae.h"

#include <assert.h>
#include <math.h>
#include <string.h>

int main(void) {
    rg_train_options options;
    rg_bic_config bic;

    assert(strcmp(rg_version_string(), "0.1.0") == 0);
    assert(rg_version_major() == 0);
    assert(rg_version_minor() == 1);
    assert(rg_version_patch() == 0);
    assert(rg_abi_version() == 28);
    assert(strcmp(rg_status_string(RG_OK), "ok") == 0);

    rg_bic_config_init_defaults(&bic);
    assert(bic.split_scorer == RG_SPLIT_SCORER_CORRECTED_BIC);
    assert(strcmp(rg_split_scorer_string(bic.split_scorer), "corrected_bic") == 0);
    assert(fabs(bic.split_prior_concentration - 1.0) < 1e-12);
    assert(fabs(bic.delta_bic_threshold) < 1e-12);
    assert(bic.min_split_observations == 2);
    assert(bic.max_split_depth == 3);
    assert(bic.min_chunk_observations == 2);
    assert(fabs(bic.long_range_delta_bic_threshold) < 1e-12);
    assert(bic.long_range_min_split_observations == 5);
    assert(fabs(bic.long_range_min_dominant_fraction - 0.6) < 1e-12);
    assert(bic.cross_dim_max_iterations == 5);
    assert(bic.cross_dim_min_rule_count == 3);
    /* 0.0 on purpose: a fixed fraction does not measure conditioning, so the
     * decision belongs to the BIC gate against the complementary environment. */
    assert(fabs(bic.cross_dim_min_rule_confidence - 0.0) < 1e-12);
    assert(fabs(bic.cross_dim_delta_bic_threshold) < 1e-12);
    assert(fabs(bic.search_penalty_gamma - 1.0) < 1e-12);
    assert(bic.multi_lect_bic_small_sample_correction == 0);
    assert(fabs(bic.multi_lect_min_commit_scale - 0.5) < 1e-12);

    rg_train_options_init_defaults(&options);
    assert(options.max_chunk_size == RG_DEFAULT_MAX_CHUNK_SIZE);
    assert(fabs(options.temperature - 1.0) < 1e-12);
    assert(fabs(options.concentration - 5.0) < 1e-12);
    assert(options.max_iter == 30);
    assert(fabs(options.convergence_eps - 1e-4) < 1e-12);
    assert(fabs(options.segment_weight - 0.7) < 1e-12);
    assert(fabs(options.displacement_weight - 0.3) < 1e-12);
    assert(fabs(options.tone_weight - 1.0) < 1e-12);
    assert(fabs(options.chunk_min_transparency - 0.0) < 1e-12);
    assert(options.bootstrap_n == 0);
    assert(options.bootstrap_seed == 0);
    assert(options.bootstrap_unit == RG_OBSERVATION_UNIT_AUTO);
    assert(options.predictive_folds == 0);
    assert(options.predictive_seed == 20260816);
    assert(options.predictive_min_groups == 2);
    assert(fabs(options.predictive_abstention_threshold - 0.5) < 1e-12);
    assert(options.predictive_top_k == 3);
    assert(strcmp(rg_observation_unit_string(RG_OBSERVATION_UNIT_ETYMON_GROUP),
                  "etymon_group") == 0);
    assert(strcmp(rg_observation_unit_string(RG_OBSERVATION_UNIT_DEPENDENCY_COMPONENT),
                  "dependency_component") == 0);
    assert(strcmp(rg_predictive_status_string(RG_PREDICTIVE_DESCRIPTIVE_ONLY),
                  "descriptive_only") == 0);

    rg_bic_config_init_defaults(0);
    rg_train_options_init_defaults(0);
    rg_string_free(0);
    return 0;
}
