#include "internal.h"

void rg_bic_config_init_defaults(rg_bic_config *config) {
    if (config == 0) {
        return;
    }
    config->split_scorer = RG_SPLIT_SCORER_CORRECTED_BIC;
    config->split_prior_concentration = 1.0;
    config->delta_bic_threshold = 0.0;
    /* Both sides of an immediate split need three observations, not two.
     *
     * At two, a partition of two aligned positions can commit a conditioned
     * rule, which is far under the roughly eight-example evidence floor the
     * search is known to need before it recovers a real environment. What came
     * out of the gap was not recall: on the Grimm fixture -- three
     * unconditioned shifts, where the README says any conditioning found is
     * invention -- it published four rules, of which `gmc:m ~ pie:m`,
     * `gmc:o ~ pie:o` and a `gmc:t ~ pie:t` carved against `gmc:t ~ pie:d`
     * were all built on two or three observations. The last reads as a
     * conditional merger of PIE *t and *d.
     *
     * Swept over every corpus in the tree at 2, 3 and 4. Three drops Grimm to
     * one rule and clears Grassmann's spurious identity row while keeping both
     * real Grassmann rules; the four restraint corpora stay silent, every
     * graded rung resolves unchanged, and no soundlaw assertion moves. Four
     * takes Grimm to zero but takes final devoicing with it, so it is past the
     * point where the cost is only noise. */
    config->min_split_observations = 3;
    config->max_split_depth = 3;
    config->min_chunk_observations = 2;
    config->long_range_delta_bic_threshold = 0.0;
    config->long_range_min_split_observations = 5;
    config->long_range_min_dominant_fraction = 0.6;
    config->cross_dim_max_iterations = 5;
    config->cross_dim_min_rule_count = 3;
    config->cross_dim_min_rule_confidence = 0.0;
    config->cross_dim_delta_bic_threshold = 0.0;
    config->multi_lect_bic_small_sample_correction = 0;
    config->multi_lect_min_commit_scale = 0.5;
    config->search_penalty_gamma = RG_SEARCH_PENALTY_GAMMA;
    config->class_outcome_mode = RG_CLASS_OUTCOME_PER_SISTER_LECT;
}

void rg_train_options_init_defaults(rg_train_options *options) {
    if (options == 0) {
        return;
    }
    options->max_chunk_size = RG_DEFAULT_MAX_CHUNK_SIZE;
    options->temperature = 1.0;
    options->concentration = 5.0;
    options->max_iter = 30;
    options->convergence_eps = 1e-4;
    options->segment_weight = 0.7;
    options->displacement_weight = 0.3;
    options->tone_weight = 1.0;
    options->chunk_min_transparency = 0.0;
    rg_bic_config_init_defaults(&options->bic);
    options->bootstrap_n = 0;
    options->bootstrap_seed = 0;
    options->bootstrap_unit = RG_OBSERVATION_UNIT_AUTO;
    options->permutation_count = 0;
    options->tune_search_penalty = 0;
    options->permutation_seed = 20260815;
    options->predictive_folds = 0;
    options->predictive_seed = 20260816;
    options->predictive_min_groups = 2;
    options->predictive_abstention_threshold = 0.5;
    options->predictive_top_k = 3;
    options->ibm1_prior = -1;
    options->progress = 0;
    options->progress_user_data = 0;
}

void rg_progress_init_internal(rg_progress_state *state, const rg_train_options *options, size_t total) {
    if (state == 0) {
        return;
    }
    state->fn = options == 0 ? 0 : options->progress;
    state->user_data = options == 0 ? 0 : options->progress_user_data;
    state->completed = 0;
    state->total = total == 0 ? 1 : total;
    state->cancelled = false;
}

bool rg_progress_step_internal(rg_progress_state *state, const char *stage) {
    if (state == 0 || state->fn == 0) {
        return false;
    }
    if (state->cancelled) {
        return true;
    }
    if (state->completed < state->total) {
        state->completed++;
    }
    if (state->fn(stage, state->completed, state->total, state->user_data)) {
        state->cancelled = true;
    }
    return state->cancelled;
}
