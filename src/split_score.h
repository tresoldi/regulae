#ifndef REGULAE_SPLIT_SCORE_H
#define REGULAE_SPLIT_SCORE_H

#include "internal.h"

typedef struct rg_split_score_config {
    rg_split_scorer scorer;
    double dirichlet_concentration;
    double bic_log_sample_size;
    double bic_extra_penalty;
    size_t candidate_count;
    double search_gamma;
} rg_split_score_config;

typedef struct rg_split_score_result {
    double delta;
    double search_charge;
    double complexity_charge;
} rg_split_score_result;

rg_status rg_categorical_split_score_internal(
    const double *pooled,
    const double *inside,
    const double *outside,
    size_t outcome_count,
    const rg_split_score_config *config,
    rg_split_score_result *out
);

#endif
