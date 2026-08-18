#ifndef REGULAE_SPLIT_SCORE_H
#define REGULAE_SPLIT_SCORE_H

#include "internal.h"

struct rg_split_observation;
struct rg_split_score_config;

typedef struct rg_split_score_result {
    double delta;
    double search_charge;
    double complexity_charge;
} rg_split_score_result;

/* Scores a partition directly from its observations rather than from pooled
 * mass vectors, for outcome models the generic categorical scorer cannot
 * express -- the per-sister-lect sum (RG_CLASS_OUTCOME_PER_SISTER_LECT), whose
 * likelihood and charge decompose over sister lects and so need each
 * observation's sister tuple, which only the multi-lect caller can read from an
 * observation's `owner`. Set on the score config alongside `group_scorer_user`;
 * when present and the mode calls for it, rg_split_find_best hands the winning
 * partition here instead of building the joint-key categorical. Must fill every
 * field of `out` and honour `config->candidate_count` for the search charge. */
typedef rg_status (*rg_split_group_scorer)(
    const struct rg_split_observation *yes,
    size_t yes_count,
    const struct rg_split_observation *no,
    size_t no_count,
    const struct rg_split_score_config *config,
    void *user,
    rg_split_score_result *out
);

typedef struct rg_split_score_config {
    rg_split_scorer scorer;
    double dirichlet_concentration;
    double bic_log_sample_size;
    double bic_extra_penalty;
    size_t candidate_count;
    double search_gamma;
    /* Selects the outcome model -- see rg_class_outcome_mode. Left SISTER_TUPLE
     * (the generic joint-key categorical) by every caller except the multi-lect
     * class search, which sets PER_SISTER_LECT and a group_scorer. */
    rg_class_outcome_mode outcome_mode;
    /* Optional. Used only when outcome_mode is RG_CLASS_OUTCOME_PER_SISTER_LECT.
     * See rg_split_group_scorer. */
    rg_split_group_scorer group_scorer;
    void *group_scorer_user;
} rg_split_score_config;

rg_status rg_categorical_split_score_internal(
    const double *pooled,
    const double *inside,
    const double *outside,
    size_t outcome_count,
    const rg_split_score_config *config,
    rg_split_score_result *out
);

#endif
