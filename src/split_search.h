#ifndef REGULAE_SPLIT_SEARCH_H
#define REGULAE_SPLIT_SEARCH_H

#include "internal.h"
#include "split_score.h"

/* The greedy categorical split search, once.
 *
 * Two stages look for conditioning by partitioning observations on a candidate
 * predicate and asking whether the split pays for itself: the pairwise stage
 * over target graphemes, and the multi-lect stage over sister tuples. They had
 * the same search written out twice at full length -- the same baseline, the
 * same search charge, the same two gates, the same score line character for
 * character, the same tie epsilon, the same search-margin formula, even the
 * same undocumented `max_depth * 4` bound -- differing only in what an
 * observation is.
 *
 * So an observation is projected to what the search actually needs: an
 * environment to test predicates against, something to group cost by, and a
 * weight. `owner` is handed straight back to the caller at commit time, which
 * is where knowing what the row really was matters again.
 *
 * The key is a string rather than an interned id on purpose. Cost is summed in
 * sorted key order so the result does not depend on the order rows arrived in,
 * and sorting ids would sort by whenever each key was first seen. Both stages
 * already sorted by the key's own text, and both keep doing so.
 */
/* How many splits one greedy pass commits before it stops looking, as a
 * function of the depth the caller allows.
 *
 * A backstop, not a tuned quantity: the loop already stops when the remaining
 * weight falls under the stage's minimum, and this only bounds a pass that
 * keeps finding thin splits. Both stages used `max_depth * 4` written out
 * inline with nothing said about it, and neither the factor nor its origin is
 * recorded anywhere. It is named here so that changing it is a decision rather
 * than an edit in two places. */
#define RG_SPLIT_MAX_COMMITS(max_depth) ((max_depth) * 4)

typedef struct rg_split_observation {
    const rg_context_spec *context;
    const char *key;
    double weight;
    const void *owner;
} rg_split_observation;

/* Scratch for one search, sized once for the largest group it will see, so the
 * candidate loop allocates nothing. `best_yes` and `best_no` hold the winning
 * partition when the search returns 1. */
typedef struct rg_split_search {
    rg_split_observation *yes;
    rg_split_observation *no;
    rg_split_observation *best_yes;
    rg_split_observation *best_no;
    size_t capacity;
} rg_split_search;

rg_status rg_split_search_init(rg_split_search *search, size_t observation_capacity);
void rg_split_search_clear(rg_split_search *search);

/* The negative log-likelihood of a group under its own key distribution,
 * summed in sorted key order. Returns 0 on allocation failure, which reads as
 * "this group costs nothing" and so as "no split improves on it". */
double rg_split_group_cost(const rg_split_observation *rows, size_t count);
double rg_split_total_weight(const rg_split_observation *rows, size_t count);
double rg_split_dominant_fraction(const rg_split_observation *rows, size_t count);

typedef struct rg_split_result {
    rg_split_candidate candidate;
    size_t yes_count;
    size_t no_count;
    rg_split_scorer scorer;
    double delta_score;
    double search_margin;
} rg_split_result;

/* The best split of `rows` under the configured categorical criterion.
 * Returns RG_OK and writes `found`; when found is true, `out` is filled and
 * the partition is in search->best_yes/best_no. */
rg_status rg_split_find_best(
    rg_split_search *search,
    const rg_split_observation *rows,
    size_t count,
    const rg_split_candidate *candidates,
    const rg_split_gate *gates,
    size_t candidate_count,
    const rg_split_score_config *score_config,
    rg_split_result *out,
    int *found
);

#endif
