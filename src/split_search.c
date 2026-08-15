#include "split_search.h"

#include "environment.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void rg_split_search_clear(rg_split_search *search) {
    if (search == 0) {
        return;
    }
    free(search->yes);
    free(search->no);
    free(search->best_yes);
    free(search->best_no);
    memset(search, 0, sizeof(*search));
}

rg_status rg_split_search_init(rg_split_search *search, size_t observation_capacity) {
    size_t n = observation_capacity == 0 ? 1 : observation_capacity;
    memset(search, 0, sizeof(*search));
    search->capacity = observation_capacity;
    search->yes = (rg_split_observation *)calloc(n, sizeof(*search->yes));
    search->no = (rg_split_observation *)calloc(n, sizeof(*search->no));
    search->best_yes = (rg_split_observation *)calloc(n, sizeof(*search->best_yes));
    search->best_no = (rg_split_observation *)calloc(n, sizeof(*search->best_no));
    if (search->yes == 0 || search->no == 0 || search->best_yes == 0 || search->best_no == 0) {
        rg_split_search_clear(search);
        return RG_ERR_OOM;
    }
    return RG_OK;
}

typedef struct key_mass {
    const char *key;
    double mass;
} key_mass;

static rg_status add_key_mass(
    key_mass **items,
    size_t *count,
    size_t *cap,
    const char *key,
    double weight
) {
    size_t i;
    for (i = 0; i < *count; i++) {
        if (strcmp((*items)[i].key, key) == 0) {
            (*items)[i].mass += weight;
            return RG_OK;
        }
    }
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 4 : *cap * 2;
        key_mass *next = (key_mass *)realloc(*items, next_cap * sizeof(*next));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *items = next;
        *cap = next_cap;
    }
    (*items)[*count].key = key;
    (*items)[*count].mass = weight;
    (*count)++;
    return RG_OK;
}

/* Accumulates mass per key in row order, then orders the keys by their own text
 * so the sum below does not depend on the order rows arrived in. */
static rg_status collect_key_mass(
    const rg_split_observation *rows,
    size_t count,
    key_mass **out_items,
    size_t *out_count,
    double *out_total
) {
    key_mass *items = 0;
    size_t item_count = 0;
    size_t cap = 0;
    double total = 0.0;
    size_t i;
    size_t j;

    for (i = 0; i < count; i++) {
        if (add_key_mass(&items, &item_count, &cap, rows[i].key, rows[i].weight) != RG_OK) {
            free(items);
            return RG_ERR_OOM;
        }
        total += rows[i].weight;
    }
    for (i = 1; i < item_count; i++) {
        key_mass key = items[i];
        j = i;
        while (j > 0 && strcmp(items[j - 1].key, key.key) > 0) {
            items[j] = items[j - 1];
            j--;
        }
        items[j] = key;
    }
    *out_items = items;
    *out_count = item_count;
    *out_total = total;
    return RG_OK;
}

double rg_split_group_cost(const rg_split_observation *rows, size_t count) {
    key_mass *items = 0;
    size_t item_count = 0;
    double total = 0.0;
    double cost = 0.0;
    size_t i;

    if (collect_key_mass(rows, count, &items, &item_count, &total) != RG_OK) {
        return 0.0;
    }
    if (total <= 0.0) {
        free(items);
        return 0.0;
    }
    for (i = 0; i < item_count; i++) {
        double p = items[i].mass / total;
        if (p > 0.0) {
            cost += -items[i].mass * log(p);
        }
    }
    free(items);
    return cost;
}

double rg_split_total_weight(const rg_split_observation *rows, size_t count) {
    double total = 0.0;
    size_t i;
    for (i = 0; i < count; i++) {
        total += rows[i].weight;
    }
    return total;
}

double rg_split_dominant_fraction(const rg_split_observation *rows, size_t count) {
    key_mass *items = 0;
    size_t item_count = 0;
    double total = 0.0;
    double mode = 0.0;
    size_t i;

    if (collect_key_mass(rows, count, &items, &item_count, &total) != RG_OK) {
        return 0.0;
    }
    for (i = 0; i < item_count; i++) {
        if (items[i].mass > mode) {
            mode = items[i].mass;
        }
    }
    free(items);
    return total > 0.0 ? mode / total : 0.0;
}

int rg_split_find_best(
    rg_split_search *search,
    const rg_split_observation *rows,
    size_t count,
    const rg_split_candidate *candidates,
    const rg_split_gate *gates,
    size_t candidate_count,
    double penalty,
    double search_gamma,
    rg_split_result *out
) {
    double baseline = rg_split_group_cost(rows, count);
    /* Charge for the search, not only for the parameter.
     *
     * BIC prices one added term against the likelihood it buys. The term that
     * survives here is not one term: it is the best of candidate_count of them,
     * and the maximum of a hundred candidates beats its bar by chance far more
     * often than one candidate does. Permuting a corpus's pairings -- which
     * removes every correspondence there is to find -- used to *raise* the
     * number of committed rules, which is what an unpriced argmax looks like.
     *
     * 2*ln(candidates) is the same currency as the BIC penalty and is the
     * standard extended-BIC shape for a large model space. It is not a
     * substitute for the shuffled baseline, which measures the inflation this
     * only models. */
    double search_penalty = candidate_count > 1 ? search_gamma * 2.0 * log((double)candidate_count) : 0.0;
    double best_margin = 0.0;
    int found = 0;
    size_t ci;

    for (ci = 0; ci < candidate_count; ci++) {
        double min_obs = gates[ci].min_obs;
        double delta_threshold = gates[ci].delta_threshold;
        double min_dominant_fraction = gates[ci].min_dominant_fraction;
        double margin;
        size_t yes_count = 0;
        size_t no_count = 0;
        size_t i;
        double split_cost;
        double delta_bic;
        for (i = 0; i < count; i++) {
            if (rg_predicate_holds_internal(rows[i].context, &candidates[ci])) {
                search->yes[yes_count++] = rows[i];
            } else {
                search->no[no_count++] = rows[i];
            }
        }
        if (rg_split_total_weight(search->yes, yes_count) < min_obs ||
            rg_split_total_weight(search->no, no_count) < min_obs) {
            continue;
        }
        if (min_dominant_fraction > 0.0 &&
            rg_split_dominant_fraction(search->yes, yes_count) < min_dominant_fraction) {
            continue;
        }
        split_cost = rg_split_group_cost(search->yes, yes_count) +
                     rg_split_group_cost(search->no, no_count);
        delta_bic = -2.0 * (baseline - split_cost) + penalty + search_penalty;
        margin = delta_threshold - delta_bic;
        /* Two predicates can carve the same partition and so clear their bar by
         * the same amount. Requiring a later candidate to beat the incumbent by
         * more than the tie epsilon hands the tie to candidate order, which is
         * the same everywhere, rather than to the last bit of a log. */
        if (margin > best_margin + RG_TIE_EPSILON) {
            best_margin = margin;
            out->delta_bic = delta_bic;
            /* How heavy a search charge this split's evidence could carry and
             * still commit. The charge is gamma * 2 * ln(candidates), so the
             * gamma at which this split stops clearing its bar is a
             * corpus-independent measure of how far the evidence stands above
             * the search that found it -- and it can be compared against the
             * same number computed on the corpus shuffled, which is what the
             * fit summary reports. */
            out->search_margin = candidate_count > 1
                ? (delta_threshold - (delta_bic - search_penalty)) / (2.0 * log((double)candidate_count))
                : 0.0;
            out->candidate = candidates[ci];
            memcpy(search->best_yes, search->yes, yes_count * sizeof(*search->yes));
            memcpy(search->best_no, search->no, no_count * sizeof(*search->no));
            out->yes_count = yes_count;
            out->no_count = no_count;
            found = 1;
        }
    }
    return found;
}
