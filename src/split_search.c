#include "split_search.h"

#include "environment.h"

#include <math.h>
#include <stdint.h>
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
    if (out_total != 0) {
        *out_total = total;
    }
    return RG_OK;
}

static double split_group_cost(
    const rg_split_observation *rows,
    size_t count,
    size_t *out_key_count
) {
    key_mass *items = 0;
    size_t item_count = 0;
    double total = 0.0;
    double cost = 0.0;
    size_t i;

    if (collect_key_mass(rows, count, &items, &item_count, &total) != RG_OK) {
        if (out_key_count != 0) {
            *out_key_count = 0;
        }
        return 0.0;
    }
    if (out_key_count != 0) {
        *out_key_count = item_count;
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

double rg_split_group_cost(const rg_split_observation *rows, size_t count) {
    return split_group_cost(rows, count, 0);
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

static void masses_in_alphabet(
    const key_mass *alphabet,
    size_t alphabet_count,
    const key_mass *items,
    size_t item_count,
    double *out
) {
    size_t a;
    size_t i = 0;
    for (a = 0; a < alphabet_count; a++) {
        out[a] = 0.0;
        while (i < item_count && strcmp(items[i].key, alphabet[a].key) < 0) {
            i++;
        }
        if (i < item_count && strcmp(items[i].key, alphabet[a].key) == 0) {
            out[a] = items[i].mass;
        }
    }
}

static int candidates_same_split(
    const rg_split_observation *rows,
    size_t count,
    const rg_split_candidate *a,
    const rg_split_candidate *b
) {
    size_t i;
    int same = 1;
    int opposite = 1;
    for (i = 0; i < count; i++) {
        int a_holds = rg_predicate_holds_internal(rows[i].context, a) != 0;
        int b_holds = rg_predicate_holds_internal(rows[i].context, b) != 0;
        same = same && a_holds == b_holds;
        opposite = opposite && a_holds != b_holds;
    }
    return same || opposite;
}

/* How many distinct OTHER features, at a DIFFERENT position, carve this split's
 * observations the same way -- the confound count for a conditioned class.
 * A predicate at another slot (another neighbour) that partitions the same rows
 * identically or exactly oppositely is a rival conditioner the corpus cannot
 * rule out: the split might be caused by that segment, not the one it names.
 *
 * The position test is deliberate, matching the cross-dimensional flag: features
 * of the SAME neighbour that co-vary with the committed one are one environment
 * to a reader, not a rival, so only a different slot counts. Fires only on a
 * perfect confound; the committed split is a real split of its rows, so a
 * degenerate candidate cannot match it. Counts distinct features. */
size_t rg_split_environment_alternatives(
    const rg_split_observation *rows,
    size_t count,
    const rg_split_candidate *committed,
    const rg_split_candidate *candidates,
    size_t candidate_count
) {
    const char *seen[64];
    size_t seen_count = 0;
    size_t c;
    for (c = 0; c < candidate_count; c++) {
        size_t s;
        int already;
        if (committed->slot != 0 && candidates[c].slot != 0 &&
            strcmp(candidates[c].slot, committed->slot) == 0) {
            continue;
        }
        if (committed->slot == 0 && candidates[c].slot == 0) {
            continue;
        }
        if (strcmp(candidates[c].feature, committed->feature) == 0) {
            continue;
        }
        if (!candidates_same_split(rows, count, committed, &candidates[c])) {
            continue;
        }
        already = 0;
        for (s = 0; s < seen_count; s++) {
            if (strcmp(seen[s], candidates[c].feature) == 0) {
                already = 1;
                break;
            }
        }
        if (!already && seen_count < sizeof(seen) / sizeof(seen[0])) {
            seen[seen_count++] = candidates[c].feature;
        }
    }
    return seen_count;
}

static rg_status distinct_partition_count(
    const rg_split_observation *rows,
    size_t count,
    const rg_split_candidate *candidates,
    size_t candidate_count,
    size_t *out
) {
    uint64_t *hashes;
    size_t distinct = 0;
    size_t c;
    hashes = (uint64_t *)calloc(candidate_count == 0 ? 1 : candidate_count, sizeof(*hashes));
    if (hashes == 0) {
        return RG_ERR_OOM;
    }
    for (c = 0; c < candidate_count; c++) {
        uint64_t hash = UINT64_C(1469598103934665603);
        uint64_t complement_hash = UINT64_C(1469598103934665603);
        size_t i;
        int duplicate = 0;
        for (i = 0; i < count; i++) {
            uint64_t holds = (uint64_t)(rg_predicate_holds_internal(
                rows[i].context, &candidates[c]) != 0);
            hash ^= holds;
            hash *= UINT64_C(1099511628211);
            complement_hash ^= UINT64_C(1) - holds;
            complement_hash *= UINT64_C(1099511628211);
        }
        hash = hash < complement_hash ? hash : complement_hash;
        for (i = 0; i < c; i++) {
            if (hashes[i] == hash &&
                candidates_same_split(rows, count, &candidates[c], &candidates[i])) {
                duplicate = 1;
                break;
            }
        }
        hashes[c] = hash;
        if (!duplicate) {
            distinct++;
        }
    }
    free(hashes);
    *out = distinct;
    return RG_OK;
}

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
) {
    key_mass *pooled_items = 0;
    size_t outcome_count = 0;
    double *pooled_mass = 0;
    double *yes_mass = 0;
    double *no_mass = 0;
    double best_margin = 0.0;
    rg_status status;
    size_t partition_count = 0;
    size_t ci;

    if (search == 0 || rows == 0 || candidates == 0 || gates == 0 ||
        score_config == 0 || out == 0 || found == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *found = 0;
    status = distinct_partition_count(rows, count, candidates, candidate_count, &partition_count);
    if (status != RG_OK) {
        return status;
    }
    status = collect_key_mass(rows, count, &pooled_items, &outcome_count, 0);
    if (status != RG_OK) {
        return status;
    }
    if (outcome_count < 2) {
        free(pooled_items);
        return RG_OK;
    }
    pooled_mass = (double *)calloc(outcome_count, sizeof(*pooled_mass));
    yes_mass = (double *)calloc(outcome_count, sizeof(*yes_mass));
    no_mass = (double *)calloc(outcome_count, sizeof(*no_mass));
    if (pooled_mass == 0 || yes_mass == 0 || no_mass == 0) {
        free(pooled_items);
        free(pooled_mass);
        free(yes_mass);
        free(no_mass);
        return RG_ERR_OOM;
    }
    masses_in_alphabet(pooled_items, outcome_count, pooled_items, outcome_count, pooled_mass);
    for (ci = 0; ci < candidate_count; ci++) {
        key_mass *yes_items = 0;
        key_mass *no_items = 0;
        size_t yes_item_count = 0;
        size_t no_item_count = 0;
        double min_obs = gates[ci].min_obs;
        double delta_threshold = gates[ci].delta_threshold;
        double min_dominant_fraction = gates[ci].min_dominant_fraction;
        double margin;
        size_t yes_count = 0;
        size_t no_count = 0;
        size_t i;
        rg_split_score_config candidate_config = *score_config;
        rg_split_score_result scored;
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
        status = collect_key_mass(search->yes, yes_count, &yes_items, &yes_item_count, 0);
        if (status == RG_OK) {
            status = collect_key_mass(search->no, no_count, &no_items, &no_item_count, 0);
        }
        if (status != RG_OK) {
            free(yes_items);
            free(no_items);
            break;
        }
        masses_in_alphabet(pooled_items, outcome_count, yes_items, yes_item_count, yes_mass);
        masses_in_alphabet(pooled_items, outcome_count, no_items, no_item_count, no_mass);
        free(yes_items);
        free(no_items);
        candidate_config.candidate_count = partition_count;
        if (score_config->outcome_mode == RG_CLASS_OUTCOME_PER_SISTER_LECT &&
            score_config->group_scorer != 0) {
            /* The per-sister-lect sum decomposes over sisters, which the caller
             * reads from each observation's owner -- the generic pooled-mass
             * categorical cannot express it. */
            status = score_config->group_scorer(search->yes, yes_count, search->no, no_count,
                                                &candidate_config, score_config->group_scorer_user,
                                                &scored);
        } else {
            status = rg_categorical_split_score_internal(pooled_mass, yes_mass, no_mass,
                                                         outcome_count, &candidate_config, &scored);
        }
        if (status != RG_OK) {
            break;
        }
        margin = delta_threshold - scored.delta;
        /* Two predicates can carve the same partition and so clear their bar by
         * the same amount. Requiring a later candidate to beat the incumbent by
         * more than the tie epsilon hands the tie to candidate order, which is
         * the same everywhere, rather than to the last bit of a log. */
        if (margin > best_margin + RG_TIE_EPSILON) {
            best_margin = margin;
            out->scorer = score_config->scorer;
            out->delta_score = scored.delta;
            /* How heavy a search charge this split's evidence could carry and
             * still commit. The charge is gamma * 2 * ln(candidates), so the
             * gamma at which this split stops clearing its bar is a
             * corpus-independent measure of how far the evidence stands above
             * the search that found it -- and it can be compared against the
             * same number computed on the corpus shuffled, which is what the
             * fit summary reports. */
            out->search_margin = partition_count > 1
                ? (delta_threshold - (scored.delta - scored.search_charge)) /
                  (2.0 * log((double)partition_count))
                : 0.0;
            out->candidate = candidates[ci];
            memcpy(search->best_yes, search->yes, yes_count * sizeof(*search->yes));
            memcpy(search->best_no, search->no, no_count * sizeof(*search->no));
            out->yes_count = yes_count;
            out->no_count = no_count;
            *found = 1;
        }
    }
    free(pooled_items);
    free(pooled_mass);
    free(yes_mass);
    free(no_mass);
    return status;
}
