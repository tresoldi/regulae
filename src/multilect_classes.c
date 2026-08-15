#include "multilect_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- multi-lect context discovery ------------------------------------- */


static const char *const multi_long_range_slots[] = {
    "same_syllable",
    "next_syllable",
    "previous_syllable",
    "preceding@2",
    "preceding@3",
    "following@2",
    "following@3",
    "somewhere_preceding",
    "somewhere_following"
};


static const char *const multi_stress_slots[] = {
    "self_stress",
    "preceding_stress",
    "following_stress"
};

typedef struct sister_tuple {
    char *key;
    char **lects;
    char **graphemes;
    size_t count;
} sister_tuple;

typedef struct pivot_obs {
    size_t sister_index;
    const rg_context_spec *context;
    double weight;
    size_t observation_index;
} pivot_obs;

typedef struct pivot_bucket {
    char *lect;
    char *grapheme;
    pivot_obs *obs;
    size_t obs_count;
    size_t obs_cap;
} pivot_bucket;

typedef struct committed_split {
    char *pivot_lect;
    char *pivot_grapheme;
    rg_context_spec context;
    size_t sister_index;
    double count;
    double bucket_size;
    /* The same sister where the environment does not hold, and what the split
     * scored. A conditioning claim is a comparison; without these the class is
     * a number with no denominator. */
    double contrast_count;
    double delta_bic;
    double search_margin;
    int decision_index;
    /* The observations that justified this split, so the class it merges into
     * can name its own evidence. */
    size_t *observation_indices;
    size_t observation_count;
} committed_split;

typedef struct merged_class {
    size_t *observation_indices;
    size_t observation_count;
    size_t observation_cap;
    char *key;
    char **lects;
    char **graphemes;
    rg_context_spec *contexts;
    size_t segment_count;
    double count;
    double confidence;
    double bucket_size;
    double winning_count;
    double contrast_count;
    double delta_bic;
    double search_margin;
    int decision_index;
} merged_class;

typedef struct discovery_state {
    sister_tuple *sisters;
    size_t sister_count;
    size_t sister_cap;
    pivot_bucket *pivots;
    size_t pivot_count;
    size_t pivot_cap;
    committed_split *committed;
    size_t committed_count;
    size_t committed_cap;
    char **stress_values;
    size_t stress_count;
    size_t stress_cap;
    /* Morphological values the corpus actually shows. A corpus without
     * boundaries reports none and pays nothing for the axis. */
    const char *morph_placements[8];
    size_t morph_placement_count;
    const char *morph_indices[8];
    size_t morph_index_count;
    rg_split_candidate *immediate;
    size_t immediate_count;
    rg_split_candidate *long_range;
    size_t long_range_count;
    /* Both pools in one array, for refinement: a narrower environment may be
     * built from either kind of predicate whatever the enclosing pass led
     * with. */
    rg_split_candidate *all;
    size_t all_count;
    /* Decisions committed so far. Publication sorts the classes and would
     * otherwise lose the order they were settled in. */
    int decision_count;
} discovery_state;

static void discovery_state_clear(discovery_state *state) {
    size_t i;
    if (state == 0) {
        return;
    }
    for (i = 0; i < state->sister_count; i++) {
        free(state->sisters[i].key);
        string_array_clear(state->sisters[i].lects, state->sisters[i].count);
        string_array_clear(state->sisters[i].graphemes, state->sisters[i].count);
    }
    free(state->sisters);
    for (i = 0; i < state->pivot_count; i++) {
        free(state->pivots[i].lect);
        free(state->pivots[i].grapheme);
        free(state->pivots[i].obs);
    }
    free(state->pivots);
    for (i = 0; i < state->committed_count; i++) {
        free(state->committed[i].pivot_lect);
        free(state->committed[i].pivot_grapheme);
        free(state->committed[i].observation_indices);
        rg_context_spec_clear_internal(&state->committed[i].context);
    }
    free(state->committed);
    string_array_clear(state->stress_values, state->stress_count);
    free(state->immediate);
    free(state->long_range);
    free(state->all);
    memset(state, 0, sizeof(*state));
}

/* "lect:grapheme|lect:grapheme" over the given items. */
static rg_status join_lect_grapheme_key(
    char *const *lects,
    char *const *graphemes,
    size_t count,
    char **out
) {
    size_t len = 1;
    size_t i;
    char *key;
    char *p;
    *out = 0;
    for (i = 0; i < count; i++) {
        len += strlen(lects[i]) + strlen(graphemes[i]) + 2;
    }
    key = (char *)malloc(len);
    if (key == 0) {
        return RG_ERR_OOM;
    }
    p = key;
    for (i = 0; i < count; i++) {
        size_t n;
        if (i > 0) {
            *p++ = '|';
        }
        n = strlen(lects[i]);
        memcpy(p, lects[i], n);
        p += n;
        *p++ = ':';
        n = strlen(graphemes[i]);
        memcpy(p, graphemes[i], n);
        p += n;
    }
    *p = '\0';
    *out = key;
    return RG_OK;
}

static rg_status sister_index_for(
    discovery_state *state,
    char *const *lects,
    char *const *graphemes,
    size_t count,
    size_t *out
) {
    char *key = 0;
    size_t i;
    rg_status status = join_lect_grapheme_key(lects, graphemes, count, &key);
    if (status != RG_OK) {
        return status;
    }
    for (i = 0; i < state->sister_count; i++) {
        if (strcmp(state->sisters[i].key, key) == 0) {
            free(key);
            *out = i;
            return RG_OK;
        }
    }
    if (state->sister_count == state->sister_cap) {
        size_t next_cap = state->sister_cap == 0 ? 16 : state->sister_cap * 2;
        sister_tuple *next = (sister_tuple *)realloc(state->sisters, next_cap * sizeof(*next));
        if (next == 0) {
            free(key);
            return RG_ERR_OOM;
        }
        state->sisters = next;
        state->sister_cap = next_cap;
    }
    memset(&state->sisters[state->sister_count], 0, sizeof(state->sisters[state->sister_count]));
    state->sisters[state->sister_count].key = key;
    state->sisters[state->sister_count].lects = (char **)calloc(count, sizeof(char *));
    state->sisters[state->sister_count].graphemes = (char **)calloc(count, sizeof(char *));
    if (state->sisters[state->sister_count].lects == 0 || state->sisters[state->sister_count].graphemes == 0) {
        free(state->sisters[state->sister_count].lects);
        free(state->sisters[state->sister_count].graphemes);
        free(key);
        memset(&state->sisters[state->sister_count], 0, sizeof(state->sisters[state->sister_count]));
        return RG_ERR_OOM;
    }
    for (i = 0; i < count; i++) {
        state->sisters[state->sister_count].lects[i] = rg_strdup_internal(lects[i]);
        state->sisters[state->sister_count].graphemes[i] = rg_strdup_internal(graphemes[i]);
        if (state->sisters[state->sister_count].lects[i] == 0 || state->sisters[state->sister_count].graphemes[i] == 0) {
            state->sisters[state->sister_count].count = i + 1;
            free(state->sisters[state->sister_count].key);
            string_array_clear(state->sisters[state->sister_count].lects, i + 1);
            string_array_clear(state->sisters[state->sister_count].graphemes, i + 1);
            memset(&state->sisters[state->sister_count], 0, sizeof(state->sisters[state->sister_count]));
            return RG_ERR_OOM;
        }
    }
    state->sisters[state->sister_count].count = count;
    *out = state->sister_count;
    state->sister_count++;
    return RG_OK;
}

static rg_status pivot_index_for(discovery_state *state, const char *lect, const char *grapheme, size_t *out) {
    size_t i;
    for (i = 0; i < state->pivot_count; i++) {
        if (strcmp(state->pivots[i].lect, lect) == 0 && strcmp(state->pivots[i].grapheme, grapheme) == 0) {
            *out = i;
            return RG_OK;
        }
    }
    if (state->pivot_count == state->pivot_cap) {
        size_t next_cap = state->pivot_cap == 0 ? 16 : state->pivot_cap * 2;
        pivot_bucket *next = (pivot_bucket *)realloc(state->pivots, next_cap * sizeof(*next));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        state->pivots = next;
        state->pivot_cap = next_cap;
    }
    memset(&state->pivots[state->pivot_count], 0, sizeof(state->pivots[state->pivot_count]));
    state->pivots[state->pivot_count].lect = rg_strdup_internal(lect);
    state->pivots[state->pivot_count].grapheme = rg_strdup_internal(grapheme);
    if (state->pivots[state->pivot_count].lect == 0 || state->pivots[state->pivot_count].grapheme == 0) {
        free(state->pivots[state->pivot_count].lect);
        free(state->pivots[state->pivot_count].grapheme);
        memset(&state->pivots[state->pivot_count], 0, sizeof(state->pivots[state->pivot_count]));
        return RG_ERR_OOM;
    }
    *out = state->pivot_count;
    state->pivot_count++;
    return RG_OK;
}

static rg_status pivot_bucket_append(pivot_bucket *bucket, size_t sister_index, const rg_context_spec *context, double weight, size_t observation_index) {
    if (bucket->obs_count == bucket->obs_cap) {
        size_t next_cap = bucket->obs_cap == 0 ? 8 : bucket->obs_cap * 2;
        pivot_obs *next = (pivot_obs *)realloc(bucket->obs, next_cap * sizeof(*next));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        bucket->obs = next;
        bucket->obs_cap = next_cap;
    }
    bucket->obs[bucket->obs_count].sister_index = sister_index;
    bucket->obs[bucket->obs_count].context = context;
    bucket->obs[bucket->obs_count].weight = weight;
    bucket->obs[bucket->obs_count].observation_index = observation_index;
    bucket->obs_count++;
    return RG_OK;
}

static rg_status record_stress_value(discovery_state *state, const char *value) {
    size_t i;
    size_t insert_at;
    for (i = 0; i < state->stress_count; i++) {
        if (strcmp(state->stress_values[i], value) == 0) {
            return RG_OK;
        }
    }
    if (state->stress_count == state->stress_cap) {
        size_t next_cap = state->stress_cap == 0 ? 4 : state->stress_cap * 2;
        char **next = (char **)realloc(state->stress_values, next_cap * sizeof(*next));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        state->stress_values = next;
        state->stress_cap = next_cap;
    }
    insert_at = state->stress_count;
    while (insert_at > 0 && strcmp(state->stress_values[insert_at - 1], value) > 0) {
        insert_at--;
    }
    if (insert_at < state->stress_count) {
        memmove(&state->stress_values[insert_at + 1], &state->stress_values[insert_at],
                (state->stress_count - insert_at) * sizeof(*state->stress_values));
    }
    state->stress_values[insert_at] = rg_strdup_internal(value);
    if (state->stress_values[insert_at] == 0) {
        return RG_ERR_OOM;
    }
    state->stress_count++;
    return RG_OK;
}

static void record_morphology_value(const char **values, size_t *count, const char *value) {
    size_t i;
    if (value == 0 || value[0] == '\0') {
        return;
    }
    for (i = 0; i < *count; i++) {
        if (strcmp(values[i], value) == 0) {
            return;
        }
    }
    if (*count < 8) {
        values[*count] = value;
        (*count)++;
    }
}

static void collect_morphology_values(discovery_state *state, const rg_context_spec *context) {
    record_morphology_value(state->morph_placements, &state->morph_placement_count, context->morphological);
    record_morphology_value(state->morph_indices, &state->morph_index_count, context->morpheme_index);
}

static rg_status collect_stress_values(discovery_state *state, const rg_context_spec *context) {
    const rg_feature_constraint *slots[3];
    size_t counts[3];
    size_t s;
    slots[0] = context->self_stress;
    counts[0] = context->self_stress_count;
    slots[1] = context->preceding_stress;
    counts[1] = context->preceding_stress_count;
    slots[2] = context->following_stress;
    counts[2] = context->following_stress_count;
    for (s = 0; s < 3; s++) {
        size_t i;
        for (i = 0; i < counts[s]; i++) {
            if (slots[s][i].feature != 0 && strcmp(slots[s][i].feature, "stress") == 0 && slots[s][i].value != 0) {
                rg_status status = record_stress_value(state, slots[s][i].value);
                if (status != RG_OK) {
                    return status;
                }
            }
        }
    }
    return RG_OK;
}

/* The candidate lists are fixed once the observed stress values are known,
 * because every multi-lect split is searched against an empty base context. */
static rg_status build_candidate_lists(discovery_state *state, const rg_feature_vocabulary *vocabulary) {
    size_t immediate_total = 2 * vocabulary->count;
    size_t slot_count = sizeof(multi_stress_slots) / sizeof(multi_stress_slots[0]);
    size_t long_slots = sizeof(multi_long_range_slots) / sizeof(multi_long_range_slots[0]);
    size_t long_features = vocabulary->count == 0 ? 1 : vocabulary->count;
    size_t total = immediate_total + slot_count * state->stress_count
        + state->morph_placement_count + state->morph_index_count;
    size_t i;
    size_t s;
    size_t n = 0;

    state->immediate = (rg_split_candidate *)calloc(total == 0 ? 1 : total, sizeof(*state->immediate));
    if (state->immediate == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < vocabulary->count; i++) {
        state->immediate[n].slot = "preceding";
        state->immediate[n].feature = vocabulary->entries[i].feature;
        state->immediate[n].value = vocabulary->entries[i].value;
        n++;
        state->immediate[n].slot = "following";
        state->immediate[n].feature = vocabulary->entries[i].feature;
        state->immediate[n].value = vocabulary->entries[i].value;
        n++;
    }
    for (s = 0; s < slot_count; s++) {
        for (i = 0; i < state->stress_count; i++) {
            state->immediate[n].slot = multi_stress_slots[s];
            state->immediate[n].feature = "stress";
            state->immediate[n].value = state->stress_values[i];
            n++;
        }
    }
    for (i = 0; i < state->morph_placement_count; i++) {
        state->immediate[n].slot = "morphological";
        state->immediate[n].feature = state->morph_placements[i];
        state->immediate[n].value = "+";
        n++;
    }
    for (i = 0; i < state->morph_index_count; i++) {
        state->immediate[n].slot = "morpheme_index";
        state->immediate[n].feature = state->morph_indices[i];
        state->immediate[n].value = "+";
        n++;
    }
    state->immediate_count = n;

    state->long_range = (rg_split_candidate *)calloc(long_slots * long_features, sizeof(*state->long_range));
    if (state->long_range == 0) {
        return RG_ERR_OOM;
    }
    n = 0;
    for (s = 0; s < long_slots; s++) {
        for (i = 0; i < vocabulary->count; i++) {
            state->long_range[n].slot = multi_long_range_slots[s];
            state->long_range[n].feature = vocabulary->entries[i].feature;
            state->long_range[n].value = vocabulary->entries[i].value;
            n++;
        }
    }
    state->long_range_count = n;

    state->all = (rg_split_candidate *)calloc(
        state->immediate_count + state->long_range_count + 1, sizeof(*state->all));
    if (state->all == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < state->immediate_count; i++) {
        state->all[i] = state->immediate[i];
    }
    for (i = 0; i < state->long_range_count; i++) {
        state->all[state->immediate_count + i] = state->long_range[i];
    }
    state->all_count = state->immediate_count + state->long_range_count;
    return RG_OK;
}

static double observation_weight(const pivot_obs *obs, size_t count) {
    double total = 0.0;
    size_t i;
    for (i = 0; i < count; i++) {
        total += obs[i].weight;
    }
    return total;
}

/* Negative log-likelihood of a group under a single unconditioned outcome.
 * Sister keys are summed in sorted order so the result is reproducible. */
static double group_cost(const discovery_state *state, const pivot_obs *obs, size_t count) {
    double *masses;
    size_t *order;
    double total = 0.0;
    double cost = 0.0;
    size_t i;
    size_t used = 0;
    if (count == 0) {
        return 0.0;
    }
    masses = (double *)calloc(state->sister_count, sizeof(*masses));
    order = (size_t *)calloc(state->sister_count, sizeof(*order));
    if (masses == 0 || order == 0) {
        free(masses);
        free(order);
        return 0.0;
    }
    for (i = 0; i < count; i++) {
        if (masses[obs[i].sister_index] == 0.0) {
            size_t insert_at = used;
            while (insert_at > 0 &&
                   strcmp(state->sisters[order[insert_at - 1]].key, state->sisters[obs[i].sister_index].key) > 0) {
                insert_at--;
            }
            if (insert_at < used) {
                memmove(&order[insert_at + 1], &order[insert_at], (used - insert_at) * sizeof(*order));
            }
            order[insert_at] = obs[i].sister_index;
            used++;
        }
        masses[obs[i].sister_index] += obs[i].weight;
        total += obs[i].weight;
    }
    if (total > 0.0) {
        for (i = 0; i < used; i++) {
            double n = masses[order[i]];
            double p = n / total;
            if (p > 0.0) {
                cost += -n * log(p);
            }
        }
    }
    free(masses);
    free(order);
    return cost;
}

static double dominant_fraction(const discovery_state *state, const pivot_obs *obs, size_t count) {
    double *masses;
    double mode = 0.0;
    double total = 0.0;
    size_t i;
    if (count == 0) {
        return 0.0;
    }
    masses = (double *)calloc(state->sister_count, sizeof(*masses));
    if (masses == 0) {
        return 0.0;
    }
    for (i = 0; i < count; i++) {
        masses[obs[i].sister_index] += obs[i].weight;
        total += obs[i].weight;
    }
    for (i = 0; i < state->sister_count; i++) {
        if (masses[i] > mode) {
            mode = masses[i];
        }
    }
    free(masses);
    return total > 0.0 ? mode / total : 0.0;
}

static int multi_lect_min_commit_count(double n_total, double scale) {
    double v;
    if (scale <= 0.0) {
        return 2;
    }
    v = ceil(scale * (log(n_total + 1.0) / log(2.0)));
    if (v < 2.0) {
        return 2;
    }
    return (int)v;
}

static rg_status append_committed_split(
    discovery_state *state,
    const char *pivot_lect,
    const char *pivot_grapheme,
    const rg_context_spec *context,
    size_t sister_index,
    double count,
    double bucket_size,
    double contrast_count,
    double delta_bic,
    double search_margin,
    int decision_index,
    const size_t *observation_indices,
    size_t observation_count
) {
    committed_split *slot;
    rg_status status;
    if (state->committed_count == state->committed_cap) {
        size_t next_cap = state->committed_cap == 0 ? 16 : state->committed_cap * 2;
        committed_split *next = (committed_split *)realloc(state->committed, next_cap * sizeof(*next));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        state->committed = next;
        state->committed_cap = next_cap;
    }
    slot = &state->committed[state->committed_count];
    memset(slot, 0, sizeof(*slot));
    slot->contrast_count = contrast_count;
    slot->delta_bic = delta_bic;
    slot->search_margin = search_margin;
    slot->decision_index = decision_index;
    slot->pivot_lect = rg_strdup_internal(pivot_lect);
    slot->pivot_grapheme = rg_strdup_internal(pivot_grapheme);
    if (slot->pivot_lect == 0 || slot->pivot_grapheme == 0) {
        free(slot->pivot_lect);
        free(slot->pivot_grapheme);
        memset(slot, 0, sizeof(*slot));
        return RG_ERR_OOM;
    }
    status = rg_context_spec_copy_internal(context, &slot->context);
    if (status != RG_OK) {
        free(slot->pivot_lect);
        free(slot->pivot_grapheme);
        memset(slot, 0, sizeof(*slot));
        return status;
    }
    slot->sister_index = sister_index;
    slot->count = count;
    slot->bucket_size = bucket_size;
    if (observation_count > 0) {
        slot->observation_indices = (size_t *)calloc(observation_count, sizeof(size_t));
        if (slot->observation_indices == 0) {
            free(slot->pivot_lect);
            free(slot->pivot_grapheme);
            rg_context_spec_clear_internal(&slot->context);
            memset(slot, 0, sizeof(*slot));
            return RG_ERR_OOM;
        }
        memcpy(slot->observation_indices, observation_indices, observation_count * sizeof(size_t));
        slot->observation_count = observation_count;
    }
    state->committed_count++;
    return RG_OK;
}

/* Commits one class per distinct sister tuple in the YES partition whose mass
 * meets the adaptive minimum, in sorted sister-key order. */
static rg_status emit_sister_classes(
    discovery_state *state,
    const char *pivot_lect,
    const char *pivot_grapheme,
    const rg_context_spec *yes_context,
    const pivot_obs *yes_obs,
    size_t yes_count,
    const pivot_obs *no_obs,
    size_t no_count,
    double delta_bic,
    double search_margin,
    int decision_index,
    double min_commit,
    double n_total
) {
    double *masses;
    double *contrast_masses;
    size_t *order;
    size_t used = 0;
    size_t i;
    rg_status status = RG_OK;

    masses = (double *)calloc(state->sister_count, sizeof(*masses));
    contrast_masses = (double *)calloc(state->sister_count, sizeof(*contrast_masses));
    order = (size_t *)calloc(state->sister_count, sizeof(*order));
    if (masses == 0 || contrast_masses == 0 || order == 0) {
        free(masses);
        free(contrast_masses);
        free(order);
        return RG_ERR_OOM;
    }
    for (i = 0; i < no_count; i++) {
        contrast_masses[no_obs[i].sister_index] += no_obs[i].weight;
    }
    for (i = 0; i < yes_count; i++) {
        if (masses[yes_obs[i].sister_index] == 0.0) {
            size_t insert_at = used;
            while (insert_at > 0 &&
                   strcmp(state->sisters[order[insert_at - 1]].key, state->sisters[yes_obs[i].sister_index].key) > 0) {
                insert_at--;
            }
            if (insert_at < used) {
                memmove(&order[insert_at + 1], &order[insert_at], (used - insert_at) * sizeof(*order));
            }
            order[insert_at] = yes_obs[i].sister_index;
            used++;
        }
        masses[yes_obs[i].sister_index] += yes_obs[i].weight;
    }
    for (i = 0; i < used && status == RG_OK; i++) {
        size_t *evidence;
        size_t evidence_count = 0;
        size_t j;
        if (masses[order[i]] < min_commit) {
            continue;
        }
        evidence = (size_t *)calloc(yes_count == 0 ? 1 : yes_count, sizeof(*evidence));
        if (evidence == 0) {
            status = RG_ERR_OOM;
            break;
        }
        for (j = 0; j < yes_count; j++) {
            if (yes_obs[j].sister_index == order[i]) {
                evidence[evidence_count++] = yes_obs[j].observation_index;
            }
        }
        status = append_committed_split(
            state,
            pivot_lect,
            pivot_grapheme,
            yes_context,
            order[i],
            masses[order[i]],
            n_total,
            contrast_masses[order[i]],
            delta_bic,
            search_margin,
            decision_index,
            evidence,
            evidence_count
        );
        free(evidence);
    }
    free(masses);
    free(contrast_masses);
    free(order);
    return status;
}

/* Greedy BIC-gated split loop over one pivot bucket. The immediate and
 * long-range passes differ only in candidate inventory, thresholds, the
 * small-sample penalty, and the dominance filter. */
/* Finds the best split of a group, weighing every candidate against its own
 * bar. Returns 0 when nothing clears one. */
static int pivot_best_split(
    discovery_state *state,
    const pivot_obs *rows,
    size_t count,
    const rg_split_candidate *candidates,
    const rg_split_gate *gates,
    size_t candidate_count,
    double penalty,
    double search_gamma,
    pivot_obs *yes,
    pivot_obs *no,
    pivot_obs *best_yes,
    pivot_obs *best_no,
    size_t *best_yes_count,
    size_t *best_no_count,
    size_t *best_candidate,
    double *best_delta_bic,
    double *best_search_margin
) {
    double baseline = group_cost(state, rows, count);
    /* The same charge for the same reason as find_best_split in model.c. */
    double search_penalty = candidate_count > 1 ? search_gamma * 2.0 * log((double)candidate_count) : 0.0;
    double best_margin = 0.0;
    int found = 0;
    size_t ci;

    for (ci = 0; ci < candidate_count; ci++) {
        size_t yes_count = 0;
        size_t no_count = 0;
        double split_cost;
        double delta_bic;
        double margin;
        size_t i;
        for (i = 0; i < count; i++) {
            if (rg_predicate_holds_internal(rows[i].context, &candidates[ci])) {
                yes[yes_count++] = rows[i];
            } else {
                no[no_count++] = rows[i];
            }
        }
        if (observation_weight(yes, yes_count) < gates[ci].min_obs ||
            observation_weight(no, no_count) < gates[ci].min_obs) {
            continue;
        }
        if (gates[ci].min_dominant_fraction > 0.0 &&
            dominant_fraction(state, yes, yes_count) < gates[ci].min_dominant_fraction) {
            continue;
        }
        split_cost = group_cost(state, yes, yes_count) + group_cost(state, no, no_count);
        delta_bic = -2.0 * (baseline - split_cost) + penalty + search_penalty;
        margin = gates[ci].delta_threshold - delta_bic;
        /* Ties go to candidate order, which is the same everywhere, rather
         * than to the last bit of a log. */
        if (margin > best_margin + RG_TIE_EPSILON) {
            best_margin = margin;
            *best_delta_bic = delta_bic;
            *best_search_margin = candidate_count > 1
                ? (gates[ci].delta_threshold - (delta_bic - search_penalty)) / (2.0 * log((double)candidate_count))
                : 0.0;
            *best_candidate = ci;
            memcpy(best_yes, yes, yes_count * sizeof(*yes));
            memcpy(best_no, no, no_count * sizeof(*no));
            *best_yes_count = yes_count;
            *best_no_count = no_count;
            found = 1;
        }
    }
    return found;
}

/* Conjoins a second predicate within a committed group, and emits the narrower
 * classes it separates. Without this the stage can only ever say one thing
 * about an environment, and a change conditioned by two -- preceded by a nasal
 * *and* before a front vowel, which is what assimilation usually looks like --
 * comes out as a single predicate with contradictory outcomes under it. */
static rg_status refine_pivot_split(
    discovery_state *state,
    const pivot_bucket *bucket,
    const rg_context_spec *base_context,
    const pivot_obs *rows,
    size_t count,
    int depth,
    int max_depth,
    const rg_split_gate *gates,
    double penalty,
    double search_gamma,
    double min_commit,
    double n_total
) {
    pivot_obs *yes;
    pivot_obs *no;
    pivot_obs *best_yes;
    pivot_obs *best_no;
    size_t best_yes_count = 0;
    size_t best_no_count = 0;
    size_t best_candidate = 0;
    double delta_bic = 0.0;
    double search_margin = 0.0;
    rg_status status = RG_OK;

    if (depth >= max_depth || count == 0) {
        return RG_OK;
    }
    yes = (pivot_obs *)calloc(count, sizeof(*yes));
    no = (pivot_obs *)calloc(count, sizeof(*no));
    best_yes = (pivot_obs *)calloc(count, sizeof(*best_yes));
    best_no = (pivot_obs *)calloc(count, sizeof(*best_no));
    if (yes == 0 || no == 0 || best_yes == 0 || best_no == 0) {
        free(yes); free(no); free(best_yes); free(best_no);
        return RG_ERR_OOM;
    }
    if (pivot_best_split(state, rows, count, state->all, gates, state->all_count,
                         penalty, search_gamma, yes, no, best_yes, best_no,
                         &best_yes_count, &best_no_count, &best_candidate, &delta_bic,
                         &search_margin)) {
        rg_context_spec narrowed;
        status = rg_context_extend_internal(base_context, &state->all[best_candidate], &narrowed);
        if (status == RG_OK) {
            status = emit_sister_classes(state, bucket->lect, bucket->grapheme,
                                         &narrowed, best_yes, best_yes_count,
                                         best_no, best_no_count, delta_bic, search_margin,
                                         state->decision_count++, min_commit, n_total);
            if (status == RG_OK) {
                status = refine_pivot_split(state, bucket, &narrowed, best_yes,
                                            best_yes_count, depth + 1, max_depth,
                                            gates, penalty, search_gamma, min_commit, n_total);
            }
            rg_context_spec_clear_internal(&narrowed);
        }
    }
    free(yes); free(no); free(best_yes); free(best_no);
    return status;
}

static rg_status commit_splits_for_pivot(
    discovery_state *state,
    const pivot_bucket *bucket,
    const rg_split_candidate *candidates,
    const rg_split_gate *gates,
    size_t candidate_count,
    const rg_split_gate *all_gates,
    double min_obs,
    int max_depth,
    double penalty,
    double search_gamma,
    double min_commit,
    double n_total
) {
    pivot_obs *remaining;
    pivot_obs *yes;
    pivot_obs *no;
    pivot_obs *best_yes;
    pivot_obs *best_no;
    size_t remaining_count = bucket->obs_count;
    int committed_count = 0;
    rg_status status = RG_OK;

    remaining = (pivot_obs *)calloc(bucket->obs_count == 0 ? 1 : bucket->obs_count, sizeof(*remaining));
    yes = (pivot_obs *)calloc(bucket->obs_count == 0 ? 1 : bucket->obs_count, sizeof(*yes));
    no = (pivot_obs *)calloc(bucket->obs_count == 0 ? 1 : bucket->obs_count, sizeof(*no));
    if (remaining == 0 || yes == 0 || no == 0) {
        free(remaining);
        free(yes);
        free(no);
        return RG_ERR_OOM;
    }
    memcpy(remaining, bucket->obs, bucket->obs_count * sizeof(*remaining));

    best_yes = (pivot_obs *)calloc(bucket->obs_count == 0 ? 1 : bucket->obs_count, sizeof(*best_yes));
    best_no = (pivot_obs *)calloc(bucket->obs_count == 0 ? 1 : bucket->obs_count, sizeof(*best_no));
    if (best_yes == 0 || best_no == 0) {
        free(remaining); free(yes); free(no); free(best_yes); free(best_no);
        return RG_ERR_OOM;
    }

    while (committed_count < max_depth * 4 && observation_weight(remaining, remaining_count) >= min_obs) {
        size_t best_candidate = 0;
        size_t best_yes_count = 0;
        size_t best_no_count = 0;
        double delta_bic = 0.0;
        double search_margin = 0.0;

        if (!pivot_best_split(state, remaining, remaining_count, candidates, gates,
                              candidate_count, penalty, search_gamma, yes, no, best_yes, best_no,
                              &best_yes_count, &best_no_count, &best_candidate,
                              &delta_bic, &search_margin)) {
            break;
        }
        {
            rg_context_spec yes_context;
            status = rg_context_from_candidate_internal(&candidates[best_candidate], &yes_context);
            if (status == RG_OK) {
                status = emit_sister_classes(
                    state,
                    bucket->lect,
                    bucket->grapheme,
                    &yes_context,
                    best_yes,
                    best_yes_count,
                    best_no,
                    best_no_count,
                    delta_bic,
                    search_margin,
                    state->decision_count++,
                    min_commit,
                    n_total
                );
                /* The group that satisfied this predicate may still be mixed;
                 * a second predicate within it is a narrower environment, not
                 * a competing rule. */
                if (status == RG_OK) {
                    status = refine_pivot_split(state, bucket, &yes_context, best_yes,
                                                best_yes_count, 1, max_depth, all_gates,
                                                penalty, search_gamma, min_commit, n_total);
                }
                rg_context_spec_clear_internal(&yes_context);
            }
        }
        memcpy(remaining, best_no, best_no_count * sizeof(*remaining));
        remaining_count = best_no_count;
        if (status != RG_OK) {
            break;
        }
        committed_count++;
    }
    free(best_yes);
    free(best_no);
    free(remaining);
    free(yes);
    free(no);
    return status;
}

static int merged_class_cmp(const void *a, const void *b) {
    const merged_class *ma = (const merged_class *)a;
    const merged_class *mb = (const merged_class *)b;
    size_t i;
    if (ma->count != mb->count) {
        return ma->count > mb->count ? -1 : 1;
    }
    for (i = 0; i < ma->segment_count && i < mb->segment_count; i++) {
        int c = strcmp(ma->lects[i], mb->lects[i]);
        if (c != 0) {
            return c;
        }
        c = strcmp(ma->graphemes[i], mb->graphemes[i]);
        if (c != 0) {
            return c;
        }
    }
    if (ma->segment_count != mb->segment_count) {
        return ma->segment_count < mb->segment_count ? -1 : 1;
    }
    return 0;
}

static void merged_classes_free(merged_class *items, size_t count) {
    size_t i;
    if (items == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        free(items[i].key);
        free(items[i].observation_indices);
        string_array_clear(items[i].lects, items[i].segment_count);
        string_array_clear(items[i].graphemes, items[i].segment_count);
        if (items[i].contexts != 0) {
            size_t j;
            for (j = 0; j < items[i].segment_count; j++) {
                rg_context_spec_clear_internal(&items[i].contexts[j]);
            }
            free(items[i].contexts);
        }
    }
    free(items);
}

/* Builds the full (lect, grapheme) tuple of a committed split, ascending by
 * lect id: the pivot merged into its sister tuple. */
static rg_status committed_split_segments(
    const discovery_state *state,
    const committed_split *split,
    char ***out_lects,
    char ***out_graphemes,
    size_t *out_count
) {
    const sister_tuple *sister = &state->sisters[split->sister_index];
    size_t total = sister->count + 1;
    char **lects = (char **)calloc(total, sizeof(*lects));
    char **graphemes = (char **)calloc(total, sizeof(*graphemes));
    size_t i;
    size_t insert_at = sister->count;
    size_t n = 0;

    if (lects == 0 || graphemes == 0) {
        free(lects);
        free(graphemes);
        return RG_ERR_OOM;
    }
    for (i = 0; i < sister->count; i++) {
        if (strcmp(sister->lects[i], split->pivot_lect) > 0) {
            insert_at = i;
            break;
        }
    }
    for (i = 0; i < total; i++) {
        const char *lect;
        const char *grapheme;
        if (i == insert_at) {
            lect = split->pivot_lect;
            grapheme = split->pivot_grapheme;
        } else {
            size_t src = i < insert_at ? i : i - 1;
            lect = sister->lects[src];
            grapheme = sister->graphemes[src];
        }
        lects[n] = rg_strdup_internal(lect);
        graphemes[n] = rg_strdup_internal(grapheme);
        if (lects[n] == 0 || graphemes[n] == 0) {
            string_array_clear(lects, n + 1);
            string_array_clear(graphemes, n + 1);
            return RG_ERR_OOM;
        }
        n++;
    }
    *out_lects = lects;
    *out_graphemes = graphemes;
    *out_count = total;
    return RG_OK;
}

static rg_status merged_class_add_evidence(
    merged_class *entry,
    const size_t *indices,
    size_t count
) {
    size_t i;
    for (i = 0; i < count; i++) {
        size_t j;
        int seen = 0;
        for (j = 0; j < entry->observation_count; j++) {
            if (entry->observation_indices[j] == indices[i]) {
                seen = 1;
                break;
            }
        }
        if (seen) {
            continue;
        }
        if (entry->observation_count == entry->observation_cap) {
            size_t next_cap = entry->observation_cap == 0 ? 8 : entry->observation_cap * 2;
            size_t *next = (size_t *)realloc(entry->observation_indices, next_cap * sizeof(*next));
            if (next == 0) {
                return RG_ERR_OOM;
            }
            entry->observation_indices = next;
            entry->observation_cap = next_cap;
        }
        entry->observation_indices[entry->observation_count++] = indices[i];
    }
    return RG_OK;
}

static rg_status merge_committed_splits(
    const discovery_state *state,
    merged_class **out,
    size_t *out_count
) {
    merged_class *merged = 0;
    size_t count = 0;
    size_t cap = 0;
    size_t i;

    for (i = 0; i < state->committed_count; i++) {
        const committed_split *split = &state->committed[i];
        char **lects = 0;
        char **graphemes = 0;
        size_t segment_count = 0;
        char *key = 0;
        double coverage;
        size_t existing;
        size_t j;
        rg_status status;

        status = committed_split_segments(state, split, &lects, &graphemes, &segment_count);
        if (status != RG_OK) {
            merged_classes_free(merged, count);
            return status;
        }
        status = join_lect_grapheme_key(lects, graphemes, segment_count, &key);
        if (status != RG_OK) {
            string_array_clear(lects, segment_count);
            string_array_clear(graphemes, segment_count);
            merged_classes_free(merged, count);
            return status;
        }
        coverage = split->bucket_size > 0.0 ? split->count / split->bucket_size : 0.0;

        existing = count;
        for (j = 0; j < count; j++) {
            if (strcmp(merged[j].key, key) == 0) {
                existing = j;
                break;
            }
        }
        if (existing < count) {
            merged_class *entry = &merged[existing];
            size_t slot;
            string_array_clear(lects, segment_count);
            string_array_clear(graphemes, segment_count);
            free(key);
            if (merged_class_add_evidence(entry, split->observation_indices,
                                          split->observation_count) != RG_OK) {
                merged_classes_free(merged, count);
                return RG_ERR_OOM;
            }
            if (split->count > entry->count) {
                entry->count = split->count;
            }
            if (coverage > entry->confidence) {
                entry->confidence = coverage;
                entry->bucket_size = split->bucket_size;
                entry->winning_count = split->count;
                /* The contrast and the score belong to the split whose
                 * coverage the class is reporting, not to whichever pivot
                 * merged in last. */
                entry->contrast_count = split->contrast_count;
                entry->delta_bic = split->delta_bic;
                entry->search_margin = split->search_margin;
                /* The earliest decision that reached this class keeps it. */
                if (split->decision_index < entry->decision_index) {
                    entry->decision_index = split->decision_index;
                }
            }
            for (slot = 0; slot < entry->segment_count; slot++) {
                if (strcmp(entry->lects[slot], split->pivot_lect) != 0) {
                    continue;
                }
                if (rg_context_spec_constraint_count(&split->context) >
                    rg_context_spec_constraint_count(&entry->contexts[slot])) {
                    rg_context_spec_clear_internal(&entry->contexts[slot]);
                    status = rg_context_spec_copy_internal(&split->context, &entry->contexts[slot]);
                    if (status != RG_OK) {
                        merged_classes_free(merged, count);
                        return status;
                    }
                }
                break;
            }
            continue;
        }

        if (count == cap) {
            size_t next_cap = cap == 0 ? 16 : cap * 2;
            merged_class *next = (merged_class *)realloc(merged, next_cap * sizeof(*next));
            if (next == 0) {
                string_array_clear(lects, segment_count);
                string_array_clear(graphemes, segment_count);
                free(key);
                merged_classes_free(merged, count);
                return RG_ERR_OOM;
            }
            merged = next;
            cap = next_cap;
        }
        memset(&merged[count], 0, sizeof(merged[count]));
        merged[count].key = key;
        merged[count].lects = lects;
        merged[count].graphemes = graphemes;
        merged[count].segment_count = segment_count;
        merged[count].count = split->count;
        merged[count].confidence = coverage;
        merged[count].bucket_size = split->bucket_size;
        merged[count].winning_count = split->count;
        merged[count].contrast_count = split->contrast_count;
        merged[count].delta_bic = split->delta_bic;
        merged[count].search_margin = split->search_margin;
        merged[count].decision_index = split->decision_index;
        if (merged_class_add_evidence(&merged[count], split->observation_indices,
                                      split->observation_count) != RG_OK) {
            merged_classes_free(merged, count + 1);
            return RG_ERR_OOM;
        }
        merged[count].contexts = (rg_context_spec *)calloc(segment_count, sizeof(*merged[count].contexts));
        if (merged[count].contexts == 0) {
            merged[count].contexts = 0;
            merged_classes_free(merged, count + 1);
            return RG_ERR_OOM;
        }
        for (j = 0; j < segment_count; j++) {
            rg_context_spec_init_empty(&merged[count].contexts[j]);
        }
        for (j = 0; j < segment_count; j++) {
            if (strcmp(merged[count].lects[j], split->pivot_lect) == 0) {
                status = rg_context_spec_copy_internal(&split->context, &merged[count].contexts[j]);
                if (status != RG_OK) {
                    merged_classes_free(merged, count + 1);
                    return status;
                }
                break;
            }
        }
        count++;
    }
    *out = merged;
    *out_count = count;
    return RG_OK;
}

/* Class-level context discovery: for each (pivot lect, pivot grapheme) that
 * appears across more than one sister tuple, a greedy BIC-driven split on the
 * pivot's own phonological context. Committed splits become conditioned
 * classes, deduplicated across pivots by their full segment tuple. */
rg_status multi_lect_context_discovery(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    rg_multi_model *model,
    const reconciled_observation *observations,
    size_t observation_count
) {
    discovery_state state;
    rg_feature_vocabulary vocabulary;
    rg_split_gate *immediate_gates = 0;
    rg_split_gate *long_gates = 0;
    rg_split_gate *all_gates = 0;
    rg_context_spec **form_contexts = 0;
    size_t *form_context_counts = 0;
    size_t cache_size = cognate_count * model->lect_count;
    merged_class *merged = 0;
    size_t merged_count = 0;
    size_t i;
    rg_status status = RG_OK;

    if (observation_count == 0 || model->lect_count == 0) {
        return RG_OK;
    }
    memset(&state, 0, sizeof(state));
    vocabulary.entries = 0;
    vocabulary.count = 0;

    form_contexts = (rg_context_spec **)calloc(cache_size == 0 ? 1 : cache_size, sizeof(*form_contexts));
    form_context_counts = (size_t *)calloc(cache_size == 0 ? 1 : cache_size, sizeof(*form_context_counts));
    if (form_contexts == 0 || form_context_counts == 0) {
        free(form_contexts);
        free(form_context_counts);
        return RG_ERR_OOM;
    }

    for (i = 0; i < observation_count && status == RG_OK; i++) {
        const reconciled_observation *obs = &observations[i];
        size_t p;
        if (obs->weight <= 0.0) {
            continue;
        }
        for (p = 0; p < obs->segment_count && status == RG_OK; p++) {
            char *sister_lects[64];
            char *sister_graphemes[64];
            size_t sister_count = 0;
            size_t sister_index = 0;
            size_t pivot_index = 0;
            size_t cache_index;
            size_t q;
            if (obs->segment_count < 2 || obs->segment_count > 64) {
                continue;
            }
            for (q = 0; q < obs->segment_count; q++) {
                if (q == p) {
                    continue;
                }
                sister_lects[sister_count] = obs->lects[q];
                sister_graphemes[sister_count] = obs->graphemes[q];
                sister_count++;
            }
            status = sister_index_for(&state, sister_lects, sister_graphemes, sister_count, &sister_index);
            if (status != RG_OK) {
                break;
            }
            cache_index = obs->cognate_index * model->lect_count + obs->lect_indices[p];
            if (form_contexts[cache_index] == 0) {
                const rg_form *form = form_for_lect(&cognates[obs->cognate_index], obs->lects[p]);
                if (form == 0) {
                    continue;
                }
                status = rg_form_position_contexts_internal(
                    ctx,
                    form,
                    &form_contexts[cache_index],
                    &form_context_counts[cache_index]
                );
                if (status != RG_OK) {
                    break;
                }
            }
            if (obs->positions[p] >= form_context_counts[cache_index]) {
                continue;
            }
            status = pivot_index_for(&state, obs->lects[p], obs->graphemes[p], &pivot_index);
            if (status != RG_OK) {
                break;
            }
            status = pivot_bucket_append(
                &state.pivots[pivot_index],
                sister_index,
                &form_contexts[cache_index][obs->positions[p]],
                obs->weight,
                i
            );
            if (status == RG_OK) {
                status = collect_stress_values(&state, &form_contexts[cache_index][obs->positions[p]]);
                collect_morphology_values(&state, &form_contexts[cache_index][obs->positions[p]]);
            }
        }
    }

    if (status == RG_OK) {
        /* The candidate lists borrow the vocabulary's strings, so it has to
         * outlive them: it is released with the rest of the discovery state. */
        status = rg_feature_vocabulary_build_from_sets_internal(ctx, cognates, cognate_count, &vocabulary);
        if (status == RG_OK) {
            status = build_candidate_lists(&state, &vocabulary);
        }
    }
    /* One bar per candidate: the immediate axes are few and cheap to trust,
     * the long-range ones many and easy to fit by chance, and refinement has
     * to weigh both at once. */
    if (status == RG_OK) {
        size_t k;
        immediate_gates = (rg_split_gate *)calloc(state.immediate_count + 1, sizeof(*immediate_gates));
        long_gates = (rg_split_gate *)calloc(state.long_range_count + 1, sizeof(*long_gates));
        all_gates = (rg_split_gate *)calloc(state.all_count + 1, sizeof(*all_gates));
        if (immediate_gates == 0 || long_gates == 0 || all_gates == 0) {
            status = RG_ERR_OOM;
        } else {
            for (k = 0; k < state.immediate_count; k++) {
                immediate_gates[k].min_obs = (double)options->bic.min_split_observations;
                immediate_gates[k].delta_threshold = options->bic.delta_bic_threshold;
                immediate_gates[k].min_dominant_fraction = 0.0;
                all_gates[k] = immediate_gates[k];
            }
            for (k = 0; k < state.long_range_count; k++) {
                long_gates[k].min_obs = (double)options->bic.long_range_min_split_observations;
                long_gates[k].delta_threshold = options->bic.long_range_delta_bic_threshold;
                long_gates[k].min_dominant_fraction = options->bic.long_range_min_dominant_fraction;
                all_gates[state.immediate_count + k] = long_gates[k];
            }
        }
    }

    /* Pivot buckets are processed in ascending (lect, grapheme) order so that
     * the cross-pivot merge below resolves ties deterministically. */
    if (status == RG_OK && state.pivot_count > 1) {
        size_t a;
        for (a = 1; a < state.pivot_count; a++) {
            pivot_bucket key = state.pivots[a];
            size_t b = a;
            while (b > 0) {
                int c = strcmp(state.pivots[b - 1].lect, key.lect);
                if (c == 0) {
                    c = strcmp(state.pivots[b - 1].grapheme, key.grapheme);
                }
                if (c <= 0) {
                    break;
                }
                state.pivots[b] = state.pivots[b - 1];
                b--;
            }
            state.pivots[b] = key;
        }
    }

    for (i = 0; i < state.pivot_count && status == RG_OK; i++) {
        const pivot_bucket *bucket = &state.pivots[i];
        double mass = observation_weight(bucket->obs, bucket->obs_count);
        size_t distinct = 0;
        size_t j;
        double n_total;
        double penalty;
        double min_commit;

        for (j = 0; j < bucket->obs_count; j++) {
            size_t k;
            int seen = 0;
            for (k = 0; k < j; k++) {
                if (bucket->obs[k].sister_index == bucket->obs[j].sister_index) {
                    seen = 1;
                    break;
                }
            }
            if (!seen) {
                distinct++;
            }
        }
        if (distinct < 2 || mass < 4.0) {
            continue;
        }

        n_total = mass;
        penalty = log(n_total);
        if (options->bic.multi_lect_bic_small_sample_correction) {
            double denom = n_total - 1.0 < 1.0 ? 1.0 : n_total - 1.0;
            penalty += 2.0 / denom;
        }
        min_commit = (double)multi_lect_min_commit_count(n_total, options->bic.multi_lect_min_commit_scale);
        if (n_total >= 4.0) {
            status = commit_splits_for_pivot(
                &state,
                bucket,
                state.immediate,
                immediate_gates,
                state.immediate_count,
                all_gates,
                (double)options->bic.min_split_observations,
                options->bic.max_split_depth,
                penalty,
                options->bic.search_penalty_gamma,
                min_commit,
                n_total
            );
        }
        if (status != RG_OK) {
            break;
        }
        if (n_total >= (double)options->bic.long_range_min_split_observations) {
            double long_min_commit = (double)multi_lect_min_commit_count(n_total, options->bic.multi_lect_min_commit_scale);
            if (long_min_commit < (double)options->bic.long_range_min_split_observations) {
                long_min_commit = (double)options->bic.long_range_min_split_observations;
            }
            status = commit_splits_for_pivot(
                &state,
                bucket,
                state.long_range,
                long_gates,
                state.long_range_count,
                all_gates,
                (double)options->bic.long_range_min_split_observations,
                options->bic.max_split_depth,
                log(n_total),
                options->bic.search_penalty_gamma,
                long_min_commit,
                n_total
            );
        }
    }

    if (status == RG_OK) {
        status = merge_committed_splits(&state, &merged, &merged_count);
    }

    if (status == RG_OK && merged_count > 0) {
        if (merged_count > 1) {
            qsort(merged, merged_count, sizeof(*merged), merged_class_cmp);
        }
        model->conditioned_classes = (rg_multi_class_owned *)calloc(merged_count, sizeof(*model->conditioned_classes));
        if (model->conditioned_classes == 0) {
            status = RG_ERR_OOM;
        } else {
            for (i = 0; i < merged_count; i++) {
                model->conditioned_classes[i].view.class_id = (int)(model->unconditioned_class_count + i);
                model->conditioned_classes[i].view.lect_ids = (const char *const *)merged[i].lects;
                model->conditioned_classes[i].view.graphemes = (const char *const *)merged[i].graphemes;
                model->conditioned_classes[i].view.contexts = merged[i].contexts;
                model->conditioned_classes[i].view.segment_count = merged[i].segment_count;
                model->conditioned_classes[i].view.count = merged[i].count;
                model->conditioned_classes[i].view.confidence = merged[i].confidence;
                model->conditioned_classes[i].view.contrast_count = merged[i].contrast_count;
                model->conditioned_classes[i].view.delta_bic = merged[i].delta_bic;
                model->conditioned_classes[i].view.search_margin = merged[i].search_margin;
                model->conditioned_classes[i].view.decision_index = merged[i].decision_index;
                model->conditioned_classes[i].view.uncertainty =
                    rg_wilson_default_internal(merged[i].winning_count, merged[i].bucket_size);
                /* The environment was chosen by the same observations, so the
                 * interval says how well the rate is pinned given it, not
                 * whether it is real. search_margin answers that. */
                model->conditioned_classes[i].view.uncertainty.post_selection = 1;
                merged[i].lects = 0;
                merged[i].graphemes = 0;
                merged[i].contexts = 0;
            }
            model->conditioned_class_count = merged_count;
        }
    }

    /* Tag each reconciled position with the conditioned classes it realises,
     * taken from the observations that justified each committed split. This is
     * not re-derivable afterwards: a class merged from two pivots carries a
     * context from each, and no single observation need satisfy both at once,
     * so testing the published contexts as a conjunction can match nothing. */
    if (status == RG_OK) {
        for (i = 0; i < merged_count && status == RG_OK; i++) {
            size_t j;
            for (j = 0; j < merged[i].observation_count && status == RG_OK; j++) {
                size_t index = merged[i].observation_indices[j];
                if (index < model->class_position_count) {
                    status = class_position_add_id(&model->class_positions[index],
                                                   (int)(model->unconditioned_class_count + i));
                }
            }
        }
    }

    merged_classes_free(merged, merged_count);
    for (i = 0; i < cache_size; i++) {
        rg_context_spec_array_free_internal(form_contexts[i], form_context_counts[i]);
    }
    free(form_contexts);
    free(form_context_counts);
    rg_feature_vocabulary_clear_internal(&vocabulary);
    discovery_state_clear(&state);
    free(immediate_gates);
    free(long_gates);
    free(all_gates);
    return status;
}

