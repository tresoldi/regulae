#include "model_internal.h"
#include "split_search.h"
#include "environment.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct context_observation {
    char *source;
    char *target;
    rg_context_spec context;
    double weight;
} context_observation;

typedef rg_split_candidate split_candidate;


static void context_observation_clear(context_observation *obs) {
    if (obs == 0) {
        return;
    }
    free(obs->source);
    free(obs->target);
    rg_context_spec_clear_internal(&obs->context);
    obs->source = 0;
    obs->target = 0;
    obs->weight = 0.0;
}

/* `swap` records the link from the target's point of view: the target grapheme
 * becomes the thing being conditioned and the source grapheme the outcome, and
 * the caller supplies the target form's context. Everything downstream --
 * bucketing, splitting, refinement -- then works unchanged, and the roles are
 * put back when the row is written. */
static rg_status append_context_observation_as(
    context_observation **items,
    size_t *count,
    size_t *cap,
    const rg_link *link,
    const rg_context_spec *context,
    int swap,
    double weight
) {
    context_observation *next;
    rg_status status;
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 32 : *cap * 2;
        next = (context_observation *)realloc(*items, next_cap * sizeof(**items));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *items = next;
        *cap = next_cap;
    }
    memset(&(*items)[*count], 0, sizeof((*items)[*count]));
    (*items)[*count].source = rg_strdup_internal(
        swap ? link->target[0].grapheme : link->source[0].grapheme);
    (*items)[*count].target = rg_strdup_internal(
        swap ? link->source[0].grapheme : link->target[0].grapheme);
    (*items)[*count].weight = weight;
    if ((*items)[*count].source == 0 || (*items)[*count].target == 0) {
        context_observation_clear(&(*items)[*count]);
        return RG_ERR_OOM;
    }
    status = rg_context_spec_copy_internal(context, &(*items)[*count].context);
    if (status != RG_OK) {
        context_observation_clear(&(*items)[*count]);
        return status;
    }
    (*count)++;
    return RG_OK;
}

typedef struct target_mass {
    const char *target;
    double mass;
} target_mass;

static rg_status add_target_mass(target_mass **items, size_t *count, size_t *cap, const char *target, double weight) {
    size_t i;
    target_mass *next;
    for (i = 0; i < *count; i++) {
        if (strcmp((*items)[i].target, target) == 0) {
            (*items)[i].mass += weight;
            return RG_OK;
        }
    }
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 4 : *cap * 2;
        next = (target_mass *)realloc(*items, next_cap * sizeof(**items));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *items = next;
        *cap = next_cap;
    }
    (*items)[*count].target = target;
    (*items)[*count].mass = weight;
    (*count)++;
    return RG_OK;
}

static int string_ptr_cmp(const void *a, const void *b) {
    const char *const *sa = (const char *const *)a;
    const char *const *sb = (const char *const *)b;
    return strcmp(*sa, *sb);
}

static rg_status append_unique_source(const char ***items, size_t *count, size_t *cap, const char *source) {
    size_t i;
    const char **next;
    for (i = 0; i < *count; i++) {
        if (strcmp((*items)[i], source) == 0) {
            return RG_OK;
        }
    }
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 16 : *cap * 2;
        next = (const char **)realloc(*items, next_cap * sizeof(**items));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *items = next;
        *cap = next_cap;
    }
    (*items)[(*count)++] = source;
    return RG_OK;
}

/* ---- pairwise context discovery ---------------------------------------- */


static const char *const split_positions[] = {"initial", "medial", "final"};


typedef struct stress_inventory {
    char **values;
    size_t count;
    size_t cap;
} stress_inventory;

static void stress_inventory_clear(stress_inventory *inventory) {
    size_t i;
    for (i = 0; i < inventory->count; i++) {
        free(inventory->values[i]);
    }
    free(inventory->values);
    memset(inventory, 0, sizeof(*inventory));
}

static rg_status stress_inventory_add(stress_inventory *inventory, const char *value) {
    size_t i;
    size_t insert_at;
    for (i = 0; i < inventory->count; i++) {
        if (strcmp(inventory->values[i], value) == 0) {
            return RG_OK;
        }
    }
    if (inventory->count == inventory->cap) {
        size_t next_cap = inventory->cap == 0 ? 4 : inventory->cap * 2;
        char **next = (char **)realloc(inventory->values, next_cap * sizeof(*next));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        inventory->values = next;
        inventory->cap = next_cap;
    }
    insert_at = inventory->count;
    while (insert_at > 0 && strcmp(inventory->values[insert_at - 1], value) > 0) {
        insert_at--;
    }
    if (insert_at < inventory->count) {
        memmove(&inventory->values[insert_at + 1], &inventory->values[insert_at],
                (inventory->count - insert_at) * sizeof(*inventory->values));
    }
    inventory->values[insert_at] = rg_strdup_internal(value);
    if (inventory->values[insert_at] == 0) {
        return RG_ERR_OOM;
    }
    inventory->count++;
    return RG_OK;
}

static rg_status stress_inventory_add_cb(void *user, const char *value) {
    return stress_inventory_add((stress_inventory *)user, value);
}

static rg_status collect_observed_stress(stress_inventory *inventory, const rg_context_spec *context) {
    return rg_env_collect_stress_values(context, stress_inventory_add_cb, inventory);
}

/* Candidate axes not already constrained by base_context. A feature is dropped
 * from the preceding/following slots once that slot constrains it, the position
 * axis disappears once a position is fixed, and a stress value disappears once
 * that slot already carries it. */
/* Which morphological values the corpus actually shows, collected the same way
 * the stress values are: from the observations, so a corpus without boundaries
 * gets no candidates and pays nothing for the axis. */
typedef struct morphology_inventory {
    const char *placements[8];
    size_t placement_count;
    const char *indices[8];
    size_t index_count;
} morphology_inventory;

static void morphology_inventory_add(const char **values, size_t *count, size_t cap, const char *value) {
    size_t i;
    if (value == 0 || value[0] == '\0') {
        return;
    }
    for (i = 0; i < *count; i++) {
        if (strcmp(values[i], value) == 0) {
            return;
        }
    }
    if (*count < cap) {
        values[*count] = value;
        (*count)++;
    }
}

static void collect_observed_morphology(morphology_inventory *inventory, const rg_context_spec *context) {
    morphology_inventory_add(inventory->placements, &inventory->placement_count, 8, context->morphological);
    morphology_inventory_add(inventory->indices, &inventory->index_count, 8, context->morpheme_index);
}

static size_t immediate_candidates_for(
    const rg_context_spec *base_context,
    const stress_inventory *stress,
    const morphology_inventory *morphology,
    const rg_feature_vocabulary *vocabulary,
    split_candidate *out,
    size_t capacity
) {
    size_t count = 0;
    size_t i;
    size_t s;
    for (i = 0; i < 2 * vocabulary->count; i++) {
        split_candidate generated;
        const split_candidate *candidate = &generated;
        int skip = 0;
        size_t entry = i / 2;
        generated.slot = (i % 2) == 0 ? "preceding" : "following";
        generated.feature = vocabulary->entries[entry].feature;
        generated.value = vocabulary->entries[entry].value;
        if (strcmp(candidate->slot, "following") == 0) {
            size_t j;
            for (j = 0; j < base_context->following_count; j++) {
                if (strcmp(base_context->following[j].feature, candidate->feature) == 0) {
                    skip = 1;
                    break;
                }
            }
        } else if (strcmp(candidate->slot, "preceding") == 0) {
            size_t j;
            for (j = 0; j < base_context->preceding_count; j++) {
                if (strcmp(base_context->preceding[j].feature, candidate->feature) == 0) {
                    skip = 1;
                    break;
                }
            }
        }
        if (!skip && count < capacity) {
            out[count++] = *candidate;
        }
    }
    if (base_context->position == 0 || base_context->position[0] == '\0') {
        for (i = 0; i < sizeof(split_positions) / sizeof(split_positions[0]); i++) {
            if (count < capacity) {
                out[count].slot = "position";
                out[count].feature = split_positions[i];
                out[count].value = "+";
                count++;
            }
        }
    }
    /* The morphological axes exist only for corpora that carry boundaries, and
     * `morphology` holds the values actually observed -- there is no point
     * asking about a fourth morpheme in a corpus whose words have two. */
    if (base_context->morphological == 0 || base_context->morphological[0] == '\0') {
        for (i = 0; i < morphology->placement_count; i++) {
            if (count < capacity) {
                out[count].slot = "morphological";
                out[count].feature = morphology->placements[i];
                out[count].value = "+";
                count++;
            }
        }
    }
    if (base_context->morpheme_index == 0 || base_context->morpheme_index[0] == '\0') {
        for (i = 0; i < morphology->index_count; i++) {
            if (count < capacity) {
                out[count].slot = "morpheme_index";
                out[count].feature = morphology->indices[i];
                out[count].value = "+";
                count++;
            }
        }
    }
    for (s = 0; s < rg_env_stress_slot_count; s++) {
        const rg_feature_constraint *existing = 0;
        size_t existing_count = 0;
        if (s == 0) {
            existing = base_context->self_stress;
            existing_count = base_context->self_stress_count;
        } else if (s == 1) {
            existing = base_context->preceding_stress;
            existing_count = base_context->preceding_stress_count;
        } else {
            existing = base_context->following_stress;
            existing_count = base_context->following_stress_count;
        }
        for (i = 0; i < stress->count; i++) {
            int skip = 0;
            size_t j;
            for (j = 0; j < existing_count; j++) {
                if (existing[j].feature != 0 && strcmp(existing[j].feature, "stress") == 0 &&
                    strcmp(existing[j].value, stress->values[i]) == 0) {
                    skip = 1;
                    break;
                }
            }
            if (!skip && count < capacity) {
                out[count].slot = rg_env_stress_slots[s];
                out[count].feature = "stress";
                out[count].value = stress->values[i];
                count++;
            }
        }
    }
    return count;
}

/* Properties of a syllable rather than of its segments, so they are not in the
 * feature vocabulary and are offered directly. A predicate true of every
 * syllable in a corpus partitions nothing and is dropped by the split gate, so
 * the cost of offering them where they do not apply is one gate test each. */
static const rg_feature_constraint syllable_shape_candidates[] = {
    {"syllable_shape", "open"},
    {"syllable_shape", "closed"},
    {"syllable_nucleus", "long"},
    {"syllable_nucleus", "short"}
};

static size_t long_range_candidates(
    const rg_feature_vocabulary *vocabulary,
    split_candidate *out,
    size_t capacity
) {
    size_t count = 0;
    size_t s;
    size_t f;
    for (s = 0; s < rg_env_long_range_slot_count; s++) {
        for (f = 0; f < vocabulary->count; f++) {
            if (count < capacity) {
                out[count].slot = rg_env_long_range_slots[s];
                out[count].feature = vocabulary->entries[f].feature;
                out[count].value = vocabulary->entries[f].value;
                count++;
            }
        }
        if (strcmp(rg_env_long_range_slots[s], "same_syllable") == 0 ||
            strcmp(rg_env_long_range_slots[s], "next_syllable") == 0 ||
            strcmp(rg_env_long_range_slots[s], "previous_syllable") == 0) {
            for (f = 0; f < sizeof(syllable_shape_candidates) / sizeof(syllable_shape_candidates[0]); f++) {
                if (count < capacity) {
                    out[count].slot = rg_env_long_range_slots[s];
                    out[count].feature = syllable_shape_candidates[f].feature;
                    out[count].value = syllable_shape_candidates[f].value;
                    count++;
                }
            }
        }
    }
    return count;
}


/* Negative log-likelihood of a group under a single unconditioned
 * correspondence. Target keys are summed in sorted order so repeated runs
 * produce bit-identical results. */
/* Writes one conditioned entry per observed target in this group. Repeating a
 * (source, target, context) key replaces the previous count rather than adding
 * to it, because each commit states the mass of that group outright. */
static rg_status commit_observation_group(
    rg_pairwise_model *model,
    const char *source,
    const rg_split_observation *rows,
    size_t count,
    const rg_context_spec *context,
    int target_side,
    double bucket_total,
    const rg_split_observation *contrast_rows,
    size_t contrast_row_count,
    double delta_bic,
    double search_margin,
    int decision_index
) {
    target_mass *targets = 0;
    target_mass *contrast_targets = 0;
    size_t contrast_target_count = 0;
    size_t contrast_target_cap = 0;
    double contrast_total = 0.0;
    size_t target_count = 0;
    size_t target_cap = 0;
    size_t conditioned_cap = model->conditioned_segment_count_count;
    /* On the target side the bucket key is a target grapheme, which the segment
     * table is not keyed by, so the denominator is the bucket's own mass. */
    double source_total = target_side
        ? bucket_total
        : source_total_for_rows(model->segment_counts, model->segment_count_count, source);
    size_t i;
    rg_status status = RG_OK;

    for (i = 0; i < count; i++) {
        status = add_target_mass(&targets, &target_count, &target_cap, rows[i].key, rows[i].weight);
        if (status != RG_OK) {
            free(targets);
            return status;
        }
    }
    /* The same correspondence where the environment does not hold. A rule
     * published without it cannot be read. */
    for (i = 0; i < contrast_row_count; i++) {
        status = add_target_mass(&contrast_targets, &contrast_target_count, &contrast_target_cap,
                                 contrast_rows[i].key, contrast_rows[i].weight);
        if (status != RG_OK) {
            free(targets);
            free(contrast_targets);
            return status;
        }
        contrast_total += contrast_rows[i].weight;
    }
    for (i = 0; i < target_count && status == RG_OK; i++) {
        size_t existing;
        int replaced = 0;
        double contrast_count = 0.0;
        for (existing = 0; existing < contrast_target_count; existing++) {
            if (strcmp(contrast_targets[existing].target, targets[i].target) == 0) {
                contrast_count = contrast_targets[existing].mass;
                break;
            }
        }
        for (existing = 0; existing < model->conditioned_segment_count_count; existing++) {
            rg_conditioned_segment_count_row *row = &model->conditioned_segment_counts[existing];
            bool a_subset_b = false;
            bool b_subset_a = false;
            const char *want_source = target_side ? targets[i].target : source;
            const char *want_target = target_side ? source : targets[i].target;
            if (row->context_is_target != target_side ||
                strcmp(row->source, want_source) != 0 ||
                strcmp(row->target, want_target) != 0) {
                continue;
            }
            if (rg_context_spec_is_subset(&row->context, context, &a_subset_b) == RG_OK &&
                rg_context_spec_is_subset(context, &row->context, &b_subset_a) == RG_OK &&
                a_subset_b && b_subset_a) {
                row->count = targets[i].mass;
                row->source_total = source_total;
                row->contrast_count = contrast_count;
                row->contrast_total = contrast_total;
                row->delta_bic = delta_bic;
                row->search_margin = search_margin;
                if (row->decision_index < 0 || decision_index < row->decision_index) {
                    row->decision_index = decision_index;
                }
                row->uncertainty = rg_wilson_default_internal(row->count, source_total);
                row->uncertainty.post_selection = 1;
                replaced = 1;
                break;
            }
        }
        if (replaced) {
            continue;
        }
        if (target_side) {
            model->has_target_conditioned = 1;
        }
        status = add_conditioned_segment_count(
            &model->conditioned_segment_counts,
            &model->conditioned_segment_count_count,
            &conditioned_cap,
            target_side ? targets[i].target : source,
            target_side ? source : targets[i].target,
            context,
            target_side,
            targets[i].mass,
            source_total,
            contrast_count,
            contrast_total,
            delta_bic,
            search_margin,
            decision_index
        );
    }
    free(targets);
    free(contrast_targets);
    return status;
}

/* Recursively refines an already-committed conditioned entry, looking for one
 * further conditioning axis that improves BIC. */
/* Deepens a committed split by conjoining a second predicate within the group
 * that satisfied the first.
 *
 * The candidates offered here are the union of every axis, not the immediate
 * neighbours the enclosing stage happened to search. A conditioning
 * environment is not obliged to be built out of one kind of predicate:
 * Grassmann's Law is word-initial *and* followed somewhere by an aspirate, and
 * assimilation is regularly "before X" *and* "after Y". Searching one kind at
 * a time can state either half and never the conjunction.
 *
 * Candidates the base context already implies need no filtering: every row in
 * the group satisfies them, so the split has an empty complement and its own
 * minimum rejects it. */
static rg_status refine_split(
    rg_pairwise_model *model,
    const char *source,
    const rg_split_observation *rows,
    size_t count,
    const rg_context_spec *base_context,
    int depth,
    int max_depth,
    double min_obs,
    const split_candidate *candidates,
    const rg_split_gate *gates,
    size_t candidate_count,
    double penalty,
    double search_gamma,
    size_t observation_capacity,
    int target_side,
    double bucket_total
) {
    rg_split_search search;
    rg_split_result best;
    rg_context_spec yes_context;
    rg_status status;

    if (depth >= max_depth || rg_split_total_weight(rows, count) < min_obs) {
        return RG_OK;
    }
    status = rg_split_search_init(&search, observation_capacity);
    if (status != RG_OK) {
        return status;
    }
    if (!rg_split_find_best(&search, rows, count, candidates, gates, candidate_count,
                            penalty, search_gamma, &best)) {
        rg_split_search_clear(&search);
        return RG_OK;
    }
    status = rg_context_extend_internal(base_context, &best.candidate, &yes_context);
    if (status == RG_OK) {
        status = commit_observation_group(model, source, search.best_yes, best.yes_count, &yes_context,
                                          target_side, bucket_total,
                                          search.best_no, best.no_count, best.delta_bic,
                                          best.search_margin, model->decision_count++);
        if (status == RG_OK) {
            status = refine_split(
                model,
                source,
                search.best_yes,
                best.yes_count,
                &yes_context,
                depth + 1,
                max_depth,
                min_obs,
                candidates,
                gates,
                candidate_count,
                penalty,
                search_gamma,
                observation_capacity,
                target_side,
                bucket_total
            );
        }
        rg_context_spec_clear_internal(&yes_context);
    }
    rg_split_search_clear(&search);
    return status;
}

/* Sequential greedy splitting for one source grapheme: repeatedly take the best
 * BIC-improving split of the remaining observations, commit it (plus a
 * refinement of its YES side when immediate axes are in play), and continue on
 * the NO side. */
static rg_status commit_splits_for_source(
    rg_pairwise_model *model,
    const char *source,
    const rg_split_observation *rows,
    size_t count,
    const split_candidate *top_candidates,
    const rg_split_gate *top_gates,
    size_t top_candidate_count,
    const split_candidate *all_candidates,
    const rg_split_gate *all_gates,
    size_t all_candidate_count,
    int max_depth,
    double min_obs,
    double penalty,
    double search_gamma,
    int target_side
) {
    rg_split_search search;
    rg_split_observation *remaining;
    size_t remaining_count = count;
    int committed = 0;
    rg_status status;

    status = rg_split_search_init(&search, count);
    if (status != RG_OK) {
        return status;
    }
    remaining = (rg_split_observation *)calloc(count == 0 ? 1 : count, sizeof(*remaining));
    if (remaining == 0) {
        rg_split_search_clear(&search);
        return RG_ERR_OOM;
    }
    memcpy(remaining, rows, count * sizeof(*remaining));

    while (committed < RG_SPLIT_MAX_COMMITS(max_depth) && rg_split_total_weight(remaining, remaining_count) >= min_obs) {
        rg_split_result best;
        rg_context_spec yes_context;
        rg_context_spec empty;

        if (!rg_split_find_best(&search, remaining, remaining_count, top_candidates, top_gates,
                                top_candidate_count, penalty, search_gamma, &best)) {
            break;
        }
        rg_context_spec_init_empty(&empty);
        status = rg_context_extend_internal(&empty, &best.candidate, &yes_context);
        rg_context_spec_clear_internal(&empty);
        if (status != RG_OK) {
            break;
        }
        status = commit_observation_group(model, source, search.best_yes, best.yes_count, &yes_context,
                                          target_side, rg_split_total_weight(rows, count),
                                          search.best_no, best.no_count, best.delta_bic,
                                          best.search_margin, model->decision_count++);
        if (status == RG_OK) {
            status = refine_split(
                model,
                source,
                search.best_yes,
                best.yes_count,
                &yes_context,
                1,
                max_depth,
                min_obs,
                all_candidates,
                all_gates,
                all_candidate_count,
                penalty,
                search_gamma,
                count,
                target_side,
                rg_split_total_weight(rows, count)
            );
        }
        rg_context_spec_clear_internal(&yes_context);
        if (status != RG_OK) {
            break;
        }
        memcpy(remaining, search.best_no, best.no_count * sizeof(*remaining));
        remaining_count = best.no_count;
        committed++;
    }
    free(remaining);
    rg_split_search_clear(&search);
    return status;
}

/* Flattens the corpus alignments into (source, target, context, weight) rows.
 * When decompose_chunks is set, multi-segment links are broken down through a
 * one-segment sub-alignment and each resulting 1-to-1 pair inherits the outer
 * link's context, matching the Go reference. */
static rg_status flatten_context_observations(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model,
    int decompose_chunks,
    int target_side,
    context_observation **out,
    size_t *out_count,
    double *out_total
) {
    context_observation *observations = 0;
    size_t observation_count = 0;
    size_t observation_cap = 0;
    double n_total = 0.0;
    int max_chunk_size = RG_DEFAULT_MAX_CHUNK_SIZE;
    rg_context_spec *target_contexts = 0;
    size_t target_context_count = 0;
    size_t target_position = 0;
    size_t i;
    rg_status status = RG_OK;

    *out = 0;
    *out_count = 0;
    *out_total = 0.0;
    if (options != 0 && options->max_chunk_size > 0) {
        max_chunk_size = options->max_chunk_size;
    }
    for (i = 0; i < pair_count && status == RG_OK; i++) {
        rg_alignment *alignment = 0;
        size_t j;
        double weight = pairs[i].weight == 0.0 ? 1.0 : pairs[i].weight;
        if (weight <= 0.0) {
            continue;
        }
        status = rg_align_forms_with_model(ctx, model, options, &pairs[i].source, &pairs[i].target, max_chunk_size, &alignment);
        if (status != RG_OK) {
            break;
        }
        /* The same alignments either way. Reversing the pair would produce
         * different ones, and the question is what the model's own alignment
         * looks like from the other side, not what a differently-trained model
         * would do. */
        if (target_side) {
            status = rg_form_position_contexts_internal(ctx, &pairs[i].target,
                                                        &target_contexts, &target_context_count);
            if (status != RG_OK) {
                rg_alignment_free(alignment);
                break;
            }
        }
        target_position = 0;
        for (j = 0; j < rg_alignment_link_count(alignment) && status == RG_OK; j++) {
            const rg_link *link = rg_alignment_link_at(alignment, j);
            size_t this_target = target_position;
            target_position += link->target_count;
            if (link->source_count == 1 && link->target_count == 1) {
                if (target_side) {
                    if (this_target < target_context_count) {
                        status = append_context_observation_as(
                            &observations, &observation_count, &observation_cap, link,
                            &target_contexts[this_target], 1, weight);
                        n_total += weight;
                    }
                } else {
                    status = append_context_observation_as(
                        &observations, &observation_count, &observation_cap, link,
                        &link->context, 0, weight);
                    n_total += weight;
                }
                continue;
            }
            if (target_side && this_target >= target_context_count) {
                continue;
            }
            if (!decompose_chunks || link->source_count == 0 || link->target_count == 0) {
                continue;
            }
            {
                rg_form sub_source;
                rg_form sub_target;
                rg_alignment *sub = 0;
                size_t k;
                memset(&sub_source, 0, sizeof(sub_source));
                memset(&sub_target, 0, sizeof(sub_target));
                sub_source.lect_id = "_sub";
                sub_source.segments = link->source;
                sub_source.segment_count = link->source_count;
                sub_target.lect_id = "_sub";
                sub_target.segments = link->target;
                sub_target.segment_count = link->target_count;
                status = rg_align_forms_with_model(ctx, model, options, &sub_source, &sub_target, 1, &sub);
                if (status != RG_OK) {
                    break;
                }
                for (k = 0; k < rg_alignment_link_count(sub) && status == RG_OK; k++) {
                    const rg_link *sub_link = rg_alignment_link_at(sub, k);
                    rg_link merged;
                    if (sub_link->source_count != 1 || sub_link->target_count != 1) {
                        continue;
                    }
                    /* The outer link's context is what conditions this pair. */
                    /* The outer link's context is what conditions this pair.
                     * On the target side that is the target form's view at the
                     * chunk's first position -- the source side uses the outer
                     * link's own span context, and dropping chunks instead
                     * would lose precisely the correspondences chunk promotion
                     * found interesting. */
                    merged = *sub_link;
                    merged.context = link->context;
                    status = append_context_observation_as(
                        &observations, &observation_count, &observation_cap, &merged,
                        target_side ? &target_contexts[this_target] : &link->context,
                        target_side, weight);
                    n_total += weight;
                }
                rg_alignment_free(sub);
            }
        }
        rg_alignment_free(alignment);
        if (target_side) {
            rg_context_spec_array_free_internal(target_contexts, target_context_count);
            target_contexts = 0;
            target_context_count = 0;
        }
    }
    if (target_side) {
        rg_context_spec_array_free_internal(target_contexts, target_context_count);
    }
    if (status != RG_OK) {
        for (i = 0; i < observation_count; i++) {
            context_observation_clear(&observations[i]);
        }
        free(observations);
        return status;
    }
    *out = observations;
    *out_count = observation_count;
    *out_total = n_total;
    return RG_OK;
}

/* Shared driver for the two context-discovery passes. */
static rg_status discover_context_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model,
    int long_range,
    int target_side,
    const rg_feature_vocabulary *vocabulary,
    context_observation *observations,
    size_t observation_count,
    double n_total
) {
    const char **sources = 0;
    size_t source_count = 0;
    size_t source_cap = 0;
    stress_inventory stress;
    morphology_inventory morphology;
    split_candidate *long_range_list = 0;
    size_t long_range_count = 0;
    rg_split_observation *rows = 0;
    size_t i;
    rg_status status;
    double min_obs;
    double delta_threshold;
    double dominant_fraction = 0.0;
    int max_depth = 3;
    split_candidate *immediate_list = 0;
    size_t immediate_count = 0;
    rg_split_gate *immediate_gates = 0;
    rg_split_gate *long_gates = 0;
    split_candidate *all_list = 0;
    rg_split_gate *all_gates = 0;
    size_t all_count = 0;
    double immediate_min_obs = 2.0;
    double immediate_delta = -1.0;
    double long_min_obs = 5.0;
    double long_delta = -5.0;
    double long_dominant = 0.6;

    if (ctx == 0 || model == 0 || (pair_count > 0 && pairs == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    memset(&stress, 0, sizeof(stress));
    memset(&morphology, 0, sizeof(morphology));
    if (options != 0) {
        max_depth = options->bic.max_split_depth > 0 ? options->bic.max_split_depth : 3;
        if (options->bic.min_split_observations > 0) {
            immediate_min_obs = (double)options->bic.min_split_observations;
        }
        immediate_delta = options->bic.delta_bic_threshold;
        if (options->bic.long_range_min_split_observations > 0) {
            long_min_obs = (double)options->bic.long_range_min_split_observations;
        }
        long_delta = options->bic.long_range_delta_bic_threshold;
        long_dominant = options->bic.long_range_min_dominant_fraction;
    }
    min_obs = long_range ? long_min_obs : immediate_min_obs;
    delta_threshold = long_range ? long_delta : immediate_delta;
    dominant_fraction = long_range ? long_dominant : 0.0;
    (void)min_obs;
    (void)delta_threshold;
    (void)dominant_fraction;

    status = RG_OK;
    if (observation_count == 0 || n_total <= 0.0) {
        return RG_OK;
    }

    for (i = 0; i < observation_count && status == RG_OK; i++) {
        status = append_unique_source(&sources, &source_count, &source_cap, observations[i].source);
        if (status == RG_OK && !long_range) {
            status = collect_observed_stress(&stress, &observations[i].context);
        }
        collect_observed_morphology(&morphology, &observations[i].context);
    }
    /* Both lists are built whatever this stage leads with: the stage decides
     * which kind of predicate opens a split, and refinement may then conjoin
     * either kind onto it. */
    if (status == RG_OK) {
        size_t immediate_cap =
            2 * vocabulary->count +
            sizeof(split_positions) / sizeof(split_positions[0]) +
            3 * stress.count + 8 + morphology.placement_count + morphology.index_count;
        size_t long_cap =
            rg_env_long_range_slot_count *
            (vocabulary->count == 0 ? 1 : vocabulary->count) +
            rg_env_long_range_slot_count *
            sizeof(syllable_shape_candidates) / sizeof(syllable_shape_candidates[0]);
        rg_context_spec empty;
        immediate_list = (split_candidate *)calloc(immediate_cap, sizeof(*immediate_list));
        if (long_range_list == 0) {
            long_range_list = (split_candidate *)calloc(long_cap, sizeof(*long_range_list));
            if (long_range_list != 0) {
                long_range_count = long_range_candidates(vocabulary, long_range_list, long_cap);
            }
        }
        if (immediate_list == 0 || long_range_list == 0) {
            status = RG_ERR_OOM;
        } else {
            rg_context_spec_init_empty(&empty);
            immediate_count = immediate_candidates_for(&empty, &stress, &morphology, vocabulary, immediate_list, immediate_cap);
            rg_context_spec_clear_internal(&empty);
        }
    }
    if (status == RG_OK) {
        size_t total = immediate_count + long_range_count;
        size_t k;
        all_list = (split_candidate *)calloc(total == 0 ? 1 : total, sizeof(*all_list));
        all_gates = (rg_split_gate *)calloc(total == 0 ? 1 : total, sizeof(*all_gates));
        immediate_gates = (rg_split_gate *)calloc(immediate_count == 0 ? 1 : immediate_count, sizeof(*immediate_gates));
        long_gates = (rg_split_gate *)calloc(long_range_count == 0 ? 1 : long_range_count, sizeof(*long_gates));
        if (all_list == 0 || all_gates == 0 || immediate_gates == 0 || long_gates == 0) {
            status = RG_ERR_OOM;
        } else {
            for (k = 0; k < immediate_count; k++) {
                immediate_gates[k].min_obs = immediate_min_obs;
                immediate_gates[k].delta_threshold = immediate_delta;
                immediate_gates[k].min_dominant_fraction = 0.0;
                all_list[k] = immediate_list[k];
                all_gates[k] = immediate_gates[k];
            }
            for (k = 0; k < long_range_count; k++) {
                long_gates[k].min_obs = long_min_obs;
                long_gates[k].delta_threshold = long_delta;
                long_gates[k].min_dominant_fraction = long_dominant;
                all_list[immediate_count + k] = long_range_list[k];
                all_gates[immediate_count + k] = long_gates[k];
            }
            all_count = total;
        }
    }
    rows = (rg_split_observation *)calloc(observation_count, sizeof(*rows));
    if (rows == 0) {
        status = RG_ERR_OOM;
    }
    if (status == RG_OK) {
        qsort(sources, source_count, sizeof(*sources), string_ptr_cmp);
    }

    for (i = 0; i < source_count && status == RG_OK; i++) {
        size_t row_count = 0;
        size_t j;
        size_t distinct_targets = 0;
        double mass;

        /* Project the stage's own rows into what the search reads: the
         * environment to test predicates against, the target to group cost by,
         * and the weight. */
        for (j = 0; j < observation_count; j++) {
            if (strcmp(observations[j].source, sources[i]) == 0) {
                rows[row_count].context = &observations[j].context;
                rows[row_count].key = observations[j].target;
                rows[row_count].weight = observations[j].weight;
                rows[row_count].owner = &observations[j];
                row_count++;
            }
        }
        for (j = 0; j < row_count; j++) {
            size_t k;
            int seen = 0;
            for (k = 0; k < j; k++) {
                if (strcmp(rows[k].key, rows[j].key) == 0) {
                    seen = 1;
                    break;
                }
            }
            if (!seen) {
                distinct_targets++;
            }
        }
        mass = rg_split_total_weight(rows, row_count);
        if (distinct_targets < 2 || mass < 4.0) {
            continue;
        }
        status = commit_splits_for_source(
            model,
            sources[i],
            rows,
            row_count,
            long_range ? long_range_list : immediate_list,
            long_range ? long_gates : immediate_gates,
            long_range ? long_range_count : immediate_count,
            all_list,
            all_gates,
            all_count,
            max_depth,
            long_range ? long_min_obs : immediate_min_obs,
            log(n_total),
            options->bic.search_penalty_gamma,
            target_side
        );
    }

    free(sources);
    free(rows);
    free(long_range_list);
    free(immediate_list);
    free(immediate_gates);
    free(long_gates);
    free(all_list);
    free(all_gates);
    stress_inventory_clear(&stress);
    if (status == RG_OK && model->conditioned_segment_count_count > 1) {
        qsort(model->conditioned_segment_counts, model->conditioned_segment_count_count,
              sizeof(*model->conditioned_segment_counts), conditioned_count_row_cmp);
    }
    return status;
}

/* Both sides, flattened before either is committed.
 *
 * The order matters and must not: committing the source-side rules first would
 * change the alignments the target-side pass then reads, so which side ran
 * first would leave a mark on the result -- and which side runs first is
 * decided by which lect the corpus happened to name first. Reading both views
 * off the same model is what makes the two passes commute. */
static rg_status discover_both_sides(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model,
    const rg_feature_vocabulary *vocabulary,
    int long_range
) {
    context_observation *observations[2] = {0, 0};
    size_t counts[2] = {0, 0};
    double totals[2] = {0.0, 0.0};
    int side;
    size_t i;
    rg_status status = RG_OK;

    for (side = 0; side < 2 && status == RG_OK; side++) {
        /* Immediate discovery looks inside promoted chunks; long-range
         * discovery does not, and that costs it evidence. Promotion runs
         * first, so by the time this stage sees the corpus the segments a
         * long-range rule would be stated over may already be inside a chunk
         * row: on place_dissimilation the rule that names the following labial
         * is left with 6 of its 18 observations, the other 12 having gone into
         * "pa ~ ta" and "pal ~ tal".
         *
         * Decomposing here recovers all 18 in both directions, and is not
         * committed because it uncovers a further asymmetry it does not cause:
         * with the extra observations in view, long-range discovery commits
         * rows in one direction that it does not commit in the other (two on
         * rhotacism, one on verner) although its inputs mirror exactly -- the
         * same source groups, the same masses, the same target counts. That is
         * its own defect and wants its own change. */
        status = flatten_context_observations(
            ctx, pairs, pair_count, options, model, long_range ? 0 : 1, side,
            &observations[side], &counts[side], &totals[side]
        );
    }
    for (side = 0; side < 2 && status == RG_OK; side++) {
        status = discover_context_counts(ctx, pairs, pair_count, options, model,
                                         long_range, side, vocabulary,
                                         observations[side], counts[side], totals[side]);
    }
    for (side = 0; side < 2; side++) {
        for (i = 0; i < counts[side]; i++) {
            context_observation_clear(&observations[side][i]);
        }
        free(observations[side]);
    }
    return status;
}

rg_status discover_immediate_context_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model,
    const rg_feature_vocabulary *vocabulary
) {
    /* Both directions. A change is only visible from the side that has the
     * split: where the daughter reflects a conditioned change, the ancestor's
     * segment answers to two daughter segments and the ancestor's environment
     * is what separates them, while from the daughter's side each segment has
     * one source and there is nothing to condition. Looking from one side only
     * left half of every pair's conditioning unreachable, and which half
     * depended on which lect happened to sort first. */
    return discover_both_sides(ctx, pairs, pair_count, options, model, vocabulary, 0);
}

/* Long-range discovery runs after cross-dimensional discovery and adds more
 * specific overlays alongside the immediate-neighbour entries. */
rg_status discover_long_range_context_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model,
    const rg_feature_vocabulary *vocabulary
) {
    return discover_both_sides(ctx, pairs, pair_count, options, model, vocabulary, 1);
}


