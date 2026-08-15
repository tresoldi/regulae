#include "internal.h"

#include <stdint.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct class_bucket {
    size_t origin;
    char **lect_ids;
    char **graphemes;
    size_t segment_count;
    double count;
    char *participant_key;
    char **supporting_cognates;
    size_t supporting_cognate_count;
    size_t supporting_cognate_cap;
} class_bucket;

static void string_array_clear(char **items, size_t count) {
    size_t i;
    if (items == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        free(items[i]);
    }
    free(items);
}

/* One reconciled class observation: the participating lects (ascending by lect
 * id), their graphemes, and the position each occupies in its own form. */
typedef struct reconciled_observation {
    size_t cognate_index;
    char **lects;
    char **graphemes;
    size_t *positions;
    size_t *lect_indices;
    size_t segment_count;
    double weight;
    /* Which unconditioned bucket this landed in, before the buckets are
     * sorted into class order. */
    size_t bucket_index;
} reconciled_observation;

static void reconciled_observation_clear(reconciled_observation *obs) {
    if (obs == 0) {
        return;
    }
    string_array_clear(obs->lects, obs->segment_count);
    string_array_clear(obs->graphemes, obs->segment_count);
    free(obs->positions);
    free(obs->lect_indices);
    memset(obs, 0, sizeof(*obs));
}

static rg_status class_position_add_id(rg_class_position *position, int class_id) {
    size_t i;
    for (i = 0; i < position->class_id_count; i++) {
        if (position->class_ids[i] == class_id) {
            return RG_OK;
        }
    }
    if (position->class_id_count == position->class_id_cap) {
        size_t next_cap = position->class_id_cap == 0 ? 4 : position->class_id_cap * 2;
        int *next = (int *)realloc(position->class_ids, next_cap * sizeof(*next));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        position->class_ids = next;
        position->class_id_cap = next_cap;
    }
    position->class_ids[position->class_id_count++] = class_id;
    return RG_OK;
}

/* Moves the reconciled observations onto the model as the class-position
 * table, so a consumer can ask which classes a given aligned position
 * realises. Only the positions and their class ids are kept; the graphemes are
 * already on the classes. */
static rg_status publish_class_positions(
    rg_multi_model *model,
    reconciled_observation *observations,
    size_t observation_count
) {
    size_t i;
    model->class_positions = (rg_class_position *)calloc(
        observation_count == 0 ? 1 : observation_count, sizeof(*model->class_positions));
    if (model->class_positions == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < observation_count; i++) {
        rg_class_position *out = &model->class_positions[i];
        size_t n = observations[i].segment_count;
        out->cognate_index = observations[i].cognate_index;
        out->segment_count = n;
        out->lect_indices = (size_t *)calloc(n == 0 ? 1 : n, sizeof(*out->lect_indices));
        out->positions = (size_t *)calloc(n == 0 ? 1 : n, sizeof(*out->positions));
        if (out->lect_indices == 0 || out->positions == 0) {
            return RG_ERR_OOM;
        }
        memcpy(out->lect_indices, observations[i].lect_indices, n * sizeof(*out->lect_indices));
        memcpy(out->positions, observations[i].positions, n * sizeof(*out->positions));
        if (observations[i].bucket_index < model->unconditioned_class_count) {
            rg_status status = class_position_add_id(out, (int)observations[i].bucket_index);
            if (status != RG_OK) {
                return status;
            }
        }
        model->class_position_count = i + 1;
    }
    return RG_OK;
}

size_t rg_model_classes_at_internal(
    const rg_multi_model *model,
    size_t cognate_index,
    size_t lect_a,
    size_t position_a,
    size_t lect_b,
    size_t position_b,
    int *out,
    size_t capacity
) {
    size_t i;
    size_t written = 0;
    if (model == 0 || out == 0) {
        return 0;
    }
    for (i = 0; i < model->class_position_count; i++) {
        const rg_class_position *entry = &model->class_positions[i];
        int has_a = 0;
        int has_b = 0;
        size_t j;
        size_t k;
        if (entry->cognate_index != cognate_index) {
            continue;
        }
        for (j = 0; j < entry->segment_count; j++) {
            if (entry->lect_indices[j] == lect_a && entry->positions[j] == position_a) {
                has_a = 1;
            }
            if (entry->lect_indices[j] == lect_b && entry->positions[j] == position_b) {
                has_b = 1;
            }
        }
        if (!has_a || !has_b) {
            continue;
        }
        for (k = 0; k < entry->class_id_count && written < capacity; k++) {
            size_t seen;
            int duplicate = 0;
            for (seen = 0; seen < written; seen++) {
                if (out[seen] == entry->class_ids[k]) {
                    duplicate = 1;
                    break;
                }
            }
            if (!duplicate) {
                out[written++] = entry->class_ids[k];
            }
        }
    }
    return written;
}

static void reconciled_observations_free(reconciled_observation *items, size_t count) {
    size_t i;
    if (items == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        reconciled_observation_clear(&items[i]);
    }
    free(items);
}

typedef struct uf_state {
    size_t *parent;
    size_t *rank;
    size_t count;
} uf_state;

typedef struct outlier_work_row {
    char *cognate_id;
    int pair_count;
    double cost_per_segment;
    double z_score;
} outlier_work_row;

static void class_bucket_clear(class_bucket *bucket) {
    if (bucket == 0) {
        return;
    }
    string_array_clear(bucket->lect_ids, bucket->segment_count);
    string_array_clear(bucket->graphemes, bucket->segment_count);
    free(bucket->participant_key);
    string_array_clear(bucket->supporting_cognates, bucket->supporting_cognate_count);
    memset(bucket, 0, sizeof(*bucket));
}

static void multi_class_clear(rg_multi_class_owned *klass) {
    size_t i;
    if (klass == 0) {
        return;
    }
    string_array_clear((char **)klass->view.lect_ids, klass->view.segment_count);
    string_array_clear((char **)klass->view.graphemes, klass->view.segment_count);
    if (klass->view.contexts != 0) {
        for (i = 0; i < klass->view.segment_count; i++) {
            rg_context_spec_clear_internal((rg_context_spec *)&klass->view.contexts[i]);
        }
        free((rg_context_spec *)klass->view.contexts);
    }
    string_array_clear((char **)klass->view.supporting_cognates, klass->view.supporting_cognate_count);
    memset(klass, 0, sizeof(*klass));
}

void rg_multi_model_free(rg_multi_model *model) {
    size_t i;
    if (model == 0) {
        return;
    }
    string_array_clear(model->lect_ids, model->lect_count);
    for (i = 0; i < model->pair_model_count; i++) {
        free(model->pair_models[i].lect_a);
        free(model->pair_models[i].lect_b);
        rg_pairwise_model_free(model->pair_models[i].model);
    }
    free(model->pair_models);
    for (i = 0; i < model->unconditioned_class_count; i++) {
        multi_class_clear(&model->unconditioned_classes[i]);
    }
    free(model->unconditioned_classes);
    for (i = 0; i < model->conditioned_class_count; i++) {
        multi_class_clear(&model->conditioned_classes[i]);
    }
    free(model->conditioned_classes);
    free(model->cross_dimensional_rows);
    for (i = 0; i < model->class_position_count; i++) {
        free(model->class_positions[i].lect_indices);
        free(model->class_positions[i].positions);
        free(model->class_positions[i].class_ids);
    }
    free(model->class_positions);
    free(model);
}

static int lect_index(const char *const *lects, size_t count, const char *lect_id) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (strcmp(lects[i], lect_id) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static rg_status append_lect(char ***lects, size_t *count, size_t *cap, const char *lect_id) {
    char **next;
    if (lect_id == 0 || lect_id[0] == '\0') {
        return RG_ERR_INVALID_ARGUMENT;
    }
    if (lect_index((const char *const *)*lects, *count, lect_id) >= 0) {
        return RG_OK;
    }
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 4 : *cap * 2;
        next = (char **)realloc(*lects, next_cap * sizeof(**lects));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *lects = next;
        *cap = next_cap;
    }
    (*lects)[*count] = rg_strdup_internal(lect_id);
    if ((*lects)[*count] == 0) {
        return RG_ERR_OOM;
    }
    (*count)++;
    return RG_OK;
}

static int outlier_cmp(const void *a, const void *b) {
    const outlier_work_row *ra = (const outlier_work_row *)a;
    const outlier_work_row *rb = (const outlier_work_row *)b;
    if (ra->z_score != rb->z_score) {
        return ra->z_score > rb->z_score ? -1 : 1;
    }
    return strcmp(ra->cognate_id, rb->cognate_id);
}

static double cognate_weight(const rg_cognate_set *cognate) {
    return cognate->confidence;
}

static const rg_form *form_for_lect(const rg_cognate_set *cognate, const char *lect_id) {
    size_t i;
    for (i = 0; i < cognate->form_count; i++) {
        if (strcmp(cognate->forms[i].lect_id, lect_id) == 0) {
            return &cognate->forms[i].form;
        }
    }
    return 0;
}

static rg_status train_pair_models(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    rg_progress_state *progress,
    rg_multi_model *model
) {
    size_t i;
    size_t j;
    size_t pair_cap = 0;
    for (i = 0; i < model->lect_count; i++) {
        for (j = i + 1; j < model->lect_count; j++) {
            rg_form_pair *pairs = 0;
            size_t pair_count = 0;
            size_t cap = 0;
            size_t c;
            rg_pairwise_model *pair_model = 0;
            rg_status status;
            for (c = 0; c < cognate_count; c++) {
                const rg_form *source = form_for_lect(&cognates[c], model->lect_ids[i]);
                const rg_form *target = form_for_lect(&cognates[c], model->lect_ids[j]);
                rg_form_pair *next;
                /* A zero-confidence cognate carries zero weight through every
                 * training stage, so excluding it here is equivalent to
                 * including it with weight zero and cheaper. */
                if (source == 0 || target == 0 || cognate_weight(&cognates[c]) <= 0.0) {
                    continue;
                }
                if (pair_count == cap) {
                    size_t next_cap = cap == 0 ? 16 : cap * 2;
                    next = (rg_form_pair *)realloc(pairs, next_cap * sizeof(*pairs));
                    if (next == 0) {
                        free(pairs);
                        return RG_ERR_OOM;
                    }
                    pairs = next;
                    cap = next_cap;
                }
                pairs[pair_count].source = *source;
                pairs[pair_count].target = *target;
                pairs[pair_count].weight = cognate_weight(&cognates[c]);
                pair_count++;
            }
            if (pair_count == 0) {
                free(pairs);
                continue;
            }
            status = rg_train_pairwise_internal(ctx, pairs, pair_count, options, progress, &pair_model);
            free(pairs);
            if (status != RG_OK) {
                return status;
            }
            if (model->pair_model_count == pair_cap) {
                size_t next_cap = pair_cap == 0 ? 4 : pair_cap * 2;
                rg_multi_pair_model_owned *next = (rg_multi_pair_model_owned *)realloc(model->pair_models, next_cap * sizeof(*model->pair_models));
                if (next == 0) {
                    rg_pairwise_model_free(pair_model);
                    return RG_ERR_OOM;
                }
                model->pair_models = next;
                pair_cap = next_cap;
            }
            memset(&model->pair_models[model->pair_model_count], 0, sizeof(model->pair_models[model->pair_model_count]));
            model->pair_models[model->pair_model_count].lect_a = rg_strdup_internal(model->lect_ids[i]);
            model->pair_models[model->pair_model_count].lect_b = rg_strdup_internal(model->lect_ids[j]);
            if (model->pair_models[model->pair_model_count].lect_a == 0 ||
                model->pair_models[model->pair_model_count].lect_b == 0) {
                rg_pairwise_model_free(pair_model);
                return RG_ERR_OOM;
            }
            model->pair_models[model->pair_model_count].model = pair_model;
            model->pair_models[model->pair_model_count].view.lect_a = model->pair_models[model->pair_model_count].lect_a;
            model->pair_models[model->pair_model_count].view.lect_b = model->pair_models[model->pair_model_count].lect_b;
            model->pair_models[model->pair_model_count].view.model = pair_model;
            model->pair_model_count++;
        }
    }
    return RG_OK;
}

static rg_status lift_cross_dimensional_rows(rg_multi_model *model) {
    size_t i;
    size_t cap = 0;
    for (i = 0; i < model->pair_model_count; i++) {
        size_t j;
        size_t row_count = rg_pairwise_model_cross_dimensional_row_count(model->pair_models[i].model);
        for (j = 0; j < row_count; j++) {
            const rg_cross_dimensional_row *row = rg_pairwise_model_cross_dimensional_row_at(model->pair_models[i].model, j);
            rg_multi_cross_dimensional_owned *next;
            if (row == 0) {
                continue;
            }
            if (model->cross_dimensional_count == cap) {
                size_t next_cap = cap == 0 ? 8 : cap * 2;
                next = (rg_multi_cross_dimensional_owned *)realloc(model->cross_dimensional_rows, next_cap * sizeof(*model->cross_dimensional_rows));
                if (next == 0) {
                    return RG_ERR_OOM;
                }
                model->cross_dimensional_rows = next;
                cap = next_cap;
            }
            memset(&model->cross_dimensional_rows[model->cross_dimensional_count], 0, sizeof(model->cross_dimensional_rows[model->cross_dimensional_count]));
            model->cross_dimensional_rows[model->cross_dimensional_count].view.source_lect = model->pair_models[i].lect_a;
            model->cross_dimensional_rows[model->cross_dimensional_count].view.target_lect = model->pair_models[i].lect_b;
            /* Borrowed from the pairwise row, which outlives the lifted view. */
            model->cross_dimensional_rows[model->cross_dimensional_count].view.source_environment = row->source_environment;
            model->cross_dimensional_rows[model->cross_dimensional_count].view.target_dimension = row->target_dimension;
            model->cross_dimensional_rows[model->cross_dimensional_count].view.target_value = row->target_value;
            model->cross_dimensional_rows[model->cross_dimensional_count].view.target_position_offset = row->target_position_offset;
            model->cross_dimensional_rows[model->cross_dimensional_count].view.count = row->count;
            model->cross_dimensional_rows[model->cross_dimensional_count].view.source_count = row->source_count;
            model->cross_dimensional_rows[model->cross_dimensional_count].view.confidence = row->confidence;
            model->cross_dimensional_rows[model->cross_dimensional_count].view.contrast_count = row->contrast_count;
            model->cross_dimensional_rows[model->cross_dimensional_count].view.contrast_source_count = row->contrast_source_count;
            model->cross_dimensional_rows[model->cross_dimensional_count].view.contrast_confidence = row->contrast_confidence;
            model->cross_dimensional_rows[model->cross_dimensional_count].view.delta_bic = row->delta_bic;
            model->cross_dimensional_rows[model->cross_dimensional_count].view.decision_index = row->decision_index;
            model->cross_dimensional_rows[model->cross_dimensional_count].view.search_margin = row->search_margin;
            model->cross_dimensional_rows[model->cross_dimensional_count].view.uncertainty = row->uncertainty;
            model->cross_dimensional_count++;
        }
    }
    return RG_OK;
}

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
static rg_status multi_lect_context_discovery(
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

static const rg_pairwise_model *pair_model_for(const rg_multi_model *model, const char *lect_a, const char *lect_b) {
    size_t i;
    for (i = 0; i < model->pair_model_count; i++) {
        if (strcmp(model->pair_models[i].lect_a, lect_a) == 0 && strcmp(model->pair_models[i].lect_b, lect_b) == 0) {
            return model->pair_models[i].model;
        }
        if (strcmp(model->pair_models[i].lect_a, lect_b) == 0 && strcmp(model->pair_models[i].lect_b, lect_a) == 0) {
            return model->pair_models[i].model;
        }
    }
    return 0;
}

static rg_status uf_init(uf_state *uf, size_t count) {
    size_t i;
    memset(uf, 0, sizeof(*uf));
    uf->parent = (size_t *)calloc(count, sizeof(*uf->parent));
    uf->rank = (size_t *)calloc(count, sizeof(*uf->rank));
    if ((uf->parent == 0 || uf->rank == 0) && count > 0) {
        free(uf->parent);
        free(uf->rank);
        memset(uf, 0, sizeof(*uf));
        return RG_ERR_OOM;
    }
    uf->count = count;
    for (i = 0; i < count; i++) {
        uf->parent[i] = i;
    }
    return RG_OK;
}

static void uf_clear(uf_state *uf) {
    free(uf->parent);
    free(uf->rank);
    memset(uf, 0, sizeof(*uf));
}

static size_t uf_find(uf_state *uf, size_t x) {
    size_t root = x;
    while (uf->parent[root] != root) {
        root = uf->parent[root];
    }
    while (uf->parent[x] != x) {
        size_t next = uf->parent[x];
        uf->parent[x] = root;
        x = next;
    }
    return root;
}

static void uf_union(uf_state *uf, size_t a, size_t b) {
    size_t ra = uf_find(uf, a);
    size_t rb = uf_find(uf, b);
    if (ra == rb) {
        return;
    }
    if (uf->rank[ra] < uf->rank[rb]) {
        uf->parent[ra] = rb;
    } else if (uf->rank[ra] > uf->rank[rb]) {
        uf->parent[rb] = ra;
    } else {
        uf->parent[rb] = ra;
        uf->rank[ra]++;
    }
}

static rg_status participant_key_from_lects(char **lects, size_t count, char **out) {
    size_t i;
    size_t len = 1;
    char *key;
    char *p;
    *out = 0;
    for (i = 0; i < count; i++) {
        len += strlen(lects[i]) + 1;
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
    }
    *p = '\0';
    *out = key;
    return RG_OK;
}

static rg_status bucket_append_support(class_bucket *bucket, const char *cognate_id) {
    char **next;
    const char *id = cognate_id == 0 ? "" : cognate_id;
    if (bucket->supporting_cognate_count == bucket->supporting_cognate_cap) {
        size_t next_cap = bucket->supporting_cognate_cap == 0 ? 4 : bucket->supporting_cognate_cap * 2;
        next = (char **)realloc(bucket->supporting_cognates, next_cap * sizeof(*bucket->supporting_cognates));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        bucket->supporting_cognates = next;
        bucket->supporting_cognate_cap = next_cap;
    }
    bucket->supporting_cognates[bucket->supporting_cognate_count] = rg_strdup_internal(id);
    if (bucket->supporting_cognates[bucket->supporting_cognate_count] == 0) {
        return RG_ERR_OOM;
    }
    bucket->supporting_cognate_count++;
    return RG_OK;
}

static int bucket_equal(const class_bucket *bucket, char **lects, char **graphemes, size_t count) {
    size_t i;
    if (bucket->segment_count != count) {
        return 0;
    }
    for (i = 0; i < count; i++) {
        if (strcmp(bucket->lect_ids[i], lects[i]) != 0 || strcmp(bucket->graphemes[i], graphemes[i]) != 0) {
            return 0;
        }
    }
    return 1;
}

static rg_status add_bucket_observation(
    class_bucket **buckets,
    size_t *bucket_count,
    size_t *bucket_cap,
    char **lects,
    char **graphemes,
    size_t count,
    double weight,
    const char *cognate_id,
    size_t *out_bucket_index
) {
    size_t i;
    for (i = 0; i < *bucket_count; i++) {
        if (bucket_equal(&(*buckets)[i], lects, graphemes, count)) {
            (*buckets)[i].count += weight;
            if (bucket_append_support(&(*buckets)[i], cognate_id) != RG_OK) {
                return RG_ERR_OOM;
            }
            string_array_clear(lects, count);
            string_array_clear(graphemes, count);
            *out_bucket_index = i;
            return RG_OK;
        }
    }
    *out_bucket_index = *bucket_count;
    if (*bucket_count == *bucket_cap) {
        size_t next_cap = *bucket_cap == 0 ? 16 : *bucket_cap * 2;
        class_bucket *next = (class_bucket *)realloc(*buckets, next_cap * sizeof(**buckets));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *buckets = next;
        *bucket_cap = next_cap;
    }
    memset(&(*buckets)[*bucket_count], 0, sizeof((*buckets)[*bucket_count]));
    (*buckets)[*bucket_count].lect_ids = lects;
    (*buckets)[*bucket_count].graphemes = graphemes;
    (*buckets)[*bucket_count].segment_count = count;
    (*buckets)[*bucket_count].count = weight;
    if (participant_key_from_lects(lects, count, &(*buckets)[*bucket_count].participant_key) != RG_OK) {
        memset(&(*buckets)[*bucket_count], 0, sizeof((*buckets)[*bucket_count]));
        return RG_ERR_OOM;
    }
    if (bucket_append_support(&(*buckets)[*bucket_count], cognate_id) != RG_OK) {
        class_bucket_clear(&(*buckets)[*bucket_count]);
        return RG_ERR_OOM;
    }
    (*bucket_count)++;
    return RG_OK;
}

static int bucket_cmp(const void *a, const void *b) {
    const class_bucket *ba = (const class_bucket *)a;
    const class_bucket *bb = (const class_bucket *)b;
    size_t i;
    if (ba->count != bb->count) {
        return ba->count > bb->count ? -1 : 1;
    }
    for (i = 0; i < ba->segment_count && i < bb->segment_count; i++) {
        int c = strcmp(ba->lect_ids[i], bb->lect_ids[i]);
        if (c != 0) {
            return c;
        }
        c = strcmp(ba->graphemes[i], bb->graphemes[i]);
        if (c != 0) {
            return c;
        }
    }
    if (ba->segment_count != bb->segment_count) {
        return ba->segment_count < bb->segment_count ? -1 : 1;
    }
    return 0;
}

/* Lect indices of the lects present in this cognate set, ascending by lect id.
 * Reconciliation walks pairs and union-find components in this order, which is
 * what makes component enumeration and supporting-cognate order deterministic
 * regardless of the corpus's first-seen lect order. */
static rg_status present_lects_sorted(
    const rg_cognate_set *cognate,
    const rg_multi_model *model,
    const rg_form **forms,
    size_t **out_order,
    size_t *out_count
) {
    size_t *order;
    size_t count = 0;
    size_t i;
    order = (size_t *)calloc(model->lect_count == 0 ? 1 : model->lect_count, sizeof(*order));
    if (order == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < model->lect_count; i++) {
        size_t insert_at;
        forms[i] = form_for_lect(cognate, model->lect_ids[i]);
        if (forms[i] == 0) {
            continue;
        }
        insert_at = count;
        while (insert_at > 0 && strcmp(model->lect_ids[order[insert_at - 1]], model->lect_ids[i]) > 0) {
            insert_at--;
        }
        if (insert_at < count) {
            memmove(&order[insert_at + 1], &order[insert_at], (count - insert_at) * sizeof(*order));
        }
        order[insert_at] = i;
        count++;
    }
    *out_order = order;
    *out_count = count;
    return RG_OK;
}

/* Position-level union-find edges induced by aligning one pair. Equal-length
 * chunks pair position by position; unequal-length non-gap chunks are
 * decomposed through a one-segment sub-alignment; pure gaps induce no edges. */
static rg_status union_pair_alignment_edges(
    const rg_context *ctx,
    const rg_pairwise_model *pair_model,
    const rg_train_options *options,
    const rg_form *source,
    const rg_form *target,
    size_t source_offset,
    size_t target_offset,
    uf_state *uf
) {
    rg_alignment *alignment = 0;
    size_t link_i;
    size_t source_pos = 0;
    size_t target_pos = 0;
    rg_status status;

    status = rg_align_forms_with_model(ctx, pair_model, options, source, target, 0, &alignment);
    if (status != RG_OK) {
        return status;
    }
    for (link_i = 0; link_i < rg_alignment_link_count(alignment); link_i++) {
        const rg_link *link = rg_alignment_link_at(alignment, link_i);
        size_t k;
        if (link->source_count == link->target_count) {
            /* Reconciliation binds the positions that answer to each other,
             * which for a reordering is not the diagonal. Binding `sk` to `ks`
             * position by position makes /s/ and /k/ members of each other's
             * class in both directions -- two false correspondences standing
             * for one reordering. */
            size_t pairing[RG_MAX_REORDER_SPAN];
            int reordering = rg_link_is_reordering_internal(link->source, link->source_count,
                                                            link->target, link->target_count, pairing);
            for (k = 0; k < link->source_count; k++) {
                size_t partner = reordering ? pairing[k] : k;
                uf_union(uf, source_offset + source_pos + k, target_offset + target_pos + partner);
            }
        } else if (link->source_count != 0 && link->target_count != 0) {
            size_t sub_source_pos = 0;
            size_t sub_target_pos = 0;
            rg_form sub_source;
            rg_form sub_target;
            rg_alignment *sub = 0;
            memset(&sub_source, 0, sizeof(sub_source));
            memset(&sub_target, 0, sizeof(sub_target));
            sub_source.lect_id = source->lect_id;
            sub_source.segments = link->source;
            sub_source.segment_count = link->source_count;
            sub_target.lect_id = target->lect_id;
            sub_target.segments = link->target;
            sub_target.segment_count = link->target_count;
            /* A one-segment chunk limit makes the pairwise chunk table
             * unreachable, matching the Go sub-alignment's emptied table. */
            status = rg_align_forms_with_model(ctx, pair_model, options, &sub_source, &sub_target, 1, &sub);
            if (status != RG_OK) {
                rg_alignment_free(alignment);
                return status;
            }
            for (k = 0; k < rg_alignment_link_count(sub); k++) {
                const rg_link *sub_link = rg_alignment_link_at(sub, k);
                if (sub_link->source_count == 1 && sub_link->target_count == 1) {
                    uf_union(uf, source_offset + source_pos + sub_source_pos, target_offset + target_pos + sub_target_pos);
                }
                sub_source_pos += sub_link->source_count;
                sub_target_pos += sub_link->target_count;
            }
            rg_alignment_free(sub);
        }
        source_pos += link->source_count;
        target_pos += link->target_count;
    }
    rg_alignment_free(alignment);
    return RG_OK;
}

static rg_status append_reconciled_observation(
    reconciled_observation **items,
    size_t *count,
    size_t *cap,
    const reconciled_observation *value
) {
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 32 : *cap * 2;
        reconciled_observation *next = (reconciled_observation *)realloc(*items, next_cap * sizeof(**items));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *items = next;
        *cap = next_cap;
    }
    (*items)[*count] = *value;
    (*count)++;
    return RG_OK;
}

static rg_status aggregate_position_classes(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    rg_multi_model *model,
    reconciled_observation **out_observations,
    size_t *out_observation_count
) {
    class_bucket *buckets = 0;
    size_t bucket_count = 0;
    size_t bucket_cap = 0;
    reconciled_observation *observations = 0;
    size_t observation_count = 0;
    size_t observation_cap = 0;
    size_t c;

    *out_observations = 0;
    *out_observation_count = 0;

    for (c = 0; c < cognate_count; c++) {
        const rg_form **forms;
        size_t *order = 0;
        size_t order_count = 0;
        size_t *offsets;
        size_t *node_lects;
        size_t *node_positions;
        size_t *node_order;
        int *seen_roots;
        uf_state uf;
        size_t node_count = 0;
        double weight = cognate_weight(&cognates[c]);
        size_t i;
        rg_status status = RG_OK;
        if (weight <= 0.0) {
            continue;
        }
        forms = (const rg_form **)calloc(model->lect_count, sizeof(*forms));
        offsets = (size_t *)calloc(model->lect_count, sizeof(*offsets));
        if (forms == 0 || offsets == 0) {
            free(forms);
            free(offsets);
            status = RG_ERR_OOM;
            goto cognate_failed;
        }
        status = present_lects_sorted(&cognates[c], model, forms, &order, &order_count);
        if (status != RG_OK) {
            free(forms);
            free(offsets);
            goto cognate_failed;
        }
        for (i = 0; i < model->lect_count; i++) {
            offsets[i] = (size_t)-1;
        }
        /* Node ids are laid out in ascending lect-id order so that walking them
         * in id order also walks them in (lect, position) order. */
        for (i = 0; i < order_count; i++) {
            offsets[order[i]] = node_count;
            node_count += forms[order[i]]->segment_count;
        }
        node_lects = (size_t *)calloc(node_count == 0 ? 1 : node_count, sizeof(*node_lects));
        node_positions = (size_t *)calloc(node_count == 0 ? 1 : node_count, sizeof(*node_positions));
        node_order = (size_t *)calloc(node_count == 0 ? 1 : node_count, sizeof(*node_order));
        seen_roots = (int *)calloc(node_count == 0 ? 1 : node_count, sizeof(*seen_roots));
        if (node_lects == 0 || node_positions == 0 || node_order == 0 || seen_roots == 0) {
            free(forms);
            free(offsets);
            free(order);
            free(node_lects);
            free(node_positions);
            free(node_order);
            free(seen_roots);
            status = RG_ERR_OOM;
            goto cognate_failed;
        }
        for (i = 0; i < order_count; i++) {
            size_t pos;
            for (pos = 0; pos < forms[order[i]]->segment_count; pos++) {
                node_lects[offsets[order[i]] + pos] = order[i];
                node_positions[offsets[order[i]] + pos] = pos;
                node_order[offsets[order[i]] + pos] = i;
            }
        }
        status = uf_init(&uf, node_count);
        if (status != RG_OK) {
            free(forms);
            free(offsets);
            free(order);
            free(node_lects);
            free(node_positions);
            free(node_order);
            free(seen_roots);
            goto cognate_failed;
        }
        for (i = 0; i < order_count && status == RG_OK; i++) {
            size_t j;
            for (j = i + 1; j < order_count && status == RG_OK; j++) {
                size_t a = order[i];
                size_t b = order[j];
                const rg_pairwise_model *pair_model = pair_model_for(model, model->lect_ids[a], model->lect_ids[b]);
                if (pair_model == 0) {
                    continue;
                }
                status = union_pair_alignment_edges(
                    ctx,
                    pair_model,
                    options,
                    forms[a],
                    forms[b],
                    offsets[a],
                    offsets[b],
                    &uf
                );
            }
        }
        for (i = 0; i < node_count && status == RG_OK; i++) {
            size_t root = uf_find(&uf, i);
            reconciled_observation obs;
            size_t item_count = 0;
            size_t n;
            int inconsistent = 0;
            if (seen_roots[root]) {
                continue;
            }
            seen_roots[root] = 1;
            memset(&obs, 0, sizeof(obs));
            obs.lects = (char **)calloc(order_count, sizeof(*obs.lects));
            obs.graphemes = (char **)calloc(order_count, sizeof(*obs.graphemes));
            obs.positions = (size_t *)calloc(order_count, sizeof(*obs.positions));
            obs.lect_indices = (size_t *)calloc(order_count, sizeof(*obs.lect_indices));
            if (obs.lects == 0 || obs.graphemes == 0 || obs.positions == 0 || obs.lect_indices == 0) {
                obs.segment_count = 0;
                reconciled_observation_clear(&obs);
                status = RG_ERR_OOM;
                break;
            }
            /* Nodes are visited in ascending (lect, position) order, so members
             * accumulate already sorted by lect id. */
            for (n = 0; n < node_count; n++) {
                size_t lect_index;
                if (uf_find(&uf, n) != root) {
                    continue;
                }
                lect_index = node_lects[n];
                if (item_count > 0 && obs.lect_indices[item_count - 1] == lect_index) {
                    inconsistent = 1;
                    break;
                }
                obs.lects[item_count] = rg_strdup_internal(model->lect_ids[lect_index]);
                obs.graphemes[item_count] = rg_strdup_internal(forms[lect_index]->segments[node_positions[n]].grapheme);
                if (obs.lects[item_count] == 0 || obs.graphemes[item_count] == 0) {
                    obs.segment_count = item_count + 1;
                    reconciled_observation_clear(&obs);
                    status = RG_ERR_OOM;
                    break;
                }
                obs.positions[item_count] = node_positions[n];
                obs.lect_indices[item_count] = lect_index;
                item_count++;
                (void)node_order;
            }
            if (status != RG_OK) {
                break;
            }
            if (inconsistent || item_count < 2) {
                obs.segment_count = item_count;
                reconciled_observation_clear(&obs);
                continue;
            }
            obs.segment_count = item_count;
            obs.cognate_index = c;
            obs.weight = weight;
            {
                char **lects_copy = (char **)calloc(item_count, sizeof(*lects_copy));
                char **graphemes_copy = (char **)calloc(item_count, sizeof(*graphemes_copy));
                size_t k;
                if (lects_copy == 0 || graphemes_copy == 0) {
                    free(lects_copy);
                    free(graphemes_copy);
                    reconciled_observation_clear(&obs);
                    status = RG_ERR_OOM;
                    break;
                }
                for (k = 0; k < item_count; k++) {
                    lects_copy[k] = rg_strdup_internal(obs.lects[k]);
                    graphemes_copy[k] = rg_strdup_internal(obs.graphemes[k]);
                    if (lects_copy[k] == 0 || graphemes_copy[k] == 0) {
                        string_array_clear(lects_copy, k + 1);
                        string_array_clear(graphemes_copy, k + 1);
                        lects_copy = 0;
                        graphemes_copy = 0;
                        break;
                    }
                }
                if (lects_copy == 0) {
                    reconciled_observation_clear(&obs);
                    status = RG_ERR_OOM;
                    break;
                }
                status = add_bucket_observation(
                    &buckets,
                    &bucket_count,
                    &bucket_cap,
                    lects_copy,
                    graphemes_copy,
                    item_count,
                    weight,
                    cognates[c].cognate_id,
                    &obs.bucket_index
                );
                if (status != RG_OK) {
                    reconciled_observation_clear(&obs);
                    break;
                }
            }
            status = append_reconciled_observation(&observations, &observation_count, &observation_cap, &obs);
            if (status != RG_OK) {
                reconciled_observation_clear(&obs);
                break;
            }
        }
        uf_clear(&uf);
        free(forms);
        free(offsets);
        free(order);
        free(node_lects);
        free(node_positions);
        free(node_order);
        free(seen_roots);
    cognate_failed:
        if (status != RG_OK) {
            size_t n;
            for (n = 0; n < bucket_count; n++) {
                class_bucket_clear(&buckets[n]);
            }
            free(buckets);
            reconciled_observations_free(observations, observation_count);
            return status;
        }
    }
    /* Sorting reorders the buckets, so remember where each one went before
     * the observations' bucket indices become meaningless. */
    {
        size_t *bucket_to_class = (size_t *)calloc(bucket_count == 0 ? 1 : bucket_count, sizeof(*bucket_to_class));
        size_t i;
        if (bucket_to_class == 0) {
            for (c = 0; c < bucket_count; c++) {
                class_bucket_clear(&buckets[c]);
            }
            free(buckets);
            reconciled_observations_free(observations, observation_count);
            return RG_ERR_OOM;
        }
        for (i = 0; i < bucket_count; i++) {
            buckets[i].origin = i;
        }
        if (bucket_count > 1) {
            qsort(buckets, bucket_count, sizeof(*buckets), bucket_cmp);
        }
        for (i = 0; i < bucket_count; i++) {
            bucket_to_class[buckets[i].origin] = i;
        }
        for (i = 0; i < observation_count; i++) {
            if (observations[i].bucket_index < bucket_count) {
                observations[i].bucket_index = bucket_to_class[observations[i].bucket_index];
            }
        }
        free(bucket_to_class);
    }
    model->unconditioned_classes = (rg_multi_class_owned *)calloc(bucket_count == 0 ? 1 : bucket_count, sizeof(*model->unconditioned_classes));
    if (model->unconditioned_classes == 0) {
        for (c = 0; c < bucket_count; c++) {
            class_bucket_clear(&buckets[c]);
        }
        free(buckets);
        reconciled_observations_free(observations, observation_count);
        return RG_ERR_OOM;
    }
    for (c = 0; c < bucket_count; c++) {
        double participant_total = 0.0;
        size_t i;
        for (i = 0; i < bucket_count; i++) {
            if (strcmp(buckets[i].participant_key, buckets[c].participant_key) == 0) {
                participant_total += buckets[i].count;
            }
        }
        model->unconditioned_classes[c].view.class_id = (int)c;
        model->unconditioned_classes[c].view.lect_ids = (const char *const *)buckets[c].lect_ids;
        model->unconditioned_classes[c].view.graphemes = (const char *const *)buckets[c].graphemes;
        model->unconditioned_classes[c].view.segment_count = buckets[c].segment_count;
        model->unconditioned_classes[c].view.count = buckets[c].count;
        model->unconditioned_classes[c].view.confidence = 1.0;
        /* An unconditioned class is aggregated, not decided. */
        model->unconditioned_classes[c].view.decision_index = -1;
        model->unconditioned_classes[c].view.supporting_cognates = (const char *const *)buckets[c].supporting_cognates;
        model->unconditioned_classes[c].view.supporting_cognate_count = buckets[c].supporting_cognate_count;
        model->unconditioned_classes[c].view.uncertainty = rg_wilson_default_internal(buckets[c].count, participant_total);
        buckets[c].lect_ids = 0;
        buckets[c].graphemes = 0;
        buckets[c].supporting_cognates = 0;
        buckets[c].supporting_cognate_count = 0;
    }
    model->unconditioned_class_count = bucket_count;
    for (c = 0; c < bucket_count; c++) {
        class_bucket_clear(&buckets[c]);
    }
    free(buckets);
    *out_observations = observations;
    *out_observation_count = observation_count;
    return RG_OK;
}

static rg_status score_cognate_set(
    const rg_context *ctx,
    const rg_multi_model *model,
    const rg_train_options *options,
    const rg_cognate_set *set,
    int max_chunk_size,
    double *out_cost,
    int *out_pair_count
);

/* xorshift64*, seeded from the options. The baseline has to be reproducible:
 * a fit statistic a user cannot recompute is not a statistic. */
static uint64_t permutation_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ULL;
}

/* Mean cost per segment over every cognate set the model can score. */
static rg_status corpus_cost_per_segment(
    const rg_context *ctx,
    const rg_multi_model *model,
    const rg_train_options *options,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    double *out_mean,
    size_t *out_scored
) {
    size_t c;
    size_t scored = 0;
    double total = 0.0;
    int max_chunk_size = options != 0 && options->max_chunk_size > 0
        ? options->max_chunk_size : RG_DEFAULT_MAX_CHUNK_SIZE;

    *out_mean = 0.0;
    *out_scored = 0;
    for (c = 0; c < cognate_count; c++) {
        double cost = 0.0;
        int pair_count = 0;
        rg_status status = score_cognate_set(ctx, model, options, &cognates[c], max_chunk_size, &cost, &pair_count);
        if (status != RG_OK) {
            return status;
        }
        if (pair_count > 0) {
            total += cost;
            scored++;
        }
    }
    if (scored > 0) {
        *out_mean = total / (double)scored;
    }
    *out_scored = scored;
    return RG_OK;
}

/* Rebuilds the corpus with the correspondences taken out of it and nothing
 * else changed: every lect keeps its whole wordlist, every set keeps its size
 * and its lect membership, and only which form answers to which is permuted.
 * Whatever the model finds in this is what the method finds in no data. */
static rg_status permute_cognate_sets(
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const char *const *lect_ids,
    size_t lect_count,
    uint64_t *rng,
    rg_cognate_set *out_sets,
    rg_cognate_form *out_forms,
    size_t *slots
) {
    size_t c;
    size_t f;
    size_t used = 0;
    size_t l;

    for (c = 0; c < cognate_count; c++) {
        out_sets[c] = cognates[c];
        out_sets[c].forms = out_forms + used;
        for (f = 0; f < cognates[c].form_count; f++) {
            out_forms[used + f] = cognates[c].forms[f];
        }
        used += cognates[c].form_count;
    }
    /* One lect at a time, so a set never ends up with two forms of the same
     * lect or loses one. Fisher-Yates over that lect's slots. */
    for (l = 0; l < lect_count; l++) {
        size_t slot_count = 0;
        size_t i;
        used = 0;
        for (c = 0; c < cognate_count; c++) {
            for (f = 0; f < cognates[c].form_count; f++) {
                if (strcmp(out_forms[used + f].lect_id, lect_ids[l]) == 0) {
                    slots[slot_count++] = used + f;
                }
            }
            used += cognates[c].form_count;
        }
        for (i = slot_count; i > 1; i--) {
            size_t j = (size_t)(permutation_next(rng) % (uint64_t)(unsigned long)i);
            rg_form swap = out_forms[slots[i - 1]].form;
            out_forms[slots[i - 1]].form = out_forms[slots[j]].form;
            out_forms[slots[j]].form = swap;
        }
    }
    return RG_OK;
}


typedef struct permutation_baseline permutation_baseline;
struct permutation_baseline {
    size_t runs;
    double cost_mean;
    double cost_sd;
    double unconditioned_mean;
    double conditioned_mean;
    double search_margin;
    double search_margin_quantile;
};

static rg_status compute_corpus_fit(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    const permutation_baseline *baseline,
    rg_multi_model *model
);
static rg_status run_permutation_baseline(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    const char *const *lect_ids,
    size_t lect_count,
    permutation_baseline *out
);
static rg_status bootstrap_class_intervals(
    rg_multi_model *model,
    size_t cognate_count,
    const rg_train_options *options
);

rg_status rg_train_model(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    rg_multi_model **out
) {
    rg_multi_model *model;
    rg_progress_state progress;
    rg_train_options resolved_options;
    permutation_baseline baseline;
    reconciled_observation *observations = 0;
    size_t observation_count = 0;
    size_t c;
    size_t lect_cap = 0;
    rg_status status;
    if (ctx == 0 || out == 0 || (cognate_count > 0 && cognates == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    model = (rg_multi_model *)calloc(1, sizeof(*model));
    if (model == 0) {
        return RG_ERR_OOM;
    }
    for (c = 0; c < cognate_count; c++) {
        size_t f;
        if (cognates[c].form_count == 0 || cognates[c].forms == 0) {
            rg_multi_model_free(model);
            return RG_ERR_INVALID_ARGUMENT;
        }
        if (cognates[c].confidence < 0.0 || cognates[c].confidence > 1.0) {
            rg_multi_model_free(model);
            return RG_ERR_INVALID_ARGUMENT;
        }
        for (f = 0; f < cognates[c].form_count; f++) {
            if (cognates[c].forms[f].form.segment_count == 0 ||
                cognates[c].forms[f].form.segments == 0 ||
                cognates[c].forms[f].lect_id == 0) {
                rg_multi_model_free(model);
                return RG_ERR_INVALID_ARGUMENT;
            }
            status = append_lect(&model->lect_ids, &model->lect_count, &lect_cap, cognates[c].forms[f].lect_id);
            if (status != RG_OK) {
                rg_multi_model_free(model);
                return status;
            }
        }
    }
    memset(&baseline, 0, sizeof(baseline));
    if (options == 0) {
        rg_train_options_init_defaults(&resolved_options);
    } else {
        resolved_options = *options;
    }
    options = &resolved_options;
    /* The baseline runs before the model, not after, because it is what sets
     * the search charge when the caller asks for a tuned one. Its own runs
     * recurse into this function with permutation_count cleared. */
    if (options->permutation_count > 0) {
        status = run_permutation_baseline(ctx, cognates, cognate_count, options,
                                          (const char *const *)model->lect_ids, model->lect_count,
                                          &baseline);
        if (status != RG_OK) {
            rg_multi_model_free(model);
            return status;
        }
        if (options->tune_search_penalty && baseline.search_margin > 0.0) {
            /* What the shuffles reached is what a rule has to beat. Bought
             * with recall: a corpus whose conditioning is weak against its own
             * noise loses rules the fixed charge would have kept. */
            resolved_options.bic.search_penalty_gamma = baseline.search_margin;
        }
    }
    /* A set with one form carries no correspondence: there is nothing to align
     * it against. That is a fact about the data, not an error in it. Every
     * cognate-coded wordlist has them -- an isolate, a loan, a unique
     * retention, or a form whose cognates are in lects this corpus did not
     * sample -- and refusing the whole corpus over one of them made regulae
     * unable to read the field's standard datasets without preprocessing. It
     * was also inconsistent: the wide loader drops such rows on its own, so the
     * same data trained when read wide and failed when read long.
     *
     * They are counted rather than silently dropped, and the count is
     * published: a user is entitled to know how much of their corpus
     * contributed nothing. */
    for (c = 0; c < cognate_count; c++) {
        if (cognates[c].form_count < 2) {
            model->unpaired_set_count++;
        }
    }
    /* Lects are held in ascending id order, which every later stage assumes.
     * They used to be held in the order they were first seen in the corpus,
     * while reconciliation, class discovery and the outlier ranking all walk
     * pairs in ascending order -- so whenever a corpus did not happen to list
     * its lects alphabetically, those stages aligned a pair in the opposite
     * direction from the one its model was trained in. A model of P(b|a) read
     * as P(a|b) misses on nearly every lookup and falls back to the prior, so
     * the classes came out of an untrained alignment. Renaming a lect changed
     * a quarter of the published classes on real data, which is how this
     * surfaced: a name is metadata, and no analysis may turn on it. */
    if (model->lect_count > 1) {
        size_t a;
        for (a = 1; a < model->lect_count; a++) {
            char *key = model->lect_ids[a];
            size_t b = a;
            while (b > 0 && strcmp(model->lect_ids[b - 1], key) > 0) {
                model->lect_ids[b] = model->lect_ids[b - 1];
                b--;
            }
            model->lect_ids[b] = key;
        }
    }
    /* One counter spans every lect pair and the multi-lect stages, so a caller
     * sees a single monotonic fraction rather than a bar that restarts. */
    {
        size_t pairs = model->lect_count < 2 ? 0 : model->lect_count * (model->lect_count - 1) / 2;
        rg_progress_init_internal(&progress, options,
                                  pairs * RG_PAIRWISE_STAGE_COUNT + RG_MULTILECT_STAGE_COUNT);
    }
    status = train_pair_models(ctx, cognates, cognate_count, options, &progress, model);
    if (status == RG_OK && rg_progress_step_internal(&progress, "reconciliation")) {
        status = RG_ERR_CANCELLED;
    }
    if (status == RG_OK) {
        status = aggregate_position_classes(
            ctx,
            cognates,
            cognate_count,
            options,
            model,
            &observations,
            &observation_count
        );
    }
    if (status == RG_OK && rg_progress_step_internal(&progress, "class discovery")) {
        status = RG_ERR_CANCELLED;
    }
    if (status == RG_OK) {
        status = publish_class_positions(model, observations, observation_count);
    }
    if (status == RG_OK) {
        status = multi_lect_context_discovery(
            ctx,
            cognates,
            cognate_count,
            options,
            model,
            observations,
            observation_count
        );
    }
    if (status == RG_OK) {
        status = lift_cross_dimensional_rows(model);
    }
    if (status == RG_OK && rg_progress_step_internal(&progress, "cross-dimensional lifting")) {
        status = RG_ERR_CANCELLED;
    }
    reconciled_observations_free(observations, observation_count);
    if (status == RG_OK) {
        status = bootstrap_class_intervals(model, cognate_count, options);
    }
    if (status == RG_OK) {
        status = compute_corpus_fit(ctx, cognates, cognate_count, options, &baseline, model);
    }
    if (status != RG_OK) {
        rg_multi_model_free(model);
        return status;
    }
    *out = model;
    return RG_OK;
}

const rg_corpus_fit *rg_multi_model_fit(const rg_multi_model *model) {
    return model == 0 ? 0 : &model->fit;
}

size_t rg_multi_model_lect_count(const rg_multi_model *model) {
    return model == 0 ? 0 : model->lect_count;
}

size_t rg_multi_model_unpaired_set_count(const rg_multi_model *model) {
    return model == 0 ? 0 : model->unpaired_set_count;
}

const char *rg_multi_model_lect_at(const rg_multi_model *model, size_t index) {
    if (model == 0 || index >= model->lect_count) {
        return 0;
    }
    return model->lect_ids[index];
}

size_t rg_multi_model_pair_model_count(const rg_multi_model *model) {
    return model == 0 ? 0 : model->pair_model_count;
}

const rg_multi_pair_model_row *rg_multi_model_pair_model_at(const rg_multi_model *model, size_t index) {
    if (model == 0 || index >= model->pair_model_count) {
        return 0;
    }
    return &model->pair_models[index].view;
}

size_t rg_multi_model_unconditioned_class_count(const rg_multi_model *model) {
    return model == 0 ? 0 : model->unconditioned_class_count;
}

const rg_multi_class_row *rg_multi_model_unconditioned_class_at(const rg_multi_model *model, size_t index) {
    if (model == 0 || index >= model->unconditioned_class_count) {
        return 0;
    }
    return &model->unconditioned_classes[index].view;
}

size_t rg_multi_model_conditioned_class_count(const rg_multi_model *model) {
    return model == 0 ? 0 : model->conditioned_class_count;
}

const rg_multi_class_row *rg_multi_model_conditioned_class_at(const rg_multi_model *model, size_t index) {
    if (model == 0 || index >= model->conditioned_class_count) {
        return 0;
    }
    return &model->conditioned_classes[index].view;
}

size_t rg_multi_model_cross_dimensional_row_count(const rg_multi_model *model) {
    return model == 0 ? 0 : model->cross_dimensional_count;
}

const rg_multi_cross_dimensional_row *rg_multi_model_cross_dimensional_row_at(const rg_multi_model *model, size_t index) {
    if (model == 0 || index >= model->cross_dimensional_count) {
        return 0;
    }
    return &model->cross_dimensional_rows[index].view;
}

void rg_cognate_outlier_rows_free(rg_cognate_outlier_row *rows, size_t count) {
    size_t i;
    if (rows == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        free((char *)rows[i].cognate_id);
    }
    free(rows);
}

/* Mean alignment cost per segment for one cognate set, over every lect pair in
 * it that has a trained model. This is the corpus's goodness of fit read one
 * set at a time: the outlier diagnostic z-scores it across sets, and the fit
 * summary averages it. */
/* Replaces the closed-form intervals on the multi-lect classes with ones
 * resampled over whole cognate sets.
 *
 * The Wilson interval's denominator counts aligned positions, and positions
 * from one word pair are not independent observations of anything: a
 * Latin-Spanish corpus has 413 of them over 97 cognate sets, so the interval
 * is narrower than the evidence supports by roughly the square root of that
 * ratio. Resampling the sets rather than the positions is what makes the
 * denominator mean what the arithmetic assumes, and it carries the corpus's
 * confidence weighting along with it, since a set is drawn or not as a whole.
 *
 * No realignment is involved. `class_positions` already records which cognate
 * each reconciled position came from, so a resample is a reweighting of a
 * table that exists. */
static rg_status bootstrap_class_intervals(
    rg_multi_model *model,
    size_t cognate_count,
    const rg_train_options *options
) {
    size_t draws = options->bootstrap_n > 0 ? (size_t)options->bootstrap_n : 0;
    size_t class_count = model->unconditioned_class_count + model->conditioned_class_count;
    size_t *multiplicity;
    double *rates;
    double *counts;
    uint64_t rng;
    size_t b;
    size_t i;

    if (draws == 0 || class_count == 0 || cognate_count == 0) {
        return RG_OK;
    }
    multiplicity = (size_t *)calloc(cognate_count, sizeof(*multiplicity));
    counts = (double *)calloc(class_count, sizeof(*counts));
    rates = (double *)calloc(class_count * draws, sizeof(*rates));
    if (multiplicity == 0 || counts == 0 || rates == 0) {
        free(multiplicity);
        free(counts);
        free(rates);
        return RG_ERR_OOM;
    }
    rng = (uint64_t)(unsigned int)options->bootstrap_seed * 6364136223846793005ULL
        + 1442695040888963407ULL;
    for (b = 0; b < draws; b++) {
        double total = 0.0;
        for (i = 0; i < cognate_count; i++) {
            multiplicity[i] = 0;
        }
        for (i = 0; i < cognate_count; i++) {
            multiplicity[(size_t)(permutation_next(&rng) % (uint64_t)(unsigned long)cognate_count)]++;
        }
        for (i = 0; i < class_count; i++) {
            counts[i] = 0.0;
        }
        for (i = 0; i < model->class_position_count; i++) {
            const rg_class_position *position = &model->class_positions[i];
            double weight;
            size_t k;
            if (position->cognate_index >= cognate_count) {
                continue;
            }
            weight = (double)multiplicity[position->cognate_index];
            total += weight;
            for (k = 0; k < position->class_id_count; k++) {
                int id = position->class_ids[k];
                if (id >= 0 && (size_t)id < class_count) {
                    counts[id] += weight;
                }
            }
        }
        for (i = 0; i < class_count; i++) {
            rates[i * draws + b] = total > 0.0 ? counts[i] / total : 0.0;
        }
    }
    for (i = 0; i < class_count; i++) {
        rg_multi_class_owned *owned = i < model->unconditioned_class_count
            ? &model->unconditioned_classes[i]
            : &model->conditioned_classes[i - model->unconditioned_class_count];
        rg_uncertainty_estimate estimate;
        int post = i >= model->unconditioned_class_count;
        if (rg_percentile_interval(&rates[i * draws], draws,
                                  owned->view.uncertainty.estimate,
                                  owned->view.uncertainty.n,
                                  owned->view.uncertainty.alpha,
                                  &estimate) != RG_OK) {
            continue;
        }
        estimate.post_selection = post;
        owned->view.uncertainty = estimate;
    }
    free(multiplicity);
    free(counts);
    free(rates);
    return RG_OK;
}

/* Trains the corpus with its correspondences taken out of it, as many times as
 * asked, and reports what the method finds in nothing.
 *
 * The shuffled runs carry no search charge at all. That is deliberate: what
 * they measure is how high an unpriced search can reach by chance, and
 * charging them would hide exactly that. The level their rules reach becomes
 * `null_search_margin`, and a real rule at or under it was findable in data
 * with no correspondences left in it. */

static int double_ascending(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static rg_status run_permutation_baseline(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    const char *const *lect_ids,
    size_t lect_count,
    permutation_baseline *out
) {
    size_t n = options->permutation_count > 0 ? (size_t)options->permutation_count : 0;
    size_t total_forms = 0;
    size_t c;
    rg_cognate_set *sets;
    rg_cognate_form *forms;
    size_t *slots;
    double *costs;
    double *margins = 0;
    size_t margin_count = 0;
    size_t margin_cap = 0;
    rg_train_options nested = *options;
    uint64_t rng = (uint64_t)(unsigned int)options->permutation_seed * 6364136223846793005ULL
        + 1442695040888963407ULL;
    double uncond = 0.0;
    double cond = 0.0;
    size_t completed = 0;
    size_t i;
    rg_status status = RG_OK;

    memset(out, 0, sizeof(*out));
    out->search_margin_quantile = 0.95;
    if (n == 0) {
        return RG_OK;
    }
    for (c = 0; c < cognate_count; c++) {
        total_forms += cognates[c].form_count;
    }
    sets = (rg_cognate_set *)calloc(cognate_count == 0 ? 1 : cognate_count, sizeof(*sets));
    forms = (rg_cognate_form *)calloc(total_forms == 0 ? 1 : total_forms, sizeof(*forms));
    slots = (size_t *)calloc(total_forms == 0 ? 1 : total_forms, sizeof(*slots));
    costs = (double *)calloc(n, sizeof(*costs));
    if (sets == 0 || forms == 0 || slots == 0 || costs == 0) {
        free(sets); free(forms); free(slots); free(costs);
        return RG_ERR_OOM;
    }
    nested.permutation_count = 0;
    nested.tune_search_penalty = 0;
    nested.progress = 0;
    nested.progress_user_data = 0;
    nested.bic.search_penalty_gamma = 0.0;
    for (i = 0; i < n && status == RG_OK; i++) {
        rg_multi_model *shuffled = 0;
        double cost = 0.0;
        size_t scored = 0;
        size_t k;
        status = permute_cognate_sets(cognates, cognate_count, lect_ids, lect_count,
                                      &rng, sets, forms, slots);
        if (status != RG_OK) {
            break;
        }
        status = rg_train_model(ctx, sets, cognate_count, &nested, &shuffled);
        if (status != RG_OK) {
            break;
        }
        status = corpus_cost_per_segment(ctx, shuffled, &nested, sets, cognate_count, &cost, &scored);
        if (status == RG_OK && scored > 0) {
            costs[completed] = cost;
            uncond += (double)rg_multi_model_unconditioned_class_count(shuffled);
            cond += (double)rg_multi_model_conditioned_class_count(shuffled);
            completed++;
        }
        for (k = 0; k < rg_multi_model_conditioned_class_count(shuffled) && status == RG_OK; k++) {
            const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(shuffled, k);
            if (margin_count == margin_cap) {
                size_t next_cap = margin_cap == 0 ? 64 : margin_cap * 2;
                double *next = (double *)realloc(margins, next_cap * sizeof(*next));
                if (next == 0) {
                    status = RG_ERR_OOM;
                    break;
                }
                margins = next;
                margin_cap = next_cap;
            }
            margins[margin_count++] = row->search_margin;
        }
        rg_multi_model_free(shuffled);
    }
    if (status == RG_OK && completed > 0) {
        double mean = 0.0;
        double variance = 0.0;
        for (i = 0; i < completed; i++) {
            mean += costs[i];
        }
        mean /= (double)completed;
        for (i = 0; i < completed; i++) {
            double d = costs[i] - mean;
            variance += d * d;
        }
        out->runs = completed;
        out->cost_mean = mean;
        out->cost_sd = completed > 1 ? sqrt(variance / (double)(completed - 1)) : 0.0;
        out->unconditioned_mean = uncond / (double)completed;
        out->conditioned_mean = cond / (double)completed;
        if (margin_count > 0) {
            size_t index;
            qsort(margins, margin_count, sizeof(*margins), double_ascending);
            index = (size_t)(out->search_margin_quantile * (double)margin_count);
            if (index >= margin_count) {
                index = margin_count - 1;
            }
            out->search_margin = margins[index];
        }
    }
    free(sets); free(forms); free(slots); free(costs); free(margins);
    return status;
}

/* Fills in the model's fit summary from the observed corpus, plus whatever the
 * shuffled baseline measured before the model was trained. */
static rg_status compute_corpus_fit(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    const permutation_baseline *baseline,
    rg_multi_model *model
) {
    rg_status status;

    model->fit.unconditioned_class_count = model->unconditioned_class_count;
    model->fit.conditioned_class_count = model->conditioned_class_count;
    {
        size_t c;
        for (c = 0; c < cognate_count; c++) {
            size_t f;
            for (f = 0; f < cognates[c].form_count; f++) {
                size_t *breaks = 0;
                size_t break_count = 0;
                int inferred = 0;
                if (rg_compute_syllable_breaks_internal(ctx, &cognates[c].forms[f].form,
                                                        &breaks, &break_count, &inferred) != RG_OK) {
                    continue;
                }
                free(breaks);
                model->fit.syllabified_form_count++;
                if (inferred) {
                    model->fit.inferred_nucleus_form_count++;
                }
            }
        }
    }
    status = corpus_cost_per_segment(ctx, model, options, cognates, cognate_count,
                                     &model->fit.cost_per_segment, &model->fit.scored_set_count);
    if (status != RG_OK || baseline->runs == 0) {
        return status;
    }
    /* The verdict a reader would otherwise have to reach by hand, on
     * twenty-five rules: does this one's evidence carry a heavier search
     * charge than the level the same search reaches on the corpus with its
     * correspondences shuffled out? Reported per rule and counted for the
     * corpus. */
    {
        size_t i;
        size_t j;
        for (i = 0; i < model->conditioned_class_count; i++) {
            rg_multi_class_row *row = &model->conditioned_classes[i].view;
            row->standing = row->search_margin > baseline->search_margin
                ? RG_RULE_STANDING_ABOVE_NOISE : RG_RULE_STANDING_WITHIN_NOISE;
            model->fit.rules_measured++;
            if (row->standing == RG_RULE_STANDING_ABOVE_NOISE) {
                model->fit.rules_above_noise++;
            }
        }
        for (i = 0; i < model->cross_dimensional_count; i++) {
            rg_multi_cross_dimensional_row *row = &model->cross_dimensional_rows[i].view;
            row->standing = row->search_margin > baseline->search_margin
                ? RG_RULE_STANDING_ABOVE_NOISE : RG_RULE_STANDING_WITHIN_NOISE;
            model->fit.rules_measured++;
            if (row->standing == RG_RULE_STANDING_ABOVE_NOISE) {
                model->fit.rules_above_noise++;
            }
        }
        for (i = 0; i < model->pair_model_count; i++) {
            rg_pairwise_model *pair = model->pair_models[i].model;
            for (j = 0; j < pair->conditioned_segment_count_count; j++) {
                rg_conditioned_segment_count_row *row = &pair->conditioned_segment_counts[j];
                row->standing = row->search_margin > baseline->search_margin
                    ? RG_RULE_STANDING_ABOVE_NOISE : RG_RULE_STANDING_WITHIN_NOISE;
            }
            for (j = 0; j < pair->cross_dimensional_count; j++) {
                rg_cross_dimensional_row *row = &pair->cross_dimensional_rows[j];
                row->standing = row->search_margin > baseline->search_margin
                    ? RG_RULE_STANDING_ABOVE_NOISE : RG_RULE_STANDING_WITHIN_NOISE;
            }
        }
    }
    model->fit.permutation_count = baseline->runs;
    model->fit.null_cost_per_segment_mean = baseline->cost_mean;
    model->fit.null_cost_per_segment_sd = baseline->cost_sd;
    model->fit.null_unconditioned_class_mean = baseline->unconditioned_mean;
    model->fit.null_conditioned_class_mean = baseline->conditioned_mean;
    model->fit.null_search_margin = baseline->search_margin;
    model->fit.null_search_margin_quantile = baseline->search_margin_quantile;
    if (baseline->cost_sd > 0.0) {
        model->fit.cost_per_segment_z =
            (model->fit.cost_per_segment - baseline->cost_mean) / baseline->cost_sd;
    }
    return RG_OK;
}

static rg_status score_cognate_set(
    const rg_context *ctx,
    const rg_multi_model *model,
    const rg_train_options *options,
    const rg_cognate_set *set,
    int max_chunk_size,
    double *out_cost,
    int *out_pair_count
) {
    const rg_form **forms;
    size_t *order = 0;
    size_t order_count = 0;
    size_t i;
    double total = 0.0;
    int pair_count = 0;
    rg_status status = RG_OK;

    *out_cost = 0.0;
    *out_pair_count = 0;
    forms = (const rg_form **)calloc(model->lect_count == 0 ? 1 : model->lect_count, sizeof(*forms));
    if (forms == 0) {
        return RG_ERR_OOM;
    }
    /* Lect pairs are walked in ascending lect-id order, which fixes the
     * alignment direction each pair is scored in. */
    if (present_lects_sorted(set, model, forms, &order, &order_count) != RG_OK) {
        free(forms);
        return RG_ERR_OOM;
    }
    for (i = 0; i < order_count && status == RG_OK; i++) {
        const rg_form *form_i = forms[order[i]];
        size_t j;
        for (j = i + 1; j < order_count; j++) {
            const rg_pairwise_model *pair_model;
            const rg_form *form_j = forms[order[j]];
            rg_alignment *alignment = 0;
            double cost = 0.0;
            double denom;
            pair_model = pair_model_for(model, model->lect_ids[order[i]], model->lect_ids[order[j]]);
            if (pair_model == 0) {
                continue;
            }
            status = rg_align_forms_with_model(ctx, pair_model, options, form_i, form_j, max_chunk_size, &alignment);
            if (status != RG_OK) {
                break;
            }
            status = rg_alignment_cost_with_model(ctx, pair_model, options, alignment, &cost);
            rg_alignment_free(alignment);
            if (status != RG_OK) {
                break;
            }
            denom = ((double)form_i->segment_count + (double)form_j->segment_count) / 2.0;
            if (denom > 0.0) {
                total += cost / denom;
                pair_count++;
            }
        }
    }
    free(forms);
    free(order);
    if (status != RG_OK) {
        return status;
    }
    *out_cost = pair_count > 0 ? total / (double)pair_count : 0.0;
    *out_pair_count = pair_count;
    return RG_OK;
}

rg_status rg_find_cognate_outliers(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_multi_model *model,
    const rg_train_options *options,
    int top_k,
    int max_chunk_size,
    rg_cognate_outlier_row **out,
    size_t *out_count
) {
    outlier_work_row *work = 0;
    size_t work_count = 0;
    size_t work_cap = 0;
    size_t c;
    double mean = 0.0;
    double stddev = 0.0;
    rg_cognate_outlier_row *rows;
    size_t limit;
    if (ctx == 0 || model == 0 || out == 0 || out_count == 0 || (cognate_count > 0 && cognates == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    *out_count = 0;
    if (max_chunk_size == 0) {
        max_chunk_size = RG_DEFAULT_MAX_CHUNK_SIZE;
    }
    if (max_chunk_size < 1) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    for (c = 0; c < cognate_count; c++) {
        double cost = 0.0;
        int pair_count = 0;
        rg_status status = score_cognate_set(ctx, model, options, &cognates[c], max_chunk_size, &cost, &pair_count);
        if (status != RG_OK) {
            size_t n;
            for (n = 0; n < work_count; n++) {
                free(work[n].cognate_id);
            }
            free(work);
            return status;
        }
        if (pair_count > 0) {
            outlier_work_row *next;
            if (work_count == work_cap) {
                size_t next_cap = work_cap == 0 ? 16 : work_cap * 2;
                next = (outlier_work_row *)realloc(work, next_cap * sizeof(*work));
                if (next == 0) {
                    size_t n;
                    for (n = 0; n < work_count; n++) {
                        free(work[n].cognate_id);
                    }
                    free(work);
                    return RG_ERR_OOM;
                }
                work = next;
                work_cap = next_cap;
            }
            memset(&work[work_count], 0, sizeof(work[work_count]));
            work[work_count].cognate_id = rg_strdup_internal(cognates[c].cognate_id == 0 ? "" : cognates[c].cognate_id);
            if (work[work_count].cognate_id == 0) {
                size_t n;
                for (n = 0; n < work_count; n++) {
                    free(work[n].cognate_id);
                }
                free(work);
                return RG_ERR_OOM;
            }
            work[work_count].pair_count = pair_count;
            work[work_count].cost_per_segment = cost;
            mean += cost;
            work_count++;
        }
    }
    if (work_count == 0) {
        free(work);
        return RG_OK;
    }
    mean /= (double)work_count;
    if (work_count >= 2) {
        double variance = 0.0;
        size_t i;
        for (i = 0; i < work_count; i++) {
            double d = work[i].cost_per_segment - mean;
            variance += d * d;
        }
        variance /= (double)(work_count - 1);
        stddev = sqrt(variance);
    }
    for (c = 0; c < work_count; c++) {
        work[c].z_score = stddev > 0.0 ? (work[c].cost_per_segment - mean) / stddev : 0.0;
    }
    qsort(work, work_count, sizeof(*work), outlier_cmp);
    limit = work_count;
    if (top_k > 0 && (size_t)top_k < limit) {
        limit = (size_t)top_k;
    }
    rows = (rg_cognate_outlier_row *)calloc(limit, sizeof(*rows));
    if (rows == 0 && limit > 0) {
        size_t i;
        for (i = 0; i < work_count; i++) {
            free(work[i].cognate_id);
        }
        free(work);
        return RG_ERR_OOM;
    }
    for (c = 0; c < limit; c++) {
        rows[c].cognate_id = work[c].cognate_id;
        rows[c].pair_count = work[c].pair_count;
        rows[c].cost_per_segment = work[c].cost_per_segment;
        rows[c].z_score = work[c].z_score;
        work[c].cognate_id = 0;
    }
    for (c = 0; c < work_count; c++) {
        free(work[c].cognate_id);
    }
    free(work);
    *out = rows;
    *out_count = limit;
    return RG_OK;
}
