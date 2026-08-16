#include "multilect_internal.h"

#include <stdint.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>



void string_array_clear(char **items, size_t count) {
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


void reconciled_observation_clear(reconciled_observation *obs) {
    if (obs == 0) {
        return;
    }
    string_array_clear(obs->lects, obs->segment_count);
    string_array_clear(obs->graphemes, obs->segment_count);
    free(obs->positions);
    free(obs->lect_indices);
    memset(obs, 0, sizeof(*obs));
}

rg_status class_position_add_id(rg_class_position *position, int class_id) {
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

void reconciled_observations_free(reconciled_observation *items, size_t count) {
    size_t i;
    if (items == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        reconciled_observation_clear(&items[i]);
    }
    free(items);
}




static void multi_class_clear(rg_multi_class_row *klass) {
    size_t i;
    if (klass == 0) {
        return;
    }
    string_array_clear(rg_owned_internal(klass->lect_ids), klass->segment_count);
    string_array_clear(rg_owned_internal(klass->graphemes), klass->segment_count);
    if (klass->contexts != 0) {
        for (i = 0; i < klass->segment_count; i++) {
            rg_context_spec_clear_internal(rg_owned_internal(&klass->contexts[i]));
        }
        rg_free_owned_internal(klass->contexts);
    }
    string_array_clear(rg_owned_internal(klass->supporting_cognates), klass->supporting_cognate_count);
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

double cognate_weight(const rg_cognate_set *cognate) {
    return cognate->confidence;
}

const rg_form *form_for_lect(const rg_cognate_set *cognate, const char *lect_id) {
    size_t i;
    for (i = 0; i < cognate->form_count; i++) {
        if (strcmp(cognate->forms[i].lect_id, lect_id) == 0) {
            return &cognate->forms[i].form;
        }
    }
    return 0;
}

static void evidence_uses_scorer(rg_rule_evidence *evidence, rg_split_scorer scorer) {
    if (evidence->decision_index < 0) {
        return;
    }
    evidence->scorer = scorer;
    evidence->delta_score = evidence->delta_bic;
}

static void stamp_split_scorer(rg_multi_model *model, rg_split_scorer scorer) {
    size_t i;
    for (i = 0; i < model->pair_model_count; i++) {
        rg_pairwise_model *pair = model->pair_models[i].model;
        size_t j;
        for (j = 0; j < pair->conditioned_segment_count_count; j++) {
            evidence_uses_scorer(&pair->conditioned_segment_counts[j].evidence, scorer);
        }
        for (j = 0; j < pair->cross_dimensional_count; j++) {
            evidence_uses_scorer(&pair->cross_dimensional_rows[j].evidence, scorer);
        }
    }
    for (i = 0; i < model->conditioned_class_count; i++) {
        evidence_uses_scorer(&model->conditioned_classes[i].evidence, scorer);
    }
    for (i = 0; i < model->cross_dimensional_count; i++) {
        evidence_uses_scorer(&model->cross_dimensional_rows[i].rule.evidence, scorer);
    }
}

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
    if ((options->bic.split_scorer != RG_SPLIT_SCORER_CORRECTED_BIC &&
         options->bic.split_scorer != RG_SPLIT_SCORER_MULTINOMIAL_NML &&
         options->bic.split_scorer != RG_SPLIT_SCORER_DIRICHLET_MARGINAL) ||
        !isfinite(options->bic.split_prior_concentration) ||
        options->bic.split_prior_concentration <= 0.0 ||
        !isfinite(options->bic.search_penalty_gamma) ||
        options->bic.search_penalty_gamma < 0.0 ||
        options->predictive_folds < 0 || options->predictive_folds == 1 ||
        options->predictive_min_groups < 1 || options->predictive_top_k < 1 ||
        !isfinite(options->predictive_abstention_threshold) ||
        options->predictive_abstention_threshold < 0.0 ||
        options->predictive_abstention_threshold > 1.0) {
        rg_multi_model_free(model);
        return RG_ERR_INVALID_ARGUMENT;
    }
    if (options->bic.split_scorer == RG_SPLIT_SCORER_MULTINOMIAL_NML) {
        for (c = 0; c < cognate_count; c++) {
            if (fabs(cognates[c].confidence - floor(cognates[c].confidence + 0.5)) > 1e-12) {
                rg_multi_model_free(model);
                return RG_ERR_UNSUPPORTED_OPTION;
            }
        }
    }
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
    if (status == RG_OK) {
        stamp_split_scorer(model, options->bic.split_scorer);
    }
    if (status == RG_OK && rg_progress_step_internal(&progress, "cross-dimensional lifting")) {
        status = RG_ERR_CANCELLED;
    }
    reconciled_observations_free(observations, observation_count);
    if (status == RG_OK) {
        status = bootstrap_class_intervals(model, cognates, cognate_count, options);
    }
    if (status == RG_OK) {
        status = compute_corpus_fit(ctx, cognates, cognate_count, options, &baseline, model);
    }
    if (status == RG_OK) {
        status = rg_predictive_evaluate_internal(ctx, cognates, cognate_count, options, model);
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

size_t rg_multi_model_unpaired_set_count(const rg_multi_model *model) {
    return model == 0 ? 0 : model->unpaired_set_count;
}

#define RG_MULTI_TABLE(fn, rowtype, field, countfield)                          \
    const rowtype *fn(const rg_multi_model *model, size_t *count) {              \
        if (model == 0) {                                                       \
            if (count != 0) { *count = 0; }                                     \
            return 0;                                                           \
        }                                                                       \
        if (count != 0) { *count = model->countfield; }                         \
        return model->field;                                                    \
    }

RG_MULTI_TABLE(rg_multi_model_unconditioned_classes, rg_multi_class_row, unconditioned_classes, unconditioned_class_count)
RG_MULTI_TABLE(rg_multi_model_conditioned_classes, rg_multi_class_row, conditioned_classes, conditioned_class_count)
RG_MULTI_TABLE(rg_multi_model_cross_dimensional_rows, rg_multi_cross_dimensional_row, cross_dimensional_rows, cross_dimensional_count)

#undef RG_MULTI_TABLE

const char *const *rg_multi_model_lects(const rg_multi_model *model, size_t *count) {
    if (model == 0) {
        if (count != 0) {
            *count = 0;
        }
        return 0;
    }
    if (count != 0) {
        *count = model->lect_count;
    }
    return (const char *const *)model->lect_ids;
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

void rg_cognate_outlier_rows_free(rg_cognate_outlier_row *rows, size_t count) {
    size_t i;
    if (rows == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        rg_free_owned_internal(rows[i].cognate_id);
    }
    free(rows);
}
