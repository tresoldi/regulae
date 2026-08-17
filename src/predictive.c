#include "multilect_internal.h"
#include "environment.h"
#include "search_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define RG_PREDICTIVE_CALIBRATION_BINS 10
#define RG_PREDICTIVE_RULE_MIN_OBSERVATIONS 8
#define RG_PREDICTIVE_FLOOR 1e-12

typedef enum predictive_model_kind {
    PREDICT_CONDITIONED = 0,
    PREDICT_UNCONDITIONED = 1,
    PREDICT_IDENTITY = 2,
    PREDICT_INVENTORY = 3,
    PREDICT_FEATURE_DISTANCE = 4
} predictive_model_kind;

typedef struct predictive_score_accumulator {
    size_t observation_count;
    size_t unseen_reflex_count;
    double weight;
    double log_loss;
    double top1;
    double top_k;
    double brier;
    double abstained;
    double accepted;
    double accepted_top1;
    double bin_weight[RG_PREDICTIVE_CALIBRATION_BINS];
    double bin_confidence[RG_PREDICTIVE_CALIBRATION_BINS];
    double bin_correct[RG_PREDICTIVE_CALIBRATION_BINS];
} predictive_score_accumulator;

typedef struct predictive_rule_accumulator {
    rg_rule_evidence *evidence;
    predictive_score_accumulator conditioned;
    predictive_score_accumulator unconditioned;
    size_t folds;
    int last_fold;
} predictive_rule_accumulator;

typedef struct predictive_pair_rule_accumulators {
    rg_pairwise_model *model;
    predictive_rule_accumulator *segment_items;
    size_t segment_count;
    predictive_rule_accumulator *cross_dimensional_items;
    size_t cross_dimensional_count;
} predictive_pair_rule_accumulators;

typedef struct predictive_direction_model {
    const char *source_lect;
    const char *target_lect;
    rg_pairwise_model *model;
} predictive_direction_model;

typedef struct predictive_inventory {
    const char **items;
    size_t count;
} predictive_inventory;

typedef struct predictive_probabilities {
    double *values;
    size_t count;
    size_t actual_index;
    size_t best_index;
    double best_probability;
    int unseen;
} predictive_probabilities;

static rg_status form_pairs_for_fold(
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const size_t *folds,
    size_t heldout_fold,
    const char *source_lect,
    const char *target_lect,
    int training,
    rg_form_pair **out,
    size_t *out_count
);
static void predictive_probabilities_clear(predictive_probabilities *prediction);

static int nonempty_equal(const char *a, const char *b) {
    return a != 0 && b != 0 && a[0] != '\0' && b[0] != '\0' && strcmp(a, b) == 0;
}

static size_t component_find(size_t *parents, size_t item) {
    size_t root = item;
    while (parents[root] != root) {
        root = parents[root];
    }
    while (parents[item] != item) {
        size_t next = parents[item];
        parents[item] = root;
        item = next;
    }
    return root;
}

static void component_union(size_t *parents, size_t a, size_t b) {
    size_t root_a = component_find(parents, a);
    size_t root_b = component_find(parents, b);
    if (root_a == root_b) {
        return;
    }
    if (root_a < root_b) {
        parents[root_b] = root_a;
    } else {
        parents[root_a] = root_b;
    }
}

static const char *component_label(
    const rg_cognate_set *cognates,
    size_t *parents,
    size_t root,
    size_t count
) {
    const char *label = 0;
    size_t i;
    for (i = 0; i < count; i++) {
        const char *candidate;
        if (component_find(parents, i) != root) {
            continue;
        }
        candidate = cognates[i].cognate_id;
        if (candidate == 0 || candidate[0] == '\0') {
            candidate = cognates[i].etymon_group;
        }
        if (candidate == 0 || candidate[0] == '\0') {
            candidate = cognates[i].source_group;
        }
        if (candidate != 0 && candidate[0] != '\0' &&
            (label == 0 || strcmp(candidate, label) < 0)) {
            label = candidate;
        }
    }
    return label == 0 ? "" : label;
}

static unsigned int predictive_next_random(unsigned int *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static rg_status assign_dependency_folds(
    const rg_cognate_set *cognates,
    size_t cognate_count,
    int requested_folds,
    int minimum_groups,
    int seed,
    size_t **out_folds,
    size_t *out_fold_count,
    size_t *out_group_count
) {
    size_t *parents = 0;
    size_t *roots = 0;
    size_t *folds = 0;
    size_t root_count = 0;
    size_t fold_count;
    size_t i;
    size_t j;
    unsigned int random_state = (unsigned int)seed;

    *out_folds = 0;
    *out_fold_count = 0;
    *out_group_count = 0;
    if (cognate_count == 0) {
        return RG_OK;
    }
    parents = (size_t *)malloc(cognate_count * sizeof(*parents));
    roots = (size_t *)malloc(cognate_count * sizeof(*roots));
    folds = (size_t *)malloc(cognate_count * sizeof(*folds));
    if (parents == 0 || roots == 0 || folds == 0) {
        free(parents);
        free(roots);
        free(folds);
        return RG_ERR_OOM;
    }
    for (i = 0; i < cognate_count; i++) {
        parents[i] = i;
    }
    for (i = 0; i < cognate_count; i++) {
        for (j = i + 1; j < cognate_count; j++) {
            if (nonempty_equal(cognates[i].cognate_id, cognates[j].cognate_id) ||
                nonempty_equal(cognates[i].etymon_group, cognates[j].etymon_group) ||
                nonempty_equal(cognates[i].source_group, cognates[j].source_group)) {
                component_union(parents, i, j);
            }
        }
    }
    for (i = 0; i < cognate_count; i++) {
        size_t root = component_find(parents, i);
        int seen = 0;
        for (j = 0; j < root_count; j++) {
            if (roots[j] == root) {
                seen = 1;
                break;
            }
        }
        if (!seen) {
            roots[root_count++] = root;
        }
    }
    *out_group_count = root_count;
    /* component_label needs the real bound while qsort's comparator has no
     * context. A sentinel cannot supply it, so roots are insertion-sorted here
     * with the count in scope. */
    for (i = 1; i < root_count; i++) {
        size_t key = roots[i];
        const char *key_label = component_label(cognates, parents, key, cognate_count);
        j = i;
        while (j > 0) {
            const char *previous_label = component_label(cognates, parents, roots[j - 1], cognate_count);
            int c = strcmp(previous_label, key_label);
            if (c < 0 || (c == 0 && roots[j - 1] < key)) {
                break;
            }
            roots[j] = roots[j - 1];
            j--;
        }
        roots[j] = key;
    }
    for (i = root_count; i > 1; i--) {
        size_t at = (size_t)(predictive_next_random(&random_state) % (unsigned int)i);
        size_t swap = roots[i - 1];
        roots[i - 1] = roots[at];
        roots[at] = swap;
    }
    if (minimum_groups < 1) {
        minimum_groups = 1;
    }
    fold_count = (size_t)requested_folds;
    if (fold_count > root_count / (size_t)minimum_groups) {
        fold_count = root_count / (size_t)minimum_groups;
    }
    if (fold_count < 2 || root_count - (root_count + fold_count - 1) / fold_count <
                              (size_t)minimum_groups) {
        free(parents);
        free(roots);
        free(folds);
        return RG_OK;
    }
    for (i = 0; i < cognate_count; i++) {
        size_t root = component_find(parents, i);
        for (j = 0; j < root_count; j++) {
            if (roots[j] == root) {
                folds[i] = j % fold_count;
                break;
            }
        }
    }
    free(parents);
    free(roots);
    *out_folds = folds;
    *out_fold_count = fold_count;
    return RG_OK;
}

static int string_pointer_compare(const void *left, const void *right) {
    const char *a = *(const char *const *)left;
    const char *b = *(const char *const *)right;
    return strcmp(a, b);
}

static rg_status predictive_inventory_build(
    const rg_pairwise_model *model,
    predictive_inventory *out
) {
    size_t cap = 0;
    size_t i;
    memset(out, 0, sizeof(*out));
    for (i = 0; i < model->segment_prior_count; i++) {
        const char *target = model->segment_priors[i].target;
        size_t j;
        int seen = 0;
        for (j = 0; j < out->count; j++) {
            if (strcmp(out->items[j], target) == 0) {
                seen = 1;
                break;
            }
        }
        if (!seen) {
            if (out->count == cap) {
                size_t next_cap = cap == 0 ? 16 : cap * 2;
                const char **next = (const char **)realloc(out->items,
                                                           next_cap * sizeof(*next));
                if (next == 0) {
                    free(out->items);
                    memset(out, 0, sizeof(*out));
                    return RG_ERR_OOM;
                }
                out->items = next;
                cap = next_cap;
            }
            out->items[out->count++] = target;
        }
    }
    if (out->count > 1) {
        qsort(out->items, out->count, sizeof(*out->items), string_pointer_compare);
    }
    return RG_OK;
}

static double prior_mass(
    const rg_pairwise_model *model,
    const char *source,
    const char *target
) {
    size_t i;
    for (i = 0; i < model->segment_prior_count; i++) {
        if (strcmp(model->segment_priors[i].source, source) == 0 &&
            strcmp(model->segment_priors[i].target, target) == 0) {
            return model->segment_priors[i].alpha;
        }
    }
    return 0.0;
}

static const rg_segment_count_row *unconditioned_row(
    const rg_pairwise_model *model,
    const char *source,
    const char *target
) {
    size_t i;
    for (i = 0; i < model->segment_count_count; i++) {
        if (strcmp(model->segment_counts[i].source, source) == 0 &&
            strcmp(model->segment_counts[i].target, target) == 0) {
            return &model->segment_counts[i];
        }
    }
    return 0;
}

static const rg_conditioned_segment_count_row *conditioned_row(
    const rg_pairwise_model *model,
    const char *source,
    const char *target,
    const rg_context_spec *source_context,
    const rg_context_spec *target_context
) {
    const rg_conditioned_segment_count_row *best = 0;
    size_t best_specificity = 0;
    size_t i;
    for (i = 0; i < model->conditioned_segment_count_count; i++) {
        const rg_conditioned_segment_count_row *row = &model->conditioned_segment_counts[i];
        const rg_context_spec *against;
        bool subset = false;
        size_t specificity;
        if (strcmp(row->source, source) != 0 || strcmp(row->target, target) != 0) {
            continue;
        }
        against = row->context_is_target ? target_context : source_context;
        if (against == 0 ||
            rg_context_spec_is_subset(&row->context, against, &subset) != RG_OK || !subset) {
            continue;
        }
        specificity = rg_context_spec_constraint_count(&row->context);
        if (best == 0 || specificity > best_specificity ||
            (specificity == best_specificity && row->evidence.decision_index < best->evidence.decision_index)) {
            best = row;
            best_specificity = specificity;
        }
    }
    return best;
}

static double inventory_target_count(const rg_pairwise_model *model, const char *target) {
    size_t i;
    double total = 0.0;
    for (i = 0; i < model->segment_count_count; i++) {
        if (strcmp(model->segment_counts[i].target, target) == 0) {
            total += model->segment_counts[i].count;
        }
    }
    return total;
}

static int any_conditioned_environment(
    const rg_pairwise_model *model,
    const char *source,
    const rg_context_spec *source_context,
    const rg_context_spec *target_context
) {
    size_t i;
    for (i = 0; i < model->conditioned_segment_count_count; i++) {
        const rg_conditioned_segment_count_row *row = &model->conditioned_segment_counts[i];
        const rg_context_spec *against;
        bool subset = false;
        if (strcmp(row->source, source) != 0) {
            continue;
        }
        against = row->context_is_target ? target_context : source_context;
        if (against != 0 &&
            rg_context_spec_is_subset(&row->context, against, &subset) == RG_OK && subset) {
            return 1;
        }
    }
    return 0;
}

static rg_status prediction_probabilities(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    const predictive_inventory *inventory,
    const char *source,
    const char *actual,
    const rg_context_spec *source_context,
    const rg_context_spec *target_context,
    predictive_model_kind kind,
    double temperature,
    predictive_probabilities *out
) {
    size_t category_count = inventory->count + 1;
    size_t i;
    double total = 0.0;
    int has_condition = kind == PREDICT_CONDITIONED &&
        any_conditioned_environment(model, source, source_context, target_context);

    predictive_probabilities_clear(out);
    out->values = (double *)calloc(category_count == 0 ? 1 : category_count,
                                   sizeof(*out->values));
    if (out->values == 0) {
        return RG_ERR_OOM;
    }
    out->count = category_count;
    out->actual_index = inventory->count;
    out->unseen = 1;
    for (i = 0; i < inventory->count; i++) {
        const char *target = inventory->items[i];
        const rg_segment_count_row *base = unconditioned_row(model, source, target);
        double raw = 0.0;
        if (strcmp(target, actual) == 0) {
            out->actual_index = i;
            out->unseen = 0;
        }
        if (kind == PREDICT_IDENTITY) {
            raw = strcmp(source, target) == 0 ? 1.0 : 1e-6;
        } else if (kind == PREDICT_INVENTORY) {
            raw = inventory_target_count(model, target) + 1.0;
        } else if (kind == PREDICT_FEATURE_DISTANCE) {
            double distance = 0.0;
            rg_status status = rg_context_segment_distance(ctx, source, target, &distance);
            if (status != RG_OK) {
                free(out->values);
                memset(out, 0, sizeof(*out));
                return status;
            }
            raw = exp(-distance / (temperature > 0.0 ? temperature : 1.0));
        } else if (has_condition) {
            const rg_conditioned_segment_count_row *row = conditioned_row(
                model, source, target, source_context, target_context);
            raw = prior_mass(model, source, target) + (row == 0 ? 0.0 : row->count);
        } else {
            raw = prior_mass(model, source, target) + (base == 0 ? 0.0 : base->count);
        }
        if (!isfinite(raw) || raw < RG_PREDICTIVE_FLOOR) {
            raw = RG_PREDICTIVE_FLOOR;
        }
        out->values[i] = raw;
        total += raw;
    }
    if (kind == PREDICT_INVENTORY) {
        out->values[inventory->count] = 1.0;
    } else {
        out->values[inventory->count] = RG_PREDICTIVE_FLOOR;
    }
    total += out->values[inventory->count];
    if (total <= 0.0 || !isfinite(total)) {
        free(out->values);
        memset(out, 0, sizeof(*out));
        return RG_ERR_INVALID_ARGUMENT;
    }
    out->best_index = 0;
    for (i = 0; i < category_count; i++) {
        out->values[i] /= total;
        if (i == 0 || out->values[i] > out->values[out->best_index]) {
            out->best_index = i;
        }
    }
    out->best_probability = out->values[out->best_index];
    return RG_OK;
}

static void score_prediction(
    predictive_score_accumulator *accumulator,
    const predictive_probabilities *prediction,
    double weight,
    int top_k,
    double abstention_threshold
) {
    size_t i;
    size_t rank = 1;
    double actual_probability = prediction->values[prediction->actual_index];
    double brier = 0.0;
    int correct = !prediction->unseen && prediction->best_index == prediction->actual_index;
    int bin;

    if (weight <= 0.0) {
        return;
    }
    for (i = 0; i < prediction->count; i++) {
        double expected = i == prediction->actual_index ? 1.0 : 0.0;
        double error = prediction->values[i] - expected;
        brier += error * error;
        if (prediction->values[i] > actual_probability ||
            (prediction->values[i] == actual_probability && i < prediction->actual_index)) {
            rank++;
        }
    }
    accumulator->observation_count++;
    if (prediction->unseen) {
        accumulator->unseen_reflex_count++;
    }
    accumulator->weight += weight;
    accumulator->log_loss += -weight * log(actual_probability < RG_PREDICTIVE_FLOOR
                                            ? RG_PREDICTIVE_FLOOR : actual_probability);
    accumulator->top1 += weight * (correct ? 1.0 : 0.0);
    accumulator->top_k += weight * ((!prediction->unseen && rank <= (size_t)top_k) ? 1.0 : 0.0);
    accumulator->brier += weight * brier;
    if (prediction->best_probability < abstention_threshold) {
        accumulator->abstained += weight;
    } else {
        accumulator->accepted += weight;
        accumulator->accepted_top1 += weight * (correct ? 1.0 : 0.0);
    }
    bin = (int)(prediction->best_probability * (double)RG_PREDICTIVE_CALIBRATION_BINS);
    if (bin >= RG_PREDICTIVE_CALIBRATION_BINS) {
        bin = RG_PREDICTIVE_CALIBRATION_BINS - 1;
    }
    accumulator->bin_weight[bin] += weight;
    accumulator->bin_confidence[bin] += weight * prediction->best_probability;
    accumulator->bin_correct[bin] += weight * (correct ? 1.0 : 0.0);
}

static rg_predictive_score predictive_score_finish(
    const predictive_score_accumulator *accumulator
) {
    rg_predictive_score out;
    size_t i;
    memset(&out, 0, sizeof(out));
    out.observation_count = accumulator->observation_count;
    out.unseen_reflex_count = accumulator->unseen_reflex_count;
    out.observation_weight = accumulator->weight;
    if (accumulator->weight <= 0.0) {
        return out;
    }
    out.log_loss = accumulator->log_loss / accumulator->weight;
    out.top1_coverage = accumulator->top1 / accumulator->weight;
    out.top_k_coverage = accumulator->top_k / accumulator->weight;
    out.brier_score = accumulator->brier / accumulator->weight;
    out.abstention_rate = accumulator->abstained / accumulator->weight;
    out.accepted_top1_coverage = accumulator->accepted <= 0.0
        ? 0.0 : accumulator->accepted_top1 / accumulator->accepted;
    for (i = 0; i < RG_PREDICTIVE_CALIBRATION_BINS; i++) {
        if (accumulator->bin_weight[i] > 0.0) {
            double confidence = accumulator->bin_confidence[i] / accumulator->bin_weight[i];
            double accuracy = accumulator->bin_correct[i] / accumulator->bin_weight[i];
            out.calibration_error += accumulator->bin_weight[i] / accumulator->weight *
                fabs(confidence - accuracy);
        }
    }
    return out;
}

static int context_equal(const rg_context_spec *a, const rg_context_spec *b) {
    return rg_context_spec_compare_internal(a, b) == 0;
}

static predictive_rule_accumulator *rule_accumulator_for_fold_row(
    predictive_rule_accumulator *items,
    size_t count,
    const rg_conditioned_segment_count_row *fold_row,
    int fold
) {
    size_t i;
    for (i = 0; i < count; i++) {
        rg_conditioned_segment_count_row *full_row =
            (rg_conditioned_segment_count_row *)((char *)items[i].evidence -
                offsetof(rg_conditioned_segment_count_row, evidence));
        if (strcmp(full_row->source, fold_row->source) == 0 &&
            strcmp(full_row->target, fold_row->target) == 0 &&
            full_row->context_is_target == fold_row->context_is_target &&
            context_equal(&full_row->context, &fold_row->context)) {
            if (items[i].last_fold != fold) {
                items[i].folds++;
                items[i].last_fold = fold;
            }
            return &items[i];
        }
    }
    return 0;
}

static predictive_rule_accumulator *cross_dimensional_accumulator_for_fold_row(
    predictive_rule_accumulator *items,
    size_t count,
    const rg_cross_dimensional_row *full_rows,
    const rg_cross_dimensional_row *fold_row,
    int fold
) {
    size_t i;
    for (i = 0; i < count; i++) {
        const rg_cross_dimensional_row *full_row = &full_rows[i];
        if (strcmp(full_row->dimension, fold_row->dimension) == 0 &&
            strcmp(full_row->value, fold_row->value) == 0 &&
            full_row->position_offset == fold_row->position_offset &&
            full_row->context_is_target == fold_row->context_is_target &&
            full_row->dimension_from_environment == fold_row->dimension_from_environment &&
            context_equal(&full_row->environment, &fold_row->environment)) {
            if (items[i].last_fold != fold) {
                items[i].folds++;
                items[i].last_fold = fold;
            }
            return &items[i];
        }
    }
    return 0;
}

static const char *segment_dimension_value(const rg_segment *segment, const char *dimension) {
    if (strcmp(dimension, "tone") == 0) {
        return segment->tone == 0 ? "" : segment->tone;
    }
    if (strcmp(dimension, "length") == 0) {
        return segment->length == 0 ? "" : segment->length;
    }
    if (strcmp(dimension, "stress") == 0) {
        return segment->stress == 0 ? "" : segment->stress;
    }
    return "";
}

static void score_binary_prediction(
    predictive_score_accumulator *accumulator,
    double probability,
    int actual_holds,
    double weight,
    const rg_train_options *options
) {
    double values[2];
    predictive_probabilities prediction;
    if (probability < RG_PREDICTIVE_FLOOR) {
        probability = RG_PREDICTIVE_FLOOR;
    }
    if (probability > 1.0 - RG_PREDICTIVE_FLOOR) {
        probability = 1.0 - RG_PREDICTIVE_FLOOR;
    }
    memset(&prediction, 0, sizeof(prediction));
    values[0] = probability;
    values[1] = 1.0 - probability;
    prediction.values = values;
    prediction.count = 2;
    prediction.actual_index = actual_holds ? 0 : 1;
    prediction.best_index = values[0] >= values[1] ? 0 : 1;
    prediction.best_probability = values[prediction.best_index];
    score_prediction(accumulator, &prediction, weight,
                     options->predictive_top_k,
                     options->predictive_abstention_threshold);
}

static rg_status score_cross_dimensional_rules(
    const rg_pairwise_model *fold_model,
    const rg_pairwise_model *full_model,
    predictive_rule_accumulator *accumulators,
    size_t accumulator_count,
    const rg_link *link,
    const rg_context_spec *target_context,
    const rg_form *source,
    size_t source_position,
    const rg_form *target,
    size_t target_position,
    double weight,
    const rg_train_options *options,
    int fold
) {
    size_t row_i;
    for (row_i = 0; row_i < fold_model->cross_dimensional_count; row_i++) {
        const rg_cross_dimensional_row *row = &fold_model->cross_dimensional_rows[row_i];
        /* Both orientations are published, so the environment is read from
         * whichever side the row states it over and the conditioned dimension
         * from the other. */
        const rg_context_spec *environment_context =
            row->context_is_target ? target_context : &link->context;
        /* Lect-internal rules read the conditioned dimension from the same form
         * that states the environment; cross-lect rules read it from the other.
         * Getting this wrong scores the rule against a form it makes no claim
         * about. */
        const rg_form *conditioned_form = row->dimension_from_environment
            ? (row->context_is_target ? target : source)
            : (row->context_is_target ? source : target);
        size_t conditioned_start = row->dimension_from_environment
            ? (row->context_is_target ? target_position : source_position)
            : (row->context_is_target ? source_position : target_position);
        bool subset = false;
        int actual_position;
        const char *actual;
        double conditioned_probability;
        double unconditioned_probability;
        double all_count;
        double all_total;
        predictive_rule_accumulator *accumulator;
        if (environment_context == 0) {
            continue;
        }
        if (rg_context_spec_is_subset(&row->environment, environment_context, &subset) != RG_OK ||
            !subset) {
            continue;
        }
        actual_position = (int)conditioned_start + row->position_offset;
        if (actual_position < 0 || (size_t)actual_position >= conditioned_form->segment_count) {
            continue;
        }
        accumulator = cross_dimensional_accumulator_for_fold_row(
            accumulators, accumulator_count, full_model->cross_dimensional_rows, row, fold);
        if (accumulator == 0) {
            continue;
        }
        actual = segment_dimension_value(&conditioned_form->segments[actual_position],
                                         row->dimension);
        conditioned_probability = (row->count + 1.0) / (row->source_count + 2.0);
        all_count = row->count + row->contrast_count;
        all_total = row->source_count + row->contrast_source_count;
        unconditioned_probability = (all_count + 1.0) / (all_total + 2.0);
        score_binary_prediction(&accumulator->conditioned,
                                conditioned_probability,
                                strcmp(actual, row->value) == 0,
                                weight, options);
        score_binary_prediction(&accumulator->unconditioned,
                                unconditioned_probability,
                                strcmp(actual, row->value) == 0,
                                weight, options);
    }
    return RG_OK;
}

static int fold_rule_applies(
    const rg_conditioned_segment_count_row *row,
    const char *source,
    const rg_context_spec *source_context,
    const rg_context_spec *target_context
) {
    const rg_context_spec *against;
    bool subset = false;
    if (strcmp(row->source, source) != 0) {
        return 0;
    }
    against = row->context_is_target ? target_context : source_context;
    return against != 0 &&
        rg_context_spec_is_subset(&row->context, against, &subset) == RG_OK && subset;
}

static void predictive_probabilities_clear(predictive_probabilities *prediction) {
    free(prediction->values);
    memset(prediction, 0, sizeof(*prediction));
    /* The analyzer loses the allocation identity when callers index and later
     * clear a buffer array. */
    /* NOLINTNEXTLINE(clang-analyzer-unix.Malloc) */
}

static const predictive_direction_model *direction_model_for(
    const predictive_direction_model *items,
    size_t count,
    const char *source_lect,
    const char *target_lect
) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (strcmp(items[i].source_lect, source_lect) == 0 &&
            strcmp(items[i].target_lect, target_lect) == 0) {
            return &items[i];
        }
    }
    return 0;
}

static void direction_models_free(predictive_direction_model *items, size_t count) {
    size_t i;
    for (i = 0; i < count; i++) {
        rg_pairwise_model_free(items[i].model);
    }
    free(items);
}

static rg_status train_direction_models_for_fold(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const size_t *folds,
    size_t heldout_fold,
    const rg_multi_model *full_model,
    const rg_train_options *options,
    predictive_direction_model **out,
    size_t *out_count
) {
    predictive_direction_model *items = 0;
    size_t count = 0;
    size_t cap = 0;
    size_t a;
    rg_status status = RG_OK;
    *out = 0;
    *out_count = 0;
    for (a = 0; a < full_model->lect_count; a++) {
        size_t b;
        for (b = 0; b < full_model->lect_count; b++) {
            rg_form_pair *training = 0;
            size_t training_count = 0;
            rg_pairwise_model *pair_model = 0;
            rg_train_options nested = *options;
            predictive_direction_model *next;
            if (a == b) {
                continue;
            }
            status = form_pairs_for_fold(cognates, cognate_count, folds, heldout_fold,
                                         full_model->lect_ids[a], full_model->lect_ids[b], 1,
                                         &training, &training_count);
            if (status != RG_OK) {
                break;
            }
            if (training_count == 0) {
                free(training);
                continue;
            }
            nested.bootstrap_n = 0;
            nested.permutation_count = 0;
            nested.tune_search_penalty = 0;
            nested.predictive_folds = 0;
            nested.progress = 0;
            nested.progress_user_data = 0;
            status = rg_train_pairwise(ctx, training, training_count, &nested, &pair_model);
            free(training);
            if (status != RG_OK) {
                break;
            }
            if (count == cap) {
                size_t next_cap = cap == 0 ? 8 : cap * 2;
                next = (predictive_direction_model *)realloc(items,
                                                              next_cap * sizeof(*next));
                if (next == 0) {
                    rg_pairwise_model_free(pair_model);
                    status = RG_ERR_OOM;
                    break;
                }
                items = next;
                cap = next_cap;
            }
            items[count].source_lect = full_model->lect_ids[a];
            items[count].target_lect = full_model->lect_ids[b];
            items[count].model = pair_model;
            count++;
        }
        if (status != RG_OK) {
            break;
        }
    }
    if (status != RG_OK) {
        direction_models_free(items, count);
        return status;
    }
    *out = items;
    *out_count = count;
    return RG_OK;
}

static rg_status target_union_inventory(
    const predictive_direction_model *models,
    size_t model_count,
    const char *target_lect,
    predictive_inventory *out
) {
    size_t cap = 0;
    size_t i;
    memset(out, 0, sizeof(*out));
    for (i = 0; i < model_count; i++) {
        predictive_inventory one;
        size_t j;
        rg_status status;
        if (strcmp(models[i].target_lect, target_lect) != 0) {
            continue;
        }
        status = predictive_inventory_build(models[i].model, &one);
        if (status != RG_OK) {
            free(out->items);
            memset(out, 0, sizeof(*out));
            return status;
        }
        for (j = 0; j < one.count; j++) {
            size_t k;
            int seen = 0;
            for (k = 0; k < out->count; k++) {
                if (strcmp(out->items[k], one.items[j]) == 0) {
                    seen = 1;
                    break;
                }
            }
            if (!seen) {
                if (out->count == cap) {
                    size_t next_cap = cap == 0 ? 16 : cap * 2;
                    const char **next = (const char **)realloc(
                        out->items, next_cap * sizeof(*next));
                    if (next == 0) {
                        free(one.items);
                        free(out->items);
                        memset(out, 0, sizeof(*out));
                        return RG_ERR_OOM;
                    }
                    out->items = next;
                    cap = next_cap;
                }
                out->items[out->count++] = one.items[j];
            }
        }
        free(one.items);
    }
    if (out->count > 1) {
        qsort(out->items, out->count, sizeof(*out->items), string_pointer_compare);
    }
    return RG_OK;
}

static rg_status score_leave_one_lect_out_target(
    const rg_context *ctx,
    const rg_cognate_set *cognate,
    const rg_cognate_form *target_form,
    const predictive_direction_model *models,
    size_t model_count,
    const rg_train_options *options,
    predictive_score_accumulator *conditioned,
    predictive_score_accumulator *unconditioned,
    size_t *case_count
) {
    predictive_inventory inventory;
    size_t category_count;
    double *conditioned_logs = 0;
    double *unconditioned_logs = 0;
    size_t *predictor_counts = 0;
    rg_context_spec *target_contexts = 0;
    size_t target_context_count = 0;
    size_t source_i;
    int scored_case = 0;
    rg_status status;

    status = target_union_inventory(models, model_count, target_form->lect_id, &inventory);
    if (status != RG_OK || inventory.count == 0) {
        free(inventory.items);
        return status;
    }
    category_count = inventory.count + 1;
    conditioned_logs = (double *)calloc(target_form->form.segment_count * category_count,
                                         sizeof(*conditioned_logs));
    unconditioned_logs = (double *)calloc(target_form->form.segment_count * category_count,
                                           sizeof(*unconditioned_logs));
    predictor_counts = (size_t *)calloc(target_form->form.segment_count,
                                         sizeof(*predictor_counts));
    if (conditioned_logs == 0 || unconditioned_logs == 0 || predictor_counts == 0) {
        free(conditioned_logs);
        free(unconditioned_logs);
        free(predictor_counts);
        free(inventory.items);
        return RG_ERR_OOM;
    }
    status = rg_form_position_contexts_internal(ctx, &target_form->form,
                                                &target_contexts, &target_context_count);
    if (status != RG_OK) {
        free(conditioned_logs);
        free(unconditioned_logs);
        free(predictor_counts);
        free(inventory.items);
        return status;
    }
    for (source_i = 0; source_i < cognate->form_count; source_i++) {
        const rg_cognate_form *source_form = &cognate->forms[source_i];
        const predictive_direction_model *direction;
        rg_alignment *alignment = 0;
        size_t target_position = 0;
        size_t link_i;
        if (source_form == target_form) {
            continue;
        }
        direction = direction_model_for(models, model_count, source_form->lect_id,
                                        target_form->lect_id);
        if (direction == 0) {
            continue;
        }
        /* Prediction is evaluated per reflex. Letting a promoted chunk absorb
         * a changed segment makes that segment disappear from the denominator
         * and rewards the model for declining its hardest cases. */
        status = rg_align_forms_with_model(ctx, direction->model, options,
                                           &source_form->form, &target_form->form,
                                           1, &alignment);
        if (status != RG_OK) {
            break;
        }
        for (link_i = 0; link_i < rg_alignment_link_count(alignment); link_i++) {
            const rg_link *link = rg_alignment_link_at(alignment, link_i);
            size_t position = target_position;
            target_position += link->target_count;
            if (link->source_count == 1 && link->target_count == 1 &&
                position < target_context_count) {
                predictive_probabilities predictions[2];
                size_t category;
                memset(predictions, 0, sizeof(predictions));
                status = prediction_probabilities(
                    ctx, direction->model, &inventory,
                    link->source[0].grapheme, link->target[0].grapheme,
                    &link->context, &target_contexts[position], PREDICT_CONDITIONED,
                    options->temperature, &predictions[0]);
                if (status == RG_OK) {
                    status = prediction_probabilities(
                        ctx, direction->model, &inventory,
                        link->source[0].grapheme, link->target[0].grapheme,
                        &link->context, &target_contexts[position], PREDICT_UNCONDITIONED,
                        options->temperature, &predictions[1]);
                }
                if (status != RG_OK) {
                    predictive_probabilities_clear(&predictions[0]);
                    predictive_probabilities_clear(&predictions[1]);
                    break;
                }
                for (category = 0; category < category_count; category++) {
                    conditioned_logs[position * category_count + category] +=
                        log(predictions[0].values[category] < RG_PREDICTIVE_FLOOR
                            ? RG_PREDICTIVE_FLOOR : predictions[0].values[category]);
                    unconditioned_logs[position * category_count + category] +=
                        log(predictions[1].values[category] < RG_PREDICTIVE_FLOOR
                            ? RG_PREDICTIVE_FLOOR : predictions[1].values[category]);
                }
                predictor_counts[position]++;
                predictive_probabilities_clear(&predictions[0]);
                predictive_probabilities_clear(&predictions[1]);
            }
        }
        rg_alignment_free(alignment);
        if (status != RG_OK) {
            break;
        }
    }
    if (status == RG_OK) {
        size_t position;
        for (position = 0; position < target_form->form.segment_count; position++) {
            predictive_probabilities pooled[2];
            size_t category;
            size_t actual_index = inventory.count;
            double totals[2] = {0.0, 0.0};
            if (predictor_counts[position] < 2) {
                continue;
            }
            memset(pooled, 0, sizeof(pooled));
            for (category = 0; category < inventory.count; category++) {
                if (strcmp(inventory.items[category],
                           target_form->form.segments[position].grapheme) == 0) {
                    actual_index = category;
                    break;
                }
            }
            for (category = 0; category < 2; category++) {
                size_t value_i;
                pooled[category].values = (double *)calloc(category_count, sizeof(double));
                if (pooled[category].values == 0) {
                    status = RG_ERR_OOM;
                    break;
                }
                pooled[category].count = category_count;
                pooled[category].actual_index = actual_index;
                pooled[category].unseen = actual_index == inventory.count;
                for (value_i = 0; value_i < category_count; value_i++) {
                    const double *logs = category == 0 ? conditioned_logs : unconditioned_logs;
                    pooled[category].values[value_i] = exp(
                        logs[position * category_count + value_i] /
                        (double)predictor_counts[position]);
                    totals[category] += pooled[category].values[value_i];
                }
                for (value_i = 0; value_i < category_count; value_i++) {
                    pooled[category].values[value_i] /= totals[category];
                    if (value_i == 0 || pooled[category].values[value_i] >
                                          pooled[category].values[pooled[category].best_index]) {
                        pooled[category].best_index = value_i;
                    }
                }
                pooled[category].best_probability =
                    pooled[category].values[pooled[category].best_index];
            }
            if (status == RG_OK) {
                double weight = cognate_weight(cognate);
                score_prediction(conditioned, &pooled[0], weight,
                                 options->predictive_top_k,
                                 options->predictive_abstention_threshold);
                score_prediction(unconditioned, &pooled[1], weight,
                                 options->predictive_top_k,
                                 options->predictive_abstention_threshold);
                scored_case = 1;
            }
            predictive_probabilities_clear(&pooled[0]);
            predictive_probabilities_clear(&pooled[1]);
            if (status != RG_OK) {
                break;
            }
        }
    }
    if (scored_case) {
        (*case_count)++;
    }
    rg_context_spec_array_free_internal(target_contexts, target_context_count);
    free(conditioned_logs);
    free(unconditioned_logs);
    free(predictor_counts);
    free(inventory.items);
    return status;
}

static rg_status score_leave_one_lect_out_fold(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const size_t *folds,
    size_t heldout_fold,
    const rg_multi_model *model,
    const rg_train_options *options,
    predictive_score_accumulator *conditioned,
    predictive_score_accumulator *unconditioned,
    size_t *case_count
) {
    predictive_direction_model *models = 0;
    size_t model_count = 0;
    size_t cognate_i;
    rg_status status;
    if (model->lect_count < 3) {
        return RG_OK;
    }
    status = train_direction_models_for_fold(ctx, cognates, cognate_count, folds,
                                             heldout_fold, model, options,
                                             &models, &model_count);
    if (status != RG_OK) {
        return status;
    }
    for (cognate_i = 0; cognate_i < cognate_count; cognate_i++) {
        size_t target_i;
        if (folds[cognate_i] != heldout_fold || cognates[cognate_i].form_count < 3 ||
            cognate_weight(&cognates[cognate_i]) <= 0.0) {
            continue;
        }
        for (target_i = 0; target_i < cognates[cognate_i].form_count; target_i++) {
            status = score_leave_one_lect_out_target(
                ctx, &cognates[cognate_i], &cognates[cognate_i].forms[target_i],
                models, model_count, options, conditioned, unconditioned, case_count);
            if (status != RG_OK) {
                break;
            }
        }
        if (status != RG_OK) {
            break;
        }
    }
    direction_models_free(models, model_count);
    return status;
}

static rg_status score_one_pair(
    const rg_context *ctx,
    const rg_pairwise_model *fold_model,
    const rg_form *source,
    const rg_form *target,
    double weight,
    const rg_train_options *options,
    predictive_score_accumulator accumulators[5],
    const rg_pairwise_model *full_model,
    predictive_rule_accumulator *segment_rule_accumulators,
    size_t segment_rule_accumulator_count,
    predictive_rule_accumulator *cross_dimensional_accumulators,
    size_t cross_dimensional_accumulator_count,
    int fold,
    size_t *unscored_spans
) {
    predictive_inventory inventory;
    rg_alignment *alignment = 0;
    rg_context_spec *target_contexts = 0;
    size_t target_context_count = 0;
    size_t target_position = 0;
    size_t source_position = 0;
    size_t link_i;
    rg_status status;

    status = predictive_inventory_build(fold_model, &inventory);
    if (status != RG_OK) {
        return status;
    }
    /* See the leave-one-lect-out path above: learned chunks still affected
     * training, but a held-out categorical reflex must remain an observation. */
    status = rg_align_forms_with_model(ctx, fold_model, options, source, target,
                                       1, &alignment);
    if (status == RG_OK) {
        status = rg_form_position_contexts_internal(ctx, target, &target_contexts,
                                                    &target_context_count);
    }
    if (status != RG_OK) {
        free(inventory.items);
        rg_alignment_free(alignment);
        return status;
    }
    for (link_i = 0; link_i < rg_alignment_link_count(alignment); link_i++) {
        const rg_link *link = rg_alignment_link_at(alignment, link_i);
        size_t link_target_position = target_position;
        size_t link_source_position = source_position;
        const rg_context_spec *target_context =
            link->target_count == 1 && target_position < target_context_count
                ? &target_contexts[target_position] : 0;
        predictive_probabilities predictions[5];
        size_t kind;
        memset(predictions, 0, sizeof(predictions));
        target_position += link->target_count;
        source_position += link->source_count;
        if (link->source_count != 1 || link->target_count != 1) {
            (*unscored_spans)++;
            continue;
        }
        for (kind = 0; kind < 5; kind++) {
            status = prediction_probabilities(
                ctx, fold_model, &inventory,
                link->source[0].grapheme, link->target[0].grapheme,
                &link->context, target_context, (predictive_model_kind)kind,
                options->temperature, &predictions[kind]);
            if (status != RG_OK) {
                size_t clear_i;
                for (clear_i = 0; clear_i <= kind; clear_i++) {
                    predictive_probabilities_clear(&predictions[clear_i]);
                }
                rg_context_spec_array_free_internal(target_contexts, target_context_count);
                rg_alignment_free(alignment);
                free(inventory.items);
                return status;
            }
            score_prediction(&accumulators[kind], &predictions[kind], weight,
                             options->predictive_top_k,
                             options->predictive_abstention_threshold);
        }
        if (segment_rule_accumulators != 0) {
            size_t row_i;
            for (row_i = 0; row_i < fold_model->conditioned_segment_count_count; row_i++) {
                const rg_conditioned_segment_count_row *fold_row =
                    &fold_model->conditioned_segment_counts[row_i];
                if (fold_rule_applies(fold_row, link->source[0].grapheme,
                                      &link->context, target_context)) {
                    predictive_rule_accumulator *rule = rule_accumulator_for_fold_row(
                        segment_rule_accumulators, segment_rule_accumulator_count,
                        fold_row, fold);
                    if (rule != 0) {
                        score_prediction(&rule->conditioned, &predictions[PREDICT_CONDITIONED],
                                         weight, options->predictive_top_k,
                                         options->predictive_abstention_threshold);
                        score_prediction(&rule->unconditioned, &predictions[PREDICT_UNCONDITIONED],
                                         weight, options->predictive_top_k,
                                         options->predictive_abstention_threshold);
                    }
                }
            }
        }
        if (full_model != 0 && cross_dimensional_accumulators != 0) {
            status = score_cross_dimensional_rules(
                fold_model, full_model, cross_dimensional_accumulators,
                cross_dimensional_accumulator_count, link, target_context,
                source, link_source_position, target, link_target_position,
                weight, options, fold);
            if (status != RG_OK) {
                for (kind = 0; kind < 5; kind++) {
                    predictive_probabilities_clear(&predictions[kind]);
                }
                rg_context_spec_array_free_internal(target_contexts, target_context_count);
                rg_alignment_free(alignment);
                free(inventory.items);
                return status;
            }
        }
        for (kind = 0; kind < 5; kind++) {
            predictive_probabilities_clear(&predictions[kind]);
        }
    }
    rg_context_spec_array_free_internal(target_contexts, target_context_count);
    rg_alignment_free(alignment);
    free(inventory.items);
    return RG_OK;
}

static rg_status form_pairs_for_fold(
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const size_t *folds,
    size_t heldout_fold,
    const char *source_lect,
    const char *target_lect,
    int training,
    rg_form_pair **out,
    size_t *out_count
) {
    rg_form_pair *pairs = 0;
    size_t count = 0;
    size_t cap = 0;
    size_t i;
    *out = 0;
    *out_count = 0;
    for (i = 0; i < cognate_count; i++) {
        const rg_form *source;
        const rg_form *target;
        rg_form_pair *next;
        if ((folds[i] == heldout_fold) == training) {
            continue;
        }
        source = form_for_lect(&cognates[i], source_lect);
        target = form_for_lect(&cognates[i], target_lect);
        if (source == 0 || target == 0 || cognate_weight(&cognates[i]) <= 0.0) {
            continue;
        }
        if (count == cap) {
            size_t next_cap = cap == 0 ? 16 : cap * 2;
            next = (rg_form_pair *)realloc(pairs, next_cap * sizeof(*next));
            if (next == 0) {
                free(pairs);
                return RG_ERR_OOM;
            }
            pairs = next;
            cap = next_cap;
        }
        pairs[count].source = *source;
        pairs[count].target = *target;
        pairs[count].weight = cognate_weight(&cognates[i]);
        count++;
    }
    *out = pairs;
    *out_count = count;
    return RG_OK;
}

static predictive_rule_accumulator *rule_accumulators_build(
    rg_pairwise_model *model,
    size_t *out_count
) {
    predictive_rule_accumulator *items;
    size_t i;
    *out_count = model == 0 ? 0 : model->conditioned_segment_count_count;
    if (*out_count == 0) {
        return 0;
    }
    items = (predictive_rule_accumulator *)calloc(*out_count, sizeof(*items));
    if (items == 0) {
        *out_count = 0;
        return 0;
    }
    for (i = 0; i < *out_count; i++) {
        items[i].evidence = &model->conditioned_segment_counts[i].evidence;
        items[i].last_fold = -1;
    }
    return items;
}

static predictive_rule_accumulator *cross_dimensional_accumulators_build(
    rg_pairwise_model *model,
    size_t *out_count
) {
    predictive_rule_accumulator *items;
    size_t i;
    *out_count = model == 0 ? 0 : model->cross_dimensional_count;
    if (*out_count == 0) {
        return 0;
    }
    items = (predictive_rule_accumulator *)calloc(*out_count, sizeof(*items));
    if (items == 0) {
        *out_count = 0;
        return 0;
    }
    for (i = 0; i < *out_count; i++) {
        items[i].evidence = &model->cross_dimensional_rows[i].evidence;
        items[i].last_fold = -1;
    }
    return items;
}

static void predictive_evidence_descriptive(rg_rule_evidence *evidence) {
    memset(&evidence->predictive, 0, sizeof(evidence->predictive));
    evidence->predictive.status = RG_PREDICTIVE_DESCRIPTIVE_ONLY;
    evidence->predictive.observation_unit = RG_OBSERVATION_UNIT_DEPENDENCY_COMPONENT;
}

static void mark_rules_descriptive(rg_multi_model *model) {
    size_t i;
    for (i = 0; i < model->pair_model_count; i++) {
        rg_pairwise_model *pair = model->pair_models[i].model;
        size_t j;
        for (j = 0; j < pair->conditioned_segment_count_count; j++) {
            predictive_evidence_descriptive(&pair->conditioned_segment_counts[j].evidence);
        }
        for (j = 0; j < pair->cross_dimensional_count; j++) {
            predictive_evidence_descriptive(&pair->cross_dimensional_rows[j].evidence);
        }
    }
    for (i = 0; i < model->conditioned_class_count; i++) {
        predictive_evidence_descriptive(&model->conditioned_classes[i].evidence);
    }
    for (i = 0; i < model->cross_dimensional_count; i++) {
        predictive_evidence_descriptive(&model->cross_dimensional_rows[i].rule.evidence);
    }
}

static void finish_rule_accumulators(
    predictive_rule_accumulator *items,
    size_t count
) {
    size_t i;
    for (i = 0; i < count; i++) {
        rg_predictive_evidence *out = &items[i].evidence->predictive;
        out->folds = items[i].folds;
        out->conditioned = predictive_score_finish(&items[i].conditioned);
        out->unconditioned = predictive_score_finish(&items[i].unconditioned);
        out->log_loss_gain = out->unconditioned.log_loss - out->conditioned.log_loss;
        if (out->folds >= 2 &&
            out->conditioned.observation_count >= RG_PREDICTIVE_RULE_MIN_OBSERVATIONS) {
            out->status = out->log_loss_gain > 0.0
                ? RG_PREDICTIVE_CONFIRMED : RG_PREDICTIVE_NOT_CONFIRMED;
        }
    }
}

static rg_pairwise_model *full_pair_model(
    rg_multi_model *model,
    const char *source_lect,
    const char *target_lect
) {
    size_t i;
    for (i = 0; i < model->pair_model_count; i++) {
        if (strcmp(model->pair_models[i].lect_a, source_lect) == 0 &&
            strcmp(model->pair_models[i].lect_b, target_lect) == 0) {
            return model->pair_models[i].model;
        }
    }
    return 0;
}

static void propagate_two_lect_class_evidence(rg_multi_model *model) {
    size_t class_i;
    for (class_i = 0; class_i < model->conditioned_class_count; class_i++) {
        rg_multi_class_row *class_row = &model->conditioned_classes[class_i];
        rg_pairwise_model *pair;
        size_t pair_i;
        if (class_row->segment_count != 2 || class_row->contexts == 0) {
            continue;
        }
        pair = full_pair_model(model, class_row->lect_ids[0], class_row->lect_ids[1]);
        if (pair == 0) {
            continue;
        }
        for (pair_i = 0; pair_i < pair->conditioned_segment_count_count; pair_i++) {
            const rg_conditioned_segment_count_row *row =
                &pair->conditioned_segment_counts[pair_i];
            size_t context_index = row->context_is_target ? 1 : 0;
            if (strcmp(row->source, class_row->graphemes[0]) == 0 &&
                strcmp(row->target, class_row->graphemes[1]) == 0 &&
                context_equal(&row->context, &class_row->contexts[context_index])) {
                class_row->evidence.predictive = row->evidence.predictive;
                break;
            }
        }
    }
}

static void propagate_cross_dimensional_evidence(rg_multi_model *model) {
    size_t lifted_i;
    for (lifted_i = 0; lifted_i < model->cross_dimensional_count; lifted_i++) {
        rg_multi_cross_dimensional_row *lifted = &model->cross_dimensional_rows[lifted_i];
        rg_pairwise_model *pair = full_pair_model(model, lifted->source_lect,
                                                   lifted->target_lect);
        size_t pair_i;
        if (pair == 0) {
            continue;
        }
        for (pair_i = 0; pair_i < pair->cross_dimensional_count; pair_i++) {
            const rg_cross_dimensional_row *row = &pair->cross_dimensional_rows[pair_i];
            if (strcmp(row->dimension, lifted->rule.dimension) == 0 &&
                strcmp(row->value, lifted->rule.value) == 0 &&
                row->position_offset == lifted->rule.position_offset &&
                row->context_is_target == lifted->rule.context_is_target &&
                context_equal(&row->environment, &lifted->rule.environment)) {
                lifted->rule.evidence.predictive = row->evidence.predictive;
                break;
            }
        }
    }
}

static predictive_pair_rule_accumulators *pair_rule_accumulators_for(
    predictive_pair_rule_accumulators *items,
    size_t count,
    rg_pairwise_model *model
) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (items[i].model == model) {
            return &items[i];
        }
    }
    return 0;
}

static void pair_rule_accumulators_free(
    predictive_pair_rule_accumulators *items,
    size_t count
) {
    size_t i;
    for (i = 0; i < count; i++) {
        free(items[i].segment_items);
        free(items[i].cross_dimensional_items);
    }
    free(items);
}

rg_status rg_predictive_evaluate_internal(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    rg_multi_model *model
) {
    size_t *folds = 0;
    size_t fold_count = 0;
    size_t group_count = 0;
    predictive_score_accumulator overall[5];
    predictive_score_accumulator leave_one_lect_out[2];
    predictive_pair_rule_accumulators *pair_rules = 0;
    size_t fold;
    size_t pair_rule_count = 0;
    rg_status status;

    memset(overall, 0, sizeof(overall));
    memset(leave_one_lect_out, 0, sizeof(leave_one_lect_out));
    memset(&model->fit.predictive, 0, sizeof(model->fit.predictive));
    model->fit.predictive_abstention_threshold = options->predictive_abstention_threshold;
    model->fit.predictive_top_k = options->predictive_top_k;
    if (options->predictive_folds == 0) {
        return RG_OK;
    }
    mark_rules_descriptive(model);
    model->fit.predictive.status = RG_PREDICTIVE_DESCRIPTIVE_ONLY;
    model->fit.predictive.observation_unit = RG_OBSERVATION_UNIT_DEPENDENCY_COMPONENT;
    model->fit.predictive_folds_requested = (size_t)options->predictive_folds;
    status = assign_dependency_folds(cognates, cognate_count, options->predictive_folds,
                                     options->predictive_min_groups,
                                     options->predictive_seed, &folds, &fold_count,
                                     &group_count);
    model->fit.predictive_group_count = group_count;
    if (status != RG_OK || fold_count == 0) {
        free(folds);
        return status;
    }
    model->fit.predictive.folds = fold_count;
    if (model->pair_model_count > 0) {
        size_t i;
        pair_rules = (predictive_pair_rule_accumulators *)calloc(
            model->pair_model_count, sizeof(*pair_rules));
        if (pair_rules == 0) {
            free(folds);
            return RG_ERR_OOM;
        }
        pair_rule_count = model->pair_model_count;
        for (i = 0; i < pair_rule_count; i++) {
            pair_rules[i].model = model->pair_models[i].model;
            pair_rules[i].segment_items = rule_accumulators_build(
                pair_rules[i].model, &pair_rules[i].segment_count);
            if (pair_rules[i].model->conditioned_segment_count_count > 0 &&
                pair_rules[i].segment_items == 0) {
                size_t j;
                for (j = 0; j < i; j++) {
                    free(pair_rules[j].segment_items);
                    free(pair_rules[j].cross_dimensional_items);
                }
                free(pair_rules);
                free(folds);
                return RG_ERR_OOM;
            }
            pair_rules[i].cross_dimensional_items = cross_dimensional_accumulators_build(
                pair_rules[i].model, &pair_rules[i].cross_dimensional_count);
            if (pair_rules[i].model->cross_dimensional_count > 0 &&
                pair_rules[i].cross_dimensional_items == 0) {
                size_t j;
                for (j = 0; j <= i; j++) {
                    free(pair_rules[j].segment_items);
                    free(pair_rules[j].cross_dimensional_items);
                }
                free(pair_rules);
                free(folds);
                return RG_ERR_OOM;
            }
        }
    }
    for (fold = 0; fold < fold_count; fold++) {
        size_t a;
        for (a = 0; a < model->lect_count; a++) {
            size_t b;
            for (b = 0; b < model->lect_count; b++) {
                rg_form_pair *training = 0;
                rg_form_pair *heldout = 0;
                size_t training_count = 0;
                size_t heldout_count = 0;
                rg_pairwise_model *fold_model = 0;
                rg_pairwise_model *full_model = 0;
                predictive_pair_rule_accumulators *pair_rule = 0;
                size_t rule_accumulator_count = 0;
                size_t i;
                rg_train_options nested = *options;
                if (a == b) {
                    continue;
                }
                status = form_pairs_for_fold(cognates, cognate_count, folds, fold,
                                             model->lect_ids[a], model->lect_ids[b], 1,
                                             &training, &training_count);
                if (status == RG_OK) {
                    status = form_pairs_for_fold(cognates, cognate_count, folds, fold,
                                                 model->lect_ids[a], model->lect_ids[b], 0,
                                                 &heldout, &heldout_count);
                }
                if (status != RG_OK) {
                    free(training);
                    free(heldout);
                    {
                        size_t clear_i;
                        for (clear_i = 0; clear_i < pair_rule_count; clear_i++) {
                            free(pair_rules[clear_i].segment_items);
                            free(pair_rules[clear_i].cross_dimensional_items);
                        }
                    }
                    free(pair_rules);
                    free(folds);
                    return status;
                }
                if (training_count == 0 || heldout_count == 0) {
                    free(training);
                    free(heldout);
                    continue;
                }
                nested.bootstrap_n = 0;
                nested.permutation_count = 0;
                nested.tune_search_penalty = 0;
                nested.predictive_folds = 0;
                nested.progress = 0;
                nested.progress_user_data = 0;
                status = rg_train_pairwise(ctx, training, training_count, &nested, &fold_model);
                free(training);
                if (status != RG_OK) {
                    free(heldout);
                    pair_rule_accumulators_free(pair_rules, pair_rule_count);
                    free(folds);
                    return status;
                }
                full_model = full_pair_model(model, model->lect_ids[a], model->lect_ids[b]);
                if (full_model != 0) {
                    pair_rule = pair_rule_accumulators_for(pair_rules, pair_rule_count,
                                                           full_model);
                    rule_accumulator_count = pair_rule == 0 ? 0 : pair_rule->segment_count;
                }
                for (i = 0; i < heldout_count; i++) {
                    status = score_one_pair(ctx, fold_model,
                                            &heldout[i].source, &heldout[i].target,
                                            heldout[i].weight, &nested, overall,
                                            full_model,
                                            pair_rule == 0 ? 0 : pair_rule->segment_items,
                                            rule_accumulator_count,
                                            pair_rule == 0 ? 0 : pair_rule->cross_dimensional_items,
                                            pair_rule == 0 ? 0 : pair_rule->cross_dimensional_count,
                                            (int)fold,
                                            &model->fit.predictive_unscored_span_count);
                    if (status != RG_OK) {
                        break;
                    }
                }
                rg_pairwise_model_free(fold_model);
                free(heldout);
                if (status != RG_OK) {
                    size_t clear_i;
                    for (clear_i = 0; clear_i < pair_rule_count; clear_i++) {
                        free(pair_rules[clear_i].segment_items);
                        free(pair_rules[clear_i].cross_dimensional_items);
                    }
                    free(pair_rules);
                    free(folds);
                    return status;
                }
            }
        }
        status = score_leave_one_lect_out_fold(
            ctx, cognates, cognate_count, folds, fold, model, options,
            &leave_one_lect_out[0], &leave_one_lect_out[1],
            &model->fit.predictive_leave_one_lect_out_cases);
        if (status != RG_OK) {
            size_t clear_i;
            for (clear_i = 0; clear_i < pair_rule_count; clear_i++) {
                free(pair_rules[clear_i].segment_items);
                free(pair_rules[clear_i].cross_dimensional_items);
            }
            free(pair_rules);
            free(folds);
            return status;
        }
    }
    free(folds);
    {
        size_t i;
        for (i = 0; i < pair_rule_count; i++) {
            finish_rule_accumulators(pair_rules[i].segment_items, pair_rules[i].segment_count);
            finish_rule_accumulators(pair_rules[i].cross_dimensional_items,
                                     pair_rules[i].cross_dimensional_count);
            free(pair_rules[i].segment_items);
            free(pair_rules[i].cross_dimensional_items);
        }
        free(pair_rules);
    }
    propagate_two_lect_class_evidence(model);
    propagate_cross_dimensional_evidence(model);
    model->fit.predictive.conditioned = predictive_score_finish(&overall[PREDICT_CONDITIONED]);
    model->fit.predictive.unconditioned = predictive_score_finish(&overall[PREDICT_UNCONDITIONED]);
    model->fit.predictive_identity = predictive_score_finish(&overall[PREDICT_IDENTITY]);
    model->fit.predictive_inventory_frequency = predictive_score_finish(&overall[PREDICT_INVENTORY]);
    model->fit.predictive_feature_distance = predictive_score_finish(&overall[PREDICT_FEATURE_DISTANCE]);
    model->fit.predictive_leave_one_lect_out_conditioned =
        predictive_score_finish(&leave_one_lect_out[0]);
    model->fit.predictive_leave_one_lect_out_unconditioned =
        predictive_score_finish(&leave_one_lect_out[1]);
    model->fit.predictive.log_loss_gain = model->fit.predictive.unconditioned.log_loss -
        model->fit.predictive.conditioned.log_loss;
    model->fit.predictive_pair_orientations = model->lect_count * (model->lect_count - 1);
    if (model->fit.predictive.conditioned.observation_count == 0) {
        model->fit.predictive.status = RG_PREDICTIVE_DESCRIPTIVE_ONLY;
    } else {
        model->fit.predictive.status = model->fit.predictive.log_loss_gain > 0.0
            ? RG_PREDICTIVE_CONFIRMED : RG_PREDICTIVE_NOT_CONFIRMED;
    }
    return RG_OK;
}
