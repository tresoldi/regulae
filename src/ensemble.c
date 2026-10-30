#include "internal.h"
#include "multilect_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Ensembles (docs/ensembles.md): refit per member on Bayesian-bootstrap
 * weights with sampled alignments. */

#define RG_ENSEMBLE_DEFAULT_MEMBERS 50

static uint64_t ensemble_mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/* A uniform draw in (0, 1]: never 0, so -log(u) is finite. */
static double ensemble_uniform(uint64_t *state) {
    *state = ensemble_mix(*state);
    return ((double)(*state >> 11) + 1.0) * (1.0 / 9007199254740992.0);
}

void rg_ensemble_options_init_defaults(rg_ensemble_options *options) {
    if (options == 0) {
        return;
    }
    options->member_count = RG_ENSEMBLE_DEFAULT_MEMBERS;
    options->seed = 20261030;
    options->alignment_temperature = 1.0;
    options->calibrate_temperature = false;
    options->calibration_folds = 5;
    options->calibration_repeats = 3;
}

static const double default_temperature_grid[RG_TEMPERATURE_GRID_DEFAULT_COUNT] = {
    0.0, 0.05, 0.1, 0.25, 0.5, 1.0
};

rg_status rg_calibrate_alignment_temperature(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    const double *grid,
    size_t grid_count,
    int folds,
    int repeats,
    rg_temperature_score *scores,
    double *best
) {
    rg_train_options base;
    size_t t;
    double best_loss = INFINITY;

    if (ctx == 0 || scores == 0 || best == 0 || (cognate_count > 0 && cognates == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    if (grid == 0) {
        grid = default_temperature_grid;
        grid_count = RG_TEMPERATURE_GRID_DEFAULT_COUNT;
    }
    if (grid_count == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    if (folds < 2) {
        folds = 5;
    }
    if (repeats < 1) {
        repeats = 1;
    }
    if (options == 0) {
        rg_train_options_init_defaults(&base);
    } else {
        base = *options;
    }
    base.permutation_count = 0;
    base.tune_search_penalty = 0;
    base.bootstrap_n = 0;
    base.progress = 0;
    base.progress_user_data = 0;
    base.predictive_folds = folds;
    *best = grid[0];
    for (t = 0; t < grid_count; t++) {
        double temperature = grid[t];
        int draws = temperature > 0.0 ? repeats : 1;
        double sum = 0.0;
        double sum_sq = 0.0;
        int r;
        if (!isfinite(temperature) || temperature < 0.0) {
            return RG_ERR_INVALID_ARGUMENT;
        }
        memset(&scores[t], 0, sizeof(scores[t]));
        scores[t].temperature = temperature;
        for (r = 0; r < draws; r++) {
            rg_train_options trial = base;
            rg_multi_model *model = 0;
            const rg_corpus_fit *fit;
            rg_status status;
            double loss;
            trial.alignment_sampling_temperature = temperature;
            trial.alignment_sampling_seed = 1 + r;
            status = train_model_internal(ctx, cognates, cognate_count, &trial, 0, &model);
            if (status != RG_OK) {
                return status;
            }
            fit = rg_multi_model_fit(model);
            if (fit == 0 || (fit->predictive.status != RG_PREDICTIVE_CONFIRMED &&
                             fit->predictive.status != RG_PREDICTIVE_NOT_CONFIRMED)) {
                rg_multi_model_free(model);
                return RG_ERR_UNSUPPORTED_OPTION;
            }
            loss = fit->predictive.conditioned.log_loss;
            scores[t].observations = fit->predictive.conditioned.observation_count;
            rg_multi_model_free(model);
            sum += loss;
            sum_sq += loss * loss;
        }
        scores[t].repeats = (size_t)draws;
        scores[t].log_loss_mean = sum / (double)draws;
        scores[t].log_loss_sd = draws > 1
            ? sqrt(fmax(0.0, (sum_sq - sum * sum / (double)draws) / (double)(draws - 1)))
            : 0.0;
        /* Ties go to the colder temperature: the grid is ascending. */
        if (scores[t].log_loss_mean < best_loss - 1e-12) {
            best_loss = scores[t].log_loss_mean;
            *best = temperature;
        }
    }
    return RG_OK;
}

const rg_temperature_score *rg_ensemble_temperature_scores(
    const rg_ensemble *ensemble,
    size_t *count
) {
    if (count != 0) {
        *count = ensemble == 0 ? 0 : ensemble->temperature_score_count;
    }
    return ensemble == 0 ? 0 : ensemble->temperature_scores;
}

static void free_strings(char **items, size_t count) {
    size_t i;
    if (items == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        free(items[i]);
    }
    free(items);
}

void rg_ensemble_free(rg_ensemble *ensemble) {
    size_t i;
    if (ensemble == 0) {
        return;
    }
    for (i = 0; i < ensemble->options.member_count; i++) {
        if (ensemble->members != 0) {
            rg_multi_model_free(ensemble->members[i]);
        }
        if (ensemble->set_weights != 0) {
            free(ensemble->set_weights[i]);
        }
        if (ensemble->set_ids != 0 && ensemble->set_counts != 0) {
            free_strings(ensemble->set_ids[i], ensemble->set_counts[i]);
        }
        if (ensemble->set_forms != 0 && ensemble->set_counts != 0) {
            free_strings(ensemble->set_forms[i], ensemble->set_counts[i]);
        }
    }
    free(ensemble->members);
    free(ensemble->set_weights);
    free(ensemble->set_ids);
    free(ensemble->set_forms);
    free(ensemble->set_counts);
    free(ensemble->member_seeds);
    free(ensemble->feature_system);
    free(ensemble->temperature_scores);
    free(ensemble);
}

/* Where a member's cognate sets come from: the caller's sets as given, or a
 * partition sampled from soft cognacy. `sets` and the forms they point to
 * stay valid until the next call or `release`. */
typedef struct member_source {
    rg_status (*build)(void *data, uint64_t *rng, const rg_cognate_set **sets, size_t *count,
                       char ***forms_out);
    void (*release)(void *data);
    void *data;
} member_source;

static rg_status fixed_build(void *data, uint64_t *rng, const rg_cognate_set **sets, size_t *count,
                             char ***forms_out) {
    const rg_cognate_set *const *fixed = (const rg_cognate_set *const *)data;
    (void)rng;
    *sets = fixed[0];
    *count = (size_t)(fixed[1] - fixed[0]);
    *forms_out = 0;
    return RG_OK;
}

static void no_release(void *data) {
    (void)data;
}

static rg_status check_options(
    const rg_train_options *options,
    const rg_ensemble_options *ensemble_options,
    rg_train_options *base,
    rg_ensemble_options *eopts
) {
    if (options == 0) {
        rg_train_options_init_defaults(base);
    } else {
        *base = *options;
    }
    if (ensemble_options == 0) {
        rg_ensemble_options_init_defaults(eopts);
    } else {
        *eopts = *ensemble_options;
    }
    if (eopts->member_count == 0 || !isfinite(eopts->alignment_temperature) ||
        eopts->alignment_temperature < 0.0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    /* NML scores counts, and Bayesian-bootstrap weights are not counts. */
    if (base->bic.split_scorer == RG_SPLIT_SCORER_MULTINOMIAL_NML) {
        return RG_ERR_UNSUPPORTED_OPTION;
    }
    return RG_OK;
}

/* Calibrates when asked; on success *calibration is owned by the caller. */
static rg_status maybe_calibrate(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *base,
    rg_ensemble_options *eopts,
    rg_temperature_score **calibration
) {
    rg_temperature_score scores[RG_TEMPERATURE_GRID_DEFAULT_COUNT];
    double chosen = eopts->alignment_temperature;
    rg_status status;
    *calibration = 0;
    if (!eopts->calibrate_temperature) {
        return RG_OK;
    }
    status = rg_calibrate_alignment_temperature(ctx, cognates, cognate_count, base, 0, 0,
                                                eopts->calibration_folds,
                                                eopts->calibration_repeats, scores, &chosen);
    if (status != RG_OK) {
        return status;
    }
    eopts->alignment_temperature = chosen;
    *calibration = (rg_temperature_score *)malloc(sizeof(scores));
    if (*calibration == 0) {
        return RG_ERR_OOM;
    }
    memcpy(*calibration, scores, sizeof(scores));
    return RG_OK;
}

/* The member loop both entry points share. */
static rg_status run_members(
    const rg_context *ctx,
    rg_train_options base,
    const rg_ensemble_options *eopts,
    rg_temperature_score *calibration,
    const member_source *source,
    int sampled_partitions,
    rg_ensemble **out
) {
    rg_ensemble *ensemble;
    rg_cognate_set *weighted = 0;
    size_t weighted_cap = 0;
    size_t m;
    rg_status status = RG_OK;
    const char *system_name = 0;

    /* What a member does not do: its spread across members is the uncertainty,
     * and the nested resampling procedures would multiply the cost by their
     * own counts. */
    base.permutation_count = 0;
    base.tune_search_penalty = 0;
    base.bootstrap_n = 0;
    base.predictive_folds = 0;
    base.progress = 0;
    base.progress_user_data = 0;
    base.alignment_sampling_temperature = eopts->alignment_temperature;

    ensemble = (rg_ensemble *)calloc(1, sizeof(*ensemble));
    if (ensemble == 0) {
        free(calibration);
        return RG_ERR_OOM;
    }
    ensemble->options = *eopts;
    ensemble->temperature_scores = calibration;
    ensemble->temperature_score_count = calibration != 0 ? RG_TEMPERATURE_GRID_DEFAULT_COUNT : 0;
    ensemble->sampled_partitions = sampled_partitions;
    ensemble->members = (rg_multi_model **)calloc(eopts->member_count, sizeof(*ensemble->members));
    ensemble->set_weights = (double **)calloc(eopts->member_count, sizeof(*ensemble->set_weights));
    ensemble->set_ids = (char ***)calloc(eopts->member_count, sizeof(*ensemble->set_ids));
    ensemble->set_forms = (char ***)calloc(eopts->member_count, sizeof(*ensemble->set_forms));
    ensemble->set_counts = (size_t *)calloc(eopts->member_count, sizeof(*ensemble->set_counts));
    ensemble->member_seeds = (int *)calloc(eopts->member_count, sizeof(*ensemble->member_seeds));
    if (ensemble->members == 0 || ensemble->set_weights == 0 || ensemble->set_ids == 0 ||
        ensemble->set_forms == 0 || ensemble->set_counts == 0 || ensemble->member_seeds == 0) {
        rg_ensemble_free(ensemble);
        return RG_ERR_OOM;
    }
    if (rg_context_system_name(ctx, &system_name) == RG_OK && system_name != 0) {
        ensemble->feature_system = rg_strdup_internal(system_name);
    }

    for (m = 0; m < eopts->member_count && status == RG_OK; m++) {
        uint64_t rng = ensemble_mix((uint64_t)(unsigned int)eopts->seed * 0x100000001B3ULL + m);
        rg_train_options member_options = base;
        const rg_cognate_set *sets = 0;
        size_t count = 0;
        char **forms = 0;
        size_t *group_of = 0;
        size_t group_count = 0;
        double *group_weight = 0;
        double total = 0.0;
        size_t g;
        size_t i;

        ensemble->member_seeds[m] = (int)(ensemble_mix(rng) & 0x7FFFFFFF);
        status = source->build(source->data, &rng, &sets, &count, &forms);
        if (status != RG_OK) {
            break;
        }
        ensemble->set_forms[m] = forms;
        ensemble->set_counts[m] = count;
        ensemble->set_ids[m] = (char **)calloc(count + 1, sizeof(char *));
        ensemble->set_weights[m] = (double *)calloc(count + 1, sizeof(double));
        if (count + 1 > weighted_cap) {
            rg_cognate_set *grown = (rg_cognate_set *)realloc(weighted, (count + 1) * sizeof(*weighted));
            if (grown == 0) {
                status = RG_ERR_OOM;
                break;
            }
            weighted = grown;
            weighted_cap = count + 1;
        }
        if (ensemble->set_ids[m] == 0 || ensemble->set_weights[m] == 0) {
            status = RG_ERR_OOM;
            break;
        }
        ensemble->unit = resolve_bootstrap_unit(sets, count, base.bootstrap_unit);
        status = observation_groups(sets, count, ensemble->unit, &group_of, &group_count);
        if (status == RG_OK) {
            group_weight = (double *)calloc(group_count + 1, sizeof(*group_weight));
            if (group_weight == 0) {
                status = RG_ERR_OOM;
            }
        }
        if (status != RG_OK) {
            free(group_of);
            break;
        }
        ensemble->group_count = group_count;
        /* Dirichlet(1, ..., 1) over the groups: normalised exponentials. */
        for (g = 0; g < group_count; g++) {
            group_weight[g] = -log(ensemble_uniform(&rng));
            total += group_weight[g];
        }
        for (i = 0; i < count; i++) {
            /* Scaled to mean 1 over the groups, so evidence floors and the BIC
             * N keep their scale on average. */
            double factor = total > 0.0
                ? group_weight[group_of[i]] * (double)group_count / total
                : 1.0;
            ensemble->set_weights[m][i] = factor;
            ensemble->set_ids[m][i] = rg_strdup_internal(sets[i].cognate_id != 0 ? sets[i].cognate_id : "");
            if (ensemble->set_ids[m][i] == 0) {
                status = RG_ERR_OOM;
            }
            weighted[i] = sets[i];
            weighted[i].confidence = sets[i].confidence * factor;
        }
        free(group_of);
        free(group_weight);
        if (status != RG_OK) {
            break;
        }
        member_options.alignment_sampling_seed = ensemble->member_seeds[m];
        status = train_model_internal(ctx, weighted, count, &member_options, 1, &ensemble->members[m]);
        if (status == RG_OK) {
            ensemble->member_count = m + 1;
        }
    }
    free(weighted);
    source->release(source->data);
    if (status != RG_OK) {
        rg_ensemble_free(ensemble);
        return status;
    }
    *out = ensemble;
    return RG_OK;
}

rg_status rg_train_ensemble(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    const rg_ensemble_options *ensemble_options,
    rg_ensemble **out
) {
    rg_train_options base;
    rg_ensemble_options eopts;
    rg_temperature_score *calibration = 0;
    const rg_cognate_set *bounds[2];
    member_source source;
    size_t i;
    rg_status status;

    if (ctx == 0 || out == 0 || (cognate_count > 0 && cognates == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    status = check_options(options, ensemble_options, &base, &eopts);
    if (status != RG_OK) {
        return status;
    }
    for (i = 0; i < cognate_count; i++) {
        if (cognates[i].confidence < 0.0 || cognates[i].confidence > 1.0) {
            return RG_ERR_INVALID_ARGUMENT;
        }
    }
    status = maybe_calibrate(ctx, cognates, cognate_count, &base, &eopts, &calibration);
    if (status != RG_OK) {
        return status;
    }
    bounds[0] = cognates;
    bounds[1] = cognates + cognate_count;
    source.build = fixed_build;
    source.release = no_release;
    source.data = (void *)bounds;
    return run_members(ctx, base, &eopts, calibration, &source, 0, out);
}

/* -- soft cognacy ------------------------------------------------------------------ */

typedef struct soft_state {
    const rg_soft_form *forms;
    size_t form_count;
    const rg_cognacy_edge *edges;
    size_t edge_count;
    double threshold; /* > 0: keep edges with weight >= threshold, no sampling */
    size_t *parent;
    rg_cognate_set *sets;
    rg_cognate_form *set_forms;
    char **ids;
    size_t set_count;
} soft_state;

static size_t find_root(size_t *parent, size_t x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

static void soft_clear(soft_state *state) {
    size_t i;
    for (i = 0; i < state->set_count; i++) {
        free(state->ids[i]);
    }
    free(state->ids);
    free(state->sets);
    free(state->set_forms);
    state->ids = 0;
    state->sets = 0;
    state->set_forms = 0;
    state->set_count = 0;
}

static void soft_release(void *data) {
    soft_state *state = (soft_state *)data;
    soft_clear(state);
    free(state->parent);
    state->parent = 0;
}

static int same_text(const char *a, const char *b) {
    return strcmp(a != 0 ? a : "", b != 0 ? b : "") == 0;
}

/* One partition: sample (or threshold) the edges, take components, and draw
 * one form per lect in each. Components with fewer than two lects are not
 * sets. */
static rg_status soft_build(void *data, uint64_t *rng, const rg_cognate_set **sets, size_t *count,
                            char ***forms_out) {
    soft_state *state = (soft_state *)data;
    size_t n = state->form_count;
    size_t *order = 0;
    size_t *chosen = 0;
    char **forms = 0;
    size_t i;
    size_t e;

    soft_clear(state);
    for (i = 0; i < n; i++) {
        state->parent[i] = i;
    }
    for (e = 0; e < state->edge_count; e++) {
        const rg_cognacy_edge *edge = &state->edges[e];
        int keep = state->threshold > 0.0
            ? edge->weight >= state->threshold
            : ensemble_uniform(rng) <= edge->weight;
        if (keep) {
            size_t ra = find_root(state->parent, edge->a);
            size_t rb = find_root(state->parent, edge->b);
            if (ra != rb) {
                state->parent[ra < rb ? rb : ra] = ra < rb ? ra : rb;
            }
        }
    }
    order = (size_t *)malloc((n + 1) * sizeof(*order));
    chosen = (size_t *)malloc((n + 1) * sizeof(*chosen));
    state->sets = (rg_cognate_set *)calloc(n + 1, sizeof(*state->sets));
    state->set_forms = (rg_cognate_form *)calloc(n + 1, sizeof(*state->set_forms));
    state->ids = (char **)calloc(n + 1, sizeof(*state->ids));
    forms = (char **)calloc(n + 1, sizeof(*forms));
    if (order == 0 || chosen == 0 || state->sets == 0 || state->set_forms == 0 ||
        state->ids == 0 || forms == 0) {
        free(order);
        free(chosen);
        free(forms);
        return RG_ERR_OOM;
    }
    {
        size_t used = 0;
        size_t root;
        for (root = 0; root < n; root++) {
            size_t members = 0;
            size_t lects = 0;
            size_t k;
            if (find_root(state->parent, root) != root) {
                continue;
            }
            for (i = 0; i < n; i++) {
                if (find_root(state->parent, i) == root) {
                    order[members++] = i;
                }
            }
            /* One form per lect: a uniform draw among that lect's forms. */
            for (i = 0; i < members; i++) {
                const char *lect = state->forms[order[i]].lect_id;
                size_t seen = 0;
                size_t pick = order[i];
                int first = 1;
                for (k = 0; k < i; k++) {
                    if (same_text(state->forms[order[k]].lect_id, lect)) {
                        first = 0;
                        break;
                    }
                }
                if (!first) {
                    continue;
                }
                for (k = i; k < members; k++) {
                    if (same_text(state->forms[order[k]].lect_id, lect)) {
                        seen++;
                        if (ensemble_uniform(rng) * (double)seen <= 1.0) {
                            pick = order[k];
                        }
                    }
                }
                chosen[lects++] = pick;
            }
            if (lects < 2) {
                continue;
            }
            {
                rg_cognate_set *set = &state->sets[state->set_count];
                size_t length = 0;
                char *list;
                for (k = 0; k < lects; k++) {
                    rg_cognate_form *form = &state->set_forms[used + k];
                    form->lect_id = state->forms[chosen[k]].lect_id;
                    form->form = state->forms[chosen[k]].form;
                    form->form.lect_id = form->lect_id;
                    length += strlen(state->forms[chosen[k]].form_id != 0 ? state->forms[chosen[k]].form_id : "") + 1;
                }
                list = (char *)malloc(length + 1);
                if (list == 0) {
                    free(order);
                    free(chosen);
                    free_strings(forms, state->set_count);
                    return RG_ERR_OOM;
                }
                list[0] = '\0';
                for (k = 0; k < lects; k++) {
                    const char *id = state->forms[chosen[k]].form_id;
                    if (k > 0) {
                        strcat(list, " ");
                    }
                    strcat(list, id != 0 ? id : "");
                }
                state->ids[state->set_count] = (char *)malloc(strlen(list) + 5);
                if (state->ids[state->set_count] == 0) {
                    free(list);
                    free(order);
                    free(chosen);
                    free_strings(forms, state->set_count);
                    return RG_ERR_OOM;
                }
                strcpy(state->ids[state->set_count], "set:");
                strcat(state->ids[state->set_count], state->forms[chosen[0]].form_id != 0
                                                    ? state->forms[chosen[0]].form_id : "");
                forms[state->set_count] = list;
                set->cognate_id = state->ids[state->set_count];
                set->etymon_group = state->forms[chosen[0]].meaning;
                set->forms = &state->set_forms[used];
                set->form_count = lects;
                set->confidence = 1.0;
                used += lects;
                state->set_count++;
            }
        }
    }
    free(order);
    free(chosen);
    *sets = state->sets;
    *count = state->set_count;
    *forms_out = forms;
    return RG_OK;
}

rg_status rg_train_ensemble_soft(
    const rg_context *ctx,
    const rg_soft_form *forms,
    size_t form_count,
    const rg_cognacy_edge *edges,
    size_t edge_count,
    const rg_train_options *options,
    const rg_ensemble_options *ensemble_options,
    rg_ensemble **out
) {
    rg_train_options base;
    rg_ensemble_options eopts;
    rg_temperature_score *calibration = 0;
    soft_state state;
    member_source source;
    size_t i;
    rg_status status;

    if (ctx == 0 || out == 0 || (form_count > 0 && forms == 0) || (edge_count > 0 && edges == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    status = check_options(options, ensemble_options, &base, &eopts);
    if (status != RG_OK) {
        return status;
    }
    for (i = 0; i < form_count; i++) {
        if (forms[i].lect_id == 0 || forms[i].form.segment_count == 0 || forms[i].form.segments == 0) {
            return RG_ERR_INVALID_ARGUMENT;
        }
    }
    for (i = 0; i < edge_count; i++) {
        if (edges[i].a >= form_count || edges[i].b >= form_count || !isfinite(edges[i].weight) ||
            edges[i].weight < 0.0 || edges[i].weight > 1.0) {
            return RG_ERR_INVALID_ARGUMENT;
        }
    }
    memset(&state, 0, sizeof(state));
    state.forms = forms;
    state.form_count = form_count;
    state.edges = edges;
    state.edge_count = edge_count;
    state.parent = (size_t *)malloc((form_count + 1) * sizeof(size_t));
    if (state.parent == 0) {
        return RG_ERR_OOM;
    }
    if (eopts.calibrate_temperature) {
        /* Calibrate on the partition of the edges at least as likely as not. */
        const rg_cognate_set *sets = 0;
        size_t count = 0;
        char **unused = 0;
        uint64_t rng = 1;
        state.threshold = 0.5;
        status = soft_build(&state, &rng, &sets, &count, &unused);
        if (status == RG_OK) {
            status = maybe_calibrate(ctx, sets, count, &base, &eopts, &calibration);
        }
        free_strings(unused, count);
        state.threshold = 0.0;
        if (status != RG_OK) {
            soft_release(&state);
            return status;
        }
    }
    source.build = soft_build;
    source.release = soft_release;
    source.data = &state;
    return run_members(ctx, base, &eopts, calibration, &source, 1, out);
}

size_t rg_ensemble_member_count(const rg_ensemble *ensemble) {
    return ensemble == 0 ? 0 : ensemble->member_count;
}

const rg_multi_model *rg_ensemble_member_at(const rg_ensemble *ensemble, size_t index) {
    if (ensemble == 0 || index >= ensemble->member_count) {
        return 0;
    }
    return ensemble->members[index];
}

const double *rg_ensemble_member_set_weights(
    const rg_ensemble *ensemble,
    size_t index,
    size_t *count
) {
    if (count != 0) {
        *count = 0;
    }
    if (ensemble == 0 || index >= ensemble->member_count) {
        return 0;
    }
    if (count != 0) {
        *count = ensemble->set_counts[index];
    }
    return ensemble->set_weights[index];
}
