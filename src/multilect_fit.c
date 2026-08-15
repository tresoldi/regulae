#include "multilect_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct outlier_work_row {
    char *cognate_id;
    int pair_count;
    double cost_per_segment;
    double z_score;
} outlier_work_row;

static int outlier_cmp(const void *a, const void *b) {
    const outlier_work_row *ra = (const outlier_work_row *)a;
    const outlier_work_row *rb = (const outlier_work_row *)b;
    if (ra->z_score != rb->z_score) {
        return ra->z_score > rb->z_score ? -1 : 1;
    }
    return strcmp(ra->cognate_id, rb->cognate_id);
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



rg_status compute_corpus_fit(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    const permutation_baseline *baseline,
    rg_multi_model *model
);
rg_status run_permutation_baseline(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    const char *const *lect_ids,
    size_t lect_count,
    permutation_baseline *out
);
rg_status bootstrap_class_intervals(
    rg_multi_model *model,
    size_t cognate_count,
    const rg_train_options *options
);


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
rg_status bootstrap_class_intervals(
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
        rg_multi_class_row *owned = i < model->unconditioned_class_count
            ? &model->unconditioned_classes[i]
            : &model->conditioned_classes[i - model->unconditioned_class_count];
        rg_uncertainty_estimate estimate;
        int post = i >= model->unconditioned_class_count;
        if (rg_percentile_interval(&rates[i * draws], draws,
                                  owned->uncertainty.estimate,
                                  owned->uncertainty.n,
                                  owned->uncertainty.alpha,
                                  &estimate) != RG_OK) {
            continue;
        }
        estimate.post_selection = post;
        owned->uncertainty = estimate;
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

rg_status run_permutation_baseline(
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
            uncond += (double)shuffled->unconditioned_class_count;
            cond += (double)shuffled->conditioned_class_count;
            completed++;
        }
        for (k = 0; k < shuffled->conditioned_class_count && status == RG_OK; k++) {
            const rg_multi_class_row *row = &shuffled->conditioned_classes[k];
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
            margins[margin_count++] = row->evidence.search_margin;
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
rg_status compute_corpus_fit(
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
            rg_multi_class_row *row = &model->conditioned_classes[i];
            rg_rule_evidence_judge_internal(&row->evidence, baseline->search_margin);
            model->fit.rules_measured++;
            if (row->evidence.standing == RG_RULE_STANDING_ABOVE_NOISE) {
                model->fit.rules_above_noise++;
            }
        }
        for (i = 0; i < model->cross_dimensional_count; i++) {
            rg_multi_cross_dimensional_row *row = &model->cross_dimensional_rows[i];
            rg_rule_evidence_judge_internal(&row->rule.evidence, baseline->search_margin);
            model->fit.rules_measured++;
            if (row->rule.evidence.standing == RG_RULE_STANDING_ABOVE_NOISE) {
                model->fit.rules_above_noise++;
            }
        }
        /* The per-pair tables are judged but not counted, and that is not an
         * oversight to be tidied away.
         *
         * lift_cross_dimensional_rows copies every pairwise cross-dimensional
         * row into the multi-lect table, so the loop above has already counted
         * each of those rules once; counting here would count it twice. Their
         * standing is still set, because the pairwise model is published in its
         * own right and a rule that says nothing about how it stands is not
         * readable.
         *
         * The pairwise *conditioned correspondences* are a different case: they
         * are not lifted anywhere, so they are genuinely outside the count.
         * Whether they belong in it is a question about what "how many rules
         * stand" is counting, not a bug -- see docs/architecture_plan.md. */
        for (i = 0; i < model->pair_model_count; i++) {
            rg_pairwise_model *pair = model->pair_models[i].model;
            for (j = 0; j < pair->conditioned_segment_count_count; j++) {
                rg_conditioned_segment_count_row *row = &pair->conditioned_segment_counts[j];
                rg_rule_evidence_judge_internal(&row->evidence, baseline->search_margin);
            }
            for (j = 0; j < pair->cross_dimensional_count; j++) {
                rg_cross_dimensional_row *row = &pair->cross_dimensional_rows[j];
                rg_rule_evidence_judge_internal(&row->evidence, baseline->search_margin);
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
