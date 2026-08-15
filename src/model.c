#include "model_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

static rg_status append_vocab(char ***vocab, size_t *count, size_t *cap, const char *grapheme) {
    size_t i;
    size_t insert_at;
    if (grapheme == 0) {
        return RG_OK;
    }
    for (i = 0; i < *count; i++) {
        if (strcmp((*vocab)[i], grapheme) == 0) {
            return RG_OK;
        }
    }
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 32 : *cap * 2;
        char **next = (char **)realloc(*vocab, next_cap * sizeof(*next));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *vocab = next;
        *cap = next_cap;
    }
    insert_at = *count;
    while (insert_at > 0 && strcmp((*vocab)[insert_at - 1], grapheme) > 0) {
        insert_at--;
    }
    if (insert_at < *count) {
        memmove(&(*vocab)[insert_at + 1], &(*vocab)[insert_at], (*count - insert_at) * sizeof(**vocab));
    }
    (*vocab)[insert_at] = rg_strdup_internal(grapheme);
    if ((*vocab)[insert_at] == 0) {
        return RG_ERR_OOM;
    }
    (*count)++;
    return RG_OK;
}

static int grapheme_is_known(const rg_context *ctx, const char *grapheme) {
    const rg_feature_set *features = 0;
    return rg_context_features_internal(ctx, grapheme, &features) == RG_OK;
}

/* Builds the starting model: no counts, a merkmal-derived Dirichlet prior over
 * the corpus grapheme inventory, and empty displacement, chunk and tonal
 * tables. The vocabulary is sorted so the softmax sums in a fixed order;
 * floating-point non-associativity would otherwise make iteration order
 * visible in the prior values. */
static rg_status build_initial_prior_model(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model **out
) {
    rg_pairwise_model *model;
    char **vocab = 0;
    size_t vocab_count = 0;
    size_t vocab_cap = 0;
    double *logits = 0;
    double *exps = 0;
    size_t *targets = 0;
    size_t prior_cap = 0;
    size_t normalizer_cap = 0;
    size_t i;
    rg_status status = RG_OK;
    double temperature = options->temperature == 0.0 ? 1.0 : options->temperature;
    double concentration = options->concentration;

    *out = 0;
    for (i = 0; i < pair_count && status == RG_OK; i++) {
        size_t j;
        double weight = pairs[i].weight == 0.0 ? 1.0 : pairs[i].weight;
        if (weight <= 0.0) {
            continue;
        }
        for (j = 0; j < pairs[i].source.segment_count && status == RG_OK; j++) {
            status = append_vocab(&vocab, &vocab_count, &vocab_cap, pairs[i].source.segments[j].grapheme);
        }
        for (j = 0; j < pairs[i].target.segment_count && status == RG_OK; j++) {
            status = append_vocab(&vocab, &vocab_count, &vocab_cap, pairs[i].target.segments[j].grapheme);
        }
    }
    model = (rg_pairwise_model *)calloc(1, sizeof(*model));
    if (model == 0 || status != RG_OK) {
        string_array_free(vocab, vocab_count);
        free(model);
        return status == RG_OK ? RG_ERR_OOM : status;
    }
    model->concentration = concentration;
    model->segment_weight = options->segment_weight;
    model->displacement_weight = options->displacement_weight;
    model->tone_weight = options->tone_weight;

    logits = (double *)calloc(vocab_count == 0 ? 1 : vocab_count, sizeof(*logits));
    exps = (double *)calloc(vocab_count == 0 ? 1 : vocab_count, sizeof(*exps));
    targets = (size_t *)calloc(vocab_count == 0 ? 1 : vocab_count, sizeof(*targets));
    if (logits == 0 || exps == 0 || targets == 0) {
        free(logits);
        free(exps);
        free(targets);
        string_array_free(vocab, vocab_count);
        rg_pairwise_model_free(model);
        return RG_ERR_OOM;
    }

    for (i = 0; i < vocab_count && status == RG_OK; i++) {
        size_t j;
        size_t logit_count = 0;
        double max_logit = 0.0;
        double total = 0.0;
        if (!grapheme_is_known(ctx, vocab[i])) {
            continue;
        }
        for (j = 0; j < vocab_count; j++) {
            double distance = 0.0;
            if (!grapheme_is_known(ctx, vocab[j])) {
                continue;
            }
            if (rg_context_segment_distance(ctx, vocab[i], vocab[j], &distance) != RG_OK) {
                continue;
            }
            logits[logit_count] = -temperature * distance;
            targets[logit_count] = j;
            logit_count++;
        }
        if (logit_count == 0) {
            continue;
        }
        max_logit = logits[0];
        for (j = 1; j < logit_count; j++) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
            }
        }
        for (j = 0; j < logit_count; j++) {
            exps[j] = exp(logits[j] - max_logit);
            total += exps[j];
        }
        if (total <= 0.0) {
            continue;
        }
        if (model->log_normalizer_count == normalizer_cap) {
            size_t next_cap = normalizer_cap == 0 ? 32 : normalizer_cap * 2;
            rg_log_normalizer_row *next = (rg_log_normalizer_row *)realloc(model->log_normalizers, next_cap * sizeof(*next));
            if (next == 0) {
                status = RG_ERR_OOM;
                break;
            }
            model->log_normalizers = next;
            normalizer_cap = next_cap;
        }
        model->log_normalizers[model->log_normalizer_count].source = rg_strdup_internal(vocab[i]);
        if (model->log_normalizers[model->log_normalizer_count].source == 0) {
            status = RG_ERR_OOM;
            break;
        }
        model->log_normalizers[model->log_normalizer_count].value = max_logit + log(total);
        /* The reverse direction has no observations yet; publish_target_totals
         * fills this in once counts exist. The row comes from realloc, so the
         * first cost evaluation reads whatever it is left as. */
        model->log_normalizers[model->log_normalizer_count].target_total = 0.0;
        model->log_normalizer_count++;

        for (j = 0; j < logit_count && status == RG_OK; j++) {
            if (model->segment_prior_count == prior_cap) {
                size_t next_cap = prior_cap == 0 ? 64 : prior_cap * 2;
                rg_segment_prior_row *next = (rg_segment_prior_row *)realloc(model->segment_priors, next_cap * sizeof(*next));
                if (next == 0) {
                    status = RG_ERR_OOM;
                    break;
                }
                model->segment_priors = next;
                prior_cap = next_cap;
            }
            model->segment_priors[model->segment_prior_count].source = rg_strdup_internal(vocab[i]);
            model->segment_priors[model->segment_prior_count].target = rg_strdup_internal(vocab[targets[j]]);
            if (model->segment_priors[model->segment_prior_count].source == 0 ||
                model->segment_priors[model->segment_prior_count].target == 0) {
                free(model->segment_priors[model->segment_prior_count].source);
                free(model->segment_priors[model->segment_prior_count].target);
                status = RG_ERR_OOM;
                break;
            }
            model->segment_priors[model->segment_prior_count].alpha = concentration * (exps[j] / total);
            model->segment_prior_count++;
        }
    }

    free(logits);
    free(exps);
    free(targets);
    string_array_free(vocab, vocab_count);
    if (status != RG_OK) {
        rg_pairwise_model_free(model);
        return status;
    }
    *out = model;
    return RG_OK;
}

/* Rebuilds the segment counts from the 1-to-1 links of the current alignments,
 * leaving the prior, the normalizers, and every other table untouched. This is
 * the M-step; the displacement, chunk and tonal tables stay empty until their
 * own stages run. */
static rg_status update_segment_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model
) {
    rg_segment_count_row *rows = 0;
    size_t row_count = 0;
    size_t row_cap = 0;
    size_t i;
    int max_chunk_size = options->max_chunk_size > 0 ? options->max_chunk_size : RG_DEFAULT_MAX_CHUNK_SIZE;
    rg_status status = RG_OK;

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
        for (j = 0; j < rg_alignment_link_count(alignment) && status == RG_OK; j++) {
            const rg_link *link = rg_alignment_link_at(alignment, j);
            if (link->source_count == 1 && link->target_count == 1) {
                status = add_segment_count(&rows, &row_count, &row_cap,
                                           link->source[0].grapheme, link->target[0].grapheme, weight);
            }
        }
        rg_alignment_free(alignment);
    }
    if (status != RG_OK) {
        for (i = 0; i < row_count; i++) {
            segment_count_row_clear(&rows[i]);
        }
        free(rows);
        return status;
    }
    fill_totals(rows, row_count);
    publish_target_totals(model, rows, row_count);
    if (row_count > 1) {
        qsort(rows, row_count, sizeof(*rows), count_row_cmp);
    }
    for (i = 0; i < model->segment_count_count; i++) {
        segment_count_row_clear(&model->segment_counts[i]);
    }
    free(model->segment_counts);
    model->segment_counts = rows;
    model->segment_count_count = row_count;
    return RG_OK;
}

/* Re-aligns under the post-EM model and builds the feature displacement
 * distribution from every 1-to-1 link. */
static rg_status aggregate_displacement_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model
) {
    rg_displacement_row *rows = 0;
    size_t row_count = 0;
    size_t row_cap = 0;
    size_t i;
    int max_chunk_size = options->max_chunk_size > 0 ? options->max_chunk_size : RG_DEFAULT_MAX_CHUNK_SIZE;
    rg_status status = RG_OK;

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
        for (j = 0; j < rg_alignment_link_count(alignment) && status == RG_OK; j++) {
            const rg_link *link = rg_alignment_link_at(alignment, j);
            if (link->source_count != 1 || link->target_count != 1) {
                continue;
            }
            status = add_displacement_vector(
                &rows,
                &row_count,
                &row_cap,
                link->feature_displacement,
                link->feature_displacement_count,
                weight
            );
        }
        rg_alignment_free(alignment);
    }
    if (status != RG_OK) {
        for (i = 0; i < row_count; i++) {
            displacement_row_clear(&rows[i]);
        }
        free(rows);
        return status;
    }
    fill_displacement_total(rows, row_count);
    if (row_count > 1) {
        qsort(rows, row_count, sizeof(*rows), displacement_row_cmp);
    }
    for (i = 0; i < model->displacement_row_count; i++) {
        displacement_row_clear(&model->displacement_rows[i]);
    }
    free(model->displacement_rows);
    model->displacement_rows = rows;
    model->displacement_row_count = row_count;
    return RG_OK;
}

/* Re-aligns under the post-chunk-promotion model and counts each 1-to-1 link's
 * tonal correspondence. The table stays empty for non-tonal corpora. */
static rg_status aggregate_tonal_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model *model
) {
    rg_tonal_count_row *rows = 0;
    size_t row_count = 0;
    size_t row_cap = 0;
    size_t i;
    int any_toned = 0;
    int max_chunk_size = options->max_chunk_size > 0 ? options->max_chunk_size : RG_DEFAULT_MAX_CHUNK_SIZE;
    rg_status status = RG_OK;

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
        for (j = 0; j < rg_alignment_link_count(alignment) && status == RG_OK; j++) {
            const rg_link *link = rg_alignment_link_at(alignment, j);
            const char *source_tone;
            const char *target_tone;
            if (link->source_count != 1 || link->target_count != 1) {
                continue;
            }
            source_tone = link->source[0].tone == 0 ? "" : link->source[0].tone;
            target_tone = link->target[0].tone == 0 ? "" : link->target[0].tone;
            if (source_tone[0] == '\0' && target_tone[0] == '\0') {
                continue;
            }
            any_toned = 1;
            status = add_tonal_count(&rows, &row_count, &row_cap, source_tone, target_tone, weight);
        }
        rg_alignment_free(alignment);
    }
    if (status != RG_OK || !any_toned) {
        for (i = 0; i < row_count; i++) {
            rg_free_owned_internal(rows[i].source_tone);
            rg_free_owned_internal(rows[i].target_tone);
        }
        free(rows);
        return status;
    }
    fill_tonal_source_totals(rows, row_count);
    if (row_count > 1) {
        qsort(rows, row_count, sizeof(*rows), tonal_row_cmp);
    }
    for (i = 0; i < model->tonal_count_count; i++) {
        rg_free_owned_internal(model->tonal_counts[i].source_tone);
        rg_free_owned_internal(model->tonal_counts[i].target_tone);
    }
    free(model->tonal_counts);
    model->tonal_counts = rows;
    model->tonal_count_count = row_count;
    return RG_OK;
}

static rg_status corpus_cost_with_model(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    const rg_pairwise_model *model,
    double *out
) {
    size_t i;
    rg_status status = RG_OK;
    int max_chunk_size = RG_DEFAULT_MAX_CHUNK_SIZE;
    double total = 0.0;
    if (ctx == 0 || (pair_count > 0 && pairs == 0) || model == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    if (options != 0 && options->max_chunk_size > 0) {
        max_chunk_size = options->max_chunk_size;
    }
    for (i = 0; i < pair_count; i++) {
        rg_alignment *alignment = 0;
        double cost = 0.0;
        double weight = pairs[i].weight == 0.0 ? 1.0 : pairs[i].weight;
        if (weight <= 0.0) {
            continue;
        }
        status = rg_align_forms_with_model(ctx, model, options, &pairs[i].source, &pairs[i].target, max_chunk_size, &alignment);
        if (status != RG_OK) {
            return status;
        }
        status = rg_alignment_cost_with_model(ctx, model, options, alignment, &cost);
        rg_alignment_free(alignment);
        if (status != RG_OK) {
            return status;
        }
        total += weight * cost;
    }
    *out = total;
    return RG_OK;
}

rg_status rg_train_pairwise_internal(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_progress_state *progress,
    rg_pairwise_model **out
) {
    rg_pairwise_model *model = 0;
    rg_train_options defaults;
    const rg_train_options *opts = options;
    int max_iter;
    int iter;
    double prev_cost = INFINITY;
    double eps;
    rg_feature_vocabulary vocabulary;
    rg_status status;

    vocabulary.entries = 0;
    vocabulary.count = 0;
    if (ctx == 0 || out == 0 || (pair_count > 0 && pairs == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    if (opts == 0) {
        rg_train_options_init_defaults(&defaults);
        opts = &defaults;
    }
    max_iter = opts->max_iter <= 0 ? 1 : opts->max_iter;
    eps = opts->convergence_eps <= 0.0 ? 1e-4 : opts->convergence_eps;

    /* Stage order is load-bearing and mirrors the Go pipeline: initial prior,
     * segment EM, displacement aggregation, immediate context discovery, chunk
     * promotion, tonal aggregation, cross-dimensional discovery, long-range
     * context discovery. Context splits need stable segment correspondences,
     * chunk promotion would otherwise steal observations, and cross-dimensional
     * rules should explain what the segmental and tonal baselines leave over. */
    status = build_initial_prior_model(ctx, pairs, pair_count, opts, &model);
    if (status != RG_OK) {
        return status;
    }
    if (rg_progress_step_internal(progress, "initial prior")) {
        rg_pairwise_model_free(model);
        return RG_ERR_CANCELLED;
    }
    for (iter = 0; iter < max_iter; iter++) {
        double cost = 0.0;
        status = corpus_cost_with_model(ctx, pairs, pair_count, opts, model, &cost);
        if (status != RG_OK) {
            rg_pairwise_model_free(model);
            return status;
        }
        if (!isinf(prev_cost)) {
            double denom = fabs(prev_cost) < 1.0 ? 1.0 : fabs(prev_cost);
            if (fabs(prev_cost - cost) / denom < eps) {
                break;
            }
        }
        prev_cost = cost;
        status = update_segment_counts(ctx, pairs, pair_count, opts, model);
        if (status != RG_OK) {
            rg_pairwise_model_free(model);
            return status;
        }
    }
    if (rg_progress_step_internal(progress, "segment em")) {
        rg_pairwise_model_free(model);
        return RG_ERR_CANCELLED;
    }

    /* The searchable vocabulary is a property of the corpus, not of the source
     * file, so it is derived once and every discovery stage searches the same
     * list. Built here rather than earlier because nothing before this point
     * uses it, and every stage from here on unwinds through RUN_STAGE, which
     * frees it. */
    status = rg_feature_vocabulary_build_internal(ctx, pairs, pair_count, &vocabulary);
    if (status != RG_OK) {
        rg_pairwise_model_free(model);
        return status;
    }

    /* Cancellation is checked between stages, where the unwind path is already
     * a plain free of the partly-built model. */
#define RUN_STAGE(name, call)                          \
    do {                                               \
        status = (call);                               \
        if (status != RG_OK) {                         \
            rg_feature_vocabulary_clear_internal(&vocabulary); \
            rg_pairwise_model_free(model);             \
            return status;                             \
        }                                              \
        if (rg_progress_step_internal(progress, name)) { \
            rg_feature_vocabulary_clear_internal(&vocabulary); \
            rg_pairwise_model_free(model);             \
            return RG_ERR_CANCELLED;                   \
        }                                              \
    } while (0)

    RUN_STAGE("displacement aggregation", aggregate_displacement_counts(ctx, pairs, pair_count, opts, model));
    RUN_STAGE("context discovery", discover_immediate_context_counts(ctx, pairs, pair_count, opts, model, &vocabulary));
    RUN_STAGE("chunk promotion", promote_chunk_rows(ctx, pairs, pair_count, opts, model));
    RUN_STAGE("tonal aggregation", aggregate_tonal_counts(ctx, pairs, pair_count, opts, model));
    RUN_STAGE("cross-dimensional discovery", discover_cross_dimensional_rows(ctx, pairs, pair_count, opts, model, &vocabulary));
    RUN_STAGE("long-range discovery", discover_long_range_context_counts(ctx, pairs, pair_count, opts, model, &vocabulary));

#undef RUN_STAGE

    rg_feature_vocabulary_clear_internal(&vocabulary);
    *out = model;
    return RG_OK;
}

rg_status rg_train_pairwise_segment_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model **out
) {
    rg_progress_state progress;
    rg_progress_init_internal(&progress, options, RG_PAIRWISE_STAGE_COUNT);
    return rg_train_pairwise_internal(ctx, pairs, pair_count, options, &progress, out);
}

rg_status rg_train_pairwise(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model **out
) {
    return rg_train_pairwise_segment_counts(ctx, pairs, pair_count, options, out);
}

#define RG_PAIRWISE_TABLE(fn, rowtype, field, countfield)                       \
    const rowtype *fn(const rg_pairwise_model *model, size_t *count) {           \
        if (model == 0) {                                                       \
            if (count != 0) { *count = 0; }                                     \
            return 0;                                                           \
        }                                                                       \
        if (count != 0) { *count = model->countfield; }                         \
        return model->field;                                                    \
    }

RG_PAIRWISE_TABLE(rg_pairwise_model_segment_counts, rg_segment_count_row, segment_counts, segment_count_count)
RG_PAIRWISE_TABLE(rg_pairwise_model_displacements, rg_displacement_row, displacement_rows, displacement_row_count)
RG_PAIRWISE_TABLE(rg_pairwise_model_tonal_counts, rg_tonal_count_row, tonal_counts, tonal_count_count)
RG_PAIRWISE_TABLE(rg_pairwise_model_conditioned_segment_counts, rg_conditioned_segment_count_row, conditioned_segment_counts, conditioned_segment_count_count)
RG_PAIRWISE_TABLE(rg_pairwise_model_chunks, rg_chunk_row, chunks, chunk_count)
RG_PAIRWISE_TABLE(rg_pairwise_model_cross_dimensional_rows, rg_cross_dimensional_row, cross_dimensional_rows, cross_dimensional_count)

#undef RG_PAIRWISE_TABLE
