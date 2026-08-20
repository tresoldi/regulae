#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define IBM1_DEFAULT_MAX_ITER 30
#define IBM1_DEFAULT_CONVERGENCE 1e-4
#define IBM1_DEFAULT_TEMPERATURE 1.0

struct rg_translation_table {
    const rg_context *ctx;
    char **source_vocab;
    size_t source_count;
    char **target_vocab;
    size_t target_count;
    double *forward;
    double *backward;
    double null_forward;
    double null_backward;
    double temperature;
    int has_matrices;
};

static size_t vocab_find(char *const *vocab, size_t count, const char *grapheme) {
    size_t lo = 0;
    size_t hi = count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int cmp = strcmp(vocab[mid], grapheme);
        if (cmp < 0) {
            lo = mid + 1;
        } else if (cmp > 0) {
            hi = mid;
        } else {
            return mid;
        }
    }
    return count;
}

static rg_status vocab_collect(
    const rg_form_pair *pairs,
    size_t pair_count,
    char ***out_source,
    size_t *out_source_count,
    char ***out_target,
    size_t *out_target_count
) {
    char **src = 0;
    size_t src_count = 0;
    size_t src_cap = 0;
    char **tgt = 0;
    size_t tgt_count = 0;
    size_t tgt_cap = 0;
    size_t i;
    rg_status status = RG_OK;

    for (i = 0; i < pair_count && status == RG_OK; i++) {
        size_t j;
        for (j = 0; j < pairs[i].source.segment_count && status == RG_OK; j++) {
            const char *g = pairs[i].source.segments[j].grapheme;
            if (g == 0) {
                continue;
            }
            if (vocab_find(src, src_count, g) < src_count) {
                continue;
            }
            if (src_count == src_cap) {
                size_t next_cap = src_cap == 0 ? 32 : src_cap * 2;
                char **next = (char **)realloc(src, next_cap * sizeof(*next));
                if (next == 0) {
                    status = RG_ERR_OOM;
                    break;
                }
                src = next;
                src_cap = next_cap;
            }
            src[src_count] = rg_strdup_internal(g);
            if (src[src_count] == 0) {
                status = RG_ERR_OOM;
                break;
            }
            src_count++;
            if (src_count > 1) {
                size_t k = src_count - 1;
                while (k > 0 && strcmp(src[k - 1], src[k]) > 0) {
                    char *tmp = src[k - 1];
                    src[k - 1] = src[k];
                    src[k] = tmp;
                    k--;
                }
            }
        }
        for (j = 0; j < pairs[i].target.segment_count && status == RG_OK; j++) {
            const char *g = pairs[i].target.segments[j].grapheme;
            if (g == 0) {
                continue;
            }
            if (vocab_find(tgt, tgt_count, g) < tgt_count) {
                continue;
            }
            if (tgt_count == tgt_cap) {
                size_t next_cap = tgt_cap == 0 ? 32 : tgt_cap * 2;
                char **next = (char **)realloc(tgt, next_cap * sizeof(*next));
                if (next == 0) {
                    status = RG_ERR_OOM;
                    break;
                }
                tgt = next;
                tgt_cap = next_cap;
            }
            tgt[tgt_count] = rg_strdup_internal(g);
            if (tgt[tgt_count] == 0) {
                status = RG_ERR_OOM;
                break;
            }
            tgt_count++;
            if (tgt_count > 1) {
                size_t k = tgt_count - 1;
                while (k > 0 && strcmp(tgt[k - 1], tgt[k]) > 0) {
                    char *tmp = tgt[k - 1];
                    tgt[k - 1] = tgt[k];
                    tgt[k] = tmp;
                    k--;
                }
            }
        }
    }
    if (status != RG_OK) {
        for (i = 0; i < src_count; i++) {
            free(src[i]);
        }
        free(src);
        for (i = 0; i < tgt_count; i++) {
            free(tgt[i]);
        }
        free(tgt);
        return status;
    }
    *out_source = src;
    *out_source_count = src_count;
    *out_target = tgt;
    *out_target_count = tgt_count;
    return RG_OK;
}

static int string_ptr_cmp(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static void free_vocab(char **vocab, size_t count) {
    size_t i;
    if (vocab == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        free(vocab[i]);
    }
    free(vocab);
}

static rg_status init_prior(
    const rg_context *ctx,
    char *const *source,
    size_t source_count,
    char *const *target,
    size_t target_count,
    double temperature,
    double null_prior,
    double *out
) {
    size_t stride = target_count + 1;
    size_t i;

    for (i = 0; i < source_count; i++) {
        double max_logit = -INFINITY;
        double total = 0.0;
        size_t j;
        for (j = 0; j < target_count; j++) {
            double distance = 0.0;
            if (rg_context_segment_distance(ctx, source[i], target[j], &distance) == RG_OK) {
                out[i * stride + j] = -temperature * distance;
            } else {
                out[i * stride + j] = -1e6;
            }
            if (out[i * stride + j] > max_logit) {
                max_logit = out[i * stride + j];
            }
        }
        for (j = 0; j < target_count; j++) {
            out[i * stride + j] = exp(out[i * stride + j] - max_logit);
            total += out[i * stride + j];
        }
        total += null_prior;
        if (total > 0.0) {
            for (j = 0; j < target_count; j++) {
                out[i * stride + j] /= total;
            }
            out[i * stride + target_count] = null_prior / total;
        } else {
            double uniform = 1.0 / (double)(target_count + 1);
            for (j = 0; j <= target_count; j++) {
                out[i * stride + j] = uniform;
            }
        }
    }
    return RG_OK;
}

static double compute_log_likelihood(
    const rg_form_pair *pairs,
    size_t pair_count,
    char *const *source_vocab,
    size_t source_count,
    char *const *target_vocab,
    size_t target_count,
    const double *prob
) {
    size_t stride = target_count + 1;
    double ll = 0.0;
    size_t i;

    for (i = 0; i < pair_count; i++) {
        size_t si;
        double weight = pairs[i].weight == 0.0 ? 1.0 : pairs[i].weight;
        if (weight <= 0.0) {
            continue;
        }
        for (si = 0; si < pairs[i].source.segment_count; si++) {
            const char *sg = pairs[i].source.segments[si].grapheme;
            size_t sv = vocab_find(source_vocab, source_count, sg);
            double total_p = 0.0;
            size_t tj;
            if (sv >= source_count) {
                continue;
            }
            for (tj = 0; tj < pairs[i].target.segment_count; tj++) {
                const char *tg = pairs[i].target.segments[tj].grapheme;
                size_t tv = vocab_find(target_vocab, target_count, tg);
                if (tv < target_count) {
                    total_p += prob[sv * stride + tv];
                }
            }
            total_p += prob[sv * stride + target_count];
            if (total_p > 0.0) {
                ll += weight * log(total_p);
            }
        }
    }
    return ll;
}

static rg_status em_iteration(
    const rg_form_pair *pairs,
    size_t pair_count,
    char *const *source_vocab,
    size_t source_count,
    char *const *target_vocab,
    size_t target_count,
    double *prob,
    double *counts
) {
    size_t stride = target_count + 1;
    size_t table_size = source_count * stride;
    size_t i;

    memset(counts, 0, table_size * sizeof(*counts));

    for (i = 0; i < pair_count; i++) {
        size_t si;
        double weight = pairs[i].weight == 0.0 ? 1.0 : pairs[i].weight;
        if (weight <= 0.0) {
            continue;
        }
        for (si = 0; si < pairs[i].source.segment_count; si++) {
            const char *sg = pairs[i].source.segments[si].grapheme;
            size_t sv = vocab_find(source_vocab, source_count, sg);
            double total_p = 0.0;
            size_t tj;
            if (sv >= source_count) {
                continue;
            }
            for (tj = 0; tj < pairs[i].target.segment_count; tj++) {
                const char *tg = pairs[i].target.segments[tj].grapheme;
                size_t tv = vocab_find(target_vocab, target_count, tg);
                if (tv < target_count) {
                    total_p += prob[sv * stride + tv];
                }
            }
            total_p += prob[sv * stride + target_count];
            if (total_p <= 0.0) {
                continue;
            }
            for (tj = 0; tj < pairs[i].target.segment_count; tj++) {
                const char *tg = pairs[i].target.segments[tj].grapheme;
                size_t tv = vocab_find(target_vocab, target_count, tg);
                if (tv < target_count) {
                    counts[sv * stride + tv] += weight * prob[sv * stride + tv] / total_p;
                }
            }
            counts[sv * stride + target_count] += weight * prob[sv * stride + target_count] / total_p;
        }
    }

    for (i = 0; i < source_count; i++) {
        double row_total = 0.0;
        size_t j;
        for (j = 0; j <= target_count; j++) {
            row_total += counts[i * stride + j];
        }
        if (row_total > 0.0) {
            for (j = 0; j <= target_count; j++) {
                prob[i * stride + j] = counts[i * stride + j] / row_total;
            }
        }
    }

    return RG_OK;
}

void rg_translation_table_options_init_defaults(rg_translation_table_options *options) {
    if (options == 0) {
        return;
    }
    options->max_iterations = 0;
    options->null_prior = -1.0;
    options->convergence_threshold = 0.0;
    options->temperature = 0.0;
}

rg_status rg_train_translation_table(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_translation_table_options *options,
    rg_translation_table **out
) {
    rg_translation_table *table;
    rg_translation_table_options defaults;
    double temperature;
    double null_prior;
    int max_iter;
    double convergence;
    double *prob_fwd = 0;
    double *prob_bwd = 0;
    double *counts = 0;
    rg_form_pair *reversed = 0;
    size_t fwd_size;
    size_t bwd_size;
    double prev_ll;
    int iter;
    size_t i;
    rg_status status;

    if (ctx == 0 || out == 0 || (pair_count > 0 && pairs == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    if (options == 0) {
        rg_translation_table_options_init_defaults(&defaults);
        options = &defaults;
    }
    temperature = options->temperature <= 0.0 ? IBM1_DEFAULT_TEMPERATURE : options->temperature;
    null_prior = options->null_prior < 0.0 ? exp(-RG_DEFAULT_GAP_COST) : options->null_prior;
    max_iter = options->max_iterations <= 0 ? IBM1_DEFAULT_MAX_ITER : options->max_iterations;
    convergence = options->convergence_threshold <= 0.0 ? IBM1_DEFAULT_CONVERGENCE : options->convergence_threshold;

    table = (rg_translation_table *)calloc(1, sizeof(*table));
    if (table == 0) {
        return RG_ERR_OOM;
    }
    table->ctx = ctx;
    table->temperature = temperature;

    status = vocab_collect(pairs, pair_count,
                           &table->source_vocab, &table->source_count,
                           &table->target_vocab, &table->target_count);
    if (status != RG_OK) {
        free(table);
        return status;
    }

    fwd_size = table->source_count * (table->target_count + 1);
    bwd_size = table->target_count * (table->source_count + 1);
    prob_fwd = (double *)calloc(fwd_size == 0 ? 1 : fwd_size, sizeof(*prob_fwd));
    prob_bwd = (double *)calloc(bwd_size == 0 ? 1 : bwd_size, sizeof(*prob_bwd));
    counts = (double *)calloc((fwd_size > bwd_size ? fwd_size : bwd_size) + 1, sizeof(*counts));
    if (prob_fwd == 0 || prob_bwd == 0 || counts == 0) {
        free(prob_fwd);
        free(prob_bwd);
        free(counts);
        rg_translation_table_free(table);
        return RG_ERR_OOM;
    }

    status = init_prior(ctx, table->source_vocab, table->source_count,
                        table->target_vocab, table->target_count,
                        temperature, null_prior, prob_fwd);
    if (status != RG_OK) {
        free(prob_fwd);
        free(prob_bwd);
        free(counts);
        rg_translation_table_free(table);
        return status;
    }

    prev_ll = -INFINITY;
    for (iter = 0; iter < max_iter; iter++) {
        double ll = compute_log_likelihood(pairs, pair_count,
                                           table->source_vocab, table->source_count,
                                           table->target_vocab, table->target_count,
                                           prob_fwd);
        if (!isinf(prev_ll)) {
            double denom = fabs(prev_ll) < 1.0 ? 1.0 : fabs(prev_ll);
            if (fabs(ll - prev_ll) / denom < convergence) {
                break;
            }
        }
        prev_ll = ll;
        status = em_iteration(pairs, pair_count,
                              table->source_vocab, table->source_count,
                              table->target_vocab, table->target_count,
                              prob_fwd, counts);
        if (status != RG_OK) {
            free(prob_fwd);
            free(prob_bwd);
            free(counts);
            rg_translation_table_free(table);
            return status;
        }
    }

    reversed = (rg_form_pair *)calloc(pair_count == 0 ? 1 : pair_count, sizeof(*reversed));
    if (reversed == 0) {
        free(prob_fwd);
        free(prob_bwd);
        free(counts);
        rg_translation_table_free(table);
        return RG_ERR_OOM;
    }
    for (i = 0; i < pair_count; i++) {
        reversed[i].source = pairs[i].target;
        reversed[i].target = pairs[i].source;
        reversed[i].weight = pairs[i].weight;
    }

    status = init_prior(ctx, table->target_vocab, table->target_count,
                        table->source_vocab, table->source_count,
                        temperature, null_prior, prob_bwd);
    if (status != RG_OK) {
        free(reversed);
        free(prob_fwd);
        free(prob_bwd);
        free(counts);
        rg_translation_table_free(table);
        return status;
    }

    prev_ll = -INFINITY;
    for (iter = 0; iter < max_iter; iter++) {
        double ll = compute_log_likelihood(reversed, pair_count,
                                           table->target_vocab, table->target_count,
                                           table->source_vocab, table->source_count,
                                           prob_bwd);
        if (!isinf(prev_ll)) {
            double denom = fabs(prev_ll) < 1.0 ? 1.0 : fabs(prev_ll);
            if (fabs(ll - prev_ll) / denom < convergence) {
                break;
            }
        }
        prev_ll = ll;
        status = em_iteration(reversed, pair_count,
                              table->target_vocab, table->target_count,
                              table->source_vocab, table->source_count,
                              prob_bwd, counts);
        if (status != RG_OK) {
            free(reversed);
            free(prob_fwd);
            free(prob_bwd);
            free(counts);
            rg_translation_table_free(table);
            return status;
        }
    }

    free(reversed);
    free(counts);

    table->forward = prob_fwd;
    table->backward = prob_bwd;
    table->null_forward = null_prior;
    table->null_backward = null_prior;
    table->has_matrices = 1;
    *out = table;
    return RG_OK;
}

rg_status rg_translation_table_from_prior(
    const rg_context *ctx,
    rg_translation_table **out
) {
    rg_translation_table *table;
    if (ctx == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    table = (rg_translation_table *)calloc(1, sizeof(*table));
    if (table == 0) {
        return RG_ERR_OOM;
    }
    table->ctx = ctx;
    table->temperature = IBM1_DEFAULT_TEMPERATURE;
    table->has_matrices = 0;
    *out = table;
    return RG_OK;
}

rg_status rg_translation_table_precompute(
    rg_translation_table *table,
    const char **source_inventory,
    size_t source_count,
    const char **target_inventory,
    size_t target_count
) {
    size_t fwd_size;
    size_t bwd_size;
    double null_prior;
    size_t i;
    rg_status status;

    if (table == 0 || (source_count > 0 && source_inventory == 0) ||
        (target_count > 0 && target_inventory == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    free_vocab(table->source_vocab, table->source_count);
    free_vocab(table->target_vocab, table->target_count);
    free(table->forward);
    free(table->backward);
    table->source_vocab = 0;
    table->source_count = 0;
    table->target_vocab = 0;
    table->target_count = 0;
    table->forward = 0;
    table->backward = 0;

    table->source_vocab = (char **)calloc(source_count == 0 ? 1 : source_count,
                                          sizeof(*table->source_vocab));
    table->target_vocab = (char **)calloc(target_count == 0 ? 1 : target_count,
                                          sizeof(*table->target_vocab));
    if (table->source_vocab == 0 || table->target_vocab == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < source_count; i++) {
        table->source_vocab[i] = rg_strdup_internal(source_inventory[i]);
        if (table->source_vocab[i] == 0) {
            table->source_count = i;
            return RG_ERR_OOM;
        }
    }
    table->source_count = source_count;
    for (i = 0; i < target_count; i++) {
        table->target_vocab[i] = rg_strdup_internal(target_inventory[i]);
        if (table->target_vocab[i] == 0) {
            table->target_count = i;
            return RG_ERR_OOM;
        }
    }
    table->target_count = target_count;

    if (source_count > 1) {
        qsort(table->source_vocab, source_count, sizeof(*table->source_vocab),
              string_ptr_cmp);
    }
    if (target_count > 1) {
        qsort(table->target_vocab, target_count, sizeof(*table->target_vocab),
              string_ptr_cmp);
    }

    null_prior = exp(-RG_DEFAULT_GAP_COST);
    fwd_size = source_count * (target_count + 1);
    bwd_size = target_count * (source_count + 1);
    table->forward = (double *)calloc(fwd_size == 0 ? 1 : fwd_size, sizeof(*table->forward));
    table->backward = (double *)calloc(bwd_size == 0 ? 1 : bwd_size, sizeof(*table->backward));
    if (table->forward == 0 || table->backward == 0) {
        return RG_ERR_OOM;
    }

    status = init_prior(table->ctx,
                        table->source_vocab, table->source_count,
                        table->target_vocab, table->target_count,
                        table->temperature, null_prior, table->forward);
    if (status != RG_OK) {
        return status;
    }
    status = init_prior(table->ctx,
                        table->target_vocab, table->target_count,
                        table->source_vocab, table->source_count,
                        table->temperature, null_prior, table->backward);
    if (status != RG_OK) {
        return status;
    }

    table->null_forward = null_prior;
    table->null_backward = null_prior;
    table->has_matrices = 1;
    return RG_OK;
}

static double lazy_prior_probability(
    const rg_context *ctx,
    const char *source,
    const char *target,
    double temperature
) {
    double distance = 0.0;
    if (rg_context_segment_distance(ctx, source, target, &distance) != RG_OK) {
        return 0.0;
    }
    return exp(-temperature * distance);
}

double rg_translation_probability(
    const rg_translation_table *table,
    const char *source,
    const char *target,
    rg_direction direction
) {
    double fwd;
    double bwd;

    if (table == 0 || source == 0 || target == 0) {
        return 0.0;
    }
    if (!table->has_matrices) {
        double p = lazy_prior_probability(table->ctx, source, target, table->temperature);
        if (direction == RG_DIR_FORWARD || direction == RG_DIR_BACKWARD) {
            return p;
        }
        return p;
    }

    fwd = 0.0;
    bwd = 0.0;
    if (direction == RG_DIR_FORWARD || direction == RG_DIR_SYMMETRIC) {
        size_t si = vocab_find(table->source_vocab, table->source_count, source);
        size_t ti = vocab_find(table->target_vocab, table->target_count, target);
        if (si < table->source_count && ti < table->target_count) {
            fwd = table->forward[si * (table->target_count + 1) + ti];
        }
    }
    if (direction == RG_DIR_BACKWARD || direction == RG_DIR_SYMMETRIC) {
        size_t ti = vocab_find(table->target_vocab, table->target_count, target);
        size_t si = vocab_find(table->source_vocab, table->source_count, source);
        if (ti < table->target_count && si < table->source_count) {
            bwd = table->backward[ti * (table->source_count + 1) + si];
        }
    }

    switch (direction) {
    case RG_DIR_FORWARD:
        return fwd;
    case RG_DIR_BACKWARD:
        return bwd;
    case RG_DIR_SYMMETRIC:
        if (fwd <= 0.0 || bwd <= 0.0) {
            return 0.0;
        }
        return sqrt(fwd * bwd);
    }
    return 0.0;
}

rg_status rg_translation_align(
    const rg_translation_table *table,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    rg_direction direction,
    rg_translation_alignment **out
) {
    rg_translation_alignment *alignment;
    size_t i;

    if (table == 0 || out == 0 || (source_count > 0 && source == 0) ||
        (target_count > 0 && target == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    alignment = (rg_translation_alignment *)calloc(1, sizeof(*alignment));
    if (alignment == 0) {
        return RG_ERR_OOM;
    }
    if (source_count == 0) {
        alignment->assignments = 0;
        alignment->count = 0;
        alignment->score = 0.0;
        *out = alignment;
        return RG_OK;
    }
    alignment->assignments = (rg_translation_assignment *)calloc(
        source_count, sizeof(*alignment->assignments));
    if (alignment->assignments == 0) {
        free(alignment);
        return RG_ERR_OOM;
    }
    alignment->count = source_count;
    alignment->score = 0.0;

    for (i = 0; i < source_count; i++) {
        const char *sg = source[i].grapheme;
        double best_p = 0.0;
        size_t best_j = (size_t)-1;
        size_t j;

        if (sg == 0) {
            alignment->assignments[i].source_index = i;
            alignment->assignments[i].target_index = (size_t)-1;
            alignment->assignments[i].probability = 0.0;
            continue;
        }

        for (j = 0; j < target_count; j++) {
            const char *tg = target[j].grapheme;
            double p;
            if (tg == 0) {
                continue;
            }
            p = rg_translation_probability(table, sg, tg, direction);
            if (p > best_p) {
                best_p = p;
                best_j = j;
            }
        }

        alignment->assignments[i].source_index = i;
        alignment->assignments[i].target_index = best_j;
        alignment->assignments[i].probability = best_p;
        if (best_p > 0.0) {
            alignment->score += log(best_p);
        }
    }

    if (source_count > 0) {
        alignment->score /= (double)source_count;
    }
    *out = alignment;
    return RG_OK;
}

rg_status rg_translation_score(
    const rg_translation_table *table,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    rg_direction direction,
    double *score
) {
    rg_translation_alignment *alignment = 0;
    rg_status status;

    if (score == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *score = 0.0;
    status = rg_translation_align(table, source, source_count, target, target_count, direction, &alignment);
    if (status != RG_OK) {
        return status;
    }
    *score = alignment->score;
    rg_translation_alignment_free(alignment);
    return RG_OK;
}

void rg_translation_table_free(rg_translation_table *table) {
    if (table == 0) {
        return;
    }
    free_vocab(table->source_vocab, table->source_count);
    free_vocab(table->target_vocab, table->target_count);
    free(table->forward);
    free(table->backward);
    free(table);
}

void rg_translation_alignment_free(rg_translation_alignment *alignment) {
    if (alignment == 0) {
        return;
    }
    free(alignment->assignments);
    free(alignment);
}

const char *const *rg_translation_table_source_vocab(
    const rg_translation_table *table,
    size_t *count
) {
    if (table == 0 || !table->has_matrices) {
        if (count != 0) {
            *count = 0;
        }
        return 0;
    }
    if (count != 0) {
        *count = table->source_count;
    }
    return (const char *const *)table->source_vocab;
}

const char *const *rg_translation_table_target_vocab(
    const rg_translation_table *table,
    size_t *count
) {
    if (table == 0 || !table->has_matrices) {
        if (count != 0) {
            *count = 0;
        }
        return 0;
    }
    if (count != 0) {
        *count = table->target_count;
    }
    return (const char *const *)table->target_vocab;
}

rg_status rg_seed_prior_from_ibm1_internal(
    const rg_translation_table *table,
    rg_pairwise_model *model
) {
    size_t i;
    if (table == 0 || model == 0 || !table->has_matrices) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < model->segment_prior_count; i++) {
        double p = rg_translation_probability(
            table,
            model->segment_priors[i].source,
            model->segment_priors[i].target,
            RG_DIR_SYMMETRIC
        );
        if (p > 0.0) {
            model->segment_priors[i].alpha = model->concentration * p;
        }
    }
    for (i = 0; i < model->log_normalizer_count; i++) {
        model->log_normalizers[i].value = 0.0;
    }
    return RG_OK;
}
