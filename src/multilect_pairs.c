#include "multilect_internal.h"

#include <stdlib.h>
#include <string.h>

rg_status train_pair_models(
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

rg_status lift_cross_dimensional_rows(rg_multi_model *model) {
    size_t i;
    size_t cap = 0;
    for (i = 0; i < model->pair_model_count; i++) {
        size_t j;
        size_t row_count = 0;
        const rg_cross_dimensional_row *pair_rows =
            rg_pairwise_model_cross_dimensional_rows(model->pair_models[i].model, &row_count);
        for (j = 0; j < row_count; j++) {
            const rg_cross_dimensional_row *row = &pair_rows[j];
            rg_multi_cross_dimensional_row *next;
            if (row == 0) {
                continue;
            }
            if (model->cross_dimensional_count == cap) {
                size_t next_cap = cap == 0 ? 8 : cap * 2;
                next = (rg_multi_cross_dimensional_row *)realloc(model->cross_dimensional_rows, next_cap * sizeof(*model->cross_dimensional_rows));
                if (next == 0) {
                    return RG_ERR_OOM;
                }
                model->cross_dimensional_rows = next;
                cap = next_cap;
            }
            memset(&model->cross_dimensional_rows[model->cross_dimensional_count], 0,
                   sizeof(model->cross_dimensional_rows[model->cross_dimensional_count]));
            /* The rule is the pairwise row. Only which pair it was found in is
             * new here, so there is nothing to copy field by field -- and
             * nothing to forget to copy, which forty lines of assignment could.
             * Borrowed from the pairwise row, which outlives the lifted view. */
            model->cross_dimensional_rows[model->cross_dimensional_count].source_lect = model->pair_models[i].lect_a;
            model->cross_dimensional_rows[model->cross_dimensional_count].target_lect = model->pair_models[i].lect_b;
            model->cross_dimensional_rows[model->cross_dimensional_count].rule = *row;
            model->cross_dimensional_count++;
        }
    }
    return RG_OK;
}

