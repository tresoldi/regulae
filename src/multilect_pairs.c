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

