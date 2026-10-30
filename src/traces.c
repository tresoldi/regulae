/* Lost-trigger traces (docs/ensembles.md, "Lost-trigger traces").
 *
 * A conditioned class states an environment on its pivot lect. The model keeps
 * which aligned positions realised the class, so the same environment can be
 * read at each sister's aligned segment, in the sister's own form: a sister
 * whose reflex patterns with the pivot's but whose form no longer shows the
 * trigger is the trace a latent segment leaves. */
#include "internal.h"
#include "multilect_internal.h"

#include <stdlib.h>
#include <string.h>

typedef struct form_context_cache {
    rg_context_spec **contexts;
    size_t *counts;
    size_t lect_count;
    size_t cognate_count;
} form_context_cache;

static size_t model_lect_index(const rg_multi_model *model, const char *lect) {
    size_t i;
    for (i = 0; i < model->lect_count; i++) {
        if (strcmp(model->lect_ids[i], lect) == 0) {
            return i;
        }
    }
    return (size_t)-1;
}

/* The environment observed at `position` of the lect's form in the set, or 0
 * when the set has no form for the lect or the position is past its end. */
static rg_status observed_context(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    const rg_multi_model *model,
    form_context_cache *cache,
    size_t cognate_index,
    size_t lect_index,
    size_t position,
    const rg_context_spec **out
) {
    size_t slot = cognate_index * cache->lect_count + lect_index;
    *out = 0;
    if (cache->contexts[slot] == 0 && cache->counts[slot] == 0) {
        const rg_form *form = form_for_lect(&cognates[cognate_index], model->lect_ids[lect_index]);
        rg_status status;
        if (form == 0) {
            return RG_OK;
        }
        status = rg_form_position_contexts_internal(ctx, form, &cache->contexts[slot],
                                                    &cache->counts[slot]);
        if (status != RG_OK) {
            return status;
        }
    }
    if (position < cache->counts[slot]) {
        *out = &cache->contexts[slot][position];
    }
    return RG_OK;
}

static int position_slot(const rg_class_position *position, size_t lect_index) {
    size_t s;
    for (s = 0; s < position->segment_count; s++) {
        if (position->lect_indices[s] == lect_index) {
            return (int)s;
        }
    }
    return -1;
}

static int position_has_class(const rg_class_position *position, int class_id) {
    size_t k;
    for (k = 0; k < position->class_id_count; k++) {
        if (position->class_ids[k] == class_id) {
            return 1;
        }
    }
    return 0;
}

static rg_status environment_holds(
    const rg_context_spec *environment,
    const rg_context_spec *observed,
    int *holds
) {
    bool subset = false;
    rg_status status = rg_context_spec_is_subset(environment, observed, &subset);
    *holds = status == RG_OK && subset;
    return status;
}

static rg_status append_row(rg_multi_model *model, size_t *cap, const rg_lost_trigger_row *row) {
    if (model->lost_trigger_count == *cap) {
        size_t grown_cap = *cap == 0 ? 16 : *cap * 2;
        rg_lost_trigger_row *grown = (rg_lost_trigger_row *)realloc(
            model->lost_triggers, grown_cap * sizeof(*grown));
        if (grown == 0) {
            return RG_ERR_OOM;
        }
        model->lost_triggers = grown;
        *cap = grown_cap;
    }
    model->lost_triggers[model->lost_trigger_count++] = *row;
    return RG_OK;
}

rg_status rg_lost_triggers_build_internal(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    rg_multi_model *model
) {
    form_context_cache cache;
    size_t *stamps = 0;
    size_t stamp = 0;
    size_t *tagged = 0;
    size_t cap = 0;
    size_t k;
    rg_status status = RG_OK;

    if (model->conditioned_class_count == 0 || model->class_position_count == 0) {
        return RG_OK;
    }
    memset(&cache, 0, sizeof(cache));
    cache.lect_count = model->lect_count;
    cache.cognate_count = cognate_count;
    cache.contexts = (rg_context_spec **)calloc(cognate_count * model->lect_count + 1,
                                                sizeof(*cache.contexts));
    cache.counts = (size_t *)calloc(cognate_count * model->lect_count + 1, sizeof(*cache.counts));
    stamps = (size_t *)calloc(cognate_count + 1, sizeof(*stamps));
    tagged = (size_t *)calloc(model->class_position_count + 1, sizeof(*tagged));
    if (cache.contexts == 0 || cache.counts == 0 || stamps == 0 || tagged == 0) {
        status = RG_ERR_OOM;
    }

    for (k = 0; k < model->conditioned_class_count && status == RG_OK; k++) {
        const rg_multi_class_row *row = &model->conditioned_classes[k];
        int class_id = (int)(model->unconditioned_class_count + k);
        size_t tagged_count = 0;
        size_t i;
        size_t j;
        for (i = 0; i < model->class_position_count; i++) {
            const rg_class_position *position = &model->class_positions[i];
            if (position->cognate_index < cognate_count && position_has_class(position, class_id)) {
                tagged[tagged_count++] = i;
            }
        }
        for (j = 0; j < row->segment_count && status == RG_OK; j++) {
            const rg_context_spec *environment = &row->contexts[j];
            size_t pivot = model_lect_index(model, row->lect_ids[j]);
            size_t s;
            if (rg_context_spec_constraint_count(environment) == 0 ||
                grapheme_is_gap(row->graphemes[j]) || pivot == (size_t)-1) {
                continue;
            }
            for (s = 0; s < row->segment_count && status == RG_OK; s++) {
                rg_lost_trigger_row out;
                size_t sister = model_lect_index(model, row->lect_ids[s]);
                int sister_gap = grapheme_is_gap(row->graphemes[s]);
                if (s == j || sister == (size_t)-1) {
                    continue;
                }
                memset(&out, 0, sizeof(out));
                out.class_id = class_id;
                out.pivot_lect = row->lect_ids[j];
                out.environment = environment;
                out.sister_lect = row->lect_ids[s];
                out.sister_grapheme = row->graphemes[s];
                stamp++;
                for (i = 0; i < tagged_count && status == RG_OK; i++) {
                    const rg_class_position *position = &model->class_positions[tagged[i]];
                    double weight = cognate_weight(&cognates[position->cognate_index]);
                    int pivot_slot = position_slot(position, pivot);
                    int sister_slot = position_slot(position, sister);
                    const rg_context_spec *observed = 0;
                    int holds = 0;
                    if (pivot_slot < 0 || sister_slot < 0) {
                        continue;
                    }
                    /* A class merged from two pivots carries a context from
                     * each, and a position need satisfy only one: the trace
                     * reads the positions where this pivot's environment
                     * holds. */
                    status = observed_context(ctx, cognates, model, &cache, position->cognate_index,
                                              pivot, position->positions[pivot_slot], &observed);
                    if (status != RG_OK || observed == 0) {
                        continue;
                    }
                    status = environment_holds(environment, observed, &holds);
                    if (status != RG_OK || !holds) {
                        continue;
                    }
                    if (sister_gap) {
                        out.untestable += weight;
                        continue;
                    }
                    status = observed_context(ctx, cognates, model, &cache, position->cognate_index,
                                              sister, position->positions[sister_slot], &observed);
                    if (status != RG_OK) {
                        break;
                    }
                    if (observed == 0) {
                        out.untestable += weight;
                        continue;
                    }
                    status = environment_holds(environment, observed, &holds);
                    if (status != RG_OK) {
                        break;
                    }
                    out.tested += weight;
                    out.holds += holds ? weight : 0.0;
                    if (stamps[position->cognate_index] != stamp) {
                        stamps[position->cognate_index] = stamp;
                        out.tested_cognate_count++;
                    }
                }
                if (status == RG_OK && (out.tested > 0.0 || out.untestable > 0.0)) {
                    status = append_row(model, &cap, &out);
                }
            }
        }
    }

    if (cache.contexts != 0) {
        for (k = 0; k < cognate_count * model->lect_count; k++) {
            rg_context_spec_array_free_internal(cache.contexts[k], cache.counts[k]);
        }
    }
    free(cache.contexts);
    free(cache.counts);
    free(stamps);
    free(tagged);
    return status;
}
