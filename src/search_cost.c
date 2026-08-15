#include "search_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Whether the source form satisfies a cross-dimensional rule's environment at
 * a position. The environment is an rg_context_spec, so this walks the slots
 * it can name: the preceding segment, the segment itself, the following one.
 * Suprasegmentals appear as features called "tone", "length" and "stress". */
static int cross_dimensional_slot_holds(
    const rg_context *ctx,
    const rg_form *form,
    int index,
    const rg_feature_constraint *constraints,
    size_t count
) {
    size_t i;
    for (i = 0; i < count; i++) {
        const char *feature = constraints[i].feature;
        const char *want = constraints[i].value == 0 ? "+" : constraints[i].value;
        int negated = strcmp(want, "-") == 0;
        int holds;
        if (index < 0 || (size_t)index >= form->segment_count) {
            return 0;
        }
        if (strcmp(feature, "tone") == 0 || strcmp(feature, "length") == 0 ||
            strcmp(feature, "stress") == 0) {
            const char *value = strcmp(feature, "tone") == 0 ? form->segments[index].tone
                : (strcmp(feature, "length") == 0 ? form->segments[index].length
                                                  : form->segments[index].stress);
            holds = value != 0 && strcmp(value, want) == 0;
            if (negated) {
                holds = !(value != 0 && value[0] != '\0');
            }
        } else {
            const rg_feature_set *features = 0;
            if (form->segments[index].grapheme == 0 ||
                rg_context_features_internal(ctx, form->segments[index].grapheme, &features) != RG_OK) {
                return 0;
            }
            holds = feature_set_contains(features, feature);
            if (negated) {
                holds = !holds;
            }
        }
        if (!holds) {
            return 0;
        }
    }
    return 1;
}

static int cross_dimensional_source_holds(
    const rg_context *ctx,
    const rg_form *form,
    size_t src_pos,
    const rg_cross_dimensional_row *row
) {
    const rg_context_spec *environment;
    if (ctx == 0 || form == 0 || row == 0) {
        return 0;
    }
    environment = &row->source_environment;
    if (rg_context_spec_constraint_count(environment) == 0) {
        return 0;
    }
    return cross_dimensional_slot_holds(ctx, form, (int)src_pos - 1,
                                        environment->preceding, environment->preceding_count) &&
           cross_dimensional_slot_holds(ctx, form, (int)src_pos,
                                        environment->self, environment->self_count) &&
           cross_dimensional_slot_holds(ctx, form, (int)src_pos + 1,
                                        environment->following, environment->following_count);
}

static double cross_dimensional_adjustment_for_row(
    const rg_pairwise_model *model,
    const rg_cross_dimensional_row *row,
    const char *actual
) {
    size_t i;
    double total_all = 0.0;
    double total_for_value = 0.0;
    double distinct = 0.0;
    double p_cond;
    double p_base;
    double alpha = 1.0;
    if (model == 0 || row == 0 || actual == 0 || actual[0] == '\0') {
        return 0.0;
    }
    for (i = 0; i < model->tonal_count_count; i++) {
        int first_for_tone = 1;
        size_t j;
        total_all += model->tonal_counts[i].count;
        if (strcmp(model->tonal_counts[i].target_tone, row->target_value) == 0) {
            total_for_value += model->tonal_counts[i].count;
        }
        if (model->tonal_counts[i].target_tone[0] == '\0') {
            first_for_tone = 0;
        }
        for (j = 0; j < i; j++) {
            if (strcmp(model->tonal_counts[i].target_tone, model->tonal_counts[j].target_tone) == 0) {
                first_for_tone = 0;
                break;
            }
        }
        if (first_for_tone) {
            distinct += 1.0;
        }
    }
    if (distinct < 2.0) {
        distinct = 2.0;
    }
    p_cond = (row->count + alpha) / (row->source_count + alpha * distinct);
    if (p_cond < 1e-12) {
        p_cond = 1e-12;
    }
    if (p_cond > 1.0 - 1e-12) {
        p_cond = 1.0 - 1e-12;
    }
    if (total_all <= 0.0) {
        p_base = 1.0 / distinct;
    } else {
        p_base = (total_for_value + alpha) / (total_all + alpha * distinct);
    }
    if (p_base < 1e-12) {
        p_base = 1e-12;
    }
    if (p_base > 1.0 - 1e-12) {
        p_base = 1.0 - 1e-12;
    }
    if (strcmp(actual, row->target_value) == 0) {
        return -(log(p_cond) - log(p_base));
    }
    return -(log(1.0 - p_cond) - log(1.0 - p_base));
}

double cross_dimensional_alignment_adjustment(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    const rg_alignment *alignment
) {
    size_t link_i;
    size_t src_pos = 0;
    size_t tgt_pos = 0;
    double total = 0.0;
    if (ctx == 0 || model == 0 || alignment == 0 || model->cross_dimensional_count == 0) {
        return 0.0;
    }
    for (link_i = 0; link_i < alignment->link_count; link_i++) {
        const rg_link *link = &alignment->links[link_i];
        if (link->source_count == 1 && link->target_count == 1) {
            size_t row_i;
            for (row_i = 0; row_i < model->cross_dimensional_count; row_i++) {
                const rg_cross_dimensional_row *row = &model->cross_dimensional_rows[row_i];
                int tgt_index = (int)tgt_pos + row->target_position_offset;
                const char *actual = "";
                if (!cross_dimensional_source_holds(ctx, &alignment->source_form, src_pos, row)) {
                    continue;
                }
                if (tgt_index < 0 || (size_t)tgt_index >= alignment->target_form.segment_count) {
                    continue;
                }
                if (strcmp(row->target_dimension, "tone") == 0) {
                    actual = alignment->target_form.segments[tgt_index].tone == 0 ? "" : alignment->target_form.segments[tgt_index].tone;
                } else if (strcmp(row->target_dimension, "length") == 0) {
                    actual = alignment->target_form.segments[tgt_index].length == 0 ? "" : alignment->target_form.segments[tgt_index].length;
                } else if (strcmp(row->target_dimension, "stress") == 0) {
                    actual = alignment->target_form.segments[tgt_index].stress == 0 ? "" : alignment->target_form.segments[tgt_index].stress;
                }
                total += cross_dimensional_adjustment_for_row(model, row, actual);
            }
        }
        src_pos += link->source_count;
        tgt_pos += link->target_count;
    }
    return total;
}

rg_status rg_alignment_cost(const rg_context *ctx, const rg_alignment *alignment, double *out) {
    size_t i;
    double total = 0.0;
    rg_status status;
    if (ctx == 0 || alignment == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0.0;
    for (i = 0; i < alignment->link_count; i++) {
        double link_cost = 0.0;
        const rg_link *link = &alignment->links[i];
        status = rg_score_link(ctx, link->source, link->source_count, link->target, link->target_count, &link_cost);
        if (status != RG_OK) {
            return status;
        }
        total += link_cost;
    }
    *out = total;
    return RG_OK;
}

rg_status rg_alignment_cost_with_model(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    const rg_train_options *options,
    const rg_alignment *alignment,
    double *out
) {
    size_t i;
    size_t target_position = 0;
    double total = 0.0;
    rg_context_spec *target_contexts = 0;
    size_t target_context_count = 0;
    rg_status status;
    if (ctx == 0 || model == 0 || alignment == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0.0;
    /* A link carries the source form's context only, so the target's is
     * rebuilt here. A rule naming the target's environment would otherwise be
     * silently inert whenever a cost is taken of an alignment already made. */
    if (model->has_target_conditioned) {
        status = rg_form_position_contexts_internal(ctx, &alignment->target_form,
                                                    &target_contexts, &target_context_count);
        if (status != RG_OK) {
            return status;
        }
    }
    for (i = 0; i < alignment->link_count; i++) {
        double link_cost = 0.0;
        const rg_link *link = &alignment->links[i];
        const rg_context_spec *target_context =
            (link->target_count == 1 && target_position < target_context_count)
                ? &target_contexts[target_position]
                : 0;
        target_position += link->target_count;
        status = rg_score_link_with_context_model_internal(ctx, model, options, link->source, link->source_count, link->target, link->target_count, &link->context, target_context, &link_cost);
        if (status != RG_OK) {
            rg_context_spec_array_free_internal(target_contexts, target_context_count);
            return status;
        }
        total += link_cost;
    }
    rg_context_spec_array_free_internal(target_contexts, target_context_count);
    total += cross_dimensional_alignment_adjustment(ctx, model, alignment);
    *out = total;
    return RG_OK;
}
