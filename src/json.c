#include "internal.h"

#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* JSON rendering of a trained model, and reading of a training-options object.
 *
 * This is a transport format for the CLI's --json output and the WebAssembly
 * adapter, not the interchange schema. The framework requires historia-facing
 * egress to carry ensembles and soft partitions; a single maximum-a-posteriori
 * correspondence system is explicitly not claim-capable data. Nothing here
 * should be mistaken for that format, which M8 designs. */

#define RG_JSON_FORMAT_VERSION 1

static cJSON *json_uncertainty(rg_uncertainty_estimate value) {
    cJSON *out = cJSON_CreateObject();
    if (out == 0) {
        return 0;
    }
    cJSON_AddNumberToObject(out, "estimate", value.estimate);
    cJSON_AddNumberToObject(out, "lower", value.lower);
    cJSON_AddNumberToObject(out, "upper", value.upper);
    cJSON_AddNumberToObject(out, "n", value.n);
    cJSON_AddNumberToObject(out, "alpha", value.alpha);
    cJSON_AddStringToObject(out, "method", rg_uncertainty_method_string(value.method));
    return out;
}

static cJSON *json_constraints(const rg_feature_constraint *items, size_t count) {
    cJSON *array = cJSON_CreateArray();
    size_t i;
    if (array == 0) {
        return 0;
    }
    for (i = 0; i < count; i++) {
        cJSON *entry = cJSON_CreateObject();
        if (entry == 0) {
            cJSON_Delete(array);
            return 0;
        }
        cJSON_AddStringToObject(entry, "feature", items[i].feature == 0 ? "" : items[i].feature);
        cJSON_AddStringToObject(entry, "value", items[i].value == 0 ? "" : items[i].value);
        cJSON_AddItemToArray(array, entry);
    }
    return array;
}

static cJSON *json_distance_constraints(const rg_distance_constraint *items, size_t count) {
    cJSON *array = cJSON_CreateArray();
    size_t i;
    if (array == 0) {
        return 0;
    }
    for (i = 0; i < count; i++) {
        cJSON *entry = cJSON_CreateObject();
        if (entry == 0) {
            cJSON_Delete(array);
            return 0;
        }
        cJSON_AddNumberToObject(entry, "offset", items[i].offset);
        cJSON_AddStringToObject(entry, "feature",
                                items[i].constraint.feature == 0 ? "" : items[i].constraint.feature);
        cJSON_AddStringToObject(entry, "value",
                                items[i].constraint.value == 0 ? "" : items[i].constraint.value);
        cJSON_AddItemToArray(array, entry);
    }
    return array;
}

/* Only constrained slots appear, so an unconditioned environment renders as an
 * empty object rather than a wall of nulls. */
static cJSON *json_context(const rg_context_spec *context) {
    cJSON *out = cJSON_CreateObject();
    if (out == 0) {
        return 0;
    }
    if (context == 0) {
        return out;
    }
    if (context->position != 0 && context->position[0] != '\0') {
        cJSON_AddStringToObject(out, "position", context->position);
    }
    if (context->morpheme_index != 0 && context->morpheme_index[0] != '\0') {
        cJSON_AddStringToObject(out, "morpheme_index", context->morpheme_index);
    }
    if (context->morphological != 0 && context->morphological[0] != '\0') {
        cJSON_AddStringToObject(out, "morphological", context->morphological);
    }

#define SLOT(name, field)                                                        \
    if (context->field##_count > 0) {                                            \
        cJSON_AddItemToObject(out, name,                                         \
                              json_constraints(context->field, context->field##_count)); \
    }

    SLOT("preceding", preceding)
    SLOT("following", following)
    SLOT("somewhere_preceding", somewhere_preceding)
    SLOT("somewhere_following", somewhere_following)
    SLOT("same_syllable", same_syllable)
    SLOT("next_syllable", next_syllable)
    SLOT("previous_syllable", previous_syllable)
    SLOT("self_stress", self_stress)
    SLOT("preceding_stress", preceding_stress)
    SLOT("following_stress", following_stress)

#undef SLOT

    if (context->preceding_at_distance_count > 0) {
        cJSON_AddItemToObject(out, "preceding_at_distance",
                              json_distance_constraints(context->preceding_at_distance,
                                                        context->preceding_at_distance_count));
    }
    if (context->following_at_distance_count > 0) {
        cJSON_AddItemToObject(out, "following_at_distance",
                              json_distance_constraints(context->following_at_distance,
                                                        context->following_at_distance_count));
    }
    return out;
}

static cJSON *json_class(const rg_multi_class_row *row, int with_contexts) {
    cJSON *out = cJSON_CreateObject();
    cJSON *segments;
    size_t i;

    if (out == 0) {
        return 0;
    }
    cJSON_AddNumberToObject(out, "id", row->class_id);
    cJSON_AddNumberToObject(out, "count", row->count);
    cJSON_AddNumberToObject(out, "confidence", row->confidence);
    /* The comparison the split was measured against. Zero on an unconditioned
     * class, which has no environment and so no complement. */
    cJSON_AddNumberToObject(out, "contrast_count", row->contrast_count);
    cJSON_AddNumberToObject(out, "delta_bic", row->delta_bic);
    cJSON_AddNumberToObject(out, "search_margin", row->search_margin);

    segments = cJSON_CreateArray();
    if (segments == 0) {
        cJSON_Delete(out);
        return 0;
    }
    for (i = 0; i < row->segment_count; i++) {
        cJSON *entry = cJSON_CreateObject();
        if (entry == 0) {
            cJSON_Delete(segments);
            cJSON_Delete(out);
            return 0;
        }
        cJSON_AddStringToObject(entry, "lect", row->lect_ids[i]);
        cJSON_AddStringToObject(entry, "grapheme", row->graphemes[i]);
        if (with_contexts && row->contexts != 0) {
            cJSON_AddItemToObject(entry, "context", json_context(&row->contexts[i]));
        }
        cJSON_AddItemToArray(segments, entry);
    }
    cJSON_AddItemToObject(out, "segments", segments);

    if (row->supporting_cognate_count > 0) {
        cJSON *support = cJSON_CreateArray();
        if (support == 0) {
            cJSON_Delete(out);
            return 0;
        }
        for (i = 0; i < row->supporting_cognate_count; i++) {
            cJSON_AddItemToArray(support, cJSON_CreateString(row->supporting_cognates[i]));
        }
        cJSON_AddItemToObject(out, "supporting_cognates", support);
    }
    cJSON_AddItemToObject(out, "uncertainty", json_uncertainty(row->uncertainty));
    return out;
}

static cJSON *json_segments(const rg_segment *segments, size_t count) {
    cJSON *array = cJSON_CreateArray();
    size_t i;
    if (array == 0) {
        return 0;
    }
    for (i = 0; i < count; i++) {
        cJSON_AddItemToArray(array,
                             cJSON_CreateString(segments[i].grapheme == 0 ? "" : segments[i].grapheme));
    }
    return array;
}

/* Alignments under the trained model, walked in ascending lect-id order so the
 * output matches the view reconciliation takes of the same data. */
static rg_status json_add_alignments(
    const rg_context *ctx,
    const rg_multi_model *model,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    cJSON *root
) {
    cJSON *array = cJSON_CreateArray();
    size_t c;

    if (array == 0) {
        return RG_ERR_OOM;
    }
    cJSON_AddItemToObject(root, "alignments", array);

    for (c = 0; c < cognate_count; c++) {
        const rg_cognate_set *cognate = &cognates[c];
        size_t i;
        size_t *order;
        size_t order_count = 0;

        order = (size_t *)calloc(cognate->form_count == 0 ? 1 : cognate->form_count, sizeof(*order));
        if (order == 0) {
            return RG_ERR_OOM;
        }
        for (i = 0; i < cognate->form_count; i++) {
            size_t insert_at = order_count;
            while (insert_at > 0 &&
                   strcmp(cognate->forms[order[insert_at - 1]].lect_id, cognate->forms[i].lect_id) > 0) {
                insert_at--;
            }
            if (insert_at < order_count) {
                memmove(&order[insert_at + 1], &order[insert_at], (order_count - insert_at) * sizeof(*order));
            }
            order[insert_at] = i;
            order_count++;
        }

        for (i = 0; i < order_count; i++) {
            size_t j;
            for (j = i + 1; j < order_count; j++) {
                const char *lect_a = cognate->forms[order[i]].lect_id;
                const char *lect_b = cognate->forms[order[j]].lect_id;
                const rg_pairwise_model *pair_model = 0;
                rg_alignment *alignment = 0;
                cJSON *entry;
                cJSON *links;
                double cost = 0.0;
                size_t p;
                size_t link_i;
                size_t source_pos;
                size_t target_pos;
                size_t lect_index_a = 0;
                size_t lect_index_b = 0;

                for (p = 0; p < rg_multi_model_pair_model_count(model); p++) {
                    const rg_multi_pair_model_row *pair = rg_multi_model_pair_model_at(model, p);
                    if ((strcmp(pair->lect_a, lect_a) == 0 && strcmp(pair->lect_b, lect_b) == 0) ||
                        (strcmp(pair->lect_a, lect_b) == 0 && strcmp(pair->lect_b, lect_a) == 0)) {
                        pair_model = pair->model;
                        break;
                    }
                }
                if (pair_model == 0) {
                    continue;
                }
                for (p = 0; p < rg_multi_model_lect_count(model); p++) {
                    const char *name = rg_multi_model_lect_at(model, p);
                    if (strcmp(name, lect_a) == 0) {
                        lect_index_a = p;
                    }
                    if (strcmp(name, lect_b) == 0) {
                        lect_index_b = p;
                    }
                }
                if (rg_align_forms_with_model(ctx, pair_model, options,
                                              &cognate->forms[order[i]].form,
                                              &cognate->forms[order[j]].form,
                                              0, &alignment) != RG_OK) {
                    free(order);
                    return RG_ERR_MERKMAL;
                }
                rg_alignment_cost_with_model(ctx, pair_model, options, alignment, &cost);

                entry = cJSON_CreateObject();
                if (entry == 0) {
                    rg_alignment_free(alignment);
                    free(order);
                    return RG_ERR_OOM;
                }
                cJSON_AddStringToObject(entry, "cognate_id", cognate->cognate_id == 0 ? "" : cognate->cognate_id);
                cJSON_AddStringToObject(entry, "lect_a", lect_a);
                cJSON_AddStringToObject(entry, "lect_b", lect_b);
                cJSON_AddNumberToObject(entry, "cost", cost);
                links = cJSON_CreateArray();
                if (links == 0) {
                    cJSON_Delete(entry);
                    rg_alignment_free(alignment);
                    free(order);
                    return RG_ERR_OOM;
                }
                source_pos = 0;
                target_pos = 0;
                for (link_i = 0; link_i < rg_alignment_link_count(alignment); link_i++) {
                    const rg_link *link = rg_alignment_link_at(alignment, link_i);
                    cJSON *link_entry = cJSON_CreateObject();
                    if (link_entry == 0) {
                        cJSON_Delete(links);
                        cJSON_Delete(entry);
                        rg_alignment_free(alignment);
                        free(order);
                        return RG_ERR_OOM;
                    }
                    cJSON_AddItemToObject(link_entry, "source", json_segments(link->source, link->source_count));
                    cJSON_AddItemToObject(link_entry, "target", json_segments(link->target, link->target_count));
                    /* Which classes this link realises, taken from the
                     * reconciliation rather than re-derived by matching
                     * graphemes, which could not tell a conditioned class from
                     * the unconditioned one over the same segments. A
                     * multi-segment link reports the union over the positions
                     * it spans; a gap reports nothing. */
                    if (link->source_count > 0 && link->target_count > 0) {
                        int ids[64];
                        size_t found = 0;
                        size_t si;
                        /* Every source position against every target position
                         * the link spans, not just the diagonal. A 2-to-1 link
                         * has one diagonal cell and two positions, and a class
                         * reconciled at the off-diagonal one used to be
                         * unreportable: the model held its evidence and the
                         * export could not name it, so the class appeared to
                         * rest on nothing. */
                        for (si = 0; si < link->source_count && found < 64; si++) {
                            size_t ti;
                            for (ti = 0; ti < link->target_count && found < 64; ti++) {
                                int candidates[64];
                                size_t n = rg_model_classes_at_internal(
                                    model, c,
                                    lect_index_a, source_pos + si,
                                    lect_index_b, target_pos + ti,
                                    candidates, 64);
                                size_t k;
                                for (k = 0; k < n && found < 64; k++) {
                                    size_t seen;
                                    int duplicate = 0;
                                    for (seen = 0; seen < found; seen++) {
                                        if (ids[seen] == candidates[k]) {
                                            duplicate = 1;
                                            break;
                                        }
                                    }
                                    if (!duplicate) {
                                        ids[found++] = candidates[k];
                                    }
                                }
                            }
                        }
                        if (found > 0) {
                            cJSON *class_ids = cJSON_CreateArray();
                            size_t n;
                            if (class_ids == 0) {
                                cJSON_Delete(link_entry);
                                cJSON_Delete(links);
                                cJSON_Delete(entry);
                                rg_alignment_free(alignment);
                                free(order);
                                return RG_ERR_OOM;
                            }
                            for (n = 0; n < found; n++) {
                                cJSON_AddItemToArray(class_ids, cJSON_CreateNumber(ids[n]));
                            }
                            cJSON_AddItemToObject(link_entry, "classes", class_ids);
                        }
                    }
                    source_pos += link->source_count;
                    target_pos += link->target_count;
                    cJSON_AddItemToArray(links, link_entry);
                }
                cJSON_AddItemToObject(entry, "links", links);
                cJSON_AddItemToArray(array, entry);
                rg_alignment_free(alignment);
            }
        }
        free(order);
    }
    return RG_OK;
}

char *rg_error_to_json(rg_status status, const char *detail) {
    return rg_json_error_internal(status, detail);
}

rg_status rg_train_options_from_json(
    const char *text,
    rg_train_options *out,
    char *error_detail,
    size_t error_detail_size
) {
    return rg_json_read_train_options_internal(text, out, error_detail, error_detail_size);
}

char *rg_segments_to_json(const rg_segment *segments, size_t count) {
    cJSON *root = cJSON_CreateObject();
    char *text;

    if (root == 0) {
        return 0;
    }
    cJSON_AddBoolToObject(root, "ok", 1);
    cJSON_AddNumberToObject(root, "format_version", RG_JSON_FORMAT_VERSION);
    cJSON_AddItemToObject(root, "segments", json_segments(segments, count));
    text = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    return text;
}

char *rg_model_to_json(
    const rg_context *ctx,
    const rg_multi_model *model,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    int include_alignments,
    int include_outliers
) {
    return rg_json_from_multi_model_internal(ctx, model, cognates, cognate_count, options,
                                             include_alignments, include_outliers);
}

char *rg_json_from_multi_model_internal(
    const rg_context *ctx,
    const rg_multi_model *model,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    int include_alignments,
    int include_outliers
) {
    cJSON *root;
    cJSON *lects;
    cJSON *classes;
    cJSON *array;
    char *text;
    size_t i;

    if (model == 0) {
        return 0;
    }
    root = cJSON_CreateObject();
    if (root == 0) {
        return 0;
    }
    cJSON_AddBoolToObject(root, "ok", 1);
    cJSON_AddNumberToObject(root, "format_version", RG_JSON_FORMAT_VERSION);
    cJSON_AddStringToObject(root, "regulae_version", rg_version_string());
    /* Labelled so nothing downstream mistakes this for claim-capable egress. */
    cJSON_AddStringToObject(root, "export_kind", "debug_map_snapshot");

    lects = cJSON_CreateArray();
    if (lects == 0) {
        cJSON_Delete(root);
        return 0;
    }
    for (i = 0; i < rg_multi_model_lect_count(model); i++) {
        cJSON_AddItemToArray(lects, cJSON_CreateString(rg_multi_model_lect_at(model, i)));
    }
    cJSON_AddItemToObject(root, "lects", lects);

    {
        const rg_corpus_fit *f = rg_multi_model_fit(model);
        cJSON *fit = cJSON_CreateObject();
        if (fit == 0) {
            cJSON_Delete(root);
            return 0;
        }
        cJSON_AddItemToObject(root, "fit", fit);
        cJSON_AddNumberToObject(fit, "cost_per_segment", f->cost_per_segment);
        cJSON_AddNumberToObject(fit, "scored_set_count", (double)f->scored_set_count);
        cJSON_AddNumberToObject(fit, "unconditioned_class_count", (double)f->unconditioned_class_count);
        cJSON_AddNumberToObject(fit, "conditioned_class_count", (double)f->conditioned_class_count);
        cJSON_AddNumberToObject(fit, "permutation_count", (double)f->permutation_count);
        if (f->permutation_count > 0) {
            cJSON_AddNumberToObject(fit, "null_cost_per_segment_mean", f->null_cost_per_segment_mean);
            cJSON_AddNumberToObject(fit, "null_cost_per_segment_sd", f->null_cost_per_segment_sd);
            cJSON_AddNumberToObject(fit, "cost_per_segment_z", f->cost_per_segment_z);
            cJSON_AddNumberToObject(fit, "null_unconditioned_class_mean", f->null_unconditioned_class_mean);
            cJSON_AddNumberToObject(fit, "null_conditioned_class_mean", f->null_conditioned_class_mean);
            cJSON_AddNumberToObject(fit, "null_search_margin", f->null_search_margin);
            cJSON_AddNumberToObject(fit, "null_search_margin_quantile", f->null_search_margin_quantile);
        }
    }

    classes = cJSON_CreateObject();
    if (classes == 0) {
        cJSON_Delete(root);
        return 0;
    }
    cJSON_AddItemToObject(root, "classes", classes);

    array = cJSON_CreateArray();
    if (array == 0) {
        cJSON_Delete(root);
        return 0;
    }
    cJSON_AddItemToObject(classes, "unconditioned", array);
    for (i = 0; i < rg_multi_model_unconditioned_class_count(model); i++) {
        cJSON_AddItemToArray(array, json_class(rg_multi_model_unconditioned_class_at(model, i), 0));
    }

    array = cJSON_CreateArray();
    if (array == 0) {
        cJSON_Delete(root);
        return 0;
    }
    cJSON_AddItemToObject(classes, "conditioned", array);
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        cJSON_AddItemToArray(array, json_class(rg_multi_model_conditioned_class_at(model, i), 1));
    }

    array = cJSON_CreateArray();
    if (array == 0) {
        cJSON_Delete(root);
        return 0;
    }
    cJSON_AddItemToObject(root, "cross_dimensional", array);
    for (i = 0; i < rg_multi_model_cross_dimensional_row_count(model); i++) {
        const rg_multi_cross_dimensional_row *row = rg_multi_model_cross_dimensional_row_at(model, i);
        cJSON *entry = cJSON_CreateObject();
        if (entry == 0) {
            cJSON_Delete(root);
            return 0;
        }
        cJSON_AddStringToObject(entry, "source_lect", row->source_lect);
        cJSON_AddStringToObject(entry, "target_lect", row->target_lect);
        cJSON_AddStringToObject(entry, "source_feature", row->source_feature);
        cJSON_AddStringToObject(entry, "source_value", row->source_value);
        cJSON_AddStringToObject(entry, "source_position", row->source_position);
        cJSON_AddStringToObject(entry, "target_dimension", row->target_dimension);
        cJSON_AddStringToObject(entry, "target_value", row->target_value);
        cJSON_AddNumberToObject(entry, "target_position_offset", row->target_position_offset);
        cJSON_AddNumberToObject(entry, "count", row->count);
        cJSON_AddNumberToObject(entry, "source_count", row->source_count);
        cJSON_AddNumberToObject(entry, "confidence", row->confidence);
        cJSON_AddNumberToObject(entry, "contrast_count", row->contrast_count);
        cJSON_AddNumberToObject(entry, "contrast_source_count", row->contrast_source_count);
        cJSON_AddNumberToObject(entry, "contrast_confidence", row->contrast_confidence);
        cJSON_AddNumberToObject(entry, "delta_bic", row->delta_bic);
        cJSON_AddItemToObject(entry, "uncertainty", json_uncertainty(row->uncertainty));
        cJSON_AddItemToArray(array, entry);
    }

    if (include_alignments && ctx != 0 && cognate_count > 0) {
        if (json_add_alignments(ctx, model, cognates, cognate_count, options, root) != RG_OK) {
            cJSON_Delete(root);
            return 0;
        }
    }

    if (include_outliers && ctx != 0 && cognate_count > 0) {
        rg_cognate_outlier_row *rows = 0;
        size_t row_count = 0;
        if (rg_find_cognate_outliers(ctx, cognates, cognate_count, model, options, 0, 0,
                                     &rows, &row_count) == RG_OK) {
            array = cJSON_CreateArray();
            if (array == 0) {
                rg_cognate_outlier_rows_free(rows, row_count);
                cJSON_Delete(root);
                return 0;
            }
            cJSON_AddItemToObject(root, "outliers", array);
            for (i = 0; i < row_count; i++) {
                cJSON *entry = cJSON_CreateObject();
                if (entry == 0) {
                    break;
                }
                cJSON_AddStringToObject(entry, "cognate_id", rows[i].cognate_id);
                cJSON_AddNumberToObject(entry, "pair_count", rows[i].pair_count);
                cJSON_AddNumberToObject(entry, "cost_per_segment", rows[i].cost_per_segment);
                cJSON_AddNumberToObject(entry, "z_score", rows[i].z_score);
                cJSON_AddItemToArray(array, entry);
            }
            rg_cognate_outlier_rows_free(rows, row_count);
        }
    }

    text = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    return text;
}

char *rg_json_error_internal(rg_status status, const char *detail) {
    cJSON *root = cJSON_CreateObject();
    char *text;
    if (root == 0) {
        return 0;
    }
    cJSON_AddBoolToObject(root, "ok", 0);
    cJSON_AddNumberToObject(root, "format_version", RG_JSON_FORMAT_VERSION);
    cJSON_AddStringToObject(root, "status", rg_status_string(status));
    cJSON_AddNumberToObject(root, "status_code", status);
    if (detail != 0 && detail[0] != '\0') {
        cJSON_AddStringToObject(root, "detail", detail);
    }
    text = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    return text;
}

/* Reads a flat options object. Unknown keys are rejected rather than ignored:
 * a caller who misspells a knob should be told, not silently given defaults. */
rg_status rg_json_read_train_options_internal(
    const char *text,
    rg_train_options *out,
    char *error_detail,
    size_t error_detail_size
) {
    cJSON *root;
    cJSON *item;
    rg_status status = RG_OK;

    if (out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    rg_train_options_init_defaults(out);
    if (error_detail != 0 && error_detail_size > 0) {
        error_detail[0] = '\0';
    }
    if (text == 0 || text[0] == '\0') {
        return RG_OK;
    }
    root = cJSON_Parse(text);
    if (root == 0) {
        if (error_detail != 0 && error_detail_size > 0) {
            snprintf(error_detail, error_detail_size, "options are not valid JSON");
        }
        return RG_ERR_PARSE;
    }
    if (!cJSON_IsObject(root)) {
        cJSON_Delete(root);
        if (error_detail != 0 && error_detail_size > 0) {
            snprintf(error_detail, error_detail_size, "options must be a JSON object");
        }
        return RG_ERR_PARSE;
    }

    cJSON_ArrayForEach(item, root) {
        const char *key = item->string;
        if (key == 0) {
            continue;
        }

#define NUMBER_FIELD(name, field, kind)                                          \
        if (strcmp(key, name) == 0) {                                            \
            if (!cJSON_IsNumber(item)) {                                         \
                status = RG_ERR_PARSE;                                           \
                if (error_detail != 0 && error_detail_size > 0) {                \
                    snprintf(error_detail, error_detail_size,                    \
                             "option \"%s\" must be a number", name);            \
                }                                                                \
                break;                                                           \
            }                                                                    \
            out->field = (kind)item->valuedouble;                                \
            continue;                                                            \
        }

        NUMBER_FIELD("max_chunk_size", max_chunk_size, int)
        NUMBER_FIELD("temperature", temperature, double)
        NUMBER_FIELD("concentration", concentration, double)
        NUMBER_FIELD("max_iter", max_iter, int)
        NUMBER_FIELD("convergence_eps", convergence_eps, double)
        NUMBER_FIELD("segment_weight", segment_weight, double)
        NUMBER_FIELD("displacement_weight", displacement_weight, double)
        NUMBER_FIELD("tone_weight", tone_weight, double)
        NUMBER_FIELD("chunk_min_transparency", chunk_min_transparency, double)
        NUMBER_FIELD("bootstrap_n", bootstrap_n, int)
        NUMBER_FIELD("bootstrap_seed", bootstrap_seed, int)
        NUMBER_FIELD("delta_bic_threshold", bic.delta_bic_threshold, double)
        NUMBER_FIELD("min_split_observations", bic.min_split_observations, int)
        NUMBER_FIELD("max_split_depth", bic.max_split_depth, int)
        NUMBER_FIELD("min_chunk_observations", bic.min_chunk_observations, int)
        NUMBER_FIELD("long_range_delta_bic_threshold", bic.long_range_delta_bic_threshold, double)
        NUMBER_FIELD("long_range_min_split_observations", bic.long_range_min_split_observations, int)
        NUMBER_FIELD("long_range_min_dominant_fraction", bic.long_range_min_dominant_fraction, double)
        NUMBER_FIELD("multi_lect_min_commit_scale", bic.multi_lect_min_commit_scale, double)

#undef NUMBER_FIELD

        status = RG_ERR_UNSUPPORTED_OPTION;
        if (error_detail != 0 && error_detail_size > 0) {
            snprintf(error_detail, error_detail_size, "unknown option \"%s\"", key);
        }
        break;
    }

    cJSON_Delete(root);
    if (status != RG_OK) {
        rg_train_options_init_defaults(out);
    }
    return status;
}
