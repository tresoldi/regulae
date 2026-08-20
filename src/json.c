#include "internal.h"
#include "environment.h"

#include "cJSON.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* JSON rendering of a trained model, and reading of a training-options object.
 *
 * This is a transport format for the CLI's --json output and the WebAssembly
 * adapter, not the interchange schema. The framework requires historia-facing
 * egress to carry ensembles and soft partitions; a single maximum-a-posteriori
 * correspondence system is explicitly not claim-capable data. Nothing here
 * should be mistaken for that format, which a downstream interchange schema
 * designs. */

#define RG_JSON_FORMAT_VERSION 2

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
    cJSON_AddStringToObject(out, "observation_unit",
                            rg_observation_unit_string(value.observation_unit));
    cJSON_AddNumberToObject(out, "effective_n", value.effective_n);
    if (value.post_selection) {
        cJSON_AddBoolToObject(out, "post_selection", 1);
    }
    return out;
}

static cJSON *json_predictive_score(rg_predictive_score value) {
    cJSON *out = cJSON_CreateObject();
    if (out == 0) {
        return 0;
    }
    cJSON_AddNumberToObject(out, "observations", (double)value.observation_count);
    cJSON_AddNumberToObject(out, "unseen_reflexes", (double)value.unseen_reflex_count);
    cJSON_AddNumberToObject(out, "observation_weight", value.observation_weight);
    cJSON_AddNumberToObject(out, "log_loss", value.log_loss);
    cJSON_AddNumberToObject(out, "top1_coverage", value.top1_coverage);
    cJSON_AddNumberToObject(out, "top_k_coverage", value.top_k_coverage);
    cJSON_AddNumberToObject(out, "brier_score", value.brier_score);
    cJSON_AddNumberToObject(out, "calibration_error", value.calibration_error);
    cJSON_AddNumberToObject(out, "abstention_rate", value.abstention_rate);
    cJSON_AddNumberToObject(out, "accepted_top1_coverage", value.accepted_top1_coverage);
    return out;
}

static cJSON *json_predictive_evidence(rg_predictive_evidence value) {
    cJSON *out = cJSON_CreateObject();
    if (out == 0) {
        return 0;
    }
    cJSON_AddStringToObject(out, "status", rg_predictive_status_string(value.status));
    cJSON_AddStringToObject(out, "observation_unit",
                            rg_observation_unit_string(value.observation_unit));
    cJSON_AddNumberToObject(out, "folds", (double)value.folds);
    cJSON_AddNumberToObject(out, "log_loss_gain", value.log_loss_gain);
    cJSON_AddItemToObject(out, "conditioned", json_predictive_score(value.conditioned));
    cJSON_AddItemToObject(out, "unconditioned", json_predictive_score(value.unconditioned));
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
#define STRING_SLOT(name, key)                                                        \
    if (context->name != 0 && context->name[0] != '\0') {                        \
        cJSON_AddStringToObject(out, #name, context->name);                      \
    }
    RG_ENV_STRING_SLOTS(STRING_SLOT)
#undef STRING_SLOT

    /* Every feature slot, then every distance slot, both in the slot list's
     * order. The key is the field's own name, so a slot added to the list
     * serialises itself. */
#define SLOT(name, label, key)                                                        \
    if (context->name##_count > 0) {                                             \
        cJSON_AddItemToObject(out, #name,                                        \
                              json_constraints(context->name, context->name##_count)); \
    }
    RG_ENV_FEATURE_SLOTS(SLOT)
#undef SLOT

#define DISTANCE_SLOT(name, label, key)                                               \
    if (context->name##_count > 0) {                                             \
        cJSON_AddItemToObject(out, #name,                                        \
                              json_distance_constraints(context->name, context->name##_count)); \
    }
    RG_ENV_DISTANCE_SLOTS(DISTANCE_SLOT)
#undef DISTANCE_SLOT

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
    /* The row holding the pivot's other reflex out of the environment -- the
     * one a reader needs, since contrast_count above is ~0 whenever the
     * conditioning is real. -1 (and alternative 0) on an unconditioned class or
     * a complement with no majority. */
    cJSON_AddNumberToObject(out, "contrast_class_id", row->contrast_class_id);
    cJSON_AddNumberToObject(out, "contrast_alternative_count", row->contrast_alternative_count);
    /* Rival conditioners at another position the corpus cannot distinguish;
     * 0 = identifiable (and always 0 on an unconditioned class). */
    cJSON_AddNumberToObject(out, "environment_alternatives", row->environment_alternatives);
    if (row->evidence.decision_index >= 0) {
        cJSON_AddStringToObject(out, "score_kind", rg_split_scorer_string(row->evidence.scorer));
        cJSON_AddNumberToObject(out, "delta_score", row->evidence.delta_score);
    }
    cJSON_AddNumberToObject(out, "delta_bic", row->evidence.delta_bic);
    cJSON_AddNumberToObject(out, "search_margin", row->evidence.search_margin);
    /* Where this rule sits in the decision list. Sorting the table by key
     * destroys the order discovery settled them in, and that order carries
     * meaning: a later rule refines what an earlier one left. */
    cJSON_AddNumberToObject(out, "decision_index", row->evidence.decision_index);
    cJSON_AddStringToObject(out, "standing", rg_rule_standing_string(row->evidence.standing));
    cJSON_AddStringToObject(out, "standing_null",
                            rg_null_model_string(row->evidence.standing_null));
    cJSON_AddItemToObject(out, "predictive", json_predictive_evidence(row->evidence.predictive));

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
        /* Suprasegmentals only when carried, so a segmental corpus's model is
         * byte-for-byte what it was and only tonal corpora gain the fields. */
        if (row->suprasegmentals != 0) {
            const rg_suprasegmentals *s = &row->suprasegmentals[i];
            if (s->tone[0] != '\0') {
                cJSON_AddStringToObject(entry, "tone", s->tone);
            }
            if (s->length[0] != '\0') {
                cJSON_AddStringToObject(entry, "length", s->length);
            }
            if (s->stress[0] != '\0') {
                cJSON_AddStringToObject(entry, "stress", s->stress);
            }
        }
        if (with_contexts && row->contexts != 0) {
            cJSON_AddItemToObject(entry, "context", json_context(&row->contexts[i]));
        }
        cJSON_AddItemToArray(segments, entry);
    }
    cJSON_AddItemToObject(out, "segments", segments);

    {
        /* Always, even when empty. Omitting the key would make a conditioned
         * class with no support behind it a KeyError in the consumer rather
         * than a row with nothing behind it, so the gap would read as a bug in
         * whoever was reading the model. A field a
         * consumer must guard for absence is a field that gets skipped. */
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

/* Conditioned classes that look like one change. A proposal: every member is
 * published in its own right under `classes.conditioned`, and `class_ids`
 * points at them rather than replacing them. */
static cJSON *json_proposed_event(const rg_proposed_event_row *row) {
    cJSON *out = cJSON_CreateObject();
    cJSON *ids;
    cJSON *members;
    cJSON *support;
    size_t i;
    size_t j;
    if (out == 0) {
        return 0;
    }
    ids = cJSON_CreateArray();
    members = cJSON_CreateArray();
    support = cJSON_CreateArray();
    if (ids == 0 || members == 0 || support == 0) {
        cJSON_Delete(out);
        cJSON_Delete(ids);
        cJSON_Delete(members);
        cJSON_Delete(support);
        return 0;
    }
    for (i = 0; i < row->class_id_count; i++) {
        cJSON_AddItemToArray(ids, cJSON_CreateNumber(row->class_ids[i]));
    }
    cJSON_AddItemToObject(out, "class_ids", ids);
    for (i = 0; i < row->member_count; i++) {
        const rg_event_member *member = &row->members[i];
        cJSON *entry = cJSON_CreateObject();
        cJSON *graphemes = cJSON_CreateArray();
        cJSON *features = cJSON_CreateArray();
        if (entry == 0 || graphemes == 0 || features == 0) {
            cJSON_Delete(entry);
            cJSON_Delete(graphemes);
            cJSON_Delete(features);
            cJSON_Delete(out);
            cJSON_Delete(support);
            return 0;
        }
        cJSON_AddStringToObject(entry, "lect", member->lect_id);
        for (j = 0; j < member->grapheme_count; j++) {
            cJSON_AddItemToArray(graphemes, cJSON_CreateString(member->graphemes[j]));
        }
        cJSON_AddItemToObject(entry, "graphemes", graphemes);
        for (j = 0; j < member->class_feature_count; j++) {
            cJSON_AddItemToArray(features, cJSON_CreateString(member->class_features[j]));
        }
        cJSON_AddItemToObject(entry, "class_features", features);
        cJSON_AddItemToArray(members, entry);
    }
    cJSON_AddItemToObject(out, "members", members);
    for (i = 0; i < row->supporting_cognate_count; i++) {
        cJSON_AddItemToArray(support, cJSON_CreateString(row->supporting_cognates[i]));
    }
    cJSON_AddItemToObject(out, "supporting_cognates", support);
    cJSON_AddNumberToObject(out, "count", row->count);
    cJSON_AddBoolToObject(out, "featurally_definable", row->featurally_definable);
    cJSON_AddNumberToObject(out, "search_margin", row->search_margin);
    cJSON_AddNumberToObject(out, "delta_score", row->delta_score);
    if (row->shared_displacement_count > 0) {
        cJSON *disp = cJSON_CreateArray();
        if (disp != 0) {
            for (i = 0; i < row->shared_displacement_count; i++) {
                cJSON *item = cJSON_CreateObject();
                if (item != 0) {
                    cJSON_AddStringToObject(item, "feature",
                                            row->shared_displacement[i].feature);
                    cJSON_AddStringToObject(item, "from",
                                            row->shared_displacement[i].from_value);
                    cJSON_AddStringToObject(item, "to",
                                            row->shared_displacement[i].to_value);
                    cJSON_AddItemToArray(disp, item);
                }
            }
            cJSON_AddItemToObject(out, "shared_displacement", disp);
        }
    }
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
                size_t lect_total = 0;
                const char *const *lect_names = rg_multi_model_lects(model, &lect_total);
                for (p = 0; p < lect_total; p++) {
                    const char *name = lect_names[p];
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
                     * it spans; a gap link looks up the real side's position
                     * against the sentinel the reconciler stored for the
                     * missing side. */
                    {
                        int ids[64];
                        size_t found = 0;
                        if (link->source_count > 0 && link->target_count > 0) {
                            size_t si;
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
                            /* The aligner may merge a gap into a
                             * multi-segment link (e.g. "an ~ a") instead
                             * of producing a separate gap link. Probe
                             * each side against the gap sentinel so the
                             * reconciler's gap class is still found. */
                            for (si = 0; si < link->source_count && found < 64; si++) {
                                int candidates[64];
                                size_t n = rg_model_classes_at_internal(
                                    model, c,
                                    lect_index_a, source_pos + si,
                                    lect_index_b, (size_t)-1,
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
                            {
                                size_t ti;
                                for (ti = 0; ti < link->target_count && found < 64; ti++) {
                                    int candidates[64];
                                    size_t n = rg_model_classes_at_internal(
                                        model, c,
                                        lect_index_a, (size_t)-1,
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
                        } else if (link->source_count > 0) {
                            size_t si;
                            for (si = 0; si < link->source_count && found < 64; si++) {
                                int candidates[64];
                                size_t n = rg_model_classes_at_internal(
                                    model, c,
                                    lect_index_a, source_pos + si,
                                    lect_index_b, (size_t)-1,
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
                        } else if (link->target_count > 0) {
                            size_t ti;
                            for (ti = 0; ti < link->target_count && found < 64; ti++) {
                                int candidates[64];
                                size_t n = rg_model_classes_at_internal(
                                    model, c,
                                    lect_index_a, (size_t)-1,
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

/* FNV-1a over a canonical serialisation of the cognate sets, so the export
 * names the exact input it was trained on. Fields are unit-separated so no two
 * distinct corpora collide by concatenation, and the double's bytes hash
 * directly -- both the native and WebAssembly targets are little-endian, so the
 * checksum agrees across them. */
static void checksum_bytes(uint64_t *h, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    size_t i;
    for (i = 0; i < n; i++) {
        *h ^= p[i];
        *h *= UINT64_C(1099511628211);
    }
}

static void checksum_field(uint64_t *h, const char *value) {
    checksum_bytes(h, value != 0 ? value : "", value != 0 ? strlen(value) : 0);
    checksum_bytes(h, "\x1f", 1);
}

static void corpus_checksum(const rg_cognate_set *cognates, size_t count, char out[32]) {
    uint64_t h = UINT64_C(1469598103934665603);
    size_t c;
    size_t f;
    size_t s;
    for (c = 0; c < count; c++) {
        checksum_field(&h, cognates[c].cognate_id);
        checksum_field(&h, cognates[c].etymon_group);
        checksum_field(&h, cognates[c].source_group);
        checksum_bytes(&h, &cognates[c].confidence, sizeof(double));
        for (f = 0; f < cognates[c].form_count; f++) {
            const rg_form *form = &cognates[c].forms[f].form;
            checksum_field(&h, cognates[c].forms[f].lect_id);
            for (s = 0; s < form->segment_count; s++) {
                checksum_field(&h, form->segments[s].grapheme);
                checksum_field(&h, form->segments[s].tone);
                checksum_field(&h, form->segments[s].length);
                checksum_field(&h, form->segments[s].stress);
            }
        }
    }
    snprintf(out, 32, "fnv1a64:%016llx", (unsigned long long)h);
}

/* What the export was produced from: the feature system and its version (the
 * conditioning vocabulary is whatever it reports), a checksum of the exact
 * cognate sets, and the fully-resolved training options including every seed.
 * Enough to reproduce the run and to cite it. */
static cJSON *json_provenance(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options
) {
    cJSON *out = cJSON_CreateObject();
    cJSON *opts;
    cJSON *bic;
    rg_train_options resolved;
    const char *system_name = 0;
    char checksum[32];
    if (out == 0) {
        return 0;
    }
    if (options == 0) {
        rg_train_options_init_defaults(&resolved);
    } else {
        resolved = *options;
    }
    if (ctx == 0 || rg_context_system_name(ctx, &system_name) != RG_OK) {
        system_name = "";
    }
    corpus_checksum(cognates, cognate_count, checksum);
    cJSON_AddStringToObject(out, "feature_system", system_name);
    cJSON_AddStringToObject(out, "merkmal_version", rg_merkmal_version_internal());
    cJSON_AddStringToObject(out, "corpus_checksum", checksum);

    opts = cJSON_CreateObject();
    bic = cJSON_CreateObject();
    if (opts == 0 || bic == 0) {
        cJSON_Delete(opts);
        cJSON_Delete(bic);
        cJSON_Delete(out);
        return 0;
    }
    cJSON_AddNumberToObject(opts, "max_chunk_size", resolved.max_chunk_size);
    cJSON_AddNumberToObject(opts, "temperature", resolved.temperature);
    cJSON_AddNumberToObject(opts, "concentration", resolved.concentration);
    cJSON_AddNumberToObject(opts, "max_iter", resolved.max_iter);
    cJSON_AddNumberToObject(opts, "convergence_eps", resolved.convergence_eps);
    cJSON_AddNumberToObject(opts, "segment_weight", resolved.segment_weight);
    cJSON_AddNumberToObject(opts, "displacement_weight", resolved.displacement_weight);
    cJSON_AddNumberToObject(opts, "tone_weight", resolved.tone_weight);
    cJSON_AddNumberToObject(opts, "chunk_min_transparency", resolved.chunk_min_transparency);
    cJSON_AddNumberToObject(opts, "bootstrap_n", resolved.bootstrap_n);
    cJSON_AddNumberToObject(opts, "bootstrap_seed", resolved.bootstrap_seed);
    cJSON_AddStringToObject(opts, "bootstrap_unit", rg_observation_unit_string(resolved.bootstrap_unit));
    cJSON_AddNumberToObject(opts, "permutation_count", resolved.permutation_count);
    cJSON_AddNumberToObject(opts, "permutation_seed", resolved.permutation_seed);
    cJSON_AddBoolToObject(opts, "tune_search_penalty", resolved.tune_search_penalty);
    cJSON_AddNumberToObject(opts, "predictive_folds", resolved.predictive_folds);
    cJSON_AddNumberToObject(opts, "predictive_seed", resolved.predictive_seed);
    cJSON_AddNumberToObject(opts, "predictive_min_groups", resolved.predictive_min_groups);
    cJSON_AddNumberToObject(opts, "predictive_abstention_threshold", resolved.predictive_abstention_threshold);
    cJSON_AddNumberToObject(opts, "predictive_top_k", resolved.predictive_top_k);

    cJSON_AddStringToObject(bic, "split_scorer", rg_split_scorer_string(resolved.bic.split_scorer));
    cJSON_AddStringToObject(bic, "class_outcome_mode", rg_class_outcome_mode_string(resolved.bic.class_outcome_mode));
    cJSON_AddNumberToObject(bic, "split_prior_concentration", resolved.bic.split_prior_concentration);
    cJSON_AddNumberToObject(bic, "delta_bic_threshold", resolved.bic.delta_bic_threshold);
    cJSON_AddNumberToObject(bic, "min_split_observations", resolved.bic.min_split_observations);
    cJSON_AddNumberToObject(bic, "max_split_depth", resolved.bic.max_split_depth);
    cJSON_AddNumberToObject(bic, "min_chunk_observations", resolved.bic.min_chunk_observations);
    cJSON_AddNumberToObject(bic, "long_range_delta_bic_threshold", resolved.bic.long_range_delta_bic_threshold);
    cJSON_AddNumberToObject(bic, "long_range_min_split_observations", resolved.bic.long_range_min_split_observations);
    cJSON_AddNumberToObject(bic, "long_range_min_dominant_fraction", resolved.bic.long_range_min_dominant_fraction);
    cJSON_AddNumberToObject(bic, "cross_dim_max_iterations", resolved.bic.cross_dim_max_iterations);
    cJSON_AddNumberToObject(bic, "cross_dim_min_rule_count", resolved.bic.cross_dim_min_rule_count);
    cJSON_AddNumberToObject(bic, "cross_dim_min_rule_confidence", resolved.bic.cross_dim_min_rule_confidence);
    cJSON_AddNumberToObject(bic, "cross_dim_delta_bic_threshold", resolved.bic.cross_dim_delta_bic_threshold);
    cJSON_AddBoolToObject(bic, "multi_lect_bic_small_sample_correction", resolved.bic.multi_lect_bic_small_sample_correction);
    cJSON_AddNumberToObject(bic, "multi_lect_min_commit_scale", resolved.bic.multi_lect_min_commit_scale);
    cJSON_AddNumberToObject(bic, "search_penalty_gamma", resolved.bic.search_penalty_gamma);
    cJSON_AddItemToObject(opts, "bic", bic);
    cJSON_AddItemToObject(out, "options", opts);
    return out;
}

char *rg_model_to_json(
    const rg_context *ctx,
    const rg_multi_model *model,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    bool include_alignments,
    bool include_outliers
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
    bool include_alignments,
    bool include_outliers
) {
    cJSON *root;
    cJSON *lects;
    cJSON *classes;
    cJSON *pairwise;
    cJSON *array;
    char *text;
    size_t i;
    size_t table_count = 0;
    const char *const *lect_names;
    const rg_multi_class_row *class_rows;
    const rg_multi_cross_dimensional_row *xdim_rows;

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
    /* A named, versioned export: the maximum-a-posteriori surface relationship
     * model, with the provenance below to reproduce and cite it. Still a single
     * correspondence system, not the ensemble the historia layer's interchange
     * schema requires -- a reader must not treat it as claim-capable. The
     * format is `format_version` and its stability rule is in consumer_guide §7. */
    cJSON_AddStringToObject(root, "export_kind", "surface_relationship_model");
    {
        cJSON *provenance = json_provenance(ctx, cognates, cognate_count, options);
        if (provenance == 0) {
            cJSON_Delete(root);
            return 0;
        }
        cJSON_AddItemToObject(root, "provenance", provenance);
    }

    lects = cJSON_CreateArray();
    if (lects == 0) {
        cJSON_Delete(root);
        return 0;
    }
    lect_names = rg_multi_model_lects(model, &table_count);
    for (i = 0; i < table_count; i++) {
        cJSON_AddItemToArray(lects, cJSON_CreateString(lect_names[i]));
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
        cJSON_AddStringToObject(fit, "split_scorer", rg_split_scorer_string(f->split_scorer));
        cJSON_AddNumberToObject(fit, "split_prior_concentration", f->split_prior_concentration);
        cJSON_AddNumberToObject(fit, "etymon_group_count", (double)f->etymon_group_count);
        cJSON_AddNumberToObject(fit, "source_group_count", (double)f->source_group_count);
        cJSON_AddNumberToObject(fit, "sets_without_etymon_group", (double)f->sets_without_etymon_group);
        cJSON_AddNumberToObject(fit, "sets_without_source_group", (double)f->sets_without_source_group);
        cJSON_AddStringToObject(fit, "bootstrap_unit", rg_observation_unit_string(f->bootstrap_unit));
        cJSON_AddNumberToObject(fit, "bootstrap_effective_unit_count",
                               (double)f->bootstrap_effective_unit_count);
        cJSON_AddNumberToObject(fit, "unconditioned_class_count", (double)f->unconditioned_class_count);
        cJSON_AddNumberToObject(fit, "conditioned_class_count", (double)f->conditioned_class_count);
        cJSON_AddNumberToObject(fit, "permutation_count", (double)f->permutation_count);
        cJSON_AddNumberToObject(fit, "inferred_nucleus_form_count", (double)f->inferred_nucleus_form_count);
        cJSON_AddNumberToObject(fit, "syllabified_form_count", (double)f->syllabified_form_count);
        cJSON_AddNumberToObject(fit, "cost_split_separation", f->cost_split_separation);
        cJSON_AddNumberToObject(fit, "cost_split_fraction", f->cost_split_fraction);
        {
            cJSON *predictive = json_predictive_evidence(f->predictive);
            cJSON_AddItemToObject(fit, "predictive", predictive);
            cJSON_AddNumberToObject(predictive, "group_count", (double)f->predictive_group_count);
            cJSON_AddNumberToObject(predictive, "folds_requested",
                                    (double)f->predictive_folds_requested);
            cJSON_AddNumberToObject(predictive, "pair_orientations",
                                    (double)f->predictive_pair_orientations);
            cJSON_AddNumberToObject(predictive, "leave_one_lect_out_cases",
                                    (double)f->predictive_leave_one_lect_out_cases);
            cJSON_AddNumberToObject(predictive, "unscored_spans",
                                    (double)f->predictive_unscored_span_count);
            cJSON_AddNumberToObject(predictive, "abstention_threshold",
                                    f->predictive_abstention_threshold);
            cJSON_AddNumberToObject(predictive, "top_k", f->predictive_top_k);
            cJSON_AddItemToObject(predictive, "identity",
                                  json_predictive_score(f->predictive_identity));
            cJSON_AddItemToObject(predictive, "inventory_frequency",
                                  json_predictive_score(f->predictive_inventory_frequency));
            cJSON_AddItemToObject(predictive, "feature_distance",
                                  json_predictive_score(f->predictive_feature_distance));
            cJSON_AddItemToObject(predictive, "leave_one_lect_out_conditioned",
                                  json_predictive_score(f->predictive_leave_one_lect_out_conditioned));
            cJSON_AddItemToObject(predictive, "leave_one_lect_out_unconditioned",
                                  json_predictive_score(f->predictive_leave_one_lect_out_unconditioned));
        }
        if (f->permutation_count > 0) {
            cJSON_AddStringToObject(fit, "corpus_fit_null", "pairing_shuffle");
            cJSON_AddStringToObject(fit, "conditioned_standing_null", "pairing_shuffle");
            cJSON_AddNumberToObject(fit, "null_cost_per_segment_mean", f->null_cost_per_segment_mean);
            cJSON_AddNumberToObject(fit, "null_cost_per_segment_sd", f->null_cost_per_segment_sd);
            cJSON_AddNumberToObject(fit, "cost_per_segment_z", f->cost_per_segment_z);
            cJSON_AddNumberToObject(fit, "null_unconditioned_class_mean", f->null_unconditioned_class_mean);
            cJSON_AddNumberToObject(fit, "null_conditioned_class_mean", f->null_conditioned_class_mean);
            cJSON_AddNumberToObject(fit, "null_search_margin", f->null_search_margin);
            cJSON_AddNumberToObject(fit, "null_search_margin_quantile", f->null_search_margin_quantile);
            cJSON_AddNumberToObject(fit, "rules_above_noise", (double)f->rules_above_noise);
            cJSON_AddNumberToObject(fit, "rules_measured", (double)f->rules_measured);
            cJSON_AddNumberToObject(fit, "pairwise_rules_above_noise", (double)f->pairwise_rules_above_noise);
            cJSON_AddNumberToObject(fit, "pairwise_rules_measured", (double)f->pairwise_rules_measured);
        }
        cJSON_AddNumberToObject(fit, "lect_count", (double)f->lect_count);
        cJSON_AddNumberToObject(fit, "pair_count", (double)f->pair_count);
        cJSON_AddNumberToObject(fit, "duplicate_lect_count", (double)f->duplicate_lect_count);
        cJSON_AddNumberToObject(fit, "missing_form_count", (double)f->missing_form_count);
    }

    classes = cJSON_CreateObject();
    if (classes == 0) {
        cJSON_Delete(root);
        return 0;
    }
    cJSON_AddItemToObject(root, "classes", classes);

    pairwise = cJSON_CreateArray();
    if (pairwise == 0) {
        cJSON_Delete(root);
        return 0;
    }
    cJSON_AddItemToObject(root, "pairwise", pairwise);
    for (i = 0; i < rg_multi_model_pair_model_count(model); i++) {
        const rg_multi_pair_model_row *pair = rg_multi_model_pair_model_at(model, i);
        const rg_conditioned_segment_count_row *rows;
        size_t row_count = 0;
        size_t row_i;
        cJSON *pair_entry = cJSON_CreateObject();
        cJSON *conditioned = cJSON_CreateArray();
        if (pair_entry == 0 || conditioned == 0) {
            cJSON_Delete(pair_entry);
            cJSON_Delete(conditioned);
            cJSON_Delete(root);
            return 0;
        }
        const rg_gap_count_row *gap_rows;
        size_t gap_row_count = 0;
        size_t gap_i;
        cJSON *gaps;
        cJSON_AddStringToObject(pair_entry, "source_lect", pair->lect_a);
        cJSON_AddStringToObject(pair_entry, "target_lect", pair->lect_b);
        cJSON_AddItemToObject(pair_entry, "conditioned", conditioned);
        /* A segment answering to nothing: the row shape the 1-to-1 table lacks.
         * `deletion` reads on the source->target direction, so its inverse on
         * the pair is an epenthesis and vice versa. */
        gaps = cJSON_CreateArray();
        if (gaps == 0) {
            cJSON_Delete(pair_entry);
            cJSON_Delete(root);
            return 0;
        }
        cJSON_AddItemToObject(pair_entry, "gaps", gaps);
        gap_rows = rg_pairwise_model_gap_counts(pair->model, &gap_row_count);
        for (gap_i = 0; gap_i < gap_row_count; gap_i++) {
            const rg_gap_count_row *row = &gap_rows[gap_i];
            cJSON *entry = cJSON_CreateObject();
            if (entry == 0) {
                cJSON_Delete(root);
                return 0;
            }
            cJSON_AddStringToObject(entry, "grapheme", row->grapheme);
            cJSON_AddBoolToObject(entry, "deletion", row->deletion);
            cJSON_AddNumberToObject(entry, "count", row->count);
            cJSON_AddNumberToObject(entry, "present_total", row->present_total);
            cJSON_AddItemToObject(entry, "uncertainty", json_uncertainty(row->uncertainty));
            cJSON_AddItemToArray(gaps, entry);
        }
        rows = rg_pairwise_model_conditioned_segment_counts(pair->model, &row_count);
        for (row_i = 0; row_i < row_count; row_i++) {
            const rg_conditioned_segment_count_row *row = &rows[row_i];
            cJSON *entry = cJSON_CreateObject();
            if (entry == 0) {
                cJSON_Delete(root);
                return 0;
            }
            cJSON_AddStringToObject(entry, "source", row->source);
            cJSON_AddStringToObject(entry, "target", row->target);
            cJSON_AddItemToObject(entry, "context", json_context(&row->context));
            cJSON_AddBoolToObject(entry, "context_is_target", row->context_is_target);
            cJSON_AddNumberToObject(entry, "count", row->count);
            cJSON_AddNumberToObject(entry, "source_total", row->source_total);
            cJSON_AddNumberToObject(entry, "contrast_count", row->contrast_count);
            cJSON_AddNumberToObject(entry, "contrast_total", row->contrast_total);
            cJSON_AddStringToObject(entry, "score_kind",
                                    rg_split_scorer_string(row->evidence.scorer));
            cJSON_AddNumberToObject(entry, "delta_score", row->evidence.delta_score);
            cJSON_AddNumberToObject(entry, "decision_index", row->evidence.decision_index);
            cJSON_AddStringToObject(entry, "standing",
                                    rg_rule_standing_string(row->evidence.standing));
            cJSON_AddItemToObject(entry, "predictive",
                                  json_predictive_evidence(row->evidence.predictive));
            cJSON_AddItemToObject(entry, "uncertainty", json_uncertainty(row->uncertainty));
            cJSON_AddItemToArray(conditioned, entry);
        }
        cJSON_AddItemToArray(pairwise, pair_entry);
    }

    array = cJSON_CreateArray();
    if (array == 0) {
        cJSON_Delete(root);
        return 0;
    }
    cJSON_AddItemToObject(classes, "unconditioned", array);
    class_rows = rg_multi_model_unconditioned_classes(model, &table_count);
    for (i = 0; i < table_count; i++) {
        cJSON_AddItemToArray(array, json_class(&class_rows[i], 0));
    }

    array = cJSON_CreateArray();
    if (array == 0) {
        cJSON_Delete(root);
        return 0;
    }
    cJSON_AddItemToObject(classes, "conditioned", array);
    class_rows = rg_multi_model_conditioned_classes(model, &table_count);
    for (i = 0; i < table_count; i++) {
        cJSON_AddItemToArray(array, json_class(&class_rows[i], 1));
    }

    array = cJSON_CreateArray();
    if (array == 0) {
        cJSON_Delete(root);
        return 0;
    }
    cJSON_AddItemToObject(root, "proposed_events", array);
    {
        const rg_proposed_event_row *events = rg_multi_model_proposed_events(model, &table_count);
        for (i = 0; i < table_count; i++) {
            cJSON *entry = json_proposed_event(&events[i]);
            if (entry == 0) {
                cJSON_Delete(root);
                return 0;
            }
            cJSON_AddItemToArray(array, entry);
        }
    }

    array = cJSON_CreateArray();
    if (array == 0) {
        cJSON_Delete(root);
        return 0;
    }
    cJSON_AddItemToObject(root, "cross_dimensional", array);
    xdim_rows = rg_multi_model_cross_dimensional_rows(model, &table_count);
    for (i = 0; i < table_count; i++) {
        const rg_multi_cross_dimensional_row *row = &xdim_rows[i];
        cJSON *entry;
        if (!rg_cross_dim_row_publishable_internal(&row->rule)) {
            continue;
        }
        entry = cJSON_CreateObject();
        if (entry == 0) {
            cJSON_Delete(root);
            return 0;
        }
        cJSON_AddStringToObject(entry, "source_lect", row->source_lect);
        cJSON_AddStringToObject(entry, "target_lect", row->target_lect);
        {
            cJSON *environment = json_context(&row->rule.environment);
            if (environment != 0) {
                cJSON_AddItemToObject(entry, "environment", environment);
            }
        }
        cJSON_AddBoolToObject(entry, "context_is_target", row->rule.context_is_target ? 1 : 0);
        cJSON_AddBoolToObject(entry, "dimension_from_environment",
                              row->rule.dimension_from_environment ? 1 : 0);
        {
            const char *environment_lect =
                row->rule.context_is_target ? row->target_lect : row->source_lect;
            const char *other_lect =
                row->rule.context_is_target ? row->source_lect : row->target_lect;
            cJSON_AddStringToObject(entry, "environment_lect", environment_lect);
            /* Lect-internal rules condition the tone in the same lect the
             * environment sits in; only cross-lect rules read it from the
             * other. */
            cJSON_AddStringToObject(entry, "conditioned_lect",
                                    row->rule.dimension_from_environment ? environment_lect : other_lect);
        }
        cJSON_AddStringToObject(entry, "dimension", row->rule.dimension);
        cJSON_AddStringToObject(entry, "value", row->rule.value);
        cJSON_AddNumberToObject(entry, "position_offset", row->rule.position_offset);
        cJSON_AddNumberToObject(entry, "count", row->rule.count);
        cJSON_AddNumberToObject(entry, "source_count", row->rule.source_count);
        cJSON_AddNumberToObject(entry, "confidence", row->rule.confidence);
        cJSON_AddNumberToObject(entry, "contrast_count", row->rule.contrast_count);
        cJSON_AddNumberToObject(entry, "contrast_source_count", row->rule.contrast_source_count);
        cJSON_AddNumberToObject(entry, "contrast_confidence", row->rule.contrast_confidence);
        /* How many other features carve this rule's observations the same way;
         * 0 = the environment is uniquely identifiable, >0 = confounded. */
        cJSON_AddNumberToObject(entry, "environment_alternatives",
                                row->rule.environment_alternatives);
        cJSON_AddStringToObject(entry, "score_kind",
                               rg_split_scorer_string(row->rule.evidence.scorer));
        cJSON_AddNumberToObject(entry, "delta_score", row->rule.evidence.delta_score);
        cJSON_AddNumberToObject(entry, "delta_bic", row->rule.evidence.delta_bic);
        cJSON_AddNumberToObject(entry, "decision_index", row->rule.evidence.decision_index);
        cJSON_AddNumberToObject(entry, "search_margin", row->rule.evidence.search_margin);
        cJSON_AddStringToObject(entry, "standing", rg_rule_standing_string(row->rule.evidence.standing));
        cJSON_AddStringToObject(entry, "standing_null",
                                rg_null_model_string(row->rule.evidence.standing_null));
        cJSON_AddItemToObject(entry, "predictive",
                              json_predictive_evidence(row->rule.evidence.predictive));
        cJSON_AddItemToObject(entry, "uncertainty", json_uncertainty(row->rule.uncertainty));
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
                cJSON_AddNumberToObject(entry, "confidence", rows[i].confidence);
                cJSON_AddItemToArray(array, entry);
            }
            rg_cognate_outlier_rows_free(rows, row_count);
        }
    }

    text = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    return text;
}

/* A chunk's span as the written string a reader recognises: the graphemes,
 * space-separated. */
static cJSON *json_span(const rg_segment *segments, size_t count) {
    char buffer[512];
    size_t used = 0;
    size_t i;
    buffer[0] = '\0';
    for (i = 0; i < count; i++) {
        int written = snprintf(buffer + used, sizeof(buffer) - used, "%s%s",
                               i == 0 ? "" : " ",
                               segments[i].grapheme == 0 ? "" : segments[i].grapheme);
        if (written < 0 || (size_t)written >= sizeof(buffer) - used) {
            break;
        }
        used += (size_t)written;
    }
    return cJSON_CreateString(buffer);
}

char *rg_multi_model_pair_chunks_json(const rg_multi_model *model) {
    cJSON *root;
    cJSON *pairs;
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
    pairs = cJSON_AddArrayToObject(root, "pairs");
    if (pairs == 0) {
        cJSON_Delete(root);
        return 0;
    }
    for (i = 0; i < rg_multi_model_pair_model_count(model); i++) {
        const rg_multi_pair_model_row *pair = rg_multi_model_pair_model_at(model, i);
        const rg_chunk_row *chunks;
        size_t chunk_count = 0;
        size_t c;
        cJSON *pair_entry;
        cJSON *chunk_array;
        if (pair == 0) {
            continue;
        }
        pair_entry = cJSON_CreateObject();
        if (pair_entry == 0) {
            cJSON_Delete(root);
            return 0;
        }
        cJSON_AddStringToObject(pair_entry, "source_lect", pair->lect_a);
        cJSON_AddStringToObject(pair_entry, "target_lect", pair->lect_b);
        chunk_array = cJSON_AddArrayToObject(pair_entry, "chunks");
        chunks = rg_pairwise_model_chunks(pair->model, &chunk_count);
        for (c = 0; c < chunk_count; c++) {
            cJSON *entry = cJSON_CreateObject();
            if (entry == 0) {
                break;
            }
            cJSON_AddItemToObject(entry, "source", json_span(chunks[c].source, chunks[c].source_count));
            cJSON_AddItemToObject(entry, "target", json_span(chunks[c].target, chunks[c].target_count));
            cJSON_AddNumberToObject(entry, "count", chunks[c].count);
            cJSON_AddBoolToObject(entry, "reordering", chunks[c].reordering);
            cJSON_AddNumberToObject(entry, "transparency", chunks[c].transparency);
            cJSON_AddItemToArray(chunk_array, entry);
        }
        cJSON_AddItemToArray(pairs, pair_entry);
    }
    text = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    return text;
}

char *rg_corpus_drift_json(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count
) {
    rg_transcription_drift_row *rows = 0;
    size_t row_count = 0;
    cJSON *root;
    cJSON *array;
    char *text;
    size_t i;

    if (ctx == 0) {
        return 0;
    }
    if (rg_find_transcription_drift(ctx, cognates, cognate_count, &rows, &row_count) != RG_OK) {
        return 0;
    }
    root = cJSON_CreateObject();
    if (root == 0) {
        rg_transcription_drift_rows_free(rows, row_count);
        return 0;
    }
    cJSON_AddBoolToObject(root, "ok", 1);
    cJSON_AddNumberToObject(root, "format_version", RG_JSON_FORMAT_VERSION);
    array = cJSON_AddArrayToObject(root, "drift");
    for (i = 0; i < row_count && array != 0; i++) {
        cJSON *entry = cJSON_CreateObject();
        if (entry == 0) {
            break;
        }
        cJSON_AddStringToObject(entry, "lect", rows[i].lect);
        cJSON_AddStringToObject(entry, "other_lect", rows[i].other_lect);
        cJSON_AddStringToObject(entry, "grapheme", rows[i].grapheme);
        cJSON_AddStringToObject(entry, "written_as", rows[i].written_as);
        cJSON_AddNumberToObject(entry, "corroborated", (double)rows[i].corroborated);
        cJSON_AddNumberToObject(entry, "forms", (double)rows[i].forms);
        cJSON_AddItemToArray(array, entry);
    }
    rg_transcription_drift_rows_free(rows, row_count);
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

static int parse_observation_unit(const char *value, rg_observation_unit *out) {
    if (strcmp(value, "auto") == 0) {
        *out = RG_OBSERVATION_UNIT_AUTO;
    } else if (strcmp(value, "cognate_set") == 0) {
        *out = RG_OBSERVATION_UNIT_COGNATE_SET;
    } else if (strcmp(value, "etymon_group") == 0) {
        *out = RG_OBSERVATION_UNIT_ETYMON_GROUP;
    } else if (strcmp(value, "source_group") == 0) {
        *out = RG_OBSERVATION_UNIT_SOURCE_GROUP;
    } else {
        return 0;
    }
    return 1;
}

static int parse_split_scorer(const char *value, rg_split_scorer *out) {
    if (strcmp(value, "corrected_bic") == 0 || strcmp(value, "bic") == 0) {
        *out = RG_SPLIT_SCORER_CORRECTED_BIC;
    } else if (strcmp(value, "multinomial_nml") == 0 || strcmp(value, "nml") == 0) {
        *out = RG_SPLIT_SCORER_MULTINOMIAL_NML;
    } else if (strcmp(value, "dirichlet_marginal") == 0 || strcmp(value, "dirichlet") == 0) {
        *out = RG_SPLIT_SCORER_DIRICHLET_MARGINAL;
    } else {
        return 0;
    }
    return 1;
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
        /* Calibrating against the shuffle sets both the corpus baseline and the
         * per-rule standing null; the provenance block already round-trips these
         * two, so a caller reading them back could not set them until here. */
        NUMBER_FIELD("permutation_count", permutation_count, int)
        NUMBER_FIELD("permutation_seed", permutation_seed, int)
        NUMBER_FIELD("predictive_folds", predictive_folds, int)
        NUMBER_FIELD("predictive_seed", predictive_seed, int)
        NUMBER_FIELD("predictive_min_groups", predictive_min_groups, int)
        NUMBER_FIELD("predictive_abstention_threshold", predictive_abstention_threshold, double)
        NUMBER_FIELD("predictive_top_k", predictive_top_k, int)
        NUMBER_FIELD("split_prior_concentration", bic.split_prior_concentration, double)
        NUMBER_FIELD("delta_bic_threshold", bic.delta_bic_threshold, double)
        NUMBER_FIELD("min_split_observations", bic.min_split_observations, int)
        NUMBER_FIELD("max_split_depth", bic.max_split_depth, int)
        NUMBER_FIELD("min_chunk_observations", bic.min_chunk_observations, int)
        NUMBER_FIELD("long_range_delta_bic_threshold", bic.long_range_delta_bic_threshold, double)
        NUMBER_FIELD("long_range_min_split_observations", bic.long_range_min_split_observations, int)
        NUMBER_FIELD("long_range_min_dominant_fraction", bic.long_range_min_dominant_fraction, double)
        NUMBER_FIELD("multi_lect_min_commit_scale", bic.multi_lect_min_commit_scale, double)
        NUMBER_FIELD("search_penalty_gamma", bic.search_penalty_gamma, double)

        if (strcmp(key, "split_scorer") == 0) {
            if (!cJSON_IsString(item) ||
                !parse_split_scorer(item->valuestring, &out->bic.split_scorer)) {
                status = RG_ERR_PARSE;
            }
            if (status != RG_OK && error_detail != 0 && error_detail_size > 0) {
                snprintf(error_detail, error_detail_size,
                         "option \"split_scorer\" must name a supported scorer");
            }
            if (status != RG_OK) {
                break;
            }
            continue;
        }

        if (strcmp(key, "bootstrap_unit") == 0) {
            if (!cJSON_IsString(item) ||
                !parse_observation_unit(item->valuestring, &out->bootstrap_unit)) {
                status = RG_ERR_PARSE;
            }
            if (status != RG_OK && error_detail != 0 && error_detail_size > 0) {
                snprintf(error_detail, error_detail_size,
                         "option \"bootstrap_unit\" must name a supported observation unit");
            }
            if (status != RG_OK) {
                break;
            }
            continue;
        }

        /* Whether the shuffle also tunes the search penalty rather than leaving
         * it at its fixed default; only meaningful with permutation_count > 0. */
        if (strcmp(key, "tune_search_penalty") == 0) {
            if (!cJSON_IsBool(item)) {
                status = RG_ERR_PARSE;
                if (error_detail != 0 && error_detail_size > 0) {
                    snprintf(error_detail, error_detail_size,
                             "option \"tune_search_penalty\" must be true or false");
                }
                break;
            }
            out->tune_search_penalty = cJSON_IsTrue(item) ? true : false;
            continue;
        }

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
