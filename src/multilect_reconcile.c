#include "multilect_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct class_bucket {
    size_t origin;
    char **lect_ids;
    char **graphemes;
    size_t segment_count;
    double count;
    char *participant_key;
    char **supporting_cognates;
    size_t supporting_cognate_count;
    size_t supporting_cognate_cap;
} class_bucket;

typedef struct uf_state {
    size_t *parent;
    size_t *rank;
    size_t count;
} uf_state;

static void class_bucket_clear(class_bucket *bucket) {
    if (bucket == 0) {
        return;
    }
    string_array_clear(bucket->lect_ids, bucket->segment_count);
    string_array_clear(bucket->graphemes, bucket->segment_count);
    free(bucket->participant_key);
    string_array_clear(bucket->supporting_cognates, bucket->supporting_cognate_count);
    memset(bucket, 0, sizeof(*bucket));
}

const rg_pairwise_model *pair_model_for(const rg_multi_model *model, const char *lect_a, const char *lect_b) {
    size_t i;
    for (i = 0; i < model->pair_model_count; i++) {
        if (strcmp(model->pair_models[i].lect_a, lect_a) == 0 && strcmp(model->pair_models[i].lect_b, lect_b) == 0) {
            return model->pair_models[i].model;
        }
        if (strcmp(model->pair_models[i].lect_a, lect_b) == 0 && strcmp(model->pair_models[i].lect_b, lect_a) == 0) {
            return model->pair_models[i].model;
        }
    }
    return 0;
}

static rg_status uf_init(uf_state *uf, size_t count) {
    size_t i;
    memset(uf, 0, sizeof(*uf));
    uf->parent = (size_t *)calloc(count, sizeof(*uf->parent));
    uf->rank = (size_t *)calloc(count, sizeof(*uf->rank));
    if ((uf->parent == 0 || uf->rank == 0) && count > 0) {
        free(uf->parent);
        free(uf->rank);
        memset(uf, 0, sizeof(*uf));
        return RG_ERR_OOM;
    }
    uf->count = count;
    for (i = 0; i < count; i++) {
        uf->parent[i] = i;
    }
    return RG_OK;
}

static void uf_clear(uf_state *uf) {
    free(uf->parent);
    free(uf->rank);
    memset(uf, 0, sizeof(*uf));
}

static size_t uf_find(uf_state *uf, size_t x) {
    size_t root = x;
    while (uf->parent[root] != root) {
        root = uf->parent[root];
    }
    while (uf->parent[x] != x) {
        size_t next = uf->parent[x];
        uf->parent[x] = root;
        x = next;
    }
    return root;
}

static void uf_union(uf_state *uf, size_t a, size_t b) {
    size_t ra = uf_find(uf, a);
    size_t rb = uf_find(uf, b);
    if (ra == rb) {
        return;
    }
    if (uf->rank[ra] < uf->rank[rb]) {
        uf->parent[ra] = rb;
    } else if (uf->rank[ra] > uf->rank[rb]) {
        uf->parent[rb] = ra;
    } else {
        uf->parent[rb] = ra;
        uf->rank[ra]++;
    }
}

static rg_status participant_key_from_lects(char **lects, size_t count, char **out) {
    size_t i;
    size_t len = 1;
    char *key;
    char *p;
    *out = 0;
    for (i = 0; i < count; i++) {
        len += strlen(lects[i]) + 1;
    }
    key = (char *)malloc(len);
    if (key == 0) {
        return RG_ERR_OOM;
    }
    p = key;
    for (i = 0; i < count; i++) {
        size_t n;
        if (i > 0) {
            *p++ = '|';
        }
        n = strlen(lects[i]);
        memcpy(p, lects[i], n);
        p += n;
    }
    *p = '\0';
    *out = key;
    return RG_OK;
}

static rg_status bucket_append_support(class_bucket *bucket, const char *cognate_id) {
    char **next;
    const char *id = cognate_id == 0 ? "" : cognate_id;
    if (bucket->supporting_cognate_count == bucket->supporting_cognate_cap) {
        size_t next_cap = bucket->supporting_cognate_cap == 0 ? 4 : bucket->supporting_cognate_cap * 2;
        next = (char **)realloc(bucket->supporting_cognates, next_cap * sizeof(*bucket->supporting_cognates));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        bucket->supporting_cognates = next;
        bucket->supporting_cognate_cap = next_cap;
    }
    bucket->supporting_cognates[bucket->supporting_cognate_count] = rg_strdup_internal(id);
    if (bucket->supporting_cognates[bucket->supporting_cognate_count] == 0) {
        return RG_ERR_OOM;
    }
    bucket->supporting_cognate_count++;
    return RG_OK;
}

static int bucket_equal(const class_bucket *bucket, char **lects, char **graphemes, size_t count) {
    size_t i;
    if (bucket->segment_count != count) {
        return 0;
    }
    for (i = 0; i < count; i++) {
        if (strcmp(bucket->lect_ids[i], lects[i]) != 0 || strcmp(bucket->graphemes[i], graphemes[i]) != 0) {
            return 0;
        }
    }
    return 1;
}

static rg_status add_bucket_observation(
    class_bucket **buckets,
    size_t *bucket_count,
    size_t *bucket_cap,
    char **lects,
    char **graphemes,
    size_t count,
    double weight,
    const char *cognate_id,
    size_t *out_bucket_index
) {
    size_t i;
    for (i = 0; i < *bucket_count; i++) {
        if (bucket_equal(&(*buckets)[i], lects, graphemes, count)) {
            (*buckets)[i].count += weight;
            if (bucket_append_support(&(*buckets)[i], cognate_id) != RG_OK) {
                return RG_ERR_OOM;
            }
            string_array_clear(lects, count);
            string_array_clear(graphemes, count);
            *out_bucket_index = i;
            return RG_OK;
        }
    }
    *out_bucket_index = *bucket_count;
    if (*bucket_count == *bucket_cap) {
        size_t next_cap = *bucket_cap == 0 ? 16 : *bucket_cap * 2;
        class_bucket *next = (class_bucket *)realloc(*buckets, next_cap * sizeof(**buckets));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *buckets = next;
        *bucket_cap = next_cap;
    }
    memset(&(*buckets)[*bucket_count], 0, sizeof((*buckets)[*bucket_count]));
    (*buckets)[*bucket_count].lect_ids = lects;
    (*buckets)[*bucket_count].graphemes = graphemes;
    (*buckets)[*bucket_count].segment_count = count;
    (*buckets)[*bucket_count].count = weight;
    if (participant_key_from_lects(lects, count, &(*buckets)[*bucket_count].participant_key) != RG_OK) {
        memset(&(*buckets)[*bucket_count], 0, sizeof((*buckets)[*bucket_count]));
        return RG_ERR_OOM;
    }
    if (bucket_append_support(&(*buckets)[*bucket_count], cognate_id) != RG_OK) {
        class_bucket_clear(&(*buckets)[*bucket_count]);
        return RG_ERR_OOM;
    }
    (*bucket_count)++;
    return RG_OK;
}

static int bucket_cmp(const void *a, const void *b) {
    const class_bucket *ba = (const class_bucket *)a;
    const class_bucket *bb = (const class_bucket *)b;
    size_t i;
    if (ba->count != bb->count) {
        return ba->count > bb->count ? -1 : 1;
    }
    for (i = 0; i < ba->segment_count && i < bb->segment_count; i++) {
        int c = strcmp(ba->lect_ids[i], bb->lect_ids[i]);
        if (c != 0) {
            return c;
        }
        c = strcmp(ba->graphemes[i], bb->graphemes[i]);
        if (c != 0) {
            return c;
        }
    }
    if (ba->segment_count != bb->segment_count) {
        return ba->segment_count < bb->segment_count ? -1 : 1;
    }
    return 0;
}

/* Lect indices of the lects present in this cognate set, ascending by lect id.
 * Reconciliation walks pairs and union-find components in this order, which is
 * what makes component enumeration and supporting-cognate order deterministic
 * regardless of the corpus's first-seen lect order. */
rg_status present_lects_sorted(
    const rg_cognate_set *cognate,
    const rg_multi_model *model,
    const rg_form **forms,
    size_t **out_order,
    size_t *out_count
) {
    size_t *order;
    size_t count = 0;
    size_t i;
    order = (size_t *)calloc(model->lect_count == 0 ? 1 : model->lect_count, sizeof(*order));
    if (order == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < model->lect_count; i++) {
        size_t insert_at;
        forms[i] = form_for_lect(cognate, model->lect_ids[i]);
        if (forms[i] == 0) {
            continue;
        }
        insert_at = count;
        while (insert_at > 0 && strcmp(model->lect_ids[order[insert_at - 1]], model->lect_ids[i]) > 0) {
            insert_at--;
        }
        if (insert_at < count) {
            memmove(&order[insert_at + 1], &order[insert_at], (count - insert_at) * sizeof(*order));
        }
        order[insert_at] = i;
        count++;
    }
    *out_order = order;
    *out_count = count;
    return RG_OK;
}

/* Position-level union-find edges induced by aligning one pair. Equal-length
 * chunks pair position by position; unequal-length non-gap chunks are
 * decomposed through a one-segment sub-alignment; pure gaps induce no edges. */
static rg_status union_pair_alignment_edges(
    const rg_context *ctx,
    const rg_pairwise_model *pair_model,
    const rg_train_options *options,
    const rg_form *source,
    const rg_form *target,
    size_t source_offset,
    size_t target_offset,
    uf_state *uf
) {
    rg_alignment *alignment = 0;
    size_t link_i;
    size_t source_pos = 0;
    size_t target_pos = 0;
    rg_status status;

    status = rg_align_forms_with_model(ctx, pair_model, options, source, target, 0, &alignment);
    if (status != RG_OK) {
        return status;
    }
    for (link_i = 0; link_i < rg_alignment_link_count(alignment); link_i++) {
        const rg_link *link = rg_alignment_link_at(alignment, link_i);
        size_t k;
        if (link->source_count == link->target_count) {
            /* Reconciliation binds the positions that answer to each other,
             * which for a reordering is not the diagonal. Binding `sk` to `ks`
             * position by position makes /s/ and /k/ members of each other's
             * class in both directions -- two false correspondences standing
             * for one reordering. */
            size_t pairing[RG_MAX_REORDER_SPAN];
            int reordering = rg_link_is_reordering_internal(link->source, link->source_count,
                                                            link->target, link->target_count, pairing);
            for (k = 0; k < link->source_count; k++) {
                size_t partner = reordering ? pairing[k] : k;
                uf_union(uf, source_offset + source_pos + k, target_offset + target_pos + partner);
            }
        } else if (link->source_count != 0 && link->target_count != 0) {
            size_t sub_source_pos = 0;
            size_t sub_target_pos = 0;
            rg_form sub_source;
            rg_form sub_target;
            rg_alignment *sub = 0;
            memset(&sub_source, 0, sizeof(sub_source));
            memset(&sub_target, 0, sizeof(sub_target));
            sub_source.lect_id = source->lect_id;
            sub_source.segments = link->source;
            sub_source.segment_count = link->source_count;
            sub_target.lect_id = target->lect_id;
            sub_target.segments = link->target;
            sub_target.segment_count = link->target_count;
            /* A one-segment chunk limit makes the pairwise chunk table
             * unreachable, matching the Go sub-alignment's emptied table. */
            status = rg_align_forms_with_model(ctx, pair_model, options, &sub_source, &sub_target, 1, &sub);
            if (status != RG_OK) {
                rg_alignment_free(alignment);
                return status;
            }
            for (k = 0; k < rg_alignment_link_count(sub); k++) {
                const rg_link *sub_link = rg_alignment_link_at(sub, k);
                if (sub_link->source_count == 1 && sub_link->target_count == 1) {
                    uf_union(uf, source_offset + source_pos + sub_source_pos, target_offset + target_pos + sub_target_pos);
                }
                sub_source_pos += sub_link->source_count;
                sub_target_pos += sub_link->target_count;
            }
            rg_alignment_free(sub);
        }
        source_pos += link->source_count;
        target_pos += link->target_count;
    }
    rg_alignment_free(alignment);
    return RG_OK;
}

static rg_status append_reconciled_observation(
    reconciled_observation **items,
    size_t *count,
    size_t *cap,
    const reconciled_observation *value
) {
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 32 : *cap * 2;
        reconciled_observation *next = (reconciled_observation *)realloc(*items, next_cap * sizeof(**items));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *items = next;
        *cap = next_cap;
    }
    (*items)[*count] = *value;
    (*count)++;
    return RG_OK;
}

rg_status aggregate_position_classes(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    rg_multi_model *model,
    reconciled_observation **out_observations,
    size_t *out_observation_count
) {
    class_bucket *buckets = 0;
    size_t bucket_count = 0;
    size_t bucket_cap = 0;
    reconciled_observation *observations = 0;
    size_t observation_count = 0;
    size_t observation_cap = 0;
    size_t c;

    *out_observations = 0;
    *out_observation_count = 0;

    for (c = 0; c < cognate_count; c++) {
        const rg_form **forms;
        size_t *order = 0;
        size_t order_count = 0;
        size_t *offsets;
        size_t *node_lects;
        size_t *node_positions;
        size_t *node_order;
        int *seen_roots;
        uf_state uf;
        size_t node_count = 0;
        double weight = cognate_weight(&cognates[c]);
        size_t i;
        rg_status status = RG_OK;
        if (weight <= 0.0) {
            continue;
        }
        forms = (const rg_form **)calloc(model->lect_count, sizeof(*forms));
        offsets = (size_t *)calloc(model->lect_count, sizeof(*offsets));
        if (forms == 0 || offsets == 0) {
            free(forms);
            free(offsets);
            status = RG_ERR_OOM;
            goto cognate_failed;
        }
        status = present_lects_sorted(&cognates[c], model, forms, &order, &order_count);
        if (status != RG_OK) {
            free(forms);
            free(offsets);
            goto cognate_failed;
        }
        for (i = 0; i < model->lect_count; i++) {
            offsets[i] = (size_t)-1;
        }
        /* Node ids are laid out in ascending lect-id order so that walking them
         * in id order also walks them in (lect, position) order. */
        for (i = 0; i < order_count; i++) {
            offsets[order[i]] = node_count;
            node_count += forms[order[i]]->segment_count;
        }
        node_lects = (size_t *)calloc(node_count == 0 ? 1 : node_count, sizeof(*node_lects));
        node_positions = (size_t *)calloc(node_count == 0 ? 1 : node_count, sizeof(*node_positions));
        node_order = (size_t *)calloc(node_count == 0 ? 1 : node_count, sizeof(*node_order));
        seen_roots = (int *)calloc(node_count == 0 ? 1 : node_count, sizeof(*seen_roots));
        if (node_lects == 0 || node_positions == 0 || node_order == 0 || seen_roots == 0) {
            free(forms);
            free(offsets);
            free(order);
            free(node_lects);
            free(node_positions);
            free(node_order);
            free(seen_roots);
            status = RG_ERR_OOM;
            goto cognate_failed;
        }
        for (i = 0; i < order_count; i++) {
            size_t pos;
            for (pos = 0; pos < forms[order[i]]->segment_count; pos++) {
                node_lects[offsets[order[i]] + pos] = order[i];
                node_positions[offsets[order[i]] + pos] = pos;
                node_order[offsets[order[i]] + pos] = i;
            }
        }
        status = uf_init(&uf, node_count);
        if (status != RG_OK) {
            free(forms);
            free(offsets);
            free(order);
            free(node_lects);
            free(node_positions);
            free(node_order);
            free(seen_roots);
            goto cognate_failed;
        }
        for (i = 0; i < order_count && status == RG_OK; i++) {
            size_t j;
            for (j = i + 1; j < order_count && status == RG_OK; j++) {
                size_t a = order[i];
                size_t b = order[j];
                const rg_pairwise_model *pair_model = pair_model_for(model, model->lect_ids[a], model->lect_ids[b]);
                if (pair_model == 0) {
                    continue;
                }
                status = union_pair_alignment_edges(
                    ctx,
                    pair_model,
                    options,
                    forms[a],
                    forms[b],
                    offsets[a],
                    offsets[b],
                    &uf
                );
            }
        }
        for (i = 0; i < node_count && status == RG_OK; i++) {
            size_t root = uf_find(&uf, i);
            reconciled_observation obs;
            size_t item_count = 0;
            size_t n;
            int inconsistent = 0;
            if (seen_roots[root]) {
                continue;
            }
            seen_roots[root] = 1;
            memset(&obs, 0, sizeof(obs));
            obs.lects = (char **)calloc(order_count, sizeof(*obs.lects));
            obs.graphemes = (char **)calloc(order_count, sizeof(*obs.graphemes));
            obs.positions = (size_t *)calloc(order_count, sizeof(*obs.positions));
            obs.lect_indices = (size_t *)calloc(order_count, sizeof(*obs.lect_indices));
            if (obs.lects == 0 || obs.graphemes == 0 || obs.positions == 0 || obs.lect_indices == 0) {
                obs.segment_count = 0;
                reconciled_observation_clear(&obs);
                status = RG_ERR_OOM;
                break;
            }
            /* Nodes are visited in ascending (lect, position) order, so members
             * accumulate already sorted by lect id. */
            for (n = 0; n < node_count; n++) {
                size_t lect_index;
                if (uf_find(&uf, n) != root) {
                    continue;
                }
                lect_index = node_lects[n];
                if (item_count > 0 && obs.lect_indices[item_count - 1] == lect_index) {
                    inconsistent = 1;
                    break;
                }
                obs.lects[item_count] = rg_strdup_internal(model->lect_ids[lect_index]);
                obs.graphemes[item_count] = rg_strdup_internal(forms[lect_index]->segments[node_positions[n]].grapheme);
                if (obs.lects[item_count] == 0 || obs.graphemes[item_count] == 0) {
                    obs.segment_count = item_count + 1;
                    reconciled_observation_clear(&obs);
                    status = RG_ERR_OOM;
                    break;
                }
                obs.positions[item_count] = node_positions[n];
                obs.lect_indices[item_count] = lect_index;
                item_count++;
                (void)node_order;
            }
            if (status != RG_OK) {
                break;
            }
            if (inconsistent || item_count < 2) {
                obs.segment_count = item_count;
                reconciled_observation_clear(&obs);
                continue;
            }
            obs.segment_count = item_count;
            obs.cognate_index = c;
            obs.weight = weight;
            {
                char **lects_copy = (char **)calloc(item_count, sizeof(*lects_copy));
                char **graphemes_copy = (char **)calloc(item_count, sizeof(*graphemes_copy));
                size_t k;
                if (lects_copy == 0 || graphemes_copy == 0) {
                    free(lects_copy);
                    free(graphemes_copy);
                    reconciled_observation_clear(&obs);
                    status = RG_ERR_OOM;
                    break;
                }
                for (k = 0; k < item_count; k++) {
                    lects_copy[k] = rg_strdup_internal(obs.lects[k]);
                    graphemes_copy[k] = rg_strdup_internal(obs.graphemes[k]);
                    if (lects_copy[k] == 0 || graphemes_copy[k] == 0) {
                        string_array_clear(lects_copy, k + 1);
                        string_array_clear(graphemes_copy, k + 1);
                        lects_copy = 0;
                        graphemes_copy = 0;
                        break;
                    }
                }
                if (lects_copy == 0) {
                    reconciled_observation_clear(&obs);
                    status = RG_ERR_OOM;
                    break;
                }
                status = add_bucket_observation(
                    &buckets,
                    &bucket_count,
                    &bucket_cap,
                    lects_copy,
                    graphemes_copy,
                    item_count,
                    weight,
                    cognates[c].cognate_id,
                    &obs.bucket_index
                );
                if (status != RG_OK) {
                    reconciled_observation_clear(&obs);
                    break;
                }
            }
            status = append_reconciled_observation(&observations, &observation_count, &observation_cap, &obs);
            if (status != RG_OK) {
                reconciled_observation_clear(&obs);
                break;
            }
        }
        uf_clear(&uf);
        free(forms);
        free(offsets);
        free(order);
        free(node_lects);
        free(node_positions);
        free(node_order);
        free(seen_roots);
    cognate_failed:
        if (status != RG_OK) {
            size_t n;
            for (n = 0; n < bucket_count; n++) {
                class_bucket_clear(&buckets[n]);
            }
            free(buckets);
            reconciled_observations_free(observations, observation_count);
            return status;
        }
    }
    /* Sorting reorders the buckets, so remember where each one went before
     * the observations' bucket indices become meaningless. */
    {
        size_t *bucket_to_class = (size_t *)calloc(bucket_count == 0 ? 1 : bucket_count, sizeof(*bucket_to_class));
        size_t i;
        if (bucket_to_class == 0) {
            for (c = 0; c < bucket_count; c++) {
                class_bucket_clear(&buckets[c]);
            }
            free(buckets);
            reconciled_observations_free(observations, observation_count);
            return RG_ERR_OOM;
        }
        for (i = 0; i < bucket_count; i++) {
            buckets[i].origin = i;
        }
        if (bucket_count > 1) {
            qsort(buckets, bucket_count, sizeof(*buckets), bucket_cmp);
        }
        for (i = 0; i < bucket_count; i++) {
            bucket_to_class[buckets[i].origin] = i;
        }
        for (i = 0; i < observation_count; i++) {
            if (observations[i].bucket_index < bucket_count) {
                observations[i].bucket_index = bucket_to_class[observations[i].bucket_index];
            }
        }
        free(bucket_to_class);
    }
    model->unconditioned_classes = (rg_multi_class_owned *)calloc(bucket_count == 0 ? 1 : bucket_count, sizeof(*model->unconditioned_classes));
    if (model->unconditioned_classes == 0) {
        for (c = 0; c < bucket_count; c++) {
            class_bucket_clear(&buckets[c]);
        }
        free(buckets);
        reconciled_observations_free(observations, observation_count);
        return RG_ERR_OOM;
    }
    for (c = 0; c < bucket_count; c++) {
        double participant_total = 0.0;
        size_t i;
        for (i = 0; i < bucket_count; i++) {
            if (strcmp(buckets[i].participant_key, buckets[c].participant_key) == 0) {
                participant_total += buckets[i].count;
            }
        }
        model->unconditioned_classes[c].view.class_id = (int)c;
        model->unconditioned_classes[c].view.lect_ids = (const char *const *)buckets[c].lect_ids;
        model->unconditioned_classes[c].view.graphemes = (const char *const *)buckets[c].graphemes;
        model->unconditioned_classes[c].view.segment_count = buckets[c].segment_count;
        model->unconditioned_classes[c].view.count = buckets[c].count;
        model->unconditioned_classes[c].view.confidence = 1.0;
        /* An unconditioned class is aggregated, not decided. */
        model->unconditioned_classes[c].view.decision_index = -1;
        model->unconditioned_classes[c].view.supporting_cognates = (const char *const *)buckets[c].supporting_cognates;
        model->unconditioned_classes[c].view.supporting_cognate_count = buckets[c].supporting_cognate_count;
        model->unconditioned_classes[c].view.uncertainty = rg_wilson_default_internal(buckets[c].count, participant_total);
        buckets[c].lect_ids = 0;
        buckets[c].graphemes = 0;
        buckets[c].supporting_cognates = 0;
        buckets[c].supporting_cognate_count = 0;
    }
    model->unconditioned_class_count = bucket_count;
    for (c = 0; c < bucket_count; c++) {
        class_bucket_clear(&buckets[c]);
    }
    free(buckets);
    *out_observations = observations;
    *out_observation_count = observation_count;
    return RG_OK;
}

