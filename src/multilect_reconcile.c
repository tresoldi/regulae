#include "multilect_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static char *supra_field_dup(const char *value) {
    return rg_strdup_internal(value != 0 ? value : "");
}

rg_status seg_supra_set_internal(seg_supra *out, const rg_segment *segment) {
    memset(out, 0, sizeof(*out));
    out->tone = supra_field_dup(segment != 0 ? segment->tone : 0);
    out->length = supra_field_dup(segment != 0 ? segment->length : 0);
    out->stress = supra_field_dup(segment != 0 ? segment->stress : 0);
    if (out->tone == 0 || out->length == 0 || out->stress == 0) {
        free(out->tone);
        free(out->length);
        free(out->stress);
        memset(out, 0, sizeof(*out));
        return RG_ERR_OOM;
    }
    return RG_OK;
}

void seg_supra_free_array_internal(seg_supra *items, size_t count) {
    size_t i;
    if (items == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        free(items[i].tone);
        free(items[i].length);
        free(items[i].stress);
    }
    free(items);
}

seg_supra *seg_supra_dup_array_internal(const seg_supra *items, size_t count) {
    seg_supra *copy;
    size_t i;
    copy = (seg_supra *)calloc(count == 0 ? 1 : count, sizeof(*copy));
    if (copy == 0) {
        return 0;
    }
    for (i = 0; i < count; i++) {
        copy[i].tone = rg_strdup_internal(items[i].tone);
        copy[i].length = rg_strdup_internal(items[i].length);
        copy[i].stress = rg_strdup_internal(items[i].stress);
        if (copy[i].tone == 0 || copy[i].length == 0 || copy[i].stress == 0) {
            seg_supra_free_array_internal(copy, i + 1);
            return 0;
        }
    }
    return copy;
}

int seg_supra_equal_internal(const seg_supra *a, const seg_supra *b) {
    return strcmp(a->tone, b->tone) == 0 &&
           strcmp(a->length, b->length) == 0 &&
           strcmp(a->stress, b->stress) == 0;
}

int seg_supra_is_bare_internal(const seg_supra *s) {
    return s->tone[0] == '\0' && s->length[0] == '\0' && s->stress[0] == '\0';
}

typedef struct class_bucket {
    size_t origin;
    char **lect_ids;
    char **graphemes;
    seg_supra *supra;
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
    seg_supra_free_array_internal(bucket->supra, bucket->segment_count);
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

/* Records a cognate set behind this class, once however often it is observed.
 *
 * A set reaches the same class once per aligned position that realises it, so
 * appending unconditionally made the list a multiplicity record wearing the
 * name of a support list: on the Verner fixture a class published thirty sets
 * as forty-two entries, `brother-fore` twice in a row. The multiplicity is
 * already in `count`; what this list is for is the question `count` cannot
 * answer, which is how much of the lexicon the row rests on. */
static rg_status bucket_append_support(class_bucket *bucket, const char *cognate_id) {
    char **next;
    size_t i;
    const char *id = cognate_id == 0 ? "" : cognate_id;
    for (i = 0; i < bucket->supporting_cognate_count; i++) {
        if (strcmp(bucket->supporting_cognates[i], id) == 0) {
            return RG_OK;
        }
    }
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

static int bucket_equal(const class_bucket *bucket, char **lects, char **graphemes,
                        seg_supra *supra, size_t count) {
    size_t i;
    if (bucket->segment_count != count) {
        return 0;
    }
    for (i = 0; i < count; i++) {
        if (strcmp(bucket->lect_ids[i], lects[i]) != 0 ||
            strcmp(bucket->graphemes[i], graphemes[i]) != 0 ||
            !seg_supra_equal_internal(&bucket->supra[i], &supra[i])) {
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
    seg_supra *supra,
    size_t count,
    double weight,
    const char *cognate_id,
    size_t *out_bucket_index
) {
    size_t i;
    for (i = 0; i < *bucket_count; i++) {
        if (bucket_equal(&(*buckets)[i], lects, graphemes, supra, count)) {
            (*buckets)[i].count += weight;
            if (bucket_append_support(&(*buckets)[i], cognate_id) != RG_OK) {
                return RG_ERR_OOM;
            }
            string_array_clear(lects, count);
            string_array_clear(graphemes, count);
            seg_supra_free_array_internal(supra, count);
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
    (*buckets)[*bucket_count].supra = supra;
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
        /* Suprasegmentals are part of the outcome identity, so two classes of
         * the same graphemes under a different tone must order deterministically
         * -- otherwise class ids depend on qsort's whim and diverge between the
         * native and WebAssembly builds. */
        c = strcmp(ba->supra[i].tone, bb->supra[i].tone);
        if (c != 0) {
            return c;
        }
        c = strcmp(ba->supra[i].length, bb->supra[i].length);
        if (c != 0) {
            return c;
        }
        c = strcmp(ba->supra[i].stress, bb->supra[i].stress);
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

/* One recorded gap: the node of the lect that kept a segment, and the index of
 * the lect that had nothing to align to it. After union-find settles, the
 * keeper's component gains the deleter as a `∅` member. */
typedef struct gap_edge {
    size_t keeper_node;
    size_t deleter_lect;
} gap_edge;

static rg_status gap_edges_append(gap_edge **edges, size_t *count, size_t *cap,
                                  size_t keeper_node, size_t deleter_lect) {
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 8 : *cap * 2;
        gap_edge *next = (gap_edge *)realloc(*edges, next_cap * sizeof(*next));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        *edges = next;
        *cap = next_cap;
    }
    (*edges)[*count].keeper_node = keeper_node;
    (*edges)[*count].deleter_lect = deleter_lect;
    (*count)++;
    return RG_OK;
}

/* Position-level union-find edges induced by aligning one pair. Equal-length
 * chunks pair position by position; unequal-length non-gap chunks are
 * decomposed through a one-segment sub-alignment; pure gaps induce no edges but
 * are recorded, so the kept side can name the lect that dropped the segment. */
static rg_status union_pair_alignment_edges(
    const rg_context *ctx,
    const rg_pairwise_model *pair_model,
    const rg_train_options *options,
    const rg_form *source,
    const rg_form *target,
    size_t source_offset,
    size_t target_offset,
    size_t source_lect,
    size_t target_lect,
    uf_state *uf,
    gap_edge **gaps,
    size_t *gap_count,
    size_t *gap_cap
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
            for (k = 0; k < rg_alignment_link_count(sub) && status == RG_OK; k++) {
                const rg_link *sub_link = rg_alignment_link_at(sub, k);
                size_t s;
                if (sub_link->source_count == 1 && sub_link->target_count == 1) {
                    uf_union(uf, source_offset + source_pos + sub_source_pos, target_offset + target_pos + sub_target_pos);
                } else if (sub_link->target_count == 0 && sub_link->source_count != 0) {
                    /* A loss inside an unequal chunk -- the common shape of a
                     * final -n dropped as `an ~ a`. The kept source segments
                     * name the target as their deleter. */
                    for (s = 0; s < sub_link->source_count; s++) {
                        status = gap_edges_append(gaps, gap_count, gap_cap,
                                                  source_offset + source_pos + sub_source_pos + s, target_lect);
                    }
                } else if (sub_link->source_count == 0 && sub_link->target_count != 0) {
                    for (s = 0; s < sub_link->target_count; s++) {
                        status = gap_edges_append(gaps, gap_count, gap_cap,
                                                  target_offset + target_pos + sub_target_pos + s, source_lect);
                    }
                }
                sub_source_pos += sub_link->source_count;
                sub_target_pos += sub_link->target_count;
            }
            rg_alignment_free(sub);
            if (status != RG_OK) {
                rg_alignment_free(alignment);
                return status;
            }
        } else if (link->target_count == 0 && link->source_count != 0) {
            /* The source kept these segments; the target dropped them. Each
             * kept node names the target as a deleter of its component. */
            for (k = 0; k < link->source_count; k++) {
                status = gap_edges_append(gaps, gap_count, gap_cap,
                                          source_offset + source_pos + k, target_lect);
                if (status != RG_OK) {
                    rg_alignment_free(alignment);
                    return status;
                }
            }
        } else if (link->source_count == 0 && link->target_count != 0) {
            for (k = 0; k < link->target_count; k++) {
                status = gap_edges_append(gaps, gap_count, gap_cap,
                                          target_offset + target_pos + k, source_lect);
                if (status != RG_OK) {
                    rg_alignment_free(alignment);
                    return status;
                }
            }
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
        gap_edge *gap_edges = 0;
        size_t gap_edge_count = 0;
        size_t gap_edge_cap = 0;
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
                    a,
                    b,
                    &uf,
                    &gap_edges,
                    &gap_edge_count,
                    &gap_edge_cap
                );
            }
        }
        for (i = 0; i < node_count && status == RG_OK; i++) {
            size_t root = uf_find(&uf, i);
            reconciled_observation obs;
            size_t item_count = 0;
            int inconsistent = 0;
            if (seen_roots[root]) {
                continue;
            }
            seen_roots[root] = 1;
            memset(&obs, 0, sizeof(obs));
            obs.lects = (char **)calloc(order_count, sizeof(*obs.lects));
            obs.graphemes = (char **)calloc(order_count, sizeof(*obs.graphemes));
            obs.supra = (seg_supra *)calloc(order_count, sizeof(*obs.supra));
            obs.positions = (size_t *)calloc(order_count, sizeof(*obs.positions));
            obs.lect_indices = (size_t *)calloc(order_count, sizeof(*obs.lect_indices));
            if (obs.lects == 0 || obs.graphemes == 0 || obs.supra == 0 || obs.positions == 0 || obs.lect_indices == 0) {
                obs.segment_count = 0;
                reconciled_observation_clear(&obs);
                status = RG_ERR_OOM;
                break;
            }
            /* Nodes are visited in ascending (lect, position) order, so the real
             * members accumulate already sorted by lect id. A lect that dropped
             * a segment aligning to this component is a member too, carrying the
             * gap grapheme; its lect id is merged into the same ascending order
             * so the class tuple reads left to right regardless of who deleted.
             * Held apart from the real members until both are known, because a
             * deleter that also has a real segment here is not a deletion. */
            {
                size_t g;
                size_t next_node = 0;
                size_t next_gap = 0;
                /* The deleter lects for this component, ascending and deduped. */
                size_t *gap_lects = (size_t *)calloc(order_count == 0 ? 1 : order_count, sizeof(*gap_lects));
                size_t gap_lects_count = 0;
                if (gap_lects == 0) {
                    reconciled_observation_clear(&obs);
                    status = RG_ERR_OOM;
                    break;
                }
                for (g = 0; g < gap_edge_count; g++) {
                    size_t dl = gap_edges[g].deleter_lect;
                    size_t existing;
                    int is_real = 0;
                    size_t nn;
                    if (uf_find(&uf, gap_edges[g].keeper_node) != root) {
                        continue;
                    }
                    /* A lect present in the component kept the segment; its own
                     * gap edge elsewhere does not make it a deleter here. */
                    for (nn = 0; nn < node_count; nn++) {
                        if (uf_find(&uf, nn) == root && node_lects[nn] == dl) {
                            is_real = 1;
                            break;
                        }
                    }
                    if (is_real) {
                        continue;
                    }
                    for (existing = 0; existing < gap_lects_count; existing++) {
                        if (gap_lects[existing] == dl) {
                            break;
                        }
                    }
                    if (existing < gap_lects_count) {
                        continue;
                    }
                    /* Insertion sort keeps the merge below a single pass. */
                    {
                        size_t at = gap_lects_count;
                        while (at > 0 && gap_lects[at - 1] > dl) {
                            gap_lects[at] = gap_lects[at - 1];
                            at--;
                        }
                        gap_lects[at] = dl;
                        gap_lects_count++;
                    }
                }
                /* Merge the real members and the gap members by lect id. */
                while ((next_node < node_count || next_gap < gap_lects_count) && status == RG_OK) {
                    size_t real_lect = (size_t)-1;
                    /* Advance to the next node that belongs to this component. */
                    while (next_node < node_count && uf_find(&uf, next_node) != root) {
                        next_node++;
                    }
                    if (next_node < node_count) {
                        real_lect = node_lects[next_node];
                    }
                    if (next_gap < gap_lects_count &&
                        (next_node >= node_count || gap_lects[next_gap] < real_lect)) {
                        obs.lects[item_count] = rg_strdup_internal(model->lect_ids[gap_lects[next_gap]]);
                        obs.graphemes[item_count] = rg_strdup_internal(RG_GAP_GRAPHEME);
                        (void)seg_supra_set_internal(&obs.supra[item_count], 0);
                        obs.positions[item_count] = (size_t)-1;
                        obs.lect_indices[item_count] = gap_lects[next_gap];
                        next_gap++;
                    } else if (next_node < node_count) {
                        if (item_count > 0 && obs.lect_indices[item_count - 1] == real_lect) {
                            inconsistent = 1;
                            break;
                        }
                        obs.lects[item_count] = rg_strdup_internal(model->lect_ids[real_lect]);
                        obs.graphemes[item_count] = rg_strdup_internal(forms[real_lect]->segments[node_positions[next_node]].grapheme);
                        (void)seg_supra_set_internal(&obs.supra[item_count],
                                                     &forms[real_lect]->segments[node_positions[next_node]]);
                        obs.positions[item_count] = node_positions[next_node];
                        obs.lect_indices[item_count] = real_lect;
                        next_node++;
                    } else {
                        break;
                    }
                    if (obs.lects[item_count] == 0 || obs.graphemes[item_count] == 0 ||
                        obs.supra[item_count].tone == 0) {
                        obs.segment_count = item_count + 1;
                        reconciled_observation_clear(&obs);
                        status = RG_ERR_OOM;
                        break;
                    }
                    item_count++;
                    (void)node_order;
                }
                free(gap_lects);
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
                seg_supra *supra_copy = seg_supra_dup_array_internal(obs.supra, item_count);
                size_t k;
                if (lects_copy == 0 || graphemes_copy == 0 || supra_copy == 0) {
                    free(lects_copy);
                    free(graphemes_copy);
                    seg_supra_free_array_internal(supra_copy, item_count);
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
                    seg_supra_free_array_internal(supra_copy, item_count);
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
                    supra_copy,
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
        free(gap_edges);
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
    model->unconditioned_classes = (rg_multi_class_row *)calloc(bucket_count == 0 ? 1 : bucket_count, sizeof(*model->unconditioned_classes));
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
        model->unconditioned_classes[c].class_id = (int)c;
        model->unconditioned_classes[c].lect_ids = (const char *const *)buckets[c].lect_ids;
        model->unconditioned_classes[c].graphemes = (const char *const *)buckets[c].graphemes;
        /* Layout-compatible with rg_suprasegmentals; the row takes ownership and
         * multi_class_clear frees it. */
        model->unconditioned_classes[c].suprasegmentals = (const rg_suprasegmentals *)buckets[c].supra;
        model->unconditioned_classes[c].segment_count = buckets[c].segment_count;
        model->unconditioned_classes[c].count = buckets[c].count;
        model->unconditioned_classes[c].confidence = 1.0;
        /* An unconditioned class is aggregated, not decided: no environment,
         * so no complement to contrast against. */
        model->unconditioned_classes[c].evidence.decision_index = -1;
        model->unconditioned_classes[c].contrast_class_id = -1;
        model->unconditioned_classes[c].supporting_cognates = (const char *const *)buckets[c].supporting_cognates;
        model->unconditioned_classes[c].supporting_cognate_count = buckets[c].supporting_cognate_count;
        model->unconditioned_classes[c].uncertainty = rg_wilson_default_internal(buckets[c].count, participant_total);
        buckets[c].lect_ids = 0;
        buckets[c].graphemes = 0;
        buckets[c].supra = 0;
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

