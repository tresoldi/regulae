/* Conditioned classes that look like one change, grouped.
 *
 * A change applying to more than one segment is published as one class per
 * segment. `testdata/soundlaws/natural_class.tsv` voices four continuants
 * between vowels and comes out as four conditioned rows differing only in
 * their graphemes, carrying the same environment and a search margin of 3.24
 * apiece; its control puts the same thirty-two observations on one segment and
 * states the change once, at 14.11. The evidence is identical. What differs is
 * that a reader of the first has to notice the four rows are one event, and
 * nothing published says so.
 *
 * This says so, and stops there. Whether the pooled description is the *better*
 * one is a model-selection question -- a pooled rule buys a shorter description
 * and pays for a class predicate -- and answering it is not the same as
 * noticing the grouping. The fixture pair exists to score any answer on; until
 * something is scored, proposing is the honest verb.
 *
 * Nothing here rewrites a class. Every member stays published exactly as it
 * was, so a consumer that disagrees with a grouping ignores this table.
 */

#include "multilect_internal.h"
#include "environment.h"

#include <stdlib.h>
#include <string.h>

static int class_is_identity(const rg_multi_class_row *row) {
    size_t i;
    if (row->segment_count == 0) {
        return 0;
    }
    for (i = 1; i < row->segment_count; i++) {
        if (strcmp(row->graphemes[i], row->graphemes[0]) != 0) {
            return 0;
        }
    }
    return 1;
}

/* Two classes are candidates for one event when they say the same thing about
 * the same lects in the same place: same participating lects, in the same
 * order, and an identical environment on each. Graphemes are what may differ,
 * and must -- otherwise it is one class seen twice. */
static int classes_share_an_environment(
    const rg_multi_class_row *a,
    const rg_multi_class_row *b
) {
    size_t i;
    int graphemes_differ = 0;
    if (a->segment_count != b->segment_count || a->segment_count == 0) {
        return 0;
    }
    for (i = 0; i < a->segment_count; i++) {
        if (strcmp(a->lect_ids[i], b->lect_ids[i]) != 0) {
            return 0;
        }
        if (rg_context_spec_compare_internal(&a->contexts[i], &b->contexts[i]) != 0) {
            return 0;
        }
        if (strcmp(a->graphemes[i], b->graphemes[i]) != 0) {
            graphemes_differ = 1;
        }
    }
    return graphemes_differ;
}

static int feature_set_has(const rg_feature_set *set, const char *feature) {
    size_t i;
    for (i = 0; i < rg_feature_set_size(set); i++) {
        const char *item = rg_feature_set_get(set, i);
        if (item != 0 && strcmp(item, feature) == 0) {
            return 1;
        }
    }
    return 0;
}

/* Two unconditioned classes share a feature displacement when the feature
 * difference between their lect-pair graphemes is identical: if class A shows
 * p~f (losing `stop`, gaining `fricative`) and class B shows t~θ (same), they
 * share the displacement and group. */
static int classes_share_displacement(
    const rg_context *ctx,
    const rg_multi_class_row *a,
    const rg_multi_class_row *b
) {
    size_t p;
    size_t q;
    int graphemes_differ = 0;
    if (a->segment_count != b->segment_count || a->segment_count < 2) {
        return 0;
    }
    for (p = 0; p < a->segment_count; p++) {
        if (strcmp(a->lect_ids[p], b->lect_ids[p]) != 0) {
            return 0;
        }
        if (strcmp(a->graphemes[p], b->graphemes[p]) != 0) {
            graphemes_differ = 1;
        }
    }
    if (!graphemes_differ) {
        return 0;
    }
    for (p = 0; p < a->segment_count; p++) {
        for (q = p + 1; q < a->segment_count; q++) {
            const rg_feature_set *sets[4];
            size_t s;
            size_t f;
            if (rg_context_features_internal(ctx, a->graphemes[p], &sets[0]) != RG_OK ||
                rg_context_features_internal(ctx, a->graphemes[q], &sets[1]) != RG_OK ||
                rg_context_features_internal(ctx, b->graphemes[p], &sets[2]) != RG_OK ||
                rg_context_features_internal(ctx, b->graphemes[q], &sets[3]) != RG_OK) {
                return 0;
            }
            for (s = 0; s < 4; s++) {
                for (f = 0; f < rg_feature_set_size(sets[s]); f++) {
                    const char *feat = rg_feature_set_get(sets[s], f);
                    int da;
                    int db;
                    if (feat == 0) {
                        continue;
                    }
                    da = feature_set_has(sets[0], feat) - feature_set_has(sets[1], feat);
                    db = feature_set_has(sets[2], feat) - feature_set_has(sets[3], feat);
                    if (da != db) {
                        return 0;
                    }
                }
            }
        }
    }
    return 1;
}

static int string_array_contains(char *const *items, size_t count, const char *value) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (strcmp(items[i], value) == 0) {
            return 1;
        }
    }
    return 0;
}

static rg_status string_array_add(char ***items, size_t *count, const char *value) {
    char **next;
    char *copy;
    if (string_array_contains(*items, *count, value)) {
        return RG_OK;
    }
    next = (char **)realloc(*items, (*count + 1) * sizeof(*next));
    if (next == 0) {
        return RG_ERR_OOM;
    }
    *items = next;
    copy = rg_strdup_internal(value);
    if (copy == 0) {
        return RG_ERR_OOM;
    }
    (*items)[(*count)++] = copy;
    return RG_OK;
}

static void string_array_free(char **items, size_t count) {
    size_t i;
    for (i = 0; i < count; i++) {
        free(items[i]);
    }
    free(items);
}

static int string_ptr_cmp(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

/* Every grapheme one lect shows anywhere in the corpus. The class test is
 * relative to this rather than to the feature system's whole inventory: a
 * feature separates a set of sounds *in this corpus*, and whether it would
 * also separate them in a language that is not here says nothing about the
 * change. */
static rg_status lect_inventory(
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const char *lect_id,
    char ***out,
    size_t *out_count
) {
    size_t i;
    size_t f;
    size_t s;
    rg_status status = RG_OK;
    *out = 0;
    *out_count = 0;
    for (i = 0; i < cognate_count && status == RG_OK; i++) {
        for (f = 0; f < cognates[i].form_count && status == RG_OK; f++) {
            const rg_cognate_form *form = &cognates[i].forms[f];
            if (strcmp(form->lect_id, lect_id) != 0) {
                continue;
            }
            for (s = 0; s < form->form.segment_count && status == RG_OK; s++) {
                const char *grapheme = form->form.segments[s].grapheme;
                if (grapheme != 0 && grapheme[0] != '\0') {
                    status = string_array_add(out, out_count, grapheme);
                }
            }
        }
    }
    return status;
}

static int grapheme_has_feature(const rg_context *ctx, const char *grapheme, const char *feature) {
    const rg_feature_set *features = 0;
    size_t i;
    if (rg_context_features_internal(ctx, grapheme, &features) != RG_OK) {
        return 0;
    }
    for (i = 0; i < rg_feature_set_size(features); i++) {
        const char *item = rg_feature_set_get(features, i);
        if (item != 0 && strcmp(item, feature) == 0) {
            return 1;
        }
    }
    return 0;
}

/* Features every member carries and no non-member in this lect's inventory
 * does -- what makes the set a class rather than a list.
 *
 * Empty is a real answer and a common one. Mielke's survey of nearly six
 * hundred languages found no feature theory expressing more than 71% of
 * attested active classes, so a set no feature separates may still be exactly
 * the set a change applied to. The event is published either way and says
 * which it was. */
/* Whether a feature, or a pair of them, holds of exactly `members` within this
 * lect's inventory. */
static int separates(
    const rg_context *ctx,
    const char *first,
    const char *second,
    char *const *members,
    size_t member_count,
    char *const *inventory,
    size_t inventory_count
) {
    size_t i;
    for (i = 0; i < member_count; i++) {
        if (!grapheme_has_feature(ctx, members[i], first)) {
            return 0;
        }
        if (second != 0 && !grapheme_has_feature(ctx, members[i], second)) {
            return 0;
        }
    }
    for (i = 0; i < inventory_count; i++) {
        int held;
        if (string_array_contains(members, member_count, inventory[i])) {
            continue;
        }
        held = grapheme_has_feature(ctx, inventory[i], first) &&
               (second == 0 || grapheme_has_feature(ctx, inventory[i], second));
        if (held) {
            return 0;
        }
    }
    return 1;
}

/* Features holding of every member and of no non-member in this lect's
 * inventory -- what makes the set a class rather than a list.
 *
 * Single features first, then pairs, and no deeper. The bound is a bound and
 * not a theory: `{v z ð ɣ}` needs `fricative` and `voiced` together because
 * either alone catches something else, which is ordinary, while a set needing
 * three conjuncts to pick it out is not obviously a class at all. Stopping
 * where the conditioning search stops would be a coincidence; stopping at two
 * is a decision, and a class this cannot name is reported unnamed rather than
 * described by a longer conjunction that would fit anything.
 *
 * Empty is a real answer and a common one. Mielke's survey of nearly six
 * hundred languages found no feature theory expressing more than 71% of
 * attested active classes, so a set no feature separates may still be exactly
 * the set a change applied to. The event is published either way and
 * `featurally_definable` says which it was. */
static rg_status class_features_for(
    const rg_context *ctx,
    char *const *members,
    size_t member_count,
    char *const *inventory,
    size_t inventory_count,
    char ***out,
    size_t *out_count
) {
    const rg_feature_set *features = 0;
    size_t count;
    size_t i;
    size_t j;
    rg_status status = RG_OK;
    *out = 0;
    *out_count = 0;
    if (member_count == 0 ||
        rg_context_features_internal(ctx, members[0], &features) != RG_OK) {
        return RG_OK;
    }
    count = rg_feature_set_size(features);
    for (i = 0; i < count && status == RG_OK; i++) {
        const char *feature = rg_feature_set_get(features, i);
        if (feature != 0 &&
            separates(ctx, feature, 0, members, member_count, inventory, inventory_count)) {
            status = string_array_add(out, out_count, feature);
        }
    }
    for (i = 0; i < count && status == RG_OK && *out_count == 0; i++) {
        const char *first = rg_feature_set_get(features, i);
        if (first == 0) {
            continue;
        }
        for (j = i + 1; j < count && status == RG_OK && *out_count == 0; j++) {
            const char *second = rg_feature_set_get(features, j);
            if (second == 0 ||
                !separates(ctx, first, second, members, member_count,
                           inventory, inventory_count)) {
                continue;
            }
            status = string_array_add(out, out_count, first);
            if (status == RG_OK) {
                status = string_array_add(out, out_count, second);
            }
        }
    }
    if (status == RG_OK && *out_count > 1) {
        qsort(*out, *out_count, sizeof(**out), string_ptr_cmp);
    }
    return status;
}

static void event_row_clear(rg_proposed_event_row *row) {
    size_t i;
    if (row == 0) {
        return;
    }
    for (i = 0; i < row->member_count; i++) {
        rg_event_member *member = (rg_event_member *)rg_owned_internal(&row->members[i]);
        rg_free_owned_internal(member->lect_id);
        string_array_free((char **)rg_owned_internal(member->graphemes), member->grapheme_count);
        string_array_free((char **)rg_owned_internal(member->class_features),
                          member->class_feature_count);
        member->graphemes = 0;
        member->class_features = 0;
    }
    free(rg_owned_internal(row->members));
    free(rg_owned_internal(row->class_ids));
    string_array_free((char **)rg_owned_internal(row->supporting_cognates),
                      row->supporting_cognate_count);
    memset(row, 0, sizeof(*row));
}

void rg_proposed_events_free_internal(rg_proposed_event_row *rows, size_t count) {
    size_t i;
    if (rows == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        event_row_clear(&rows[i]);
    }
    free(rows);
}

/* One event out of the classes at `indices`. */
static rg_status build_event(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_multi_class_row *classes,
    const size_t *indices,
    size_t index_count,
    rg_proposed_event_row *out
) {
    const rg_multi_class_row *first = &classes[indices[0]];
    rg_event_member *members = 0;
    int *class_ids = 0;
    char **cognate_ids = 0;
    size_t cognate_id_count = 0;
    size_t slot;
    size_t i;
    rg_status status = RG_OK;

    memset(out, 0, sizeof(*out));
    members = (rg_event_member *)calloc(first->segment_count, sizeof(*members));
    class_ids = (int *)calloc(index_count, sizeof(*class_ids));
    if (members == 0 || class_ids == 0) {
        free(members);
        free(class_ids);
        return RG_ERR_OOM;
    }

    for (i = 0; i < index_count; i++) {
        const rg_multi_class_row *row = &classes[indices[i]];
        size_t k;
        class_ids[i] = row->class_id;
        out->count += row->count;
        for (k = 0; k < row->supporting_cognate_count && status == RG_OK; k++) {
            status = string_array_add(&cognate_ids, &cognate_id_count,
                                      row->supporting_cognates[k]);
        }
    }
    out->search_margin = first->evidence.search_margin;
    out->delta_score = first->evidence.delta_score;
    out->featurally_definable = true;

    for (slot = 0; slot < first->segment_count && status == RG_OK; slot++) {
        char **graphemes = 0;
        size_t grapheme_count = 0;
        char **inventory = 0;
        size_t inventory_count = 0;
        char **features = 0;
        size_t feature_count = 0;
        for (i = 0; i < index_count && status == RG_OK; i++) {
            status = string_array_add(&graphemes, &grapheme_count,
                                      classes[indices[i]].graphemes[slot]);
        }
        if (status == RG_OK && grapheme_count > 1) {
            qsort(graphemes, grapheme_count, sizeof(*graphemes), string_ptr_cmp);
        }
        if (status == RG_OK) {
            status = lect_inventory(cognates, cognate_count, first->lect_ids[slot],
                                    &inventory, &inventory_count);
        }
        if (status == RG_OK) {
            status = class_features_for(ctx, graphemes, grapheme_count,
                                        inventory, inventory_count,
                                        &features, &feature_count);
        }
        string_array_free(inventory, inventory_count);
        if (status != RG_OK) {
            string_array_free(graphemes, grapheme_count);
            string_array_free(features, feature_count);
            break;
        }
        members[slot].lect_id = rg_strdup_internal(first->lect_ids[slot]);
        members[slot].graphemes = (const char *const *)graphemes;
        members[slot].grapheme_count = grapheme_count;
        members[slot].class_features = (const char *const *)features;
        members[slot].class_feature_count = feature_count;
        if (members[slot].lect_id == 0) {
            status = RG_ERR_OOM;
            break;
        }
        /* A slot contributing one grapheme is not a class and needs no feature
         * to name it; only the slots that vary have to be nameable. */
        if (grapheme_count > 1 && feature_count == 0) {
            out->featurally_definable = false;
        }
    }

    out->members = members;
    out->member_count = first->segment_count;
    out->class_ids = class_ids;
    out->class_id_count = index_count;
    if (status == RG_OK && cognate_id_count > 1) {
        qsort(cognate_ids, cognate_id_count, sizeof(*cognate_ids), string_ptr_cmp);
    }
    out->supporting_cognates = (const char *const *)cognate_ids;
    out->supporting_cognate_count = cognate_id_count;
    if (status != RG_OK) {
        event_row_clear(out);
    }
    return status;
}

static int has_conditioned_counterpart(
    const rg_multi_class_row *cond_classes,
    size_t cond_count,
    const rg_multi_class_row *row
) {
    size_t i;
    for (i = 0; i < cond_count; i++) {
        const rg_multi_class_row *c = &cond_classes[i];
        size_t j;
        int match = 1;
        if (c->segment_count != row->segment_count) {
            continue;
        }
        for (j = 0; j < c->segment_count; j++) {
            if (strcmp(c->lect_ids[j], row->lect_ids[j]) != 0 ||
                strcmp(c->graphemes[j], row->graphemes[j]) != 0) {
                match = 0;
                break;
            }
        }
        if (match) {
            return 1;
        }
    }
    return 0;
}

static rg_status group_classes(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_multi_class_row *classes,
    size_t class_count,
    int use_displacement,
    const rg_multi_class_row *exclude_if_in,
    size_t exclude_count,
    rg_proposed_event_row **rows,
    size_t *row_count
) {
    char *grouped = 0;
    size_t *indices = 0;
    size_t i;
    rg_status status = RG_OK;

    if (class_count < 2) {
        return RG_OK;
    }
    grouped = (char *)calloc(class_count, 1);
    indices = (size_t *)calloc(class_count, sizeof(*indices));
    if (grouped == 0 || indices == 0) {
        free(grouped);
        free(indices);
        return RG_ERR_OOM;
    }
    for (i = 0; i < class_count; i++) {
        if (class_is_identity(&classes[i])) {
            grouped[i] = 1;
        } else if (exclude_if_in != 0 &&
                   has_conditioned_counterpart(exclude_if_in, exclude_count, &classes[i])) {
            grouped[i] = 1;
        }
    }
    for (i = 0; i < class_count && status == RG_OK; i++) {
        size_t member_count = 0;
        size_t j;
        rg_proposed_event_row *next;
        if (grouped[i]) {
            continue;
        }
        indices[member_count++] = i;
        for (j = i + 1; j < class_count; j++) {
            if (grouped[j]) {
                continue;
            }
            if (use_displacement
                    ? classes_share_displacement(ctx, &classes[i], &classes[j])
                    : classes_share_an_environment(&classes[i], &classes[j])) {
                grouped[j] = 1;
                indices[member_count++] = j;
            }
        }
        if (member_count < 2) {
            continue;
        }
        grouped[i] = 1;
        next = (rg_proposed_event_row *)realloc(*rows, (*row_count + 1) * sizeof(*next));
        if (next == 0) {
            status = RG_ERR_OOM;
            break;
        }
        *rows = next;
        status = build_event(ctx, cognates, cognate_count, classes,
                             indices, member_count, &(*rows)[*row_count]);
        if (status == RG_OK) {
            (*row_count)++;
        }
    }
    free(grouped);
    free(indices);
    return status;
}

rg_status rg_propose_events_internal(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    rg_multi_model *model
) {
    rg_proposed_event_row *rows = 0;
    size_t row_count = 0;
    rg_status status;

    if (model == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    status = group_classes(ctx, cognates, cognate_count,
                           model->conditioned_classes,
                           model->conditioned_class_count,
                           0, 0, 0, &rows, &row_count);
    if (status == RG_OK) {
        status = group_classes(ctx, cognates, cognate_count,
                               model->unconditioned_classes,
                               model->unconditioned_class_count,
                               1,
                               model->conditioned_classes,
                               model->conditioned_class_count,
                               &rows, &row_count);
    }
    if (status != RG_OK) {
        rg_proposed_events_free_internal(rows, row_count);
        return status;
    }
    model->proposed_events = rows;
    model->proposed_event_count = row_count;
    return RG_OK;
}
