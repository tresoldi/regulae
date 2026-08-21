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
 *
 * Three axes, because "these rows are one change" is three claims:
 *
 *   GROUP_BY_ENVIRONMENT   one environment, several outcomes -- a change over
 *                          a class of segments (the natural-class fixture)
 *   GROUP_BY_OUTCOME       one outcome, several environments -- a change stated
 *                          as a decision list (the disjunction fixture)
 *   GROUP_BY_DISPLACEMENT  no environment -- held together by what it displaces
 *                          (Grimm, the tone fixtures)
 *
 * What holds a displacement grouping together is a *shared* displacement and
 * not an identical one. Requiring identity was the reason the most cited sound
 * change in the literature could not be grouped: p~f, t~θ and k~x agree on
 * stop→fricative and disagree on place, so no two of their feature deltas are
 * equal, and Grimm's first shift came out as three unrelated rows. The
 * intersection is the rule, and `compute_shared_displacement` was already
 * computing it in order to *report* what the grouping meant.
 *
 * Loosening the predicate that far needs something to hold the other end, and
 * two things do. The group's own displacement -- intersected over every member,
 * not checked pairwise -- has to stay non-empty, because sharing a displacement
 * is not transitive and a chain of pairwise agreement can end up asserting
 * nothing. And the members have to read as one regular change from some lect's
 * side: distinct inputs, each the leading answer for its own segment. Without
 * the second, the unrelated-wordlist corpus proposes fourteen-class groupings
 * of everything that happens to lean the same way; with it, it proposes about
 * as many events as it has real correspondences, which is the restraint that
 * corpus exists to measure.
 */

#include "multilect_internal.h"
#include "environment.h"

#include <stdlib.h>
#include <string.h>

/* Which suprasegmental dimensions each lect writes anywhere in this table.
 *
 * A dimension one lect never writes is a transcription convention, not a
 * property it lacks. Verner's fixture marks stress on Proto-Germanic and not on
 * Gothic, so every vowel pair in it reads as `a[str:primary] ~ a` -- and read
 * literally that is five vowels agreeing on one change, which is a well
 * attested regularity of the file and nothing at all about the language. The
 * same annotation asymmetry would put "lost its tone" on any corpus that
 * transcribes tone on one side of the comparison.
 *
 * So a dimension counts toward an outcome only where both lects write it
 * somewhere. Where they do, a difference is a real difference; where they do
 * not, it is the file, and this is the same restraint `lect_inventory` shows
 * for graphemes: a class is a class relative to what the corpus attests. */
typedef struct supra_scope {
    const char **lect_ids;
    char *tone;
    char *length;
    char *stress;
    size_t count;
} supra_scope;

static void supra_scope_free(supra_scope *scope) {
    free((void *)scope->lect_ids);
    free(scope->tone);
    free(scope->length);
    free(scope->stress);
    memset(scope, 0, sizeof(*scope));
}

static rg_status supra_scope_build(
    const rg_multi_class_row *classes,
    size_t class_count,
    supra_scope *out
) {
    size_t i;
    size_t slot;
    memset(out, 0, sizeof(*out));
    if (class_count == 0) {
        return RG_OK;
    }
    for (i = 0; i < class_count; i++) {
        for (slot = 0; slot < classes[i].segment_count; slot++) {
            const char *lect = classes[i].lect_ids[slot];
            size_t k;
            int known = 0;
            for (k = 0; k < out->count; k++) {
                if (strcmp(out->lect_ids[k], lect) == 0) {
                    known = 1;
                    break;
                }
            }
            if (!known) {
                const char **ids = (const char **)realloc(
                    out->lect_ids, (out->count + 1) * sizeof(*ids));
                char *tone = (char *)realloc(out->tone, out->count + 1);
                char *length = (char *)realloc(out->length, out->count + 1);
                char *stress = (char *)realloc(out->stress, out->count + 1);
                if (ids != 0) {
                    out->lect_ids = ids;
                }
                if (tone != 0) {
                    out->tone = tone;
                }
                if (length != 0) {
                    out->length = length;
                }
                if (stress != 0) {
                    out->stress = stress;
                }
                if (ids == 0 || tone == 0 || length == 0 || stress == 0) {
                    supra_scope_free(out);
                    return RG_ERR_OOM;
                }
                k = out->count++;
                out->lect_ids[k] = lect;
                out->tone[k] = 0;
                out->length[k] = 0;
                out->stress[k] = 0;
            }
            if (classes[i].suprasegmentals != 0) {
                const rg_suprasegmentals *s = &classes[i].suprasegmentals[slot];
                if (s->tone != 0 && s->tone[0] != '\0') {
                    out->tone[k] = 1;
                }
                if (s->length != 0 && s->length[0] != '\0') {
                    out->length[k] = 1;
                }
                if (s->stress != 0 && s->stress[0] != '\0') {
                    out->stress[k] = 1;
                }
            }
        }
    }
    return RG_OK;
}

/* Whether every lect this class names writes `dimension` somewhere. */
static int row_lects_all_attest(
    const supra_scope *scope,
    const rg_multi_class_row *row,
    const char *dimension
) {
    size_t slot;
    if (scope == 0) {
        return 0;
    }
    for (slot = 0; slot < row->segment_count; slot++) {
        size_t k;
        int attested = 0;
        for (k = 0; k < scope->count; k++) {
            const char *marks = dimension[0] == 't' ? scope->tone
                              : dimension[0] == 'l' ? scope->length
                                                    : scope->stress;
            if (strcmp(scope->lect_ids[k], row->lect_ids[slot]) == 0) {
                attested = marks[k];
                break;
            }
        }
        if (!attested) {
            return 0;
        }
    }
    return 1;
}

/* One slot's suprasegmentals, blanked in every dimension some lect of the class
 * does not write. NULL `suprasegmentals` on a conditioned class, whose outcome
 * is the segmental split -- see rg_multi_class_row.
 *
 * Blanked for the whole class, not per lect: the asymmetry is the point. A
 * dimension one side never writes cannot tell a difference between the sides
 * apart from a difference between the transcriptions, so it says nothing here
 * either way. */
static rg_suprasegmentals slot_suprasegmentals(
    const supra_scope *scope,
    const rg_multi_class_row *row,
    size_t slot
) {
    rg_suprasegmentals value;
    value.tone = "";
    value.length = "";
    value.stress = "";
    if (row->suprasegmentals == 0) {
        return value;
    }
    if (row_lects_all_attest(scope, row, "tone")) {
        value.tone = row->suprasegmentals[slot].tone;
    }
    if (row_lects_all_attest(scope, row, "length")) {
        value.length = row->suprasegmentals[slot].length;
    }
    if (row_lects_all_attest(scope, row, "stress")) {
        value.stress = row->suprasegmentals[slot].stress;
    }
    return value;
}

static int nullable_str_equal(const char *a, const char *b) {
    if (a == 0 || b == 0) {
        return (a == 0) && (b == 0);
    }
    return strcmp(a, b) == 0;
}

static int suprasegmentals_equal(const rg_suprasegmentals *a, const rg_suprasegmentals *b) {
    return nullable_str_equal(a->tone, b->tone) &&
           nullable_str_equal(a->length, b->length) &&
           nullable_str_equal(a->stress, b->stress);
}

/* Whether two slots state the same outcome.
 *
 * A class's outcome is its grapheme *and* its suprasegmentals: `a ¹¹ ~ a ³³` is
 * a tone change, and reading the graphemes alone calls it no change at all.
 * For Sinitic, Hmong-Mien, Tai-Kadai, Bantu register and much of Otomanguean
 * the tone correspondence is the correspondence, and grouping that read only
 * `graphemes` had every class in those corpora down as a retention -- all
 * forty-four of them on the three-lect tone fixture -- so no tone change could
 * ever reach this table. */
static int slot_outcomes_equal(
    const supra_scope *scope,
    const rg_multi_class_row *a,
    size_t a_slot,
    const rg_multi_class_row *b,
    size_t b_slot
) {
    rg_suprasegmentals sa = slot_suprasegmentals(scope, a, a_slot);
    rg_suprasegmentals sb = slot_suprasegmentals(scope, b, b_slot);
    return strcmp(a->graphemes[a_slot], b->graphemes[b_slot]) == 0 &&
           suprasegmentals_equal(&sa, &sb);
}

/* A class that states no change: every lect shows the same outcome.
 *
 * Retentions never become event members, and the exclusion is deliberate
 * rather than an oversight about half the split. On a conditioned split the
 * identity row is real evidence -- "θ stays θ after primary stress" is the
 * contrast that makes Verner's voicing visible -- but an event built out of
 * such rows says `{θ,x} ~ {θ,x}`, which is a grouping of things that did not
 * happen. The reader who wants the retention beside the change has a link for
 * it already: `contrast_class_id` on the member row names the class the split
 * was scored against, which is a sharper answer than a second event would be.
 *
 * On the natural-class fixture the four "stays" rows are what makes the
 * control a control, and `test_sound_laws.c` asserts they stay out. */
static int class_is_identity(const supra_scope *scope, const rg_multi_class_row *row) {
    size_t i;
    if (row->segment_count == 0) {
        return 0;
    }
    for (i = 1; i < row->segment_count; i++) {
        if (!slot_outcomes_equal(scope, row, i, row, 0)) {
            return 0;
        }
    }
    return 1;
}

/* Two classes are candidates for one event when they say the same thing about
 * the same lects in the same place: same participating lects, in the same
 * order, and the same environment on each. Outcomes are what may differ, and
 * must -- otherwise it is one class seen twice.
 *
 * "The same environment" is one environment being no narrower than the other,
 * not the two being written identically. Each split is searched on its own, so
 * one member routinely carries a conjunct the others did not need, and byte
 * equality then refuses the grouping over a difference in what the search found
 * convenient. On Verner's fixture `z~s` and `b~f` both turn on primary stress
 * on the Proto-Germanic side, and `z~s` alone picked up "before a vowel" on the
 * Gothic side; they are one change and equality could not say so.
 *
 * The subsumption has to run the same way at every slot -- one class uniformly
 * at least as general as the other. Letting each slot pick its own direction
 * would group a pair where neither environment is the other's, which is two
 * rules that overlap rather than one rule written twice. */
static int classes_share_an_environment(
    const supra_scope *scope,
    const rg_multi_class_row *a,
    const rg_multi_class_row *b
) {
    size_t i;
    int outcomes_differ = 0;
    int a_within_b = 1;
    int b_within_a = 1;
    if (a->segment_count != b->segment_count || a->segment_count == 0) {
        return 0;
    }
    for (i = 0; i < a->segment_count; i++) {
        bool forward = 0;
        bool backward = 0;
        if (strcmp(a->lect_ids[i], b->lect_ids[i]) != 0) {
            return 0;
        }
        if (rg_context_spec_is_subset(&a->contexts[i], &b->contexts[i], &forward) != RG_OK ||
            rg_context_spec_is_subset(&b->contexts[i], &a->contexts[i], &backward) != RG_OK) {
            return 0;
        }
        if (!forward) {
            a_within_b = 0;
        }
        if (!backward) {
            b_within_a = 0;
        }
        if (!a_within_b && !b_within_a) {
            return 0;
        }
        if (!slot_outcomes_equal(scope, a, i, b, i)) {
            outcomes_differ = 1;
        }
    }
    return outcomes_differ;
}

/* Two unconditioned classes share a feature displacement when the feature
 * difference between their lect-pair graphemes is identical: if class A shows
 * p~f (losing `stop`, gaining `fricative`) and class B shows t~θ (same), they
 * share the displacement and group.
 *
 * Suprasegmentals have to agree slot for slot, which is the same rule read on
 * the other dimension. `a[¹¹] ~ a[³³]` and `e[¹¹] ~ e[³³]` are one tone shift
 * over two vowels and group; `a[¹¹] ~ a[³³]` and `a[⁵⁵] ~ a[³³]` are two
 * different shifts that happen to land together and do not, exactly as p~f and
 * k~f do not group under a segmental reading. */
/* The displacement between two graphemes, with ∅ read as the segment carrying
 * no features.
 *
 * A loss is a correspondence to ∅ and nothing else about it is special, but the
 * feature system has no entry for ∅ and answers an error -- so every class with
 * a gap in it used to leave this table by the error path, 393 of the 2001
 * unconditioned classes across the example corpora, and the loss of a whole
 * natural class could never be one event. Read as the featureless segment, a
 * loss displaces every feature the other side carries, which is what a loss is.
 *
 * The result is owned by the caller either way, so the two paths free alike. */
static rg_status event_displacement(
    const rg_context *ctx,
    const char *from,
    const char *to,
    rg_feature_displacement **out,
    size_t *out_count
) {
    const rg_feature_displacement *borrowed = 0;
    size_t count = 0;
    rg_feature_displacement *owned = 0;
    size_t k;
    int from_gap = strcmp(from, RG_GAP_GRAPHEME) == 0;
    int to_gap = strcmp(to, RG_GAP_GRAPHEME) == 0;
    rg_status status;

    *out = 0;
    *out_count = 0;
    if (from_gap && to_gap) {
        return RG_OK;
    }
    if (from_gap || to_gap) {
        const rg_feature_set *present = 0;
        status = rg_context_features_internal(ctx, from_gap ? to : from, &present);
        if (status != RG_OK) {
            return RG_OK;
        }
        count = rg_feature_set_size(present);
        if (count == 0) {
            return RG_OK;
        }
        owned = (rg_feature_displacement *)calloc(count, sizeof(*owned));
        if (owned == 0) {
            return RG_ERR_OOM;
        }
        for (k = 0; k < count; k++) {
            owned[k].feature = rg_strdup_internal(rg_feature_set_get(present, k));
            owned[k].from_value = rg_strdup_internal(from_gap ? "absent" : "present");
            owned[k].to_value = rg_strdup_internal(from_gap ? "present" : "absent");
            if (owned[k].feature == 0 || owned[k].from_value == 0 || owned[k].to_value == 0) {
                rg_feature_displacement_free(owned, k + 1);
                return RG_ERR_OOM;
            }
        }
        *out = owned;
        *out_count = count;
        return RG_OK;
    }
    status = rg_context_displacement_internal(ctx, from, to, &borrowed, &count);
    if (status != RG_OK || count == 0) {
        return status == RG_OK ? RG_OK : status;
    }
    owned = (rg_feature_displacement *)calloc(count, sizeof(*owned));
    if (owned == 0) {
        return RG_ERR_OOM;
    }
    for (k = 0; k < count; k++) {
        owned[k].feature = rg_strdup_internal(borrowed[k].feature);
        owned[k].from_value = rg_strdup_internal(borrowed[k].from_value);
        owned[k].to_value = rg_strdup_internal(borrowed[k].to_value);
        if (owned[k].feature == 0 || owned[k].from_value == 0 || owned[k].to_value == 0) {
            rg_feature_displacement_free(owned, k + 1);
            return RG_ERR_OOM;
        }
    }
    *out = owned;
    *out_count = count;
    return RG_OK;
}

static int displacement_contains(
    const rg_feature_displacement *items,
    size_t count,
    const rg_feature_displacement *item
) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (strcmp(items[i].feature, item->feature) == 0 &&
            strcmp(items[i].from_value, item->from_value) == 0 &&
            strcmp(items[i].to_value, item->to_value) == 0) {
            return 1;
        }
    }
    return 0;
}

/* Whether two classes move in the same direction on at least one feature,
 * between the same two slots.
 *
 * This asks for a shared displacement, where it used to ask for an identical
 * one. The difference is Grimm's first shift: p~f loses `bilabial` and gains
 * `labio-dental`, t~θ loses `alveolar` and gains `dental`, and k~x moves no
 * place at all, so the three deltas are never equal and the most cited sound
 * change in the literature could not be grouped. What they share is
 * stop→fricative, which is the shift, and which is exactly what
 * `compute_shared_displacement` already intersected out in order to *report*
 * the rule. Membership now asks the question the report answers. */
static int displacements_intersect(
    const rg_context *ctx,
    const rg_multi_class_row *a,
    const rg_multi_class_row *b,
    size_t p,
    size_t q
) {
    rg_feature_displacement *da = 0;
    rg_feature_displacement *db = 0;
    size_t da_count = 0;
    size_t db_count = 0;
    size_t k;
    int shared = 0;
    if (event_displacement(ctx, a->graphemes[p], a->graphemes[q], &da, &da_count) != RG_OK ||
        event_displacement(ctx, b->graphemes[p], b->graphemes[q], &db, &db_count) != RG_OK) {
        rg_feature_displacement_free(da, da_count);
        rg_feature_displacement_free(db, db_count);
        return 0;
    }
    for (k = 0; k < da_count && !shared; k++) {
        shared = displacement_contains(db, db_count, &da[k]);
    }
    rg_feature_displacement_free(da, da_count);
    rg_feature_displacement_free(db, db_count);
    return shared;
}

static int classes_share_displacement(
    const rg_context *ctx,
    const supra_scope *scope,
    const rg_multi_class_row *a,
    const rg_multi_class_row *b
) {
    size_t p;
    size_t q;
    int outcomes_differ = 0;
    if (a->segment_count != b->segment_count || a->segment_count < 2) {
        return 0;
    }
    for (p = 0; p < a->segment_count; p++) {
        rg_suprasegmentals sa = slot_suprasegmentals(scope, a, p);
        rg_suprasegmentals sb = slot_suprasegmentals(scope, b, p);
        if (strcmp(a->lect_ids[p], b->lect_ids[p]) != 0) {
            return 0;
        }
        if (!suprasegmentals_equal(&sa, &sb)) {
            return 0;
        }
        /* A loss and a shift are not one event. Read as the featureless
         * segment, ∅ displaces every feature the other side carries, so it
         * shares a displacement with nearly anything and would otherwise
         * collect the corpus: `{k,p,t,x} ~ {b,d,g,∅}` groups a lost velar with
         * Grimm's second shift because losing a segment also loses its voice.
         * Members have to agree on which slots are losses. */
        if ((strcmp(a->graphemes[p], RG_GAP_GRAPHEME) == 0) !=
            (strcmp(b->graphemes[p], RG_GAP_GRAPHEME) == 0)) {
            return 0;
        }
        if (!slot_outcomes_equal(scope, a, p, b, p)) {
            outcomes_differ = 1;
        }
    }
    if (!outcomes_differ) {
        return 0;
    }
    for (p = 0; p < a->segment_count; p++) {
        for (q = p + 1; q < a->segment_count; q++) {
            rg_suprasegmentals sp = slot_suprasegmentals(scope, a, p);
            rg_suprasegmentals sq = slot_suprasegmentals(scope, a, q);
            /* A tone shift moves no segmental feature: between `a` and `a` the
             * displacement is empty for every member, so asking the segments
             * what these two lects did to each other answers nothing. The
             * suprasegmental step is the displacement, and both classes carry
             * the same one -- checked slot by slot above. */
            if (!suprasegmentals_equal(&sp, &sq)) {
                continue;
            }
            if (!displacements_intersect(ctx, a, b, p, q)) {
                return 0;
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
        if (member->suprasegmentals != 0) {
            rg_suprasegmentals *supra =
                (rg_suprasegmentals *)rg_owned_internal(member->suprasegmentals);
            rg_free_owned_internal(supra->tone);
            rg_free_owned_internal(supra->length);
            rg_free_owned_internal(supra->stress);
            free(supra);
        }
        member->graphemes = 0;
        member->class_features = 0;
        member->suprasegmentals = 0;
    }
    free(rg_owned_internal(row->members));
    free(rg_owned_internal(row->class_ids));
    string_array_free((char **)rg_owned_internal(row->supporting_cognates),
                      row->supporting_cognate_count);
    rg_feature_displacement_free(
        rg_owned_internal(row->shared_displacement),
        row->shared_displacement_count);
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

/* The displacement shared by every member class between two slots.
 *
 * The displacement is what changed -- the rule the grouping implies.
 * `classes_share_displacement` already verified every member agrees on
 * it; this function reads the same data and keeps a copy, so the event
 * can say what it found rather than only that it found something.
 *
 * For environment-grouped events the displacement is computed the same
 * way but may come back empty when the members change different
 * features. That is fine: `shared_displacement_count == 0` says the
 * rule is not statable as a single feature change. */
static rg_status compute_shared_displacement(
    const rg_context *ctx,
    const rg_multi_class_row *classes,
    const size_t *indices,
    size_t index_count,
    size_t slot_a,
    size_t slot_b,
    rg_feature_displacement **out,
    size_t *out_count
) {
    rg_feature_displacement *kept = 0;
    size_t kept_count = 0;
    size_t i;
    size_t k;
    rg_status status;

    *out = 0;
    *out_count = 0;
    if (index_count == 0) {
        return RG_OK;
    }
    /* The first member's displacement is the running intersection. */
    status = event_displacement(
        ctx, classes[indices[0]].graphemes[slot_a],
        classes[indices[0]].graphemes[slot_b], &kept, &kept_count);
    if (status != RG_OK || kept_count == 0) {
        return status;
    }

    /* Intersect with every subsequent member. */
    for (i = 1; i < index_count && kept_count > 0; i++) {
        rg_feature_displacement *other = 0;
        size_t other_count = 0;
        status = event_displacement(
            ctx, classes[indices[i]].graphemes[slot_a],
            classes[indices[i]].graphemes[slot_b], &other, &other_count);
        if (status != RG_OK) {
            rg_feature_displacement_free(kept, kept_count);
            return status;
        }
        /* Remove items from `kept` that are not in `other`. */
        for (k = 0; k < kept_count; ) {
            size_t j;
            int found = 0;
            for (j = 0; j < other_count; j++) {
                if (strcmp(kept[k].feature, other[j].feature) == 0 &&
                    strcmp(kept[k].from_value, other[j].from_value) == 0 &&
                    strcmp(kept[k].to_value, other[j].to_value) == 0) {
                    found = 1;
                    break;
                }
            }
            if (found) {
                k++;
            } else {
                rg_free_owned_internal(kept[k].feature);
                rg_free_owned_internal(kept[k].from_value);
                rg_free_owned_internal(kept[k].to_value);
                kept[k] = kept[kept_count - 1];
                kept_count--;
            }
        }
        rg_feature_displacement_free(other, other_count);
    }
    if (kept_count == 0) {
        free(kept);
        kept = 0;
    }
    *out = kept;
    *out_count = kept_count;
    return RG_OK;
}

/* The suprasegmentals every member of the event agrees on at this slot, or
 * NULL where they carry none or do not agree. `classes_share_displacement`
 * already requires agreement, so a disagreement here means the conditioned
 * path grouped rows whose suprasegmentals it does not read -- and the honest
 * answer for a member is then to say nothing. */
static rg_status member_suprasegmentals(
    const supra_scope *scope,
    const rg_multi_class_row *classes,
    const size_t *indices,
    size_t index_count,
    size_t slot,
    rg_suprasegmentals **out
) {
    rg_suprasegmentals agreed = slot_suprasegmentals(scope, &classes[indices[0]], slot);
    rg_suprasegmentals *kept;
    size_t i;
    *out = 0;
    for (i = 1; i < index_count; i++) {
        rg_suprasegmentals other = slot_suprasegmentals(scope, &classes[indices[i]], slot);
        if (!suprasegmentals_equal(&agreed, &other)) {
            return RG_OK;
        }
    }
    if (agreed.tone[0] == '\0' && agreed.length[0] == '\0' && agreed.stress[0] == '\0') {
        return RG_OK;
    }
    kept = (rg_suprasegmentals *)calloc(1, sizeof(*kept));
    if (kept == 0) {
        return RG_ERR_OOM;
    }
    kept->tone = rg_strdup_internal(agreed.tone);
    kept->length = rg_strdup_internal(agreed.length);
    kept->stress = rg_strdup_internal(agreed.stress);
    if (kept->tone == 0 || kept->length == 0 || kept->stress == 0) {
        rg_free_owned_internal(kept->tone);
        rg_free_owned_internal(kept->length);
        rg_free_owned_internal(kept->stress);
        free(kept);
        return RG_ERR_OOM;
    }
    *out = kept;
    return RG_OK;
}

/* One event out of the classes at `indices`. */
static rg_status build_event(
    const rg_context *ctx,
    const supra_scope *scope,
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
        /* The weakest member, not the first one. The members' scores are close
         * but they are not equal -- lenition groups three rows at margins 2.18,
         * 1.73 and 1.53 -- and publishing whichever happened to seed the group
         * states a number no member is answerable for. The floor is: every
         * member class clears at least this, which is what a reader asking how
         * far the grouping can be trusted is asking. A more negative
         * `delta_score` is the better one, so its floor is the maximum. */
        if (i == 0 || row->evidence.search_margin < out->search_margin) {
            out->search_margin = row->evidence.search_margin;
        }
        if (i == 0 || row->evidence.delta_score > out->delta_score) {
            out->delta_score = row->evidence.delta_score;
        }
        for (k = 0; k < row->supporting_cognate_count && status == RG_OK; k++) {
            status = string_array_add(&cognate_ids, &cognate_id_count,
                                      row->supporting_cognates[k]);
        }
    }
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
        {
            rg_suprasegmentals *supra = 0;
            status = member_suprasegmentals(scope, classes, indices, index_count,
                                            slot, &supra);
            if (status != RG_OK) {
                break;
            }
            members[slot].suprasegmentals = supra;
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

    /* The feature displacement shared by every member class between the
     * first two lect slots -- the rule the grouping implies. */
    if (status == RG_OK && first->segment_count >= 2) {
        rg_feature_displacement *disp = 0;
        size_t disp_count = 0;
        status = compute_shared_displacement(
            ctx, classes, indices, index_count, 0, 1, &disp, &disp_count);
        if (status == RG_OK) {
            out->shared_displacement = disp;
            out->shared_displacement_count = disp_count;
        }
    }

    if (status != RG_OK) {
        event_row_clear(out);
    }
    return status;
}

/* Whether the conditioned path already states this whole grouping.
 *
 * The unconditioned pass runs over the same corpus the conditioned one did, so
 * a change the search conditioned also sits in the unconditioned table as the
 * aggregate of its own observations. Publishing both would state one event
 * twice, once with its environment and once without.
 *
 * The test is on the group, not on its members. Applied per member -- which is
 * what it did until this was written -- one covered member is removed before
 * pairing, the rest fall below the two needed to group, and an event nothing
 * else states disappears without trace: compensatory lengthening lost the whole
 * `{iː,uː} ~ {i,u}` grouping because `uː~u` alone had been conditioned, and the
 * transcription-drift fixture lost twenty-two observations of vowel lengthening
 * the same way. A grouping is a duplicate only when every member of it is. */
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

static int group_is_already_conditioned(
    const rg_multi_class_row *cond_classes,
    size_t cond_count,
    const rg_multi_class_row *classes,
    const size_t *indices,
    size_t index_count
) {
    size_t i;
    for (i = 0; i < index_count; i++) {
        if (!has_conditioned_counterpart(cond_classes, cond_count, &classes[indices[i]])) {
            return 0;
        }
    }
    return index_count > 0;
}

/* Whether the members read as one regular change from some lect's side.
 *
 * Two conditions, on the same slot. Every member contributes a *different*
 * grapheme there, so the grouping states one outcome per input rather than a
 * fan; and each member is the leading correspondence for its own grapheme, so
 * what is grouped is what those segments usually do rather than a handful of
 * their rarer answers collected because the rare answers happen to lean the
 * same way.
 *
 * Distinctness alone is not enough, and its failure is instructive: on the
 * unrelated-wordlist corpus, alpha's `s` answers to six different omega
 * segments, and while that fans hopelessly from alpha it is a *function* read
 * from omega, where each of the six appears once. Every one-to-many pile is a
 * many-to-one merger read backwards. Regularity is what tells them apart --
 * none of those six is what omega's `l`, `m`, `n`, `p`, `t` or `w` mostly
 * answers to, whereas Grimm's `f~p`, `x~k` and `θ~t` are each the main answer
 * on both sides.
 *
 * Some lect, not every lect: a merger is a real event and satisfies this from
 * one side only. Grimm's second shift takes both `k` and `x` to `g`, and asking
 * it of both sides would refuse the merger along with the pile. Which side
 * happens to satisfy it is not a claim about direction -- see "computational
 * orientation" -- only about whether the grouping states anything. */
static int member_leads_its_outcome(
    const supra_scope *scope,
    const rg_multi_class_row *classes,
    size_t class_count,
    const rg_multi_class_row *member,
    size_t slot
) {
    size_t i;
    for (i = 0; i < class_count; i++) {
        const rg_multi_class_row *other = &classes[i];
        if (other == member || other->segment_count != member->segment_count) {
            continue;
        }
        if (strcmp(other->lect_ids[slot], member->lect_ids[slot]) != 0 ||
            !slot_outcomes_equal(scope, other, slot, member, slot)) {
            continue;
        }
        if (other->count > member->count) {
            return 0;
        }
    }
    return 1;
}

static int group_is_regular(
    const supra_scope *scope,
    const rg_multi_class_row *classes,
    size_t class_count,
    const size_t *indices,
    size_t index_count
) {
    size_t slot;
    if (index_count < 2) {
        return 1;
    }
    for (slot = 0; slot < classes[indices[0]].segment_count; slot++) {
        size_t a;
        int regular = 1;
        for (a = 0; a < index_count && regular; a++) {
            size_t b;
            regular = member_leads_its_outcome(scope, classes, class_count,
                                               &classes[indices[a]], slot);
            for (b = a + 1; b < index_count && regular; b++) {
                if (slot_outcomes_equal(scope, &classes[indices[a]], slot,
                                        &classes[indices[b]], slot)) {
                    regular = 0;
                }
            }
        }
        if (regular) {
            return 1;
        }
    }
    return 0;
}

/* Whether these classes, taken together, still state one displacement between
 * the first two slots. See the call site for why pairwise agreement does not
 * settle it. */
static int group_states_a_displacement(
    const rg_context *ctx,
    const supra_scope *scope,
    const rg_multi_class_row *classes,
    const size_t *indices,
    size_t index_count
) {
    rg_feature_displacement *shared = 0;
    size_t shared_count = 0;
    const rg_multi_class_row *first = &classes[indices[0]];
    if (first->segment_count < 2) {
        return 0;
    }
    /* A tone shift displaces no segmental feature -- `a[¹¹] ~ a[³³]` moves
     * nothing merkmal measures between `a` and `a` -- so the segmental
     * intersection is empty and says nothing about whether the grouping states
     * a change. The suprasegmental difference is the change, and every member
     * carries the same one by the time it is a member. */
    {
        rg_suprasegmentals a = slot_suprasegmentals(scope, first, 0);
        rg_suprasegmentals b = slot_suprasegmentals(scope, first, 1);
        if (!suprasegmentals_equal(&a, &b)) {
            return 1;
        }
    }
    if (compute_shared_displacement(ctx, classes, indices, index_count, 0, 1,
                                    &shared, &shared_count) != RG_OK) {
        return 0;
    }
    rg_feature_displacement_free(shared, shared_count);
    return shared_count > 0;
}

/* The three shapes a grouping can take.
 *
 * A change split per segment states one environment over several outcomes;
 * a change stated over a disjunction states one outcome over several
 * environments; an unconditioned change has no environment and is held
 * together by what it displaces. They are the same claim -- these rows are
 * one change -- read along different axes, and each needs its own predicate. */
typedef enum group_mode {
    /* Same environment, differing outcomes: the natural-class shape. */
    GROUP_BY_ENVIRONMENT = 0,
    /* Same outcome, differing environments: the decision-list shape. RUKI is
     * four rules with one outcome, and `graded_7_disjunction` is built to be
     * exactly this -- four conditioned rows, all of them `f ~ p`, under four
     * environments no single predicate covers. Every one of them is one change,
     * and the table could not say so because the environment predicate
     * *required* the outcomes to differ. */
    GROUP_BY_OUTCOME = 1,
    /* No environment: held together by a shared feature displacement. */
    GROUP_BY_DISPLACEMENT = 2
} group_mode;

/* Two conditioned classes state one outcome under different environments when
 * every slot agrees on the outcome and some slot disagrees on the environment.
 * The mirror of classes_share_an_environment, and deliberately not its
 * complement: both require the same lects in the same order. */
static int classes_share_an_outcome(
    const supra_scope *scope,
    const rg_multi_class_row *a,
    const rg_multi_class_row *b
) {
    size_t i;
    int environments_differ = 0;
    if (a->segment_count != b->segment_count || a->segment_count == 0) {
        return 0;
    }
    for (i = 0; i < a->segment_count; i++) {
        if (strcmp(a->lect_ids[i], b->lect_ids[i]) != 0) {
            return 0;
        }
        if (!slot_outcomes_equal(scope, a, i, b, i)) {
            return 0;
        }
        if (rg_context_spec_compare_internal(&a->contexts[i], &b->contexts[i]) != 0) {
            environments_differ = 1;
        }
    }
    return environments_differ;
}

static rg_status group_classes(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_multi_class_row *classes,
    size_t class_count,
    group_mode mode,
    const rg_multi_class_row *exclude_if_in,
    size_t exclude_count,
    rg_proposed_event_row **rows,
    size_t *row_count
) {
    char *grouped = 0;
    size_t *indices = 0;
    supra_scope scope;
    size_t i;
    rg_status status = RG_OK;

    if (class_count < 2) {
        return RG_OK;
    }
    status = supra_scope_build(classes, class_count, &scope);
    if (status != RG_OK) {
        return status;
    }
    grouped = (char *)calloc(class_count, 1);
    indices = (size_t *)calloc(class_count, sizeof(*indices));
    if (grouped == 0 || indices == 0) {
        supra_scope_free(&scope);
        free(grouped);
        free(indices);
        return RG_ERR_OOM;
    }
    for (i = 0; i < class_count; i++) {
        if (class_is_identity(&scope, &classes[i])) {
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
            size_t m;
            int agrees = 1;
            if (grouped[j]) {
                continue;
            }
            /* Against every member so far, not against the seed alone.
             *
             * An event asserts that all of its members are one change, which is
             * a claim about every pair of them; testing only the seed asserts
             * it of a star. The two agree exactly while the predicate is an
             * equivalence relation, which equality of environments and equality
             * of displacements both are -- so this changes nothing today and is
             * the difference between a group and a chain the moment either is
             * loosened. Chained on a four-lect Romance model, seed-only
             * grouping joins `v|v|v|b` to `ʃ|k|k|k` to `a|e|e|e` because each
             * agreed with the seed on something and with each other on
             * nothing. */
            for (m = 0; m < member_count && agrees; m++) {
                const rg_multi_class_row *held = &classes[indices[m]];
                switch (mode) {
                case GROUP_BY_DISPLACEMENT:
                    agrees = classes_share_displacement(ctx, &scope, held, &classes[j]);
                    break;
                case GROUP_BY_OUTCOME:
                    agrees = classes_share_an_outcome(&scope, held, &classes[j]);
                    break;
                default:
                    agrees = classes_share_an_environment(&scope, held, &classes[j]);
                    break;
                }
            }
            /* Pairwise agreement is not group agreement. Sharing a displacement
             * is not transitive: p~f shares stop→fricative with t~θ, and t~θ
             * shares alveolar→dental with t~s, and the three together share
             * nothing. So the candidate has to keep the *group's* displacement
             * -- the intersection over every member, which is the rule the
             * event publishes -- from going empty. Without this the chance
             * corpus proposes a fourteen-class event whose shared displacement
             * is the empty set: an event asserting nothing in particular. */
            if (agrees && mode != GROUP_BY_OUTCOME) {
                /* Both of these ask what the varying outcomes have in common,
                 * so neither has anything to ask of a grouping whose outcome is
                 * the one thing that does not vary. */
                indices[member_count] = j;
                agrees = group_is_regular(&scope, classes, class_count, indices, member_count + 1);
                if (agrees && mode == GROUP_BY_DISPLACEMENT) {
                    agrees = group_states_a_displacement(ctx, &scope, classes, indices,
                                                         member_count + 1);
                }
            }
            if (agrees) {
                grouped[j] = 1;
                indices[member_count++] = j;
            }
        }
        if (member_count < 2) {
            continue;
        }
        grouped[i] = 1;
        if (exclude_if_in != 0 &&
            group_is_already_conditioned(exclude_if_in, exclude_count,
                                         classes, indices, member_count)) {
            continue;
        }
        next = (rg_proposed_event_row *)realloc(*rows, (*row_count + 1) * sizeof(*next));
        if (next == 0) {
            status = RG_ERR_OOM;
            break;
        }
        *rows = next;
        status = build_event(ctx, &scope, cognates, cognate_count, classes,
                             indices, member_count, &(*rows)[*row_count]);
        if (status == RG_OK) {
            (*row_count)++;
        }
    }
    supra_scope_free(&scope);
    free(grouped);
    free(indices);
    return status;
}

/* Heaviest first, and the pooled count is the weight.
 *
 * The three passes run in the order the axes were written, which is an
 * implementation detail of this file and was the published order of the table:
 * a reader working down the pane met the conditioned groupings first and the
 * displacement groupings last however much or little was behind them. On the
 * denser corpora that puts eleven-observation events above sixty-observation
 * ones, and the first row of a table is read as its strongest claim.
 *
 * `count` is the aligned positions across every member class -- the number no
 * single member row carries, and the one the grouping exists to state. It is
 * the same key the class tables are ordered by, so the two panes read the same
 * way down the page.
 *
 * Ties break on the first member's class id, which is arbitrary but fixed:
 * without it qsort is free to permute equal rows and the model's JSON would
 * stop being reproducible. */
static int event_rank_cmp(const void *a, const void *b) {
    const rg_proposed_event_row *ea = (const rg_proposed_event_row *)a;
    const rg_proposed_event_row *eb = (const rg_proposed_event_row *)b;
    if (ea->count > eb->count) {
        return -1;
    }
    if (ea->count < eb->count) {
        return 1;
    }
    if (ea->class_id_count > 0 && eb->class_id_count > 0 &&
        ea->class_ids[0] != eb->class_ids[0]) {
        return ea->class_ids[0] < eb->class_ids[0] ? -1 : 1;
    }
    return 0;
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
                           GROUP_BY_ENVIRONMENT, 0, 0, &rows, &row_count);
    if (status == RG_OK) {
        status = group_classes(ctx, cognates, cognate_count,
                               model->conditioned_classes,
                               model->conditioned_class_count,
                               GROUP_BY_OUTCOME, 0, 0, &rows, &row_count);
    }
    if (status == RG_OK) {
        status = group_classes(ctx, cognates, cognate_count,
                               model->unconditioned_classes,
                               model->unconditioned_class_count,
                               GROUP_BY_DISPLACEMENT,
                               model->conditioned_classes,
                               model->conditioned_class_count,
                               &rows, &row_count);
    }
    if (status != RG_OK) {
        rg_proposed_events_free_internal(rows, row_count);
        return status;
    }
    if (row_count > 1) {
        qsort(rows, row_count, sizeof(*rows), event_rank_cmp);
    }
    model->proposed_events = rows;
    model->proposed_event_count = row_count;
    return RG_OK;
}
