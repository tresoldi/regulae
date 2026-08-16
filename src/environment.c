#include "environment.h"

#include "internal.h"

#include <stdlib.h>
#include <string.h>

static int string_absent(const char *value) {
    return value == 0 || value[0] == '\0';
}

static int string_equal(const char *a, const char *b) {
    if (string_absent(a) && string_absent(b)) {
        return 1;
    }
    if (a == 0 || b == 0) {
        return 0;
    }
    return strcmp(a, b) == 0;
}

static int feature_constraint_equal(rg_feature_constraint a, rg_feature_constraint b) {
    return string_equal(a.feature, b.feature) && string_equal(a.value, b.value);
}

static int contains_feature_constraint(
    const rg_feature_constraint *items,
    size_t count,
    rg_feature_constraint needle
) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (feature_constraint_equal(items[i], needle)) {
            return 1;
        }
    }
    return 0;
}

static int distance_constraint_equal(rg_distance_constraint a, rg_distance_constraint b) {
    return a.offset == b.offset && feature_constraint_equal(a.constraint, b.constraint);
}

static int contains_distance_constraint(
    const rg_distance_constraint *items,
    size_t count,
    rg_distance_constraint needle
) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (distance_constraint_equal(items[i], needle)) {
            return 1;
        }
    }
    return 0;
}

static int feature_slice_is_subset(
    const rg_feature_constraint *subset,
    size_t subset_count,
    const rg_feature_constraint *other,
    size_t other_count
) {
    size_t i;
    for (i = 0; i < subset_count; i++) {
        if (!contains_feature_constraint(other, other_count, subset[i])) {
            return 0;
        }
    }
    return 1;
}

static int distance_slice_is_subset(
    const rg_distance_constraint *subset,
    size_t subset_count,
    const rg_distance_constraint *other,
    size_t other_count
) {
    size_t i;
    for (i = 0; i < subset_count; i++) {
        if (!contains_distance_constraint(other, other_count, subset[i])) {
            return 0;
        }
    }
    return 1;
}

void rg_context_spec_init_empty(rg_context_spec *context) {
    if (context == 0) {
        return;
    }
    memset(context, 0, sizeof(*context));
}

size_t rg_context_spec_constraint_count(const rg_context_spec *context) {
    size_t count = 0;
    if (context == 0) {
        return 0;
    }
#define COUNT_STRING(name, key) \
    if (!string_absent(context->name)) { count++; }
    RG_ENV_STRING_SLOTS(COUNT_STRING)
#undef COUNT_STRING
#define COUNT_SLOT(name, label, key) count += context->name##_count;
    RG_ENV_SLOTS(COUNT_SLOT, COUNT_SLOT)
#undef COUNT_SLOT
    return count;
}

rg_status rg_context_spec_is_subset(
    const rg_context_spec *subset,
    const rg_context_spec *other,
    bool *out
) {
    if (subset == 0 || other == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
#define SUBSET_STRING(name, key)                                                     \
    if (!string_absent(subset->name) && !string_equal(subset->name, other->name)) { \
        return RG_OK;                                                           \
    }
    RG_ENV_STRING_SLOTS(SUBSET_STRING)
#undef SUBSET_STRING
#define SUBSET_FEATURES(name, label, key)                                            \
    if (!feature_slice_is_subset(subset->name, subset->name##_count,            \
                                 other->name, other->name##_count)) {           \
        return RG_OK;                                                           \
    }
#define SUBSET_DISTANCES(name, label, key)                                           \
    if (!distance_slice_is_subset(subset->name, subset->name##_count,           \
                                  other->name, other->name##_count)) {          \
        return RG_OK;                                                           \
    }
    RG_ENV_SLOTS(SUBSET_FEATURES, SUBSET_DISTANCES)
#undef SUBSET_FEATURES
#undef SUBSET_DISTANCES
    *out = 1;
    return RG_OK;
}

static int nullable_strcmp(const char *a, const char *b) {
    return strcmp(a == 0 ? "" : a, b == 0 ? "" : b);
}

static int constraint_list_cmp(
    const rg_feature_constraint *a, size_t a_count,
    const rg_feature_constraint *b, size_t b_count
) {
    size_t i;
    if (a_count != b_count) {
        return a_count < b_count ? -1 : 1;
    }
    for (i = 0; i < a_count; i++) {
        int c = nullable_strcmp(a[i].feature, b[i].feature);
        if (c != 0) {
            return c;
        }
        c = nullable_strcmp(a[i].value, b[i].value);
        if (c != 0) {
            return c;
        }
    }
    return 0;
}

static int distance_list_cmp(
    const rg_distance_constraint *a, size_t a_count,
    const rg_distance_constraint *b, size_t b_count
) {
    size_t i;
    if (a_count != b_count) {
        return a_count < b_count ? -1 : 1;
    }
    for (i = 0; i < a_count; i++) {
        int c;
        if (a[i].offset != b[i].offset) {
            return a[i].offset < b[i].offset ? -1 : 1;
        }
        c = nullable_strcmp(a[i].constraint.feature, b[i].constraint.feature);
        if (c != 0) {
            return c;
        }
        c = nullable_strcmp(a[i].constraint.value, b[i].constraint.value);
        if (c != 0) {
            return c;
        }
    }
    return 0;
}

int rg_context_spec_compare_internal(const rg_context_spec *a, const rg_context_spec *b) {
    /* The three string slots, in the order that is the total order over them.
     * It is not RG_ENV_STRING_SLOTS' order, which is the order the reports
     * print; written out here because this is where it is decided. */
    int c = nullable_strcmp(a->position, b->position);
    if (c != 0) {
        return c;
    }
    c = nullable_strcmp(a->morphological, b->morphological);
    if (c != 0) {
        return c;
    }
    c = nullable_strcmp(a->morpheme_index, b->morpheme_index);
    if (c != 0) {
        return c;
    }
#define CMP_FEATURES(name, label, key)                                               \
    c = constraint_list_cmp(a->name, a->name##_count, b->name, b->name##_count); \
    if (c != 0) {                                                               \
        return c;                                                               \
    }
#define CMP_DISTANCES(name, label, key)                                              \
    c = distance_list_cmp(a->name, a->name##_count, b->name, b->name##_count);  \
    if (c != 0) {                                                               \
        return c;                                                               \
    }
    RG_ENV_SLOTS(CMP_FEATURES, CMP_DISTANCES)
#undef CMP_FEATURES
#undef CMP_DISTANCES
    return 0;
}

static rg_status distance_constraint_array_copy(
    const rg_distance_constraint *src,
    size_t count,
    const rg_distance_constraint **out
) {
    rg_distance_constraint *copy;
    rg_status status;
    size_t i;
    *out = 0;
    if (count == 0) {
        return RG_OK;
    }
    copy = (rg_distance_constraint *)calloc(count, sizeof(*copy));
    if (copy == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < count; i++) {
        copy[i].offset = src[i].offset;
        status = rg_feature_constraint_copy_internal(&src[i].constraint, &copy[i].constraint);
        if (status != RG_OK) {
            while (i > 0) {
                i--;
                rg_free_owned_internal(copy[i].constraint.feature);
                rg_free_owned_internal(copy[i].constraint.value);
            }
            free(copy);
            return status;
        }
    }
    *out = copy;
    return RG_OK;
}

rg_status rg_context_spec_copy_internal(const rg_context_spec *src, rg_context_spec *out) {
    rg_status status;
    if (src == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    rg_context_spec_init_empty(out);
#define COPY_STRING(name, key)                                                       \
    if (src->name != 0) {                                                       \
        out->name = rg_strdup_internal(src->name);                              \
        if (out->name == 0) {                                                   \
            rg_context_spec_clear_internal(out);                                \
            return RG_ERR_OOM;                                                  \
        }                                                                       \
    }
    RG_ENV_STRING_SLOTS(COPY_STRING)
#undef COPY_STRING
#define COPY_FEATURES(name, label, key)                                              \
    status = rg_feature_constraint_array_copy_internal(src->name, src->name##_count, &out->name); \
    if (status != RG_OK) {                                                      \
        rg_context_spec_clear_internal(out);                                    \
        return status;                                                          \
    }                                                                           \
    out->name##_count = src->name##_count;
#define COPY_DISTANCES(name, label, key)                                             \
    status = distance_constraint_array_copy(src->name, src->name##_count, &out->name); \
    if (status != RG_OK) {                                                      \
        rg_context_spec_clear_internal(out);                                    \
        return status;                                                          \
    }                                                                           \
    out->name##_count = src->name##_count;
    RG_ENV_SLOTS(COPY_FEATURES, COPY_DISTANCES)
#undef COPY_FEATURES
#undef COPY_DISTANCES
    return RG_OK;
}

void rg_context_spec_clear_internal(rg_context_spec *context) {
    if (context == 0) {
        return;
    }
#define CLEAR_STRING(name, key) rg_free_owned_internal(context->name);
    RG_ENV_STRING_SLOTS(CLEAR_STRING)
#undef CLEAR_STRING
#define CLEAR_FEATURES(name, label, key) \
    rg_feature_constraint_array_clear_internal(context->name, context->name##_count);
#define CLEAR_DISTANCES(name, label, key) \
    rg_distance_constraint_array_clear_internal(context->name, context->name##_count);
    RG_ENV_SLOTS(CLEAR_FEATURES, CLEAR_DISTANCES)
#undef CLEAR_FEATURES
#undef CLEAR_DISTANCES
    rg_context_spec_init_empty(context);
}

/* ---- The candidate slot vocabulary -------------------------------------
 *
 * A candidate names its slot as a string. Three of the eighteen slots hold a
 * single string and are matched on the candidate's `feature` alone; eleven hold
 * a feature-constraint list and are matched on `feature` and `value`; the two
 * distance slots are named with the offset attached -- "preceding@2",
 * "following@3" -- because one field holds constraints at several distances and
 * the name has to say which. That is the whole vocabulary, and these three
 * functions are the only readers of it: one asks whether a candidate holds of
 * an environment, one builds an environment from a candidate, one conjoins a
 * candidate onto an environment. They agreed by inspection across forty-five
 * strcmp branches in three chains until they were expanded from one list.
 */

#define RG_ENV_DISTANCE_PREFIX_LEN 10

/* Splits "preceding@2" into the field it names and the offset it carries.
 * Returns 0 when the slot is not a distance slot. */
static int distance_slot(
    const char *slot,
    const rg_context_spec *context,
    const rg_distance_constraint **items,
    size_t *count,
    int *offset,
    int *is_preceding
) {
    if (strncmp(slot, "preceding@", RG_ENV_DISTANCE_PREFIX_LEN) == 0) {
        *is_preceding = 1;
        *items = context == 0 ? 0 : context->preceding_at_distance;
        *count = context == 0 ? 0 : context->preceding_at_distance_count;
    } else if (strncmp(slot, "following@", RG_ENV_DISTANCE_PREFIX_LEN) == 0) {
        *is_preceding = 0;
        *items = context == 0 ? 0 : context->following_at_distance;
        *count = context == 0 ? 0 : context->following_at_distance_count;
    } else {
        return 0;
    }
    *offset = atoi(slot + RG_ENV_DISTANCE_PREFIX_LEN);
    return 1;
}

static int context_has_constraint(
    const rg_feature_constraint *items,
    size_t count,
    const char *feature,
    const char *value
) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (strcmp(items[i].feature, feature) == 0 && strcmp(items[i].value, value) == 0) {
            return 1;
        }
    }
    return 0;
}

static int context_has_distance_constraint(
    const rg_distance_constraint *items,
    size_t count,
    int offset,
    const char *feature,
    const char *value
) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (items[i].offset == offset &&
            strcmp(items[i].constraint.feature, feature) == 0 &&
            strcmp(items[i].constraint.value, value) == 0) {
            return 1;
        }
    }
    return 0;
}

int rg_predicate_holds_internal(const rg_context_spec *context, const rg_split_candidate *candidate) {
    const rg_distance_constraint *items;
    size_t count;
    int offset;
    int is_preceding;
#define HOLDS_STRING(name, key)                                                      \
    if (strcmp(candidate->slot, #name) == 0) {                                  \
        return context->name != 0 && strcmp(context->name, candidate->feature) == 0; \
    }
    RG_ENV_STRING_SLOTS(HOLDS_STRING)
#undef HOLDS_STRING
#define HOLDS_FEATURES(name, label, key)                                             \
    if (strcmp(candidate->slot, #name) == 0) {                                  \
        return context_has_constraint(context->name, context->name##_count,     \
                                      candidate->feature, candidate->value);    \
    }
    RG_ENV_FEATURE_SLOTS(HOLDS_FEATURES)
#undef HOLDS_FEATURES
    if (distance_slot(candidate->slot, context, &items, &count, &offset, &is_preceding)) {
        return context_has_distance_constraint(items, count, offset,
                                               candidate->feature, candidate->value);
    }
    return 0;
}

static rg_status distance_singleton(
    int offset,
    const rg_feature_constraint *constraint,
    const rg_distance_constraint **out,
    size_t *out_count
) {
    rg_distance_constraint *items = (rg_distance_constraint *)calloc(1, sizeof(*items));
    rg_status status;
    if (items == 0) {
        return RG_ERR_OOM;
    }
    items[0].offset = offset;
    status = rg_feature_constraint_copy_internal(constraint, &items[0].constraint);
    if (status != RG_OK) {
        free(items);
        return status;
    }
    *out = items;
    *out_count = 1;
    return RG_OK;
}

rg_status rg_context_from_candidate_internal(const rg_split_candidate *candidate, rg_context_spec *out) {
    rg_feature_constraint constraint;
    const rg_distance_constraint *unused_items;
    size_t unused_count;
    int offset;
    int is_preceding;
    rg_status status;

    rg_context_spec_init_empty(out);
#define FROM_STRING(name, key)                                                       \
    if (strcmp(candidate->slot, #name) == 0) {                                  \
        out->name = rg_strdup_internal(candidate->feature);                     \
        return out->name == 0 ? RG_ERR_OOM : RG_OK;                             \
    }
    RG_ENV_STRING_SLOTS(FROM_STRING)
#undef FROM_STRING
    constraint.feature = candidate->feature;
    constraint.value = candidate->value;
#define FROM_FEATURES(name, label, key)                                              \
    if (strcmp(candidate->slot, #name) == 0) {                                  \
        status = rg_feature_constraint_array_copy_internal(&constraint, 1, &out->name); \
        if (status == RG_OK) {                                                  \
            out->name##_count = 1;                                              \
        }                                                                       \
        return status;                                                          \
    }
    RG_ENV_FEATURE_SLOTS(FROM_FEATURES)
#undef FROM_FEATURES
    if (distance_slot(candidate->slot, 0, &unused_items, &unused_count, &offset, &is_preceding)) {
        return is_preceding
            ? distance_singleton(offset, &constraint,
                                 &out->preceding_at_distance, &out->preceding_at_distance_count)
            : distance_singleton(offset, &constraint,
                                 &out->following_at_distance, &out->following_at_distance_count);
    }
    return RG_ERR_INVALID_ARGUMENT;
}

/* base_context with one more predicate conjoined. Environments are immutable by
 * convention, so this always allocates a fresh value. The multi-lect stage needs
 * the same operation the pairwise refinement does, and an environment built from
 * two predicates is one environment, not two rules. */
rg_status rg_context_extend_internal(
    const rg_context_spec *base_context,
    const rg_split_candidate *candidate,
    rg_context_spec *out
) {
    rg_feature_constraint *merged = 0;
    const rg_feature_constraint **slot = 0;
    size_t *slot_count = 0;
    const rg_feature_constraint *existing = 0;
    size_t existing_count = 0;
    rg_feature_constraint addition;
    rg_status status;

    status = rg_context_spec_copy_internal(base_context, out);
    if (status != RG_OK) {
        return status;
    }
    /* A string slot is replaced rather than conjoined: a position is one
     * position. */
#define EXTEND_STRING(name, key)                                                     \
    if (strcmp(candidate->slot, #name) == 0) {                                  \
        rg_free_owned_internal(out->name);                                      \
        out->name = rg_strdup_internal(candidate->feature);                     \
        if (out->name == 0) {                                                   \
            rg_context_spec_clear_internal(out);                                \
            return RG_ERR_OOM;                                                  \
        }                                                                       \
        return RG_OK;                                                           \
    }
    RG_ENV_STRING_SLOTS(EXTEND_STRING)
#undef EXTEND_STRING
    addition.feature = candidate->feature;
    addition.value = candidate->value;
#define PICK(name, label, key)                                     \
    if (strcmp(candidate->slot, #name) == 0) {                \
        slot = &out->name;                                    \
        slot_count = &out->name##_count;                      \
        existing = out->name;                                 \
        existing_count = out->name##_count;                   \
    }
    RG_ENV_FEATURE_SLOTS(PICK)
#undef PICK

    if (slot == 0) {
        rg_distance_constraint *items;
        const rg_distance_constraint *base_items;
        size_t base_count;
        size_t i;
        int offset;
        int is_preceding;
        if (!distance_slot(candidate->slot, out, &base_items, &base_count, &offset, &is_preceding)) {
            rg_context_spec_clear_internal(out);
            return RG_ERR_INVALID_ARGUMENT;
        }
        items = (rg_distance_constraint *)calloc(base_count + 1, sizeof(*items));
        if (items == 0) {
            rg_context_spec_clear_internal(out);
            return RG_ERR_OOM;
        }
        for (i = 0; i < base_count; i++) {
            items[i].offset = base_items[i].offset;
            if (rg_feature_constraint_copy_internal(&base_items[i].constraint, &items[i].constraint) != RG_OK) {
                rg_distance_constraint_array_clear_internal(items, i);
                rg_context_spec_clear_internal(out);
                return RG_ERR_OOM;
            }
        }
        items[base_count].offset = offset;
        if (rg_feature_constraint_copy_internal(&addition, &items[base_count].constraint) != RG_OK) {
            rg_distance_constraint_array_clear_internal(items, base_count);
            rg_context_spec_clear_internal(out);
            return RG_ERR_OOM;
        }
        if (is_preceding) {
            rg_distance_constraint_array_clear_internal(rg_owned_internal(out->preceding_at_distance), out->preceding_at_distance_count);
            out->preceding_at_distance = items;
            out->preceding_at_distance_count = base_count + 1;
        } else {
            rg_distance_constraint_array_clear_internal(rg_owned_internal(out->following_at_distance), out->following_at_distance_count);
            out->following_at_distance = items;
            out->following_at_distance_count = base_count + 1;
        }
        return RG_OK;
    }

    merged = (rg_feature_constraint *)calloc(existing_count + 1, sizeof(*merged));
    if (merged == 0) {
        rg_context_spec_clear_internal(out);
        return RG_ERR_OOM;
    }
    {
        size_t i;
        for (i = 0; i < existing_count; i++) {
            if (rg_feature_constraint_copy_internal(&existing[i], &merged[i]) != RG_OK) {
                rg_feature_constraint_array_clear_internal(merged, i);
                rg_context_spec_clear_internal(out);
                return RG_ERR_OOM;
            }
        }
        if (rg_feature_constraint_copy_internal(&addition, &merged[existing_count]) != RG_OK) {
            rg_feature_constraint_array_clear_internal(merged, existing_count);
            rg_context_spec_clear_internal(out);
            return RG_ERR_OOM;
        }
    }
    rg_feature_constraint_array_clear_internal(existing, existing_count);
    *slot = merged;
    *slot_count = existing_count + 1;
    return RG_OK;
}

const char *const rg_env_long_range_slots[] = {
    "same_syllable",
    "next_syllable",
    "previous_syllable",
    "preceding@2",
    "preceding@3",
    "following@2",
    "following@3",
    "somewhere_preceding",
    "somewhere_following"
};
const size_t rg_env_long_range_slot_count =
    sizeof(rg_env_long_range_slots) / sizeof(rg_env_long_range_slots[0]);

const char *const rg_env_stress_slots[] = {
    "self_stress",
    "preceding_stress",
    "following_stress"
};
const size_t rg_env_stress_slot_count =
    sizeof(rg_env_stress_slots) / sizeof(rg_env_stress_slots[0]);

/* The three slots that hold a whole syllable, and so can be asked about the
 * syllable's own shape and weight rather than about a segment in it. The same
 * three appear in the long-range list, where they carry the segment
 * predicates; a slot belongs to both families because the two ask different
 * questions of it. */
const char *const rg_env_syllable_slots[] = {
    "same_syllable",
    "next_syllable",
    "previous_syllable"
};
const size_t rg_env_syllable_slot_count =
    sizeof(rg_env_syllable_slots) / sizeof(rg_env_syllable_slots[0]);

rg_status rg_env_collect_stress_values(
    const rg_context_spec *context,
    rg_status (*add)(void *user, const char *value),
    void *user
) {
    const rg_feature_constraint *slots[3];
    size_t counts[3];
    size_t s;
    slots[0] = context->self_stress;
    counts[0] = context->self_stress_count;
    slots[1] = context->preceding_stress;
    counts[1] = context->preceding_stress_count;
    slots[2] = context->following_stress;
    counts[2] = context->following_stress_count;
    for (s = 0; s < 3; s++) {
        size_t i;
        for (i = 0; i < counts[s]; i++) {
            if (slots[s][i].feature != 0 && strcmp(slots[s][i].feature, "stress") == 0 &&
                slots[s][i].value != 0) {
                rg_status status = add(user, slots[s][i].value);
                if (status != RG_OK) {
                    return status;
                }
            }
        }
    }
    return RG_OK;
}
