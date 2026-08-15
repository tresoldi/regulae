#include "internal.h"

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
    if (!string_absent(context->position)) {
        count++;
    }
    if (!string_absent(context->morpheme_index)) {
        count++;
    }
    if (!string_absent(context->morphological)) {
        count++;
    }
    count += context->preceding_count;
    count += context->following_count;
    count += context->preceding_at_distance_count;
    count += context->following_at_distance_count;
    count += context->somewhere_preceding_count;
    count += context->somewhere_following_count;
    count += context->same_syllable_count;
    count += context->next_syllable_count;
    count += context->previous_syllable_count;
    count += context->self_count;
    count += context->self_stress_count;
    count += context->preceding_stress_count;
    count += context->following_stress_count;
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
    if (!string_absent(subset->position) && !string_equal(subset->position, other->position)) {
        return RG_OK;
    }
    if (!string_absent(subset->morpheme_index) && !string_equal(subset->morpheme_index, other->morpheme_index)) {
        *out = 0;
        return RG_OK;
    }
    if (!string_absent(subset->morphological) && !string_equal(subset->morphological, other->morphological)) {
        return RG_OK;
    }
    if (!feature_slice_is_subset(subset->preceding, subset->preceding_count, other->preceding, other->preceding_count)) {
        return RG_OK;
    }
    if (!feature_slice_is_subset(subset->following, subset->following_count, other->following, other->following_count)) {
        return RG_OK;
    }
    if (!distance_slice_is_subset(subset->preceding_at_distance, subset->preceding_at_distance_count, other->preceding_at_distance, other->preceding_at_distance_count)) {
        return RG_OK;
    }
    if (!distance_slice_is_subset(subset->following_at_distance, subset->following_at_distance_count, other->following_at_distance, other->following_at_distance_count)) {
        return RG_OK;
    }
    if (!feature_slice_is_subset(subset->somewhere_preceding, subset->somewhere_preceding_count, other->somewhere_preceding, other->somewhere_preceding_count)) {
        return RG_OK;
    }
    if (!feature_slice_is_subset(subset->somewhere_following, subset->somewhere_following_count, other->somewhere_following, other->somewhere_following_count)) {
        return RG_OK;
    }
    if (!feature_slice_is_subset(subset->same_syllable, subset->same_syllable_count, other->same_syllable, other->same_syllable_count)) {
        return RG_OK;
    }
    if (!feature_slice_is_subset(subset->next_syllable, subset->next_syllable_count, other->next_syllable, other->next_syllable_count)) {
        return RG_OK;
    }
    if (!feature_slice_is_subset(subset->previous_syllable, subset->previous_syllable_count, other->previous_syllable, other->previous_syllable_count)) {
        return RG_OK;
    }
    if (!feature_slice_is_subset(subset->self, subset->self_count, other->self, other->self_count)) {
        *out = 0;
        return RG_OK;
    }
    if (!feature_slice_is_subset(subset->self_stress, subset->self_stress_count, other->self_stress, other->self_stress_count)) {
        return RG_OK;
    }
    if (!feature_slice_is_subset(subset->preceding_stress, subset->preceding_stress_count, other->preceding_stress, other->preceding_stress_count)) {
        return RG_OK;
    }
    if (!feature_slice_is_subset(subset->following_stress, subset->following_stress_count, other->following_stress, other->following_stress_count)) {
        return RG_OK;
    }
    *out = 1;
    return RG_OK;
}

int rg_link_is_reordering_internal(
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    size_t *pairing
) {
    size_t taken[RG_MAX_REORDER_SPAN];
    size_t i;
    int moved = 0;

    if (source_count != target_count || source_count < 2 || source_count > RG_MAX_REORDER_SPAN) {
        return 0;
    }
    for (i = 0; i < source_count; i++) {
        taken[i] = 0;
    }
    /* Greedy left-to-right matching on the grapheme. Ambiguity only arises
     * when a grapheme repeats, and then any consistent choice describes the
     * same reordering. */
    for (i = 0; i < source_count; i++) {
        size_t j;
        int found = 0;
        for (j = 0; j < target_count; j++) {
            if (taken[j]) {
                continue;
            }
            if (strcmp(source[i].grapheme == 0 ? "" : source[i].grapheme,
                       target[j].grapheme == 0 ? "" : target[j].grapheme) == 0) {
                taken[j] = 1;
                pairing[i] = j;
                if (j != i) {
                    moved = 1;
                }
                found = 1;
                break;
            }
        }
        if (!found) {
            return 0;
        }
    }
    return moved;
}
