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
#define COUNT_STRING(name) \
    if (!string_absent(context->name)) { count++; }
    RG_ENV_STRING_SLOTS(COUNT_STRING)
#undef COUNT_STRING
#define COUNT_SLOT(name, label) count += context->name##_count;
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
#define SUBSET_STRING(name)                                                     \
    if (!string_absent(subset->name) && !string_equal(subset->name, other->name)) { \
        return RG_OK;                                                           \
    }
    RG_ENV_STRING_SLOTS(SUBSET_STRING)
#undef SUBSET_STRING
#define SUBSET_FEATURES(name, label)                                            \
    if (!feature_slice_is_subset(subset->name, subset->name##_count,            \
                                 other->name, other->name##_count)) {           \
        return RG_OK;                                                           \
    }
#define SUBSET_DISTANCES(name, label)                                           \
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
#define CMP_FEATURES(name, label)                                               \
    c = constraint_list_cmp(a->name, a->name##_count, b->name, b->name##_count); \
    if (c != 0) {                                                               \
        return c;                                                               \
    }
#define CMP_DISTANCES(name, label)                                              \
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
#define COPY_STRING(name)                                                       \
    if (src->name != 0) {                                                       \
        out->name = rg_strdup_internal(src->name);                              \
        if (out->name == 0) {                                                   \
            rg_context_spec_clear_internal(out);                                \
            return RG_ERR_OOM;                                                  \
        }                                                                       \
    }
    RG_ENV_STRING_SLOTS(COPY_STRING)
#undef COPY_STRING
#define COPY_FEATURES(name, label)                                              \
    status = rg_feature_constraint_array_copy_internal(src->name, src->name##_count, &out->name); \
    if (status != RG_OK) {                                                      \
        rg_context_spec_clear_internal(out);                                    \
        return status;                                                          \
    }                                                                           \
    out->name##_count = src->name##_count;
#define COPY_DISTANCES(name, label)                                             \
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
#define CLEAR_STRING(name) rg_free_owned_internal(context->name);
    RG_ENV_STRING_SLOTS(CLEAR_STRING)
#undef CLEAR_STRING
#define CLEAR_FEATURES(name, label) \
    rg_feature_constraint_array_clear_internal(context->name, context->name##_count);
#define CLEAR_DISTANCES(name, label) \
    rg_distance_constraint_array_clear_internal(context->name, context->name##_count);
    RG_ENV_SLOTS(CLEAR_FEATURES, CLEAR_DISTANCES)
#undef CLEAR_FEATURES
#undef CLEAR_DISTANCES
    rg_context_spec_init_empty(context);
}
