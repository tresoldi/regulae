#include "internal.h"

#include <stdlib.h>
#include <string.h>

/* Releases storage a public struct advertises as `const` but the library owns.
 *
 * The `const` on `rg_segment.grapheme`, on a feature name, on the arrays behind
 * an `rg_context_spec` is a statement to the caller: this is borrowed, valid
 * while the owner handle lives, and yours to read. It is not a statement about
 * who allocated it. The library did, so the library has to release it, and C
 * has no way to spell "const to you, mine to me".
 *
 * So the qualifier has to come off somewhere. It comes off here, once, with the
 * reason attached, rather than at ninety-odd call sites with none -- which is
 * what `-Wcast-qual` was reporting before 2026-08-15, and why it could not be
 * enabled. Going through a copy of the pointer value rather than a cast keeps
 * the conversion defined without an intervening integer type.
 *
 * Takes NULL, like every other free in this file. */
void *rg_owned_internal(const void *owned) {
    void *value;
    memcpy(&value, &owned, sizeof(value));
    return value;
}

void rg_free_owned_internal(const void *owned) {
    free(rg_owned_internal(owned));
}

char *rg_strdup_internal(const char *value) {
    char *out;
    size_t len;
    if (value == 0) {
        return 0;
    }
    len = strlen(value);
    out = (char *)malloc(len + 1);
    if (out == 0) {
        return 0;
    }
    memcpy(out, value, len + 1);
    return out;
}

/* strndup is not in C99. */
char *rg_strndup_internal(const char *text, size_t length) {
    char *copy;
    if (text == 0) {
        return 0;
    }
    copy = (char *)malloc(length + 1);
    if (copy == 0) {
        return 0;
    }
    memcpy(copy, text, length);
    copy[length] = '\0';
    return copy;
}

void rg_segment_clear_internal(rg_segment *segment) {
    if (segment == 0) {
        return;
    }
    rg_free_owned_internal(segment->grapheme);
    rg_free_owned_internal(segment->tone);
    rg_free_owned_internal(segment->length);
    rg_free_owned_internal(segment->stress);
    segment->grapheme = 0;
    segment->tone = 0;
    segment->length = 0;
    segment->stress = 0;
}

rg_status rg_segment_copy_internal(const rg_segment *src, rg_segment *out) {
    if (src == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    out->grapheme = 0;
    out->tone = 0;
    out->length = 0;
    out->stress = 0;
    if (src->grapheme != 0) {
        out->grapheme = rg_strdup_internal(src->grapheme);
        if (out->grapheme == 0) {
            return RG_ERR_OOM;
        }
    }
    if (src->tone != 0) {
        out->tone = rg_strdup_internal(src->tone);
        if (out->tone == 0) {
            rg_segment_clear_internal(out);
            return RG_ERR_OOM;
        }
    }
    if (src->length != 0) {
        out->length = rg_strdup_internal(src->length);
        if (out->length == 0) {
            rg_segment_clear_internal(out);
            return RG_ERR_OOM;
        }
    }
    if (src->stress != 0) {
        out->stress = rg_strdup_internal(src->stress);
        if (out->stress == 0) {
            rg_segment_clear_internal(out);
            return RG_ERR_OOM;
        }
    }
    return RG_OK;
}

static void feature_constraint_clear(rg_feature_constraint *constraint) {
    if (constraint == 0) {
        return;
    }
    rg_free_owned_internal(constraint->feature);
    rg_free_owned_internal(constraint->value);
    constraint->feature = 0;
    constraint->value = 0;
}

rg_status rg_feature_constraint_copy_internal(
    const rg_feature_constraint *src,
    rg_feature_constraint *out
) {
    if (src == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    out->feature = 0;
    out->value = 0;
    if (src->feature != 0) {
        out->feature = rg_strdup_internal(src->feature);
        if (out->feature == 0) {
            return RG_ERR_OOM;
        }
    }
    if (src->value != 0) {
        out->value = rg_strdup_internal(src->value);
        if (out->value == 0) {
            feature_constraint_clear(out);
            return RG_ERR_OOM;
        }
    }
    return RG_OK;
}

rg_status rg_feature_constraint_array_copy_internal(
    const rg_feature_constraint *src,
    size_t count,
    const rg_feature_constraint **out
) {
    rg_feature_constraint *copy;
    size_t i;
    rg_status status;
    if (out == 0 || (count > 0 && src == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    if (count == 0) {
        return RG_OK;
    }
    copy = (rg_feature_constraint *)calloc(count, sizeof(*copy));
    if (copy == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < count; i++) {
        status = rg_feature_constraint_copy_internal(&src[i], &copy[i]);
        if (status != RG_OK) {
            while (i > 0) {
                i--;
                feature_constraint_clear(&copy[i]);
            }
            free(copy);
            return status;
        }
    }
    *out = copy;
    return RG_OK;
}

void rg_feature_constraint_array_clear_internal(const rg_feature_constraint *items, size_t count) {
    size_t i;
    if (items == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        feature_constraint_clear(rg_owned_internal(&items[i]));
    }
    rg_free_owned_internal(items);
}

void rg_distance_constraint_array_clear_internal(const rg_distance_constraint *items, size_t count) {
    size_t i;
    if (items == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        feature_constraint_clear(rg_owned_internal(&items[i].constraint));
    }
    rg_free_owned_internal(items);
}

void rg_link_clear_internal(rg_link *link) {
    size_t i;
    if (link == 0) {
        return;
    }
    for (i = 0; i < link->source_count; i++) {
        rg_segment_clear_internal(rg_owned_internal(&link->source[i]));
    }
    for (i = 0; i < link->target_count; i++) {
        rg_segment_clear_internal(rg_owned_internal(&link->target[i]));
    }
    rg_free_owned_internal(link->source);
    rg_free_owned_internal(link->target);
    rg_context_spec_clear_internal(&link->context);
    rg_feature_displacement_free(rg_owned_internal(link->feature_displacement), link->feature_displacement_count);
    link->source = 0;
    link->source_count = 0;
    link->target = 0;
    link->target_count = 0;
    link->feature_displacement = 0;
    link->feature_displacement_count = 0;
}

rg_status rg_link_copy_chunks_internal(
    rg_link *link,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count
) {
    rg_segment *source_copy = 0;
    rg_segment *target_copy = 0;
    size_t i;
    rg_status status;
    if (link == 0 || (source_count > 0 && source == 0) || (target_count > 0 && target == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    if (source_count > 0) {
        source_copy = (rg_segment *)calloc(source_count, sizeof(*source_copy));
        if (source_copy == 0) {
            return RG_ERR_OOM;
        }
        for (i = 0; i < source_count; i++) {
            status = rg_segment_copy_internal(&source[i], &source_copy[i]);
            if (status != RG_OK) {
                while (i > 0) {
                    i--;
                    rg_segment_clear_internal(&source_copy[i]);
                }
                free(source_copy);
                return status;
            }
        }
    }
    if (target_count > 0) {
        target_copy = (rg_segment *)calloc(target_count, sizeof(*target_copy));
        if (target_copy == 0) {
            for (i = 0; i < source_count; i++) {
                rg_segment_clear_internal(&source_copy[i]);
            }
            free(source_copy);
            return RG_ERR_OOM;
        }
        for (i = 0; i < target_count; i++) {
            status = rg_segment_copy_internal(&target[i], &target_copy[i]);
            if (status != RG_OK) {
                while (i > 0) {
                    i--;
                    rg_segment_clear_internal(&target_copy[i]);
                }
                free(target_copy);
                for (i = 0; i < source_count; i++) {
                    rg_segment_clear_internal(&source_copy[i]);
                }
                free(source_copy);
                return status;
            }
        }
    }
    link->source = source_copy;
    link->source_count = source_count;
    link->target = target_copy;
    link->target_count = target_count;
    link->confidence = 1.0;
    rg_context_spec_init_empty(&link->context);
    return RG_OK;
}
