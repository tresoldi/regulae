#include "internal.h"

#include <stdlib.h>
#include <string.h>

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

void rg_segment_clear_internal(rg_segment *segment) {
    if (segment == 0) {
        return;
    }
    free((char *)segment->grapheme);
    free((char *)segment->tone);
    free((char *)segment->length);
    free((char *)segment->stress);
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

void rg_link_clear_internal(rg_link *link) {
    size_t i;
    if (link == 0) {
        return;
    }
    for (i = 0; i < link->source_count; i++) {
        rg_segment_clear_internal((rg_segment *)&link->source[i]);
    }
    for (i = 0; i < link->target_count; i++) {
        rg_segment_clear_internal((rg_segment *)&link->target[i]);
    }
    free((rg_segment *)link->source);
    free((rg_segment *)link->target);
    rg_feature_displacement_free((rg_feature_displacement *)link->feature_displacement, link->feature_displacement_count);
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
