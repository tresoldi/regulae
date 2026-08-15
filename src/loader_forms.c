#include "loader_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- segment parsing ---------------------------------------------------- */

/* Splits a whitespace-separated segment cell. "-" gap markers are dropped and,
 * when track_boundaries is set, "+" tokens record a morpheme boundary at the
 * current position instead of producing a segment. */
static rg_status lift_stress_mark(rg_segment *segment);

rg_status parse_segments(
    const char *raw,
    int track_boundaries,
    rg_segment **out_segments,
    size_t *out_count,
    int **out_breaks,
    size_t *out_break_count
) {
    rg_segment *segments = 0;
    size_t count = 0;
    size_t cap = 0;
    int *breaks = 0;
    size_t break_count = 0;
    size_t break_cap = 0;
    const char *p = raw == 0 ? "" : raw;

    *out_segments = 0;
    *out_count = 0;
    if (out_breaks != 0) {
        *out_breaks = 0;
        *out_break_count = 0;
    }
    for (;;) {
        const char *start;
        size_t length;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
            p++;
        }
        if (*p == '\0') {
            break;
        }
        start = p;
        while (*p != '\0' && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') {
            p++;
        }
        length = (size_t)(p - start);
        if (length == 1 && start[0] == '-') {
            continue;
        }
        if (length == 1 && start[0] == '+') {
            if (!track_boundaries || count == 0) {
                continue;
            }
            if (break_count == break_cap) {
                size_t next_cap = break_cap == 0 ? 4 : break_cap * 2;
                int *next = (int *)realloc(breaks, next_cap * sizeof(*next));
                if (next == 0) {
                    goto fail;
                }
                breaks = next;
                break_cap = next_cap;
            }
            breaks[break_count++] = (int)count;
            continue;
        }
        if (count == cap) {
            size_t next_cap = cap == 0 ? 8 : cap * 2;
            rg_segment *next = (rg_segment *)realloc(segments, next_cap * sizeof(*next));
            if (next == 0) {
                goto fail;
            }
            segments = next;
            cap = next_cap;
        }
        memset(&segments[count], 0, sizeof(segments[count]));
        {
            char *grapheme = (char *)malloc(length + 1);
            if (grapheme == 0) {
                goto fail;
            }
            memcpy(grapheme, start, length);
            grapheme[length] = '\0';
            segments[count].grapheme = grapheme;
        }
        count++;
    }
    {
        size_t i;
        for (i = 0; i < count; i++) {
            if (lift_stress_mark(&segments[i]) != RG_OK) {
                goto fail;
            }
        }
    }
    *out_segments = segments;
    *out_count = count;
    if (out_breaks != 0) {
        *out_breaks = breaks;
        *out_break_count = break_count;
    } else {
        free(breaks);
    }
    return RG_OK;

fail:
    {
        size_t i;
        for (i = 0; i < count; i++) {
            rg_free_owned_internal(segments[i].grapheme);
        }
    }
    free(segments);
    free(breaks);
    return RG_ERR_OOM;
}

/* Attaches a suprasegmental cell's whitespace-separated values to
 * already-parsed segments, by position. Tone and stress take the same shape,
 * so they take the same code: both are properties of a segment that must never
 * reach feature lookup, and both are annotated one value per segment. A value replaces whatever the word itself carried, so
 * an explicit column wins over tone written into the transcription. "-" and
 * empty tokens leave the segment as segmentation found it, so a tone-bearing
 * corpus can mark its consonants without inventing a tone for them, and a
 * corpus that annotates only some segments does not erase the rest.
 * Rejects a cell whose token count disagrees with the segment count:
 * silently truncating would tone the wrong vowels, and a corpus that annotates
 * tone at all is annotating it deliberately. */
/* Lifts a leading IPA stress mark off a pre-segmented token.
 *
 * merkmal resolves "ˈa" as a grapheme, so a mark left in place makes a stressed
 * segment a different segment from its unstressed self, splitting every
 * correspondence it takes part in. The unsegmented path lifts the mark onto the
 * syllable nucleus, because there the mark stands before a syllable and the
 * writer has not chosen a segment. Here they have: a pre-segmented corpus
 * writes the mark on the token it means, and that is where the stress goes. */
static rg_status lift_stress_mark(rg_segment *segment) {
    const char *grapheme = segment->grapheme;
    const char *value;
    char *stripped;
    /* Tested in order: a lone 0xCB is a one-byte string, and reaching [2] to
     * find out whether anything follows the mark reads past its NUL. A stress
     * mark truncated mid-character is not exotic -- it is what a file cut at a
     * byte boundary, or written in the wrong encoding, hands over. */
    if (grapheme == 0 || grapheme[0] != '\xcb' || grapheme[1] == '\0' || grapheme[2] == '\0') {
        return RG_OK;
    }
    if (grapheme[1] == '\x88') {
        value = "primary";
    } else if (grapheme[1] == '\x8c') {
        value = "secondary";
    } else {
        return RG_OK;
    }
    stripped = rg_strdup_internal(grapheme + 2);
    if (stripped == 0) {
        return RG_ERR_OOM;
    }
    rg_free_owned_internal(segment->grapheme);
    segment->grapheme = stripped;
    rg_free_owned_internal(segment->stress);
    segment->stress = rg_strdup_internal(value);
    return segment->stress == 0 ? RG_ERR_OOM : RG_OK;
}

rg_status attach_dimension(
    const char *raw,
    rg_segment *segments,
    size_t segment_count,
    int dimension
) {
    const char *p = raw == 0 ? "" : raw;
    size_t index = 0;
    for (;;) {
        const char *start;
        size_t length;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
            p++;
        }
        if (*p == '\0') {
            break;
        }
        start = p;
        while (*p != '\0' && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') {
            p++;
        }
        length = (size_t)(p - start);
        if (index >= segment_count) {
            return RG_ERR_PARSE;
        }
        if (length == 1 && start[0] == '-') {
            index++;
            continue;
        }
        {
            char *value = (char *)malloc(length + 1);
            const char **slot = dimension == RG_DIMENSION_STRESS ? &segments[index].stress
                : (dimension == RG_DIMENSION_LENGTH ? &segments[index].length
                                                    : &segments[index].tone);
            if (value == 0) {
                return RG_ERR_OOM;
            }
            memcpy(value, start, length);
            value[length] = '\0';
            rg_free_owned_internal(*slot);
            *slot = value;
        }
        index++;
    }
    if (index != 0 && index != segment_count) {
        return RG_ERR_PARSE;
    }
    return RG_OK;
}

/* Counts the tokens of an alignment cell, where "-" marks a gap. Only the
 * length is meaningful to the loaders, which use it to reject cognate sets
 * whose alignment hints disagree across lects. */
long alignment_token_count(const char *raw) {
    long count = 0;
    const char *p = raw == 0 ? "" : raw;
    for (;;) {
        while (*p == ' ' || *p == '\t') {
            p++;
        }
        if (*p == '\0') {
            break;
        }
        while (*p != '\0' && *p != ' ' && *p != '\t') {
            p++;
        }
        count++;
    }
    return count;
}

