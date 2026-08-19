/* How readable a promoted chunk is as a single historical process.
 *
 * A chunk row says "this run of segments answers to that run", and says nothing
 * about why. Some of those are one comprehensible event -- a vowel nasalized
 * and swallowed the nasal after it -- and some are a bundle of unrelated
 * changes that happened to sit next to each other and pay for themselves
 * jointly under BIC. Both look the same in the table.
 *
 * The score here separates them, on structural grounds: a chunk is more
 * transparent when it is short, when its two sides are the same length, when
 * it decomposes into matched segments without many gaps, when it does not
 * simply contain a smaller chunk that was also promoted, and when its
 * decomposition matches one of a few recognisable shapes. It is a heuristic
 * and it is advisory. `chunk_min_transparency` is what acts on it, and its
 * default of zero drops nothing.
 *
 * Two of the process profiles are inherently directional: nasal fusion
 * requires the *target* to have gained nasalization and a *source* segment to
 * have gone, and reading the same correspondence from the other lect it would
 * match nothing, so the same chunk would score 0.05 higher in one direction
 * than the other. That would make a corpus's analysis depend on which lect its
 * file names first, which is an invariant this library enforces elsewhere and
 * in chunk promotion itself. Transparency is a property of the
 * correspondence, not of a direction of change -- "a~ã before a nasal" is
 * exactly as interpretable read either way -- so each directional profile is
 * tested in both orientations and matches if either holds.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

typedef struct chunk_decomposition {
    size_t matched;
    size_t identity_matched;
    size_t changed_matched;
    size_t deletions;
    size_t insertions;
    size_t gap_links;
    size_t total_links;
    /* The single changed pair and the single dropped segment, when there is
     * exactly one of each: that is the shape every directional profile is
     * stated over. These borrow the sub-alignment's own segments, whose
     * graphemes it owns and frees, so every profile is tested before it goes.
     */
    const rg_segment *changed_source;
    const rg_segment *changed_target;
    const rg_segment *deleted;
    const rg_segment *inserted;
} chunk_decomposition;

static int chunk_segments_equal(
    const rg_segment *a, size_t a_count,
    const rg_segment *b, size_t b_count
) {
    size_t i;
    if (a_count != b_count) {
        return 0;
    }
    for (i = 0; i < a_count; i++) {
        if (a[i].grapheme == 0 || b[i].grapheme == 0) {
            return a[i].grapheme == b[i].grapheme;
        }
        if (strcmp(a[i].grapheme, b[i].grapheme) != 0) {
            return 0;
        }
    }
    return 1;
}

static int segment_has(const rg_context *ctx, const rg_segment *segment, const char *feature) {
    const rg_feature_set *features = 0;
    size_t i;
    if (ctx == 0 || segment == 0 || segment->grapheme == 0) {
        return 0;
    }
    if (rg_context_features_internal(ctx, segment->grapheme, &features) != RG_OK) {
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

static int segment_gained(
    const rg_context *ctx,
    const rg_segment *from,
    const rg_segment *to,
    const char *feature
) {
    return !segment_has(ctx, from, feature) && segment_has(ctx, to, feature);
}

/* A vowel absorbs an adjacent nasal and carries its nasality. Tested in both
 * orientations; see the note at the head of this file. */
static int looks_like_nasal_fusion(const rg_context *ctx, const chunk_decomposition *d) {
    const rg_segment *gone;
    if (d->changed_matched != 1 || d->deletions + d->insertions != 1) {
        return 0;
    }
    if (!segment_has(ctx, d->changed_source, "vowel") || !segment_has(ctx, d->changed_target, "vowel")) {
        return 0;
    }
    gone = d->deletions == 1 ? d->deleted : d->inserted;
    if (!segment_has(ctx, gone, "nasal")) {
        return 0;
    }
    return segment_gained(ctx, d->changed_source, d->changed_target, "nasalized") ||
           segment_gained(ctx, d->changed_target, d->changed_source, "nasalized");
}

/* A segment becomes, or resolves to, an approximant while neighbouring vocalic
 * or liquid material goes. Tested in both orientations. */
static int looks_like_glide_fusion(const rg_context *ctx, const chunk_decomposition *d) {
    const rg_segment *gone;
    const rg_segment *a;
    const rg_segment *b;
    if (d->changed_matched != 1 || d->deletions + d->insertions != 1) {
        return 0;
    }
    gone = d->deletions == 1 ? d->deleted : d->inserted;
    a = d->changed_source;
    b = d->changed_target;
    if (!segment_has(ctx, a, "approximant") && !segment_has(ctx, b, "approximant")) {
        return 0;
    }
    return segment_has(ctx, a, "lateral") || segment_has(ctx, b, "lateral") ||
           segment_has(ctx, a, "approximant") || segment_has(ctx, b, "approximant") ||
           segment_has(ctx, gone, "vowel") || segment_has(ctx, gone, "approximant");
}

static const char *classify_chunk_process(const rg_context *ctx, const chunk_decomposition *d) {
    size_t events = d->changed_matched + d->identity_matched + d->deletions + d->insertions;
    size_t gaps = d->deletions + d->insertions;

    if (looks_like_nasal_fusion(ctx, d)) {
        return "nasal_fusion";
    }
    if (looks_like_glide_fusion(ctx, d)) {
        return "glide_or_vocalization_fusion";
    }
    if (d->changed_matched == 1 && gaps == 1 && events <= 2) {
        return "compact_fusion";
    }
    if (d->identity_matched == 1 && gaps == 1 && events <= 2) {
        return "residual_reduction";
    }
    if (d->deletions > 0 && d->insertions > 0 && d->changed_matched + d->identity_matched <= 1) {
        return "balanced_restructuring";
    }
    if (events >= 3 && gaps > 0 && d->changed_matched + d->identity_matched > 0) {
        return "bundled_reduction";
    }
    return "mixed_or_unclear";
}

/* Re-aligns the chunk against itself one segment at a time, with the chunk
 * table emptied so the decomposition cannot lean on the very rows being
 * judged. */
static rg_status decompose_chunk(
    const rg_context *ctx,
    const rg_train_options *options,
    const rg_pairwise_model *model,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    chunk_decomposition *out,
    const char **out_profile
) {
    rg_form sub_source;
    rg_form sub_target;
    rg_alignment *sub = 0;
    rg_pairwise_model no_chunks;
    size_t i;
    rg_status status;

    memset(out, 0, sizeof(*out));
    memset(&sub_source, 0, sizeof(sub_source));
    memset(&sub_target, 0, sizeof(sub_target));
    sub_source.lect_id = "_chunk_src";
    sub_source.segments = source;
    sub_source.segment_count = source_count;
    sub_target.lect_id = "_chunk_tgt";
    sub_target.segments = target;
    sub_target.segment_count = target_count;

    no_chunks = *model;
    no_chunks.chunks = 0;
    no_chunks.chunk_count = 0;

    status = rg_align_forms_with_model(ctx, &no_chunks, options, &sub_source, &sub_target, 1, &sub);
    if (status != RG_OK) {
        return status;
    }
    out->total_links = rg_alignment_link_count(sub);
    for (i = 0; i < rg_alignment_link_count(sub); i++) {
        const rg_link *link = rg_alignment_link_at(sub, i);
        int has_source = link->source_count > 0;
        int has_target = link->target_count > 0;
        if (has_source && has_target) {
            out->matched++;
            if (chunk_segments_equal(link->source, link->source_count, link->target, link->target_count)) {
                out->identity_matched++;
            } else {
                out->changed_matched++;
                out->changed_source = &link->source[0];
                out->changed_target = &link->target[0];
            }
        } else if (has_source) {
            out->deletions += link->source_count;
            out->deleted = &link->source[0];
        } else if (has_target) {
            out->insertions += link->target_count;
            out->inserted = &link->target[0];
        }
        if (!has_source || !has_target) {
            out->gap_links++;
        }
    }
    /* Before the free: every profile below reads segments the alignment owns. */
    *out_profile = classify_chunk_process(ctx, out);
    rg_alignment_free(sub);
    if (out->total_links < 1) {
        out->total_links = 1;
    }
    return RG_OK;
}

static double process_score_adjustment(const char *profile) {
    if (strcmp(profile, "compact_fusion") == 0) {
        return 0.03;
    }
    if (strcmp(profile, "residual_reduction") == 0) {
        return 0.01;
    }
    if (strcmp(profile, "nasal_fusion") == 0) {
        return 0.05;
    }
    if (strcmp(profile, "glide_or_vocalization_fusion") == 0) {
        return 0.04;
    }
    if (strcmp(profile, "balanced_restructuring") == 0) {
        return 0.02;
    }
    if (strcmp(profile, "bundled_reduction") == 0) {
        return -0.04;
    }
    return 0.0;
}

/* Whether needle occurs as a contiguous run inside haystack. */
static int chunk_segments_contain(
    const rg_segment *haystack, size_t haystack_count,
    const rg_segment *needle, size_t needle_count
) {
    size_t start;
    if (needle_count == 0 || needle_count > haystack_count) {
        return 0;
    }
    for (start = 0; start + needle_count <= haystack_count; start++) {
        if (chunk_segments_equal(haystack + start, needle_count, needle, needle_count)) {
            return 1;
        }
    }
    return 0;
}

/* Promoted chunks strictly smaller on both sides that sit inside this one. A
 * chunk that merely wraps a chunk already in the table adds little. */
static size_t chunk_overlap_count(
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    const rg_chunk_row *rows,
    size_t row_count
) {
    size_t i;
    size_t count = 0;
    for (i = 0; i < row_count; i++) {
        const rg_chunk_row *other = &rows[i];
        if (chunk_segments_equal(other->source, other->source_count, source, source_count) &&
            chunk_segments_equal(other->target, other->target_count, target, target_count)) {
            continue;
        }
        if (other->source_count >= source_count && other->target_count >= target_count) {
            continue;
        }
        if (chunk_segments_contain(source, source_count, other->source, other->source_count) &&
            chunk_segments_contain(target, target_count, other->target, other->target_count)) {
            count++;
        }
    }
    return count;
}

rg_status rg_chunk_transparency_internal(
    const rg_context *ctx,
    const rg_train_options *options,
    const rg_pairwise_model *model,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    const rg_chunk_row *rows,
    size_t row_count,
    double *out
) {
    chunk_decomposition d;
    const char *profile;
    size_t overlaps;
    size_t total_length = source_count + target_count;
    size_t asymmetry = source_count > target_count
        ? source_count - target_count
        : target_count - source_count;
    size_t gaps;
    double gap_ratio;
    double score = 1.0;
    rg_status status;

    *out = 0.0;
    status = decompose_chunk(ctx, options, model, source, source_count, target, target_count,
                             &d, &profile);
    if (status != RG_OK) {
        return status;
    }
    overlaps = chunk_overlap_count(source, source_count, target, target_count, rows, row_count);
    gap_ratio = (double)d.gap_links / (double)d.total_links;
    gaps = d.deletions + d.insertions;

    score -= 0.18 * (double)asymmetry;
    score -= 0.24 * gap_ratio;
    score -= 0.14 * (double)(total_length > 3 ? total_length - 3 : 0);
    score -= 0.09 * (double)(overlaps < 3 ? overlaps : 3) / 3.0;
    if (total_length >= 5 && asymmetry > 0) {
        score -= 0.10;
    }
    if (d.deletions >= 2) {
        score -= 0.10;
    }
    if (d.insertions >= 2) {
        score -= 0.10;
    }
    /* One side lost material and the other gained none, which is a reduction
     * read one way and an expansion read the other. Both are charged, so the
     * pair of tests is symmetric even though each is not. */
    if (d.deletions > 0 && d.insertions == 0 && asymmetry > 0) {
        score -= 0.10;
    }
    if (d.insertions > 0 && d.deletions == 0 && asymmetry > 0) {
        score -= 0.10;
    }
    if (total_length <= 4 && asymmetry == 1 && d.matched == 1 && d.changed_matched == 1 && gaps == 1) {
        score += 0.08;
    }
    if (total_length <= 4 && asymmetry == 1 && d.matched == 1 && d.identity_matched == 1 && gaps == 1) {
        score += 0.03;
    }
    if (asymmetry == 0 && d.deletions > 0 && d.insertions > 0) {
        score += 0.12;
    }
    if (d.matched >= 2 && asymmetry <= 1) {
        score += 0.05;
    }
    if (asymmetry == 0 && gap_ratio == 0.0 && total_length <= 4) {
        score += 0.08;
    } else if (total_length <= 4 && asymmetry <= 1) {
        score += 0.05;
    }
    score += process_score_adjustment(profile);

    if (score < 0.0) {
        score = 0.0;
    } else if (score > 1.0) {
        score = 1.0;
    }
    *out = score;
    return RG_OK;
}
