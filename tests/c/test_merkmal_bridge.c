#include "regulae.h"

#include <assert.h>
#include <math.h>
#include <string.h>

static int feature_set_contains(const rg_feature_set *features, const char *needle) {
    size_t i;
    for (i = 0; i < rg_feature_set_size(features); i++) {
        const char *item = rg_feature_set_get(features, i);
        if (item != 0 && strcmp(item, needle) == 0) {
            return 1;
        }
    }
    return 0;
}

/* The tie bar is how a transcription says "one segment", and the default
 * reading honours it: "t͡ʃ" is one segment where untied "tʃ" is two. Longest
 * match against the inventory reads both as one, and on the same rule reads
 * the geminate "kk" as one, which is why it is not the default. */
static void test_segmentation_reads_the_tie_bar(rg_context *ctx) {
    rg_segment *segments = 0;
    size_t count = 0;

    assert(rg_context_segmentation(ctx) == RG_SEGMENT_ORTHOGRAPHIC);
    assert(rg_context_segment_word(ctx, "let\xcd\xa1\xca\x83""e", &segments, &count) == RG_OK);
    assert(count == 4);
    assert(strcmp(segments[2].grapheme, "t\xcd\xa1\xca\x83") == 0);
    rg_segments_free(segments, count);

    assert(rg_context_segment_word(ctx, "bukka", &segments, &count) == RG_OK);
    assert(count == 5);
    rg_segments_free(segments, count);

    assert(rg_context_set_segmentation(ctx, RG_SEGMENT_SYSTEM_LONGEST_MATCH) == RG_OK);
    assert(rg_context_segment_word(ctx, "bukka", &segments, &count) == RG_OK);
    assert(count == 4);
    assert(strcmp(segments[2].grapheme, "kk") == 0);
    rg_segments_free(segments, count);
    assert(rg_context_set_segmentation(ctx, RG_SEGMENT_ORTHOGRAPHIC) == RG_OK);

    assert(rg_context_set_segmentation(0, RG_SEGMENT_ORTHOGRAPHIC) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_context_set_segmentation(ctx, (rg_segmentation)7) == RG_ERR_UNSUPPORTED_OPTION);
}

/* Tone written on the word reaches the model as the segment's own dimension,
 * never as part of the grapheme, whether it is bound to the nucleus or spelled
 * as a token of its own the way CLDF wordlists publish it. */
static void test_tone_leaves_the_grapheme(rg_context *ctx) {
    rg_segment *segments = 0;
    size_t count = 0;

    assert(rg_context_segment_word(ctx, "ma\xc2\xb3\xc2\xb3", &segments, &count) == RG_OK);
    assert(count == 2);
    assert(strcmp(segments[1].grapheme, "a") == 0);
    assert(segments[1].tone != 0 && strcmp(segments[1].tone, "\xc2\xb3\xc2\xb3") == 0);
    rg_segments_free(segments, count);

    assert(rg_context_segment_word(ctx, "ma \xc2\xb3\xc2\xb3", &segments, &count) == RG_OK);
    assert(count == 2);
    assert(strcmp(segments[1].grapheme, "a") == 0);
    assert(segments[1].tone != 0 && strcmp(segments[1].tone, "\xc2\xb3\xc2\xb3") == 0);
    rg_segments_free(segments, count);

    /* An untoned word carries no tone, rather than an empty one. */
    assert(rg_context_segment_word(ctx, "ma", &segments, &count) == RG_OK);
    assert(count == 2);
    assert(segments[1].tone == 0);
    rg_segments_free(segments, count);
}

int main(void) {
    rg_context *ctx = 0;
    rg_feature_set *features = 0;
    const char *name = 0;
    int is_segment = 0;
    double distance = 0.0;

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    assert(ctx != 0);
    assert(rg_context_system_name(ctx, &name) == RG_OK);
    assert(strcmp(name, RG_DEFAULT_FEATURE_SYSTEM) == 0);

    assert(rg_context_is_segment(ctx, "p", &is_segment) == RG_OK);
    assert(is_segment == 1);
    assert(rg_context_is_segment(ctx, "not-ipa", &is_segment) == RG_OK);
    assert(is_segment == 0);

    assert(rg_context_segment_distance(ctx, "p", "p", &distance) == RG_OK);
    assert(fabs(distance) < 1e-12);
    assert(rg_context_segment_distance(ctx, "p", "b", &distance) == RG_OK);
    assert(distance > 0.0);

    assert(rg_context_grapheme_features(ctx, "p", &features) == RG_OK);
    assert(features != 0);
    assert(rg_feature_set_size(features) > 0);
    assert(feature_set_contains(features, "consonant"));
    rg_feature_set_free(features);
    features = 0;

    assert(rg_context_grapheme_features(ctx, "not-ipa", &features) == RG_ERR_UNKNOWN_GRAPHEME);
    assert(features == 0);

    test_segmentation_reads_the_tie_bar(ctx);
    test_tone_leaves_the_grapheme(ctx);

    assert(rg_context_use_system(ctx, "phoible") == RG_OK);
    assert(rg_context_system_name(ctx, &name) == RG_OK);
    assert(strcmp(name, "phoible") == 0);
    assert(rg_context_use_system(ctx, "missing") == RG_ERR_MERKMAL);

    assert(rg_context_new_builtin(0) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_context_system_name(0, &name) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_context_is_segment(ctx, 0, &is_segment) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_context_segment_distance(ctx, "p", 0, &distance) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_context_grapheme_features(ctx, 0, &features) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_feature_set_size(0) == 0);
    assert(rg_feature_set_get(0, 0) == 0);
    rg_feature_set_free(0);

    rg_context_free(ctx);
    rg_context_free(0);
    return 0;
}
