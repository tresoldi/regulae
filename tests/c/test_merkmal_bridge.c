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

int main(void) {
    rg_context *ctx = 0;
    rg_feature_set *features = 0;
    const char *name = 0;
    int is_segment = 0;
    double distance = 0.0;

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    assert(ctx != 0);
    assert(rg_context_system_name(ctx, &name) == RG_OK);
    assert(strcmp(name, "descriptive") == 0);

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
