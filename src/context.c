#include "regulae.h"

#include "merkmal.h"

#include <stdlib.h>

struct rg_context {
    mk_registry *registry;
    const mk_system *system;
};

struct rg_feature_set {
    mk_feature_set *inner;
};

static rg_status map_merkmal_status(mk_status status) {
    switch (status) {
    case MK_OK:
        return RG_OK;
    case MK_ERR_INVALID_ARGUMENT:
        return RG_ERR_INVALID_ARGUMENT;
    case MK_ERR_UNKNOWN_GRAPHEME:
        return RG_ERR_UNKNOWN_GRAPHEME;
    case MK_ERR_OOM:
        return RG_ERR_OOM;
    default:
        return RG_ERR_MERKMAL;
    }
}

rg_status rg_context_new_builtin(rg_context **out) {
    rg_context *ctx;
    mk_status status;
    if (out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    ctx = (rg_context *)calloc(1, sizeof(*ctx));
    if (ctx == 0) {
        return RG_ERR_OOM;
    }
    status = mk_registry_new_builtin(&ctx->registry);
    if (status != MK_OK) {
        free(ctx);
        return map_merkmal_status(status);
    }
    status = mk_registry_get_system(ctx->registry, "descriptive", &ctx->system);
    if (status != MK_OK) {
        mk_registry_free(ctx->registry);
        free(ctx);
        return map_merkmal_status(status);
    }
    *out = ctx;
    return RG_OK;
}

void rg_context_free(rg_context *ctx) {
    if (ctx == 0) {
        return;
    }
    mk_registry_free(ctx->registry);
    free(ctx);
}

rg_status rg_context_use_system(rg_context *ctx, const char *system_name) {
    const mk_system *system = 0;
    mk_status status;
    if (ctx == 0 || system_name == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    status = mk_registry_get_system(ctx->registry, system_name, &system);
    if (status != MK_OK) {
        return map_merkmal_status(status);
    }
    ctx->system = system;
    return RG_OK;
}

rg_status rg_context_system_name(const rg_context *ctx, const char **out) {
    mk_status status;
    if (ctx == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    status = mk_system_name(ctx->system, out);
    return map_merkmal_status(status);
}

rg_status rg_context_is_segment(const rg_context *ctx, const char *grapheme, int *out) {
    mk_status status;
    if (ctx == 0 || grapheme == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    status = mk_system_is_segment(ctx->system, grapheme, out);
    return map_merkmal_status(status);
}

rg_status rg_context_segment_distance(
    const rg_context *ctx,
    const char *a,
    const char *b,
    double *out
) {
    mk_status status;
    if (ctx == 0 || a == 0 || b == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0.0;
    status = mk_system_segment_distance(ctx->system, a, b, out);
    return map_merkmal_status(status);
}

rg_status rg_context_grapheme_features(
    const rg_context *ctx,
    const char *grapheme,
    rg_feature_set **out
) {
    rg_feature_set *features;
    mk_feature_set *inner = 0;
    mk_status status;
    if (ctx == 0 || grapheme == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    status = mk_system_grapheme_features(ctx->system, grapheme, &inner);
    if (status != MK_OK) {
        return map_merkmal_status(status);
    }
    features = (rg_feature_set *)calloc(1, sizeof(*features));
    if (features == 0) {
        mk_feature_set_free(inner);
        return RG_ERR_OOM;
    }
    features->inner = inner;
    *out = features;
    return RG_OK;
}

size_t rg_feature_set_size(const rg_feature_set *features) {
    if (features == 0) {
        return 0;
    }
    return mk_feature_set_size(features->inner);
}

const char *rg_feature_set_get(const rg_feature_set *features, size_t index) {
    if (features == 0) {
        return 0;
    }
    return mk_feature_set_get(features->inner, index);
}

void rg_feature_set_free(rg_feature_set *features) {
    if (features == 0) {
        return;
    }
    mk_feature_set_free(features->inner);
    free(features);
}
