#include "internal.h"

#include "merkmal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Every merkmal lookup goes through this handle, and both feature bundles and
 * pairwise distances are memoised here. The alignment DP asks for the same few
 * dozen graphemes millions of times per training run, so without these caches
 * merkmal string matching dominates the profile. Caches are keyed by the
 * resolved feature system and dropped when it changes. */
/* The feature dimensions a link context can constrain on, in the fixed order
 * they are emitted. Alphabetical, so a union over a span comes out sorted
 * without a sort step. */
const char *const rg_context_feature_names[] = {
    "back",
    "close",
    "consonant",
    "fricative",
    "front",
    "long",
    "nasal",
    "open",
    "sonorant",
    "stop",
    "voiced",
    "voiceless",
    "vowel"
};
const size_t rg_context_feature_name_count =
    sizeof(rg_context_feature_names) / sizeof(rg_context_feature_names[0]);

typedef struct feature_cache_entry {
    char *grapheme;
    rg_feature_set *features;
    rg_feature_constraint *constraints;
    size_t constraint_count;
    /* An entry can be created by either lookup, so each answer tracks its own
     * state: 0 not yet asked, 1 resolved, -1 resolved as unknown. Collapsing
     * these into one flag makes an is_segment insert look like a failed
     * feature lookup. */
    int features_state;
    int is_segment_state;
    int is_segment;
} feature_cache_entry;

typedef struct distance_cache_entry {
    char *a;
    char *b;
    double value;
    int resolved;
} distance_cache_entry;

struct rg_context {
    mk_registry *registry;
    const mk_system *system;
    /* The grapheme that last failed to resolve. The Go reference carries this
     * on its error value; a C status code cannot, and "unknown grapheme" with
     * no indication of which one is unactionable on a corpus of any size. */
    char *unknown_grapheme;
    feature_cache_entry *features;
    size_t feature_count;
    size_t feature_cap;
    distance_cache_entry *distances;
    size_t distance_count;
    size_t distance_cap;
};

/* FNV-1a, with the constants chosen for the width of size_t. WebAssembly is
 * 32-bit, where the 64-bit prime truncates to an even number and the hash
 * degenerates. */
#if SIZE_MAX > 0xFFFFFFFFu
#define RG_FNV_OFFSET ((size_t)14695981039346656037ULL)
#define RG_FNV_PRIME ((size_t)1099511628211ULL)
#else
#define RG_FNV_OFFSET ((size_t)2166136261u)
#define RG_FNV_PRIME ((size_t)16777619u)
#endif

static size_t hash_string(const char *value, size_t seed) {
    size_t hash = seed;
    while (*value != 0) {
        hash ^= (size_t)(unsigned char)*value++;
        hash *= RG_FNV_PRIME;
    }
    return hash;
}

static void context_caches_clear(rg_context *ctx) {
    size_t i;
    for (i = 0; i < ctx->feature_cap; i++) {
        if (ctx->features[i].grapheme != 0) {
            free(ctx->features[i].grapheme);
            rg_feature_set_free(ctx->features[i].features);
            free(ctx->features[i].constraints);
        }
    }
    free(ctx->features);
    ctx->features = 0;
    ctx->feature_cap = 0;
    ctx->feature_count = 0;
    for (i = 0; i < ctx->distance_cap; i++) {
        if (ctx->distances[i].a != 0) {
            free(ctx->distances[i].a);
            free(ctx->distances[i].b);
        }
    }
    free(ctx->distances);
    ctx->distances = 0;
    ctx->distance_cap = 0;
    ctx->distance_count = 0;
}

static rg_status feature_cache_grow(rg_context *ctx) {
    size_t next_cap = ctx->feature_cap == 0 ? 64 : ctx->feature_cap * 2;
    feature_cache_entry *next = (feature_cache_entry *)calloc(next_cap, sizeof(*next));
    size_t i;
    if (next == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < ctx->feature_cap; i++) {
        size_t slot;
        if (ctx->features[i].grapheme == 0) {
            continue;
        }
        slot = hash_string(ctx->features[i].grapheme, RG_FNV_OFFSET) & (next_cap - 1);
        while (next[slot].grapheme != 0) {
            slot = (slot + 1) & (next_cap - 1);
        }
        next[slot] = ctx->features[i];
    }
    free(ctx->features);
    ctx->features = next;
    ctx->feature_cap = next_cap;
    return RG_OK;
}

static rg_status distance_cache_grow(rg_context *ctx) {
    size_t next_cap = ctx->distance_cap == 0 ? 256 : ctx->distance_cap * 2;
    distance_cache_entry *next = (distance_cache_entry *)calloc(next_cap, sizeof(*next));
    size_t i;
    if (next == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < ctx->distance_cap; i++) {
        size_t slot;
        if (ctx->distances[i].a == 0) {
            continue;
        }
        slot = hash_string(ctx->distances[i].b, hash_string(ctx->distances[i].a, RG_FNV_OFFSET)) & (next_cap - 1);
        while (next[slot].a != 0) {
            slot = (slot + 1) & (next_cap - 1);
        }
        next[slot] = ctx->distances[i];
    }
    free(ctx->distances);
    ctx->distances = next;
    ctx->distance_cap = next_cap;
    return RG_OK;
}

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
    free(ctx->unknown_grapheme);
    context_caches_clear(ctx);
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
    free(ctx->unknown_grapheme);
    ctx->unknown_grapheme = 0;
    context_caches_clear(ctx);
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

/* Memoised: the scoring path asks this for both graphemes of every link it
 * considers, and resolving a grapheme in merkmal is not cheap. */
rg_status rg_context_is_segment(const rg_context *ctx, const char *grapheme, int *out) {
    rg_context *mutable_ctx = (rg_context *)ctx;
    mk_status status;
    size_t slot;

    if (ctx == 0 || grapheme == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    if (mutable_ctx->feature_cap == 0 || (mutable_ctx->feature_count + 1) * 10 >= mutable_ctx->feature_cap * 7) {
        if (feature_cache_grow(mutable_ctx) != RG_OK) {
            status = mk_system_is_segment(ctx->system, grapheme, out);
            return map_merkmal_status(status);
        }
    }
    slot = hash_string(grapheme, RG_FNV_OFFSET) & (mutable_ctx->feature_cap - 1);
    while (mutable_ctx->features[slot].grapheme != 0) {
        if (strcmp(mutable_ctx->features[slot].grapheme, grapheme) == 0) {
            if (mutable_ctx->features[slot].is_segment_state != 0) {
                *out = mutable_ctx->features[slot].is_segment;
                return RG_OK;
            }
            break;
        }
        slot = (slot + 1) & (mutable_ctx->feature_cap - 1);
    }
    status = mk_system_is_segment(ctx->system, grapheme, out);
    if (status != MK_OK) {
        return map_merkmal_status(status);
    }
    if (mutable_ctx->features[slot].grapheme == 0) {
        mutable_ctx->features[slot].grapheme = rg_strdup_internal(grapheme);
        if (mutable_ctx->features[slot].grapheme == 0) {
            return RG_OK;
        }
        mutable_ctx->feature_count++;
    }
    mutable_ctx->features[slot].is_segment = *out;
    mutable_ctx->features[slot].is_segment_state = 1;
    return RG_OK;
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
    {
        rg_context *mutable_ctx = (rg_context *)ctx;
        size_t slot;
        if (mutable_ctx->distance_cap == 0 || (mutable_ctx->distance_count + 1) * 10 >= mutable_ctx->distance_cap * 7) {
            if (distance_cache_grow(mutable_ctx) != RG_OK) {
                status = mk_system_segment_distance(ctx->system, a, b, out);
                return map_merkmal_status(status);
            }
        }
        slot = hash_string(b, hash_string(a, RG_FNV_OFFSET)) & (mutable_ctx->distance_cap - 1);
        while (mutable_ctx->distances[slot].a != 0) {
            if (strcmp(mutable_ctx->distances[slot].a, a) == 0 && strcmp(mutable_ctx->distances[slot].b, b) == 0) {
                if (!mutable_ctx->distances[slot].resolved) {
                    return RG_ERR_MERKMAL;
                }
                *out = mutable_ctx->distances[slot].value;
                return RG_OK;
            }
            slot = (slot + 1) & (mutable_ctx->distance_cap - 1);
        }
        status = mk_system_segment_distance(ctx->system, a, b, out);
        if (status != MK_OK) {
            return map_merkmal_status(status);
        }
        mutable_ctx->distances[slot].a = rg_strdup_internal(a);
        mutable_ctx->distances[slot].b = rg_strdup_internal(b);
        if (mutable_ctx->distances[slot].a == 0 || mutable_ctx->distances[slot].b == 0) {
            free(mutable_ctx->distances[slot].a);
            free(mutable_ctx->distances[slot].b);
            mutable_ctx->distances[slot].a = 0;
            mutable_ctx->distances[slot].b = 0;
            return RG_OK;
        }
        mutable_ctx->distances[slot].value = *out;
        mutable_ctx->distances[slot].resolved = 1;
        mutable_ctx->distance_count++;
    }
    return RG_OK;
}

void rg_context_note_unknown_grapheme_internal(const rg_context *ctx, const char *grapheme) {
    rg_context *mutable_ctx = (rg_context *)ctx;
    if (ctx == 0 || grapheme == 0) {
        return;
    }
    free(mutable_ctx->unknown_grapheme);
    mutable_ctx->unknown_grapheme = rg_strdup_internal(grapheme);
}

void rg_context_last_error(const rg_context *ctx, const char **grapheme, const char **feature_system) {
    if (grapheme != 0) {
        *grapheme = ctx == 0 ? 0 : ctx->unknown_grapheme;
    }
    if (feature_system != 0) {
        const char *name = 0;
        *feature_system = 0;
        if (ctx != 0 && mk_system_name(ctx->system, &name) == MK_OK) {
            *feature_system = name;
        }
    }
}

/* Borrowed feature bundle for a grapheme, valid while the context lives and
 * its feature system is unchanged. Returns RG_ERR_UNKNOWN_GRAPHEME, and
 * remembers that verdict, for graphemes the system does not cover. */
rg_status rg_context_features_internal(
    const rg_context *ctx,
    const char *grapheme,
    const rg_feature_set **out
) {
    rg_context *mutable_ctx = (rg_context *)ctx;
    rg_feature_set *features = 0;
    size_t slot;
    rg_status status;

    if (ctx == 0 || grapheme == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    if (mutable_ctx->feature_cap == 0 || (mutable_ctx->feature_count + 1) * 10 >= mutable_ctx->feature_cap * 7) {
        status = feature_cache_grow(mutable_ctx);
        if (status != RG_OK) {
            return status;
        }
    }
    slot = hash_string(grapheme, RG_FNV_OFFSET) & (mutable_ctx->feature_cap - 1);
    while (mutable_ctx->features[slot].grapheme != 0) {
        if (strcmp(mutable_ctx->features[slot].grapheme, grapheme) == 0) {
            if (mutable_ctx->features[slot].features_state < 0) {
                rg_context_note_unknown_grapheme_internal(ctx, grapheme);
                return RG_ERR_UNKNOWN_GRAPHEME;
            }
            if (mutable_ctx->features[slot].features_state > 0) {
                *out = mutable_ctx->features[slot].features;
                return RG_OK;
            }
            break;
        }
        slot = (slot + 1) & (mutable_ctx->feature_cap - 1);
    }
    status = rg_context_grapheme_features(ctx, grapheme, &features);
    if (mutable_ctx->features[slot].grapheme == 0) {
        mutable_ctx->features[slot].grapheme = rg_strdup_internal(grapheme);
        if (mutable_ctx->features[slot].grapheme == 0) {
            rg_feature_set_free(features);
            return status == RG_OK ? RG_ERR_OOM : status;
        }
        mutable_ctx->feature_count++;
    }
    if (status != RG_OK) {
        mutable_ctx->features[slot].features_state = -1;
        if (status == RG_ERR_UNKNOWN_GRAPHEME) {
            rg_context_note_unknown_grapheme_internal(ctx, grapheme);
        }
        return status;
    }
    mutable_ctx->features[slot].features_state = 1;
    mutable_ctx->features[slot].features = features;
    *out = features;
    return RG_OK;
}

/* The context-feature constraints of a grapheme, derived once and shared. The
 * alignment DP asks for these for every segment of every form it aligns, so
 * rebuilding the array per call showed up as the dominant cost. Borrowed:
 * valid while the context lives and its feature system is unchanged. */
rg_status rg_context_constraints_internal(
    const rg_context *ctx,
    const char *grapheme,
    const rg_feature_constraint **out,
    size_t *out_count
) {
    rg_context *mutable_ctx = (rg_context *)ctx;
    const rg_feature_set *features = 0;
    size_t slot;
    size_t i;
    size_t count = 0;
    rg_feature_constraint *constraints;
    rg_status status;

    if (ctx == 0 || grapheme == 0 || out == 0 || out_count == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    *out_count = 0;
    status = rg_context_features_internal(ctx, grapheme, &features);
    if (status != RG_OK) {
        return status;
    }
    slot = hash_string(grapheme, RG_FNV_OFFSET) & (mutable_ctx->feature_cap - 1);
    while (mutable_ctx->features[slot].grapheme != 0 &&
           strcmp(mutable_ctx->features[slot].grapheme, grapheme) != 0) {
        slot = (slot + 1) & (mutable_ctx->feature_cap - 1);
    }
    if (mutable_ctx->features[slot].grapheme == 0) {
        return RG_ERR_MERKMAL;
    }
    if (mutable_ctx->features[slot].constraints != 0) {
        *out = mutable_ctx->features[slot].constraints;
        *out_count = mutable_ctx->features[slot].constraint_count;
        return RG_OK;
    }
    constraints = (rg_feature_constraint *)calloc(rg_context_feature_name_count, sizeof(*constraints));
    if (constraints == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < rg_context_feature_name_count; i++) {
        size_t j;
        size_t size = rg_feature_set_size(features);
        for (j = 0; j < size; j++) {
            const char *item = rg_feature_set_get(features, j);
            if (item != 0 && strcmp(item, rg_context_feature_names[i]) == 0) {
                constraints[count].feature = rg_context_feature_names[i];
                constraints[count].value = "+";
                count++;
                break;
            }
        }
    }
    mutable_ctx->features[slot].constraints = constraints;
    mutable_ctx->features[slot].constraint_count = count;
    *out = constraints;
    *out_count = count;
    return RG_OK;
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

/* Splits a written word into segments through merkmal, merging trailing tone
 * digits into the segment they belong to. This is the only correct way to get
 * from "pater" to p/a/t/e/r: a naive character split breaks multi-codepoint
 * graphemes such as affricates, digraphs and combining diacritics. */
rg_status rg_context_segment_word(
    const rg_context *ctx,
    const char *word,
    rg_segment **out,
    size_t *out_count
) {
    mk_string_list *list = 0;
    rg_segment *segments;
    size_t count;
    size_t i;
    mk_status status;

    if (ctx == 0 || word == 0 || out == 0 || out_count == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    *out_count = 0;
    status = mk_segment_ipa_merged(word, &list);
    if (status != MK_OK) {
        return map_merkmal_status(status);
    }
    count = mk_string_list_size(list);
    if (count == 0) {
        mk_string_list_free(list);
        return RG_OK;
    }
    segments = (rg_segment *)calloc(count, sizeof(*segments));
    if (segments == 0) {
        mk_string_list_free(list);
        return RG_ERR_OOM;
    }
    for (i = 0; i < count; i++) {
        const char *item = mk_string_list_get(list, i);
        segments[i].grapheme = rg_strdup_internal(item == 0 ? "" : item);
        if (segments[i].grapheme == 0) {
            rg_segments_free(segments, i + 1);
            mk_string_list_free(list);
            return RG_ERR_OOM;
        }
    }
    mk_string_list_free(list);
    *out = segments;
    *out_count = count;
    return RG_OK;
}

void rg_segments_free(rg_segment *segments, size_t count) {
    size_t i;
    if (segments == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        free((char *)segments[i].grapheme);
        free((char *)segments[i].tone);
        free((char *)segments[i].length);
        free((char *)segments[i].stress);
    }
    free(segments);
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
