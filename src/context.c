#include "internal.h"

#include "merkmal.h"

#include <stdbool.h>
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
/* The conditioning vocabulary: every feature name merkmal's system reports,
 * which is what a context can be stated in terms of.
 *
 * This was a hand-picked 27 until 2026-08-15, and the hand that picked it was
 * writing about Latin. It carried no rounding, no vowel nasalisation, no
 * lateral, trill, tap or retroflex, no ejective, implosive or click, no
 * breathy or creaky, no syllabicity, and no vowel height between close and
 * open -- so a change conditioned by rounding, which is the organising fact of
 * Turkic and Uralic vowel harmony, could not be found however regular it was.
 * A vocabulary that fits one family is a claim about the others.
 *
 * Which of these are worth *searching* is a separate question, answered per
 * corpus: see rg_feature_vocabulary_build_internal. This list is the
 * representation, and it stays a pure function of the grapheme so the
 * constraint cache remains valid across runs. */
const char *const rg_context_feature_names[] = {
    "vowel",
    "consonant",
    "sonorant",
    "obstruent",
    "continuant",
    "voiced",
    "voiceless",
    "nasal",
    "stop",
    "fricative",
    "affricate",
    "approximant",
    "lateral",
    "trill",
    "tap",
    "sibilant",
    "aspirated",
    "ejective",
    "implosive",
    "click",
    "breathy",
    "creaky",
    "devoiced",
    "long",
    "syllabic",
    "front",
    "back",
    "central",
    "close",
    "close-mid",
    "mid",
    "open-mid",
    "open",
    "near-open",
    "rounded",
    "unrounded",
    "nasalized",
    "labial",
    "coronal",
    "dorsal",
    "guttural",
    "labio-velar",
    "bilabial",
    "dental",
    "alveolar",
    "post-alveolar",
    "retroflex",
    "palatal",
    "velar",
    "uvular",
    "pharyngeal",
    "glottal",
    "labialized",
    "palatalized",
    "velarized",
    "pharyngealized",
    "anterior",
    "non-anterior",
    "distributed",
    "non-distributed",
    "consonantal",
    "vocoid",
    "non-continuant",
    "non-pulmonic",
    "dorsal-closure"
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
    /* Why the lookup was refused, so a cache hit repeats the reason rather
     * than flattening every refusal to "unknown grapheme". */
    rg_status refusal;
    int is_segment_state;
    int is_segment;
} feature_cache_entry;

typedef struct distance_cache_entry {
    char *a;
    char *b;
    double value;
    int resolved;
} distance_cache_entry;

/* The feature displacement between two graphemes: which features one has and
 * the other lacks. A pure function of the pair, and the alignment DP asks for
 * it on every candidate link it costs, so without this it is recomputed
 * millions of times per training run -- it was 20% of the profile, and the
 * richer feature bundles merkmal 1.0 brought made it worse. The feature names
 * are borrowed from the cached feature sets, which outlive the entry, so a
 * cached displacement allocates nothing per lookup. */
typedef struct displacement_cache_entry {
    char *a;
    char *b;
    rg_feature_displacement *items;
    size_t count;
    int resolved;
} displacement_cache_entry;

struct rg_context {
    mk_registry *registry;
    const mk_system *system;
    /* The grapheme that last failed to resolve. The Go reference carries this
     * on its error value; a C status code cannot, and "unknown grapheme" with
     * no indication of which one is unactionable on a corpus of any size. */
    char *unknown_grapheme;
    rg_segmentation segmentation;
    rg_grapheme_diagnosis last_diagnosis;
    int has_diagnosis;
    feature_cache_entry *features;
    size_t feature_count;
    size_t feature_cap;
    distance_cache_entry *distances;
    size_t distance_count;
    size_t distance_cap;
    displacement_cache_entry *displacements;
    size_t displacement_count;
    size_t displacement_cap;
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
    for (i = 0; i < ctx->displacement_cap; i++) {
        if (ctx->displacements[i].a != 0) {
            free(ctx->displacements[i].a);
            free(ctx->displacements[i].b);
            /* The feature names are borrowed from the feature-set cache; only
             * the array of pairs is ours. */
            free(ctx->displacements[i].items);
        }
    }
    free(ctx->displacements);
    ctx->displacements = 0;
    ctx->displacement_cap = 0;
    ctx->displacement_count = 0;
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

static rg_status displacement_cache_grow(rg_context *ctx) {
    size_t next_cap = ctx->displacement_cap == 0 ? 256 : ctx->displacement_cap * 2;
    displacement_cache_entry *next =
        (displacement_cache_entry *)calloc(next_cap, sizeof(*next));
    size_t i;
    if (next == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < ctx->displacement_cap; i++) {
        size_t slot;
        if (ctx->displacements[i].a == 0) {
            continue;
        }
        slot = hash_string(ctx->displacements[i].b,
                           hash_string(ctx->displacements[i].a, RG_FNV_OFFSET)) & (next_cap - 1);
        while (next[slot].a != 0) {
            slot = (slot + 1) & (next_cap - 1);
        }
        next[slot] = ctx->displacements[i];
    }
    free(ctx->displacements);
    ctx->displacements = next;
    ctx->displacement_cap = next_cap;
    return RG_OK;
}

struct rg_feature_set {
    mk_string_list *inner;
};

static rg_status map_merkmal_status(mk_status status) {
    switch (status) {
    case MK_OK:
        return RG_OK;
    case MK_ERR_INVALID_ARGUMENT:
        return RG_ERR_INVALID_ARGUMENT;
    case MK_ERR_UNKNOWN_GRAPHEME:
        return RG_ERR_UNKNOWN_GRAPHEME;
    /* A CLDF/CLTS marker is a documented gap in the source, not a sound the
     * feature system is missing. Folding it into RG_ERR_UNKNOWN_GRAPHEME told
     * a user to widen their transcription when the honest answer is that the
     * dataset never transcribed that token. */
    case MK_ERR_SOURCE_MARKER:
        return RG_ERR_SOURCE_MARKER;
    case MK_ERR_PARSE:
        return RG_ERR_PARSE;
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
    status = mk_registry_get_system(ctx->registry, RG_DEFAULT_FEATURE_SYSTEM, &ctx->system);
    if (status != MK_OK) {
        mk_registry_free(ctx->registry);
        free(ctx);
        return map_merkmal_status(status);
    }
    *out = ctx;
    return RG_OK;
}

rg_status rg_context_set_segmentation(rg_context *ctx, rg_segmentation mode) {
    if (ctx == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    if (mode != RG_SEGMENT_ORTHOGRAPHIC && mode != RG_SEGMENT_SYSTEM_LONGEST_MATCH) {
        return RG_ERR_UNSUPPORTED_OPTION;
    }
    ctx->segmentation = mode;
    return RG_OK;
}

rg_segmentation rg_context_segmentation(const rg_context *ctx) {
    return ctx == 0 ? RG_SEGMENT_ORTHOGRAPHIC : ctx->segmentation;
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
            bool recognised = false;
            status = mk_system_is_segment(ctx->system, grapheme, &recognised);
            *out = recognised ? 1 : 0;
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
    {
        bool recognised = false;
        status = mk_system_is_segment(ctx->system, grapheme, &recognised);
        if (status != MK_OK) {
            return map_merkmal_status(status);
        }
        *out = recognised ? 1 : 0;
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


/* Distinct graphemes in a form, accumulated into a sorted unique list. */
static rg_status vocabulary_collect_form(
    const rg_form *form,
    char ***graphemes,
    size_t *count,
    size_t *cap
) {
    size_t j;
    for (j = 0; j < form->segment_count; j++) {
        const char *grapheme = form->segments[j].grapheme;
        size_t low = 0;
        size_t high = *count;
        if (grapheme == 0) {
            continue;
        }
        while (low < high) {
            size_t mid = low + (high - low) / 2;
            int c = strcmp((*graphemes)[mid], grapheme);
            if (c == 0) {
                low = *count + 1;
                break;
            }
            if (c < 0) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        if (low > *count) {
            continue;
        }
        if (*count == *cap) {
            size_t next_cap = *cap == 0 ? 32 : *cap * 2;
            char **next = (char **)realloc(*graphemes, next_cap * sizeof(*next));
            if (next == 0) {
                return RG_ERR_OOM;
            }
            *graphemes = next;
            *cap = next_cap;
        }
        memmove(&(*graphemes)[low + 1], &(*graphemes)[low], (*count - low) * sizeof(**graphemes));
        (*graphemes)[low] = rg_strdup_internal(grapheme);
        if ((*graphemes)[low] == 0) {
            return RG_ERR_OOM;
        }
        (*count)++;
    }
    return RG_OK;
}

/* Decides the searchable vocabulary from the corpus's own segment inventory.
 *
 * Two filters, and the second matters as much as the first.
 *
 * Contrastive: some segment carries the feature and some does not. A feature
 * nothing carries is dead weight; a feature everything carries is the
 * predicate that partitions nothing, which is a documented way to commit a
 * rule on no evidence.
 *
 * Distinct: no two features that pick out exactly the same segments both stay.
 * merkmal's vocabulary is not orthogonal -- `vowel`, `vocoid` and `syllabic`
 * separate the same segments in most corpora, `stop` and `non-continuant`
 * almost always do -- and keeping all of them widens the argmax without
 * widening what can be found, then reports the environment under whichever
 * synonym the search happened to reach first. Equivalence is a fact about this
 * corpus: features that coincide in Latin come apart in a language that
 * contrasts syllabic consonants, and there they are kept separately.
 *
 * The survivor of a tie is the earliest in rg_context_feature_names, which is
 * ordered so the name a linguist would reach for comes first. */
static rg_status vocabulary_from_graphemes(
    const rg_context *ctx,
    char *const *graphemes,
    size_t grapheme_count,
    rg_feature_vocabulary *out
) {
    unsigned char *masks;
    size_t f;
    size_t kept = 0;
    size_t *kept_index;

    out->contrastive = (unsigned char *)calloc(rg_context_feature_name_count, sizeof(*out->contrastive));
    masks = (unsigned char *)calloc(rg_context_feature_name_count * (grapheme_count == 0 ? 1 : grapheme_count),
                                    sizeof(*masks));
    kept_index = (size_t *)calloc(rg_context_feature_name_count, sizeof(*kept_index));
    if (out->contrastive == 0 || masks == 0 || kept_index == 0) {
        free(out->contrastive);
        free(masks);
        free(kept_index);
        out->contrastive = 0;
        return RG_ERR_OOM;
    }
    for (f = 0; f < grapheme_count; f++) {
        const rg_feature_constraint *constraints = 0;
        size_t constraint_count = 0;
        size_t c;
        if (rg_context_constraints_internal(ctx, graphemes[f], &constraints, &constraint_count) != RG_OK) {
            continue;
        }
        for (c = 0; c < constraint_count; c++) {
            size_t k;
            for (k = 0; k < rg_context_feature_name_count; k++) {
                if (strcmp(constraints[c].feature, rg_context_feature_names[k]) == 0) {
                    masks[k * grapheme_count + f] = 1;
                    break;
                }
            }
        }
    }
    for (f = 0; f < rg_context_feature_name_count; f++) {
        const unsigned char *mask = &masks[f * (grapheme_count == 0 ? 1 : grapheme_count)];
        size_t carriers = 0;
        size_t g;
        size_t k;
        int duplicate = 0;
        for (g = 0; g < grapheme_count; g++) {
            carriers += mask[g];
        }
        if (carriers == 0 || carriers == grapheme_count) {
            continue;
        }
        for (k = 0; k < kept && !duplicate; k++) {
            const unsigned char *other = &masks[kept_index[k] * grapheme_count];
            duplicate = memcmp(mask, other, grapheme_count) == 0;
        }
        if (duplicate) {
            continue;
        }
        kept_index[kept++] = f;
        out->contrastive[f] = 1;
        out->contrastive_count++;
    }
    free(masks);
    free(kept_index);
    return RG_OK;
}

static void vocabulary_graphemes_free(char **graphemes, size_t count) {
    size_t i;
    for (i = 0; i < count; i++) {
        free(graphemes[i]);
    }
    free(graphemes);
}

rg_status rg_feature_vocabulary_build_internal(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    rg_feature_vocabulary *out
) {
    char **graphemes = 0;
    size_t count = 0;
    size_t cap = 0;
    size_t i;
    rg_status status = RG_OK;

    if (ctx == 0 || out == 0 || (pair_count > 0 && pairs == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    out->contrastive = 0;
    out->contrastive_count = 0;
    for (i = 0; i < pair_count && status == RG_OK; i++) {
        status = vocabulary_collect_form(&pairs[i].source, &graphemes, &count, &cap);
        if (status == RG_OK) {
            status = vocabulary_collect_form(&pairs[i].target, &graphemes, &count, &cap);
        }
    }
    if (status == RG_OK) {
        status = vocabulary_from_graphemes(ctx, graphemes, count, out);
    }
    vocabulary_graphemes_free(graphemes, count);
    return status;
}

rg_status rg_feature_vocabulary_build_from_sets_internal(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    rg_feature_vocabulary *out
) {
    char **graphemes = 0;
    size_t count = 0;
    size_t cap = 0;
    size_t i;
    rg_status status = RG_OK;

    if (ctx == 0 || out == 0 || (cognate_count > 0 && cognates == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    out->contrastive = 0;
    out->contrastive_count = 0;
    for (i = 0; i < cognate_count && status == RG_OK; i++) {
        size_t j;
        for (j = 0; j < cognates[i].form_count && status == RG_OK; j++) {
            status = vocabulary_collect_form(&cognates[i].forms[j].form, &graphemes, &count, &cap);
        }
    }
    if (status == RG_OK) {
        status = vocabulary_from_graphemes(ctx, graphemes, count, out);
    }
    vocabulary_graphemes_free(graphemes, count);
    return status;
}

void rg_feature_vocabulary_clear_internal(rg_feature_vocabulary *vocabulary) {
    if (vocabulary == 0) {
        return;
    }
    free(vocabulary->contrastive);
    vocabulary->contrastive = 0;
    vocabulary->contrastive_count = 0;
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

rg_status rg_context_diagnose(
    const rg_context *ctx,
    const char *grapheme,
    rg_grapheme_diagnosis *out
) {
    mk_diagnosis diagnosis;
    mk_status status;
    if (ctx == 0 || grapheme == 0 || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    memset(&diagnosis, 0, sizeof(diagnosis));
    status = mk_system_diagnose(ctx->system, grapheme, &diagnosis);
    if (status != MK_OK) {
        return map_merkmal_status(status);
    }
    out->status = map_merkmal_status(diagnosis.status);
    out->valid_prefix_bytes = diagnosis.valid_prefix_bytes;
    out->offending_offset = diagnosis.offending_offset;
    memcpy(out->offending, diagnosis.offending, sizeof(out->offending));
    out->offending[sizeof(out->offending) - 1] = '\0';
    return RG_OK;
}

int rg_context_last_diagnosis(const rg_context *ctx, rg_grapheme_diagnosis *out) {
    if (ctx == 0 || out == 0 || !ctx->has_diagnosis) {
        return 0;
    }
    *out = ctx->last_diagnosis;
    return 1;
}

void rg_context_note_unknown_grapheme_internal(const rg_context *ctx, const char *grapheme) {
    rg_context *mutable_ctx = (rg_context *)ctx;
    if (ctx == 0 || grapheme == 0) {
        return;
    }
    free(mutable_ctx->unknown_grapheme);
    mutable_ctx->unknown_grapheme = rg_strdup_internal(grapheme);
    mutable_ctx->has_diagnosis =
        rg_context_diagnose(ctx, grapheme, &mutable_ctx->last_diagnosis) == RG_OK;
}

/* The status a refused grapheme deserves. merkmal separates a sound it does
 * not cover from CLDF markup that never transcribed a sound, and the two ask
 * different things of the user, so the refusal carries the distinction out
 * rather than flattening everything to "unknown grapheme". */
rg_status rg_context_refusal_status_internal(const rg_context *ctx, const char *grapheme) {
    rg_context_note_unknown_grapheme_internal(ctx, grapheme);
    if (ctx != 0 && ctx->has_diagnosis && ctx->last_diagnosis.status != RG_OK) {
        return ctx->last_diagnosis.status;
    }
    return RG_ERR_UNKNOWN_GRAPHEME;
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
                return mutable_ctx->features[slot].refusal;
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
        mutable_ctx->features[slot].refusal = status;
        if (status == RG_ERR_UNKNOWN_GRAPHEME || status == RG_ERR_SOURCE_MARKER ||
            status == RG_ERR_PARSE) {
            rg_context_note_unknown_grapheme_internal(ctx, grapheme);
        }
        return status;
    }
    mutable_ctx->features[slot].features_state = 1;
    mutable_ctx->features[slot].features = features;
    *out = features;
    return RG_OK;
}

static int displacement_feature_cmp(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static int feature_set_has(const rg_feature_set *features, const char *name) {
    size_t i;
    for (i = 0; i < rg_feature_set_size(features); i++) {
        const char *item = rg_feature_set_get(features, i);
        if (item != 0 && strcmp(item, name) == 0) {
            return 1;
        }
    }
    return 0;
}

/* Features of `left` that `right` lacks, appended in sorted order so the
 * displacement of a pair is one canonical vector rather than one per
 * enumeration order -- the learned table keys on the whole vector. */
static rg_status append_missing(
    const rg_feature_set *left,
    const rg_feature_set *right,
    const char *from_value,
    const char *to_value,
    rg_feature_displacement **items,
    size_t *count,
    size_t *cap
) {
    const char **names;
    size_t name_count = 0;
    size_t size = rg_feature_set_size(left);
    size_t i;

    if (size == 0) {
        return RG_OK;
    }
    names = (const char **)calloc(size, sizeof(*names));
    if (names == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < size; i++) {
        const char *name = rg_feature_set_get(left, i);
        if (name != 0 && !feature_set_has(right, name)) {
            names[name_count++] = name;
        }
    }
    qsort(names, name_count, sizeof(*names), displacement_feature_cmp);
    for (i = 0; i < name_count; i++) {
        if (*count == *cap) {
            size_t next_cap = *cap == 0 ? 8 : *cap * 2;
            rg_feature_displacement *next = (rg_feature_displacement *)realloc(
                *items, next_cap * sizeof(**items));
            if (next == 0) {
                free(names);
                return RG_ERR_OOM;
            }
            *items = next;
            *cap = next_cap;
        }
        /* Borrowed: the name belongs to the cached feature set, and the values
         * are literals. Nothing here is freed with the entry. */
        (*items)[*count].feature = names[i];
        (*items)[*count].from_value = from_value;
        (*items)[*count].to_value = to_value;
        (*count)++;
    }
    free(names);
    return RG_OK;
}

/* The feature displacement of a grapheme pair, derived once and shared.
 * Borrowed: valid while the context lives and its feature system is unchanged.
 * The alignment DP costs every candidate link with this, so it is asked for
 * far more often than there are distinct pairs to ask about. */
rg_status rg_context_displacement_internal(
    const rg_context *ctx,
    const char *source,
    const char *target,
    const rg_feature_displacement **out,
    size_t *out_count
) {
    rg_context *mutable_ctx = (rg_context *)ctx;
    const rg_feature_set *source_features = 0;
    const rg_feature_set *target_features = 0;
    rg_feature_displacement *items = 0;
    size_t count = 0;
    size_t cap = 0;
    size_t slot;
    rg_status status;

    if (ctx == 0 || source == 0 || target == 0 || out == 0 || out_count == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    *out_count = 0;
    if (mutable_ctx->displacement_cap == 0 ||
        (mutable_ctx->displacement_count + 1) * 10 >= mutable_ctx->displacement_cap * 7) {
        status = displacement_cache_grow(mutable_ctx);
        if (status != RG_OK) {
            return status;
        }
    }
    slot = hash_string(target, hash_string(source, RG_FNV_OFFSET)) &
           (mutable_ctx->displacement_cap - 1);
    while (mutable_ctx->displacements[slot].a != 0) {
        if (strcmp(mutable_ctx->displacements[slot].a, source) == 0 &&
            strcmp(mutable_ctx->displacements[slot].b, target) == 0) {
            *out = mutable_ctx->displacements[slot].items;
            *out_count = mutable_ctx->displacements[slot].count;
            return RG_OK;
        }
        slot = (slot + 1) & (mutable_ctx->displacement_cap - 1);
    }
    status = rg_context_features_internal(ctx, source, &source_features);
    if (status != RG_OK) {
        return status;
    }
    status = rg_context_features_internal(ctx, target, &target_features);
    if (status != RG_OK) {
        return status;
    }
    status = append_missing(source_features, target_features, "present", "absent",
                            &items, &count, &cap);
    if (status == RG_OK) {
        status = append_missing(target_features, source_features, "absent", "present",
                                &items, &count, &cap);
    }
    if (status != RG_OK) {
        free(items);
        return status;
    }
    mutable_ctx->displacements[slot].a = rg_strdup_internal(source);
    mutable_ctx->displacements[slot].b = rg_strdup_internal(target);
    if (mutable_ctx->displacements[slot].a == 0 || mutable_ctx->displacements[slot].b == 0) {
        free(mutable_ctx->displacements[slot].a);
        free(mutable_ctx->displacements[slot].b);
        mutable_ctx->displacements[slot].a = 0;
        mutable_ctx->displacements[slot].b = 0;
        free(items);
        return RG_ERR_OOM;
    }
    mutable_ctx->displacements[slot].items = items;
    mutable_ctx->displacements[slot].count = count;
    mutable_ctx->displacements[slot].resolved = 1;
    mutable_ctx->displacement_count++;
    *out = items;
    *out_count = count;
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
    mk_string_list *inner = 0;
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
        mk_string_list_free(inner);
        return RG_ERR_OOM;
    }
    features->inner = inner;
    *out = features;
    return RG_OK;
}

/* Appends the segments of one stress-free piece of a word. Tone leaves the
 * grapheme here: "ma³³" gives m and a carrying ³³, and a token that is nothing
 * but tone attaches to the segment before it, which is how CLDF wordlists
 * publish it. */
static rg_status append_word_piece(
    const rg_context *ctx,
    const char *piece,
    rg_segment **segments,
    size_t *count,
    size_t *cap
) {
    mk_string_list *list = 0;
    size_t n;
    size_t i;
    mk_status status;

    status = ctx->segmentation == RG_SEGMENT_SYSTEM_LONGEST_MATCH
                 ? mk_system_segment_ipa(ctx->system, piece, &list)
                 : mk_segment_ipa_merged(piece, &list);
    if (status != MK_OK) {
        return map_merkmal_status(status);
    }
    n = mk_string_list_size(list);
    for (i = 0; i < n; i++) {
        const char *item = mk_string_list_get(list, i);
        char *base = 0;
        char *tone = 0;
        const char *grapheme;

        if (item == 0) {
            item = "";
        }
        if (mk_split_tone(item, &base, &tone) == MK_ERR_UNKNOWN_GRAPHEME &&
            *count > 0 && (*segments)[*count - 1].tone == 0) {
            (*segments)[*count - 1].tone = rg_strdup_internal(item);
            if ((*segments)[*count - 1].tone == 0) {
                mk_string_list_free(list);
                return RG_ERR_OOM;
            }
            continue;
        }
        if (*count == *cap) {
            size_t next_cap = *cap == 0 ? 8 : *cap * 2;
            rg_segment *next = (rg_segment *)realloc(*segments, next_cap * sizeof(**segments));
            if (next == 0) {
                mk_string_free(base);
                mk_string_free(tone);
                mk_string_list_free(list);
                return RG_ERR_OOM;
            }
            *segments = next;
            *cap = next_cap;
        }
        memset(&(*segments)[*count], 0, sizeof((*segments)[*count]));
        grapheme = base != 0 ? base : item;
        (*segments)[*count].grapheme = rg_strdup_internal(grapheme);
        if ((*segments)[*count].grapheme == 0) {
            mk_string_free(base);
            mk_string_free(tone);
            mk_string_list_free(list);
            return RG_ERR_OOM;
        }
        if (tone != 0 && tone[0] != '\0') {
            (*segments)[*count].tone = rg_strdup_internal(tone);
            if ((*segments)[*count].tone == 0) {
                mk_string_free(base);
                mk_string_free(tone);
                (*count)++;
                mk_string_list_free(list);
                return RG_ERR_OOM;
            }
        }
        mk_string_free(base);
        mk_string_free(tone);
        (*count)++;
    }
    mk_string_list_free(list);
    return RG_OK;
}

static int segment_is_vowel(const rg_context *ctx, const char *grapheme) {
    const rg_feature_set *features = 0;
    size_t i;
    if (rg_context_features_internal(ctx, grapheme, &features) != RG_OK) {
        return 0;
    }
    for (i = 0; i < rg_feature_set_size(features); i++) {
        const char *item = rg_feature_set_get(features, i);
        if (item != 0 && strcmp(item, "vowel") == 0) {
            return 1;
        }
    }
    return 0;
}

/* Splits a written word into segments through merkmal. This is the only
 * correct way to get from "pater" to p/a/t/e/r: a naive character split breaks
 * multi-codepoint graphemes such as affricates, digraphs and combining
 * diacritics. Which cut is taken is the context's rg_segmentation setting; the
 * default reads the tie bar and leaves untied sequences apart.
 *
 * The word is cut at its stress marks first, and each piece segmented on its
 * own. Two things make that necessary rather than tidy. merkmal resolves "ˈp"
 * and "aˈ" as graphemes, so a mark left in place becomes part of a segment --
 * and a stressed segment that is a different segment from its unstressed self
 * splits every correspondence it takes part in. And the orthographic tokeniser
 * attaches a mid-word mark to the vowel *before* it, which is the opposite of
 * what the notation means: "paˈter" is stress on "ter".
 *
 * The mark stands before a syllable but stress is realised on its nucleus, and
 * that is where the conditioning needs it. Verner's Law turns on whether the
 * preceding vowel carried the accent, which a mark sitting on an onset cannot
 * answer. */
rg_status rg_context_segment_word(
    const rg_context *ctx,
    const char *word,
    rg_segment **out,
    size_t *out_count
) {
    rg_segment *segments = 0;
    size_t count = 0;
    size_t cap = 0;
    const char *cursor;
    const char *pending_stress = 0;
    rg_status status = RG_OK;

    if (ctx == 0 || word == 0 || out == 0 || out_count == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    *out_count = 0;

    cursor = word;
    while (*cursor != '\0' && status == RG_OK) {
        const char *mark = cursor;
        const char *stress = 0;
        size_t length;
        char *piece;
        size_t first = count;

        /* Find the next mark, which ends this piece. */
        while (*mark != '\0') {
            if (mark[0] == '\xcb' && (mark[1] == '\x88' || mark[1] == '\x8c')) {
                stress = mark[1] == '\x88' ? "primary" : "secondary";
                break;
            }
            mark++;
        }
        length = (size_t)(mark - cursor);
        if (length > 0) {
            piece = (char *)malloc(length + 1);
            if (piece == 0) {
                rg_segments_free(segments, count);
                return RG_ERR_OOM;
            }
            memcpy(piece, cursor, length);
            piece[length] = '\0';
            status = append_word_piece(ctx, piece, &segments, &count, &cap);
            free(piece);
        }
        if (status == RG_OK && pending_stress != 0) {
            size_t i;
            for (i = first; i < count; i++) {
                if (segment_is_vowel(ctx, segments[i].grapheme)) {
                    segments[i].stress = rg_strdup_internal(pending_stress);
                    if (segments[i].stress == 0) {
                        status = RG_ERR_OOM;
                    }
                    break;
                }
            }
            pending_stress = 0;
        }
        if (stress == 0) {
            break;
        }
        pending_stress = stress;
        cursor = mark + 2;
    }
    /* A mark with nothing after it stresses nothing; the word simply ends. */
    if (status == RG_OK && pending_stress != 0) {
        size_t i;
        for (i = 0; i < count; i++) {
            if (segments[i].stress == 0 && segment_is_vowel(ctx, segments[i].grapheme)) {
                segments[i].stress = rg_strdup_internal(pending_stress);
                break;
            }
        }
    }
    if (status != RG_OK) {
        rg_segments_free(segments, count);
        return status;
    }
    if (count == 0) {
        rg_segments_free(segments, 0);
        return RG_OK;
    }
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
    return mk_string_list_size(features->inner);
}

const char *rg_feature_set_get(const rg_feature_set *features, size_t index) {
    if (features == 0) {
        return 0;
    }
    return mk_string_list_get(features->inner, index);
}

void rg_feature_set_free(rg_feature_set *features) {
    if (features == 0) {
        return;
    }
    mk_string_list_free(features->inner);
    free(features);
}
