#include "internal.h"

#include <stdlib.h>
#include <string.h>

/* The searchable feature vocabulary: which features actually contrast in a
 * given corpus, and what to call them.
 *
 * This lived in context.c, the merkmal bridge, because it needs a grapheme's
 * features and the bridge is the only thing that may ask merkmal for them. But
 * it never calls merkmal itself -- it goes through
 * rg_context_constraints_internal like any other caller -- and it is not a
 * translation. It is a search: over the graphemes a corpus actually uses, for
 * the features that distinguish them, discarding those that duplicate a
 * distinction already kept.
 *
 * There is no list of feature names in regulae, and there must not be. A
 * grapheme's features are whatever the merkmal system in use reports, and the
 * searchable vocabulary is derived per corpus, here. The hand-picked list that
 * stood until 2026-08-15 had no term for rounding, so a perfectly regular
 * rounding harmony produced no conditioned class at all.
 */

/* A readability preference, and nothing more.
 *
 * When two predicates separate a corpus identically the search cannot tell
 * them apart, and which name gets printed is arbitrary. This orders the names
 * merkmal's categorical systems use so that the one a linguist would reach for
 * wins, and so that a cover term beats its own subtype -- `coronal` before
 * `alveolar`, `labial` before `bilabial` -- because when a corpus cannot
 * distinguish two environments the weaker claim is the honest report.
 *
 * A name absent from this list is not penalised: unlisted names simply order
 * after listed ones, alphabetically. Nothing here decides what can be found,
 * only what a tie is called, so a system with an entirely different vocabulary
 * loses readability and no capability. */
static const char *const feature_name_preference[] = {
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
static const size_t feature_name_preference_count =
    sizeof(feature_name_preference) / sizeof(feature_name_preference[0]);

/* Where a name sorts. Listed names keep their order; unlisted ones follow. */
static size_t feature_preference_rank(const char *name) {
    size_t i;
    for (i = 0; i < feature_name_preference_count; i++) {
        if (strcmp(feature_name_preference[i], name) == 0) {
            return i;
        }
    }
    return feature_name_preference_count;
}

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

/* Every (feature, value) pair any of these graphemes carries, once each, then
 * filtered to the ones worth searching. See rg_feature_vocabulary in
 * internal.h for what the two filters are and why. */
static rg_status vocabulary_from_graphemes(
    const rg_context *ctx,
    char *const *graphemes,
    size_t grapheme_count,
    rg_feature_vocabulary *out
) {
    rg_feature_constraint *universe = 0;
    size_t universe_count = 0;
    size_t universe_cap = 0;
    unsigned char *masks = 0;
    size_t *kept = 0;
    size_t kept_count = 0;
    size_t g;
    size_t f;
    rg_status status = RG_OK;

    for (g = 0; g < grapheme_count; g++) {
        const rg_feature_constraint *constraints = 0;
        size_t constraint_count = 0;
        size_t c;
        if (rg_context_constraints_internal(ctx, graphemes[g], &constraints, &constraint_count) != RG_OK) {
            continue;
        }
        for (c = 0; c < constraint_count; c++) {
            size_t u;
            int seen = 0;
            for (u = 0; u < universe_count && !seen; u++) {
                seen = strcmp(universe[u].feature, constraints[c].feature) == 0 &&
                       strcmp(universe[u].value, constraints[c].value) == 0;
            }
            if (seen) {
                continue;
            }
            if (universe_count == universe_cap) {
                size_t next_cap = universe_cap == 0 ? 32 : universe_cap * 2;
                rg_feature_constraint *next =
                    (rg_feature_constraint *)realloc(universe, next_cap * sizeof(*next));
                if (next == 0) {
                    free(universe);
                    return RG_ERR_OOM;
                }
                universe = next;
                universe_cap = next_cap;
            }
            universe[universe_count] = constraints[c];
            universe_count++;
        }
    }
    /* Sorted before anything is decided, so which synonym survives a tie is a
     * property of the vocabulary rather than of the order graphemes happened
     * to appear in the corpus. */
    for (f = 1; f < universe_count; f++) {
        rg_feature_constraint key = universe[f];
        size_t rank = feature_preference_rank(key.feature);
        size_t j = f;
        while (j > 0) {
            size_t other = feature_preference_rank(universe[j - 1].feature);
            int c = other < rank ? -1 : (other > rank ? 1 : strcmp(universe[j - 1].feature, key.feature));
            if (c == 0) {
                c = strcmp(universe[j - 1].value, key.value);
            }
            if (c <= 0) {
                break;
            }
            universe[j] = universe[j - 1];
            j--;
        }
        universe[j] = key;
    }

    masks = (unsigned char *)calloc((universe_count == 0 ? 1 : universe_count) *
                                    (grapheme_count == 0 ? 1 : grapheme_count), sizeof(*masks));
    kept = (size_t *)calloc(universe_count == 0 ? 1 : universe_count, sizeof(*kept));
    if (masks == 0 || kept == 0) {
        free(universe);
        free(masks);
        free(kept);
        return RG_ERR_OOM;
    }
    for (g = 0; g < grapheme_count; g++) {
        const rg_feature_constraint *constraints = 0;
        size_t constraint_count = 0;
        size_t c;
        if (rg_context_constraints_internal(ctx, graphemes[g], &constraints, &constraint_count) != RG_OK) {
            continue;
        }
        for (c = 0; c < constraint_count; c++) {
            for (f = 0; f < universe_count; f++) {
                if (strcmp(universe[f].feature, constraints[c].feature) == 0 &&
                    strcmp(universe[f].value, constraints[c].value) == 0) {
                    masks[f * grapheme_count + g] = 1;
                    break;
                }
            }
        }
    }
    for (f = 0; f < universe_count; f++) {
        const unsigned char *mask = &masks[f * (grapheme_count == 0 ? 1 : grapheme_count)];
        size_t carriers = 0;
        size_t k;
        int duplicate = 0;
        for (g = 0; g < grapheme_count; g++) {
            carriers += mask[g];
        }
        if (carriers == 0 || carriers == grapheme_count) {
            continue;
        }
        for (k = 0; k < kept_count && !duplicate; k++) {
            duplicate = memcmp(mask, &masks[kept[k] * grapheme_count], grapheme_count) == 0;
        }
        if (!duplicate) {
            kept[kept_count++] = f;
        }
    }
    out->entries = (rg_feature_constraint *)calloc(kept_count == 0 ? 1 : kept_count, sizeof(*out->entries));
    if (out->entries == 0) {
        status = RG_ERR_OOM;
    }
    if (status == RG_OK) {
        size_t k;
        for (k = 0; k < kept_count; k++) {
            out->entries[k].feature = rg_strdup_internal(universe[kept[k]].feature);
            out->entries[k].value = rg_strdup_internal(universe[kept[k]].value);
            if (out->entries[k].feature == 0 || out->entries[k].value == 0) {
                status = RG_ERR_OOM;
                break;
            }
            out->count = k + 1;
        }
    }
    free(universe);
    free(masks);
    free(kept);
    if (status != RG_OK) {
        rg_feature_vocabulary_clear_internal(out);
    }
    return status;
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
    out->entries = 0;
    out->count = 0;
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
    out->entries = 0;
    out->count = 0;
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
    size_t i;
    if (vocabulary == 0) {
        return;
    }
    /* NOLINTNEXTLINE(clang-analyzer-core.NullDereference): entries is null only
     * when count is zero, which the loop then does not enter. The analyzer does
     * not relate the two, because they are set together by the append helper
     * and it does not follow that far. */
    for (i = 0; i < vocabulary->count; i++) {
        /* NOLINTNEXTLINE(clang-analyzer-core.NullDereference) */
        rg_free_owned_internal(vocabulary->entries[i].feature);
        rg_free_owned_internal(vocabulary->entries[i].value);
    }
    free(vocabulary->entries);
    vocabulary->entries = 0;
    vocabulary->count = 0;
}
