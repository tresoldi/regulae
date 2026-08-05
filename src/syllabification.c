#include "internal.h"

#include <stdlib.h>
#include <string.h>

/* Minimal sonority-based syllabification via the maximum onset principle under
 * the sonority sequencing principle. Deliberately language-agnostic; callers
 * with language-specific phonotactics populate rg_form.syllable_breaks
 * externally, and that value is respected unchanged. Mirrors syllabification.go.
 */

#define RG_SONORITY_STOP 1
#define RG_SONORITY_FRICATIVE 2
#define RG_SONORITY_NASAL 3
#define RG_SONORITY_LIQUID 4
#define RG_SONORITY_GLIDE 5
#define RG_SONORITY_VOWEL 6
#define RG_SONORITY_UNKNOWN 3

static int feature_present(const rg_feature_set *features, const char *name) {
    size_t i;
    size_t count;
    if (features == 0 || name == 0) {
        return 0;
    }
    count = rg_feature_set_size(features);
    for (i = 0; i < count; i++) {
        const char *item = rg_feature_set_get(features, i);
        if (item != 0 && strcmp(item, name) == 0) {
            return 1;
        }
    }
    return 0;
}

/* sonority_from_features applies stop < fricative < nasal < liquid < glide <
 * vowel. Non-lateral approximants are glides; lateral approximants and
 * trills/taps are liquids. An absent feature set scores neutral. */
static int sonority_from_features(const rg_feature_set *features) {
    if (features == 0) {
        return RG_SONORITY_UNKNOWN;
    }
    if (feature_present(features, "vowel")) {
        return RG_SONORITY_VOWEL;
    }
    if (feature_present(features, "approximant")) {
        if (feature_present(features, "lateral")) {
            return RG_SONORITY_LIQUID;
        }
        return RG_SONORITY_GLIDE;
    }
    if (feature_present(features, "trill") || feature_present(features, "tap")) {
        return RG_SONORITY_LIQUID;
    }
    if (feature_present(features, "nasal")) {
        return RG_SONORITY_NASAL;
    }
    if (feature_present(features, "fricative")) {
        return RG_SONORITY_FRICATIVE;
    }
    if (feature_present(features, "stop") || feature_present(features, "affricate")) {
        return RG_SONORITY_STOP;
    }
    return RG_SONORITY_UNKNOWN;
}

/* segment_sonority returns the sonority score for a segment, or -1 for
 * tone-only segments (empty grapheme) so callers can skip them. An unknown
 * grapheme scores neutral rather than failing, matching the Go bridge, which
 * maps an unresolved grapheme to a nil feature set. */
static int segment_sonority(const rg_context *ctx, const rg_segment *segment) {
    const rg_feature_set *features = 0;
    rg_status status;
    if (segment == 0 || segment->grapheme == 0 || segment->grapheme[0] == '\0') {
        return -1;
    }
    status = rg_context_features_internal(ctx, segment->grapheme, &features);
    if (status != RG_OK) {
        return RG_SONORITY_UNKNOWN;
    }
    return sonority_from_features(features);
}

/* find_nuclei fills out_nuclei with the sorted nucleus positions. Any
 * vowel-sonority segment is a nucleus; a strict sonority peak at liquid or
 * above is also a nucleus (syllabic liquids). If none are found the single
 * highest-sonority position becomes the default nucleus, so every non-empty
 * form has at least one syllable. */
static size_t find_nuclei(const int *scores, size_t n, size_t *out_nuclei) {
    size_t i;
    size_t count = 0;
    for (i = 0; i < n; i++) {
        int s = scores[i];
        if (s < 0) {
            continue;
        }
        if (s >= RG_SONORITY_VOWEL) {
            out_nuclei[count++] = i;
            continue;
        }
        if (s >= RG_SONORITY_LIQUID) {
            int left_less = (i == 0) || scores[i - 1] < 0 || scores[i - 1] < s;
            int right_less = (i + 1 == n) || scores[i + 1] < 0 || scores[i + 1] < s;
            if (left_less && right_less) {
                out_nuclei[count++] = i;
            }
        }
    }
    if (count == 0) {
        size_t best = 0;
        int best_score = -1;
        int found = 0;
        for (i = 0; i < n; i++) {
            if (scores[i] < 0) {
                continue;
            }
            if (scores[i] > best_score) {
                best_score = scores[i];
                best = i;
                found = 1;
            }
        }
        if (found) {
            out_nuclei[count++] = best;
        }
    }
    return count;
}

/* place_break_between returns the syllable-break position between two nuclei
 * under the maximum onset principle subject to sonority sequencing: the onset
 * cluster of the right syllable must have non-decreasing sonority toward the
 * nucleus, and the break is placed as far left as possible. */
static size_t place_break_between(size_t left_nucleus, size_t right_nucleus, const int *scores) {
    size_t break_pos = right_nucleus;
    int last_included = -1;
    int have_last = 0;
    size_t j = right_nucleus;
    while (j > left_nucleus + 1) {
        int s;
        j--;
        s = scores[j];
        if (s < 0) {
            continue;
        }
        if (!have_last) {
            last_included = s;
            have_last = 1;
            break_pos = j;
        } else if (s <= last_included) {
            last_included = s;
            break_pos = j;
        } else {
            break;
        }
    }
    return break_pos;
}

rg_status rg_compute_syllable_breaks_internal(
    const rg_context *ctx,
    const rg_form *form,
    size_t **out,
    size_t *out_count
) {
    int *scores = 0;
    size_t *nuclei = 0;
    size_t *breaks = 0;
    size_t nucleus_count;
    size_t i;
    size_t n;

    if (ctx == 0 || form == 0 || out == 0 || out_count == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    *out_count = 0;

    /* Caller-supplied breaks are the escape hatch for language-specific
     * phonotactics and are returned unchanged. */
    if (form->syllable_break_count > 0 && form->syllable_breaks != 0) {
        size_t *copy = (size_t *)calloc(form->syllable_break_count, sizeof(*copy));
        if (copy == 0) {
            return RG_ERR_OOM;
        }
        for (i = 0; i < form->syllable_break_count; i++) {
            int value = form->syllable_breaks[i];
            copy[i] = value < 0 ? 0 : (size_t)value;
        }
        *out = copy;
        *out_count = form->syllable_break_count;
        return RG_OK;
    }

    n = form->segment_count;
    if (n <= 1) {
        return RG_OK;
    }

    scores = (int *)calloc(n, sizeof(*scores));
    nuclei = (size_t *)calloc(n, sizeof(*nuclei));
    if (scores == 0 || nuclei == 0) {
        free(scores);
        free(nuclei);
        return RG_ERR_OOM;
    }
    for (i = 0; i < n; i++) {
        scores[i] = segment_sonority(ctx, &form->segments[i]);
    }
    nucleus_count = find_nuclei(scores, n, nuclei);
    if (nucleus_count > 1) {
        breaks = (size_t *)calloc(nucleus_count - 1, sizeof(*breaks));
        if (breaks == 0) {
            free(scores);
            free(nuclei);
            return RG_ERR_OOM;
        }
        for (i = 0; i + 1 < nucleus_count; i++) {
            breaks[i] = place_break_between(nuclei[i], nuclei[i + 1], scores);
        }
        *out = breaks;
        *out_count = nucleus_count - 1;
    }
    free(scores);
    free(nuclei);
    return RG_OK;
}

rg_status rg_compute_syllable_breaks(
    const rg_context *ctx,
    const rg_form *form,
    int **out,
    size_t *out_count
) {
    size_t *internal = 0;
    size_t count = 0;
    int *result;
    size_t i;
    rg_status status;

    if (out == 0 || out_count == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    *out_count = 0;
    status = rg_compute_syllable_breaks_internal(ctx, form, &internal, &count);
    if (status != RG_OK) {
        return status;
    }
    if (count == 0) {
        free(internal);
        return RG_OK;
    }
    result = (int *)calloc(count, sizeof(*result));
    if (result == 0) {
        free(internal);
        return RG_ERR_OOM;
    }
    for (i = 0; i < count; i++) {
        result[i] = (int)internal[i];
    }
    free(internal);
    *out = result;
    *out_count = count;
    return RG_OK;
}

void rg_syllable_breaks_free(int *breaks) {
    free(breaks);
}
