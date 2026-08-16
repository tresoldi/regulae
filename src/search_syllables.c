#include "search_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>


void syllable_data_clear(syllable_data *data) {
    size_t i;
    if (data == 0) {
        return;
    }
    if (data->syllable_features != 0) {
        for (i = 0; i < data->syllable_count; i++) {
            rg_feature_constraint_array_clear_internal(
                data->syllable_features[i],
                data->syllable_feature_counts == 0 ? 0 : data->syllable_feature_counts[i]
            );
        }
    }
    if (data->same_syllable_excluding != 0) {
        for (i = 0; i < data->segment_count; i++) {
            rg_feature_constraint_array_clear_internal(
                data->same_syllable_excluding[i],
                data->same_syllable_excluding_counts == 0 ? 0 : data->same_syllable_excluding_counts[i]
            );
        }
    }
    if (data->left_cumulative != 0) {
        for (i = 0; i <= data->segment_count; i++) {
            rg_feature_constraint_array_clear_internal(
                data->left_cumulative[i],
                data->left_cumulative_counts == 0 ? 0 : data->left_cumulative_counts[i]
            );
        }
    }
    if (data->right_cumulative != 0) {
        for (i = 0; i <= data->segment_count; i++) {
            rg_feature_constraint_array_clear_internal(
                data->right_cumulative[i],
                data->right_cumulative_counts == 0 ? 0 : data->right_cumulative_counts[i]
            );
        }
    }
    free(data->syllable_of);
    free(data->syllable_features);
    free(data->syllable_feature_counts);
    free(data->same_syllable_excluding);
    free(data->same_syllable_excluding_counts);
    if (data->preceding_at_distance != 0) {
        for (i = 0; i <= data->segment_count; i++) {
            rg_distance_constraint_array_clear_internal(
                data->preceding_at_distance[i],
                data->preceding_at_distance_counts == 0 ? 0 : data->preceding_at_distance_counts[i]
            );
        }
    }
    if (data->following_at_distance != 0) {
        for (i = 0; i <= data->segment_count; i++) {
            rg_distance_constraint_array_clear_internal(
                data->following_at_distance[i],
                data->following_at_distance_counts == 0 ? 0 : data->following_at_distance_counts[i]
            );
        }
    }
    if (data->stress != 0) {
        for (i = 0; i < data->segment_count; i++) {
            rg_feature_constraint_array_clear_internal(
                data->stress[i],
                data->stress_counts == 0 ? 0 : data->stress_counts[i]
            );
        }
    }
    free(data->left_cumulative);
    free(data->left_cumulative_counts);
    free(data->right_cumulative);
    free(data->right_cumulative_counts);
    free(data->preceding_at_distance);
    free(data->preceding_at_distance_counts);
    free(data->following_at_distance);
    free(data->following_at_distance_counts);
    free(data->stress);
    free(data->stress_counts);
    memset(data, 0, sizeof(*data));
}

/* Properties of a syllable that its segments' features do not carry, and that
 * quantity-sensitive changes are stated in: whether it ends in a coda, whether
 * its nucleus is long, and the verdict the first two are usually read for.
 *
 * `syllable_shape` and `syllable_nucleus` are facts. `syllable_weight` is not,
 * and it is here anyway, so it needs its terms stated. Heavy means a long
 * nucleus **or** a coda, which is the majority convention and the one Latin,
 * Ancient Greek, Arabic and Sanskrit metrics use. It is not universal: plenty
 * of quantity systems count CVC light, some count only CVV heavy, and a few
 * weigh the coda by its sonority. A rule reported over `syllable_weight` in a
 * language whose tradition draws the line elsewhere is a rule stated in
 * somebody else's terms, and the two facts underneath it are there so that it
 * can be restated -- `prev-syl[syllable_shape:closed]` and
 * `prev-syl[syllable_nucleus:long]` are the same partition, said without the
 * verdict.
 *
 * What buys the verdict its place is that weight is disjunctive over segments
 * and a context is a conjunction. Without the term the search states the same
 * environment as two rules, `pre[long:+]` and an equivalent of "there is a
 * coda", which is correct and is not what a metrist wants to read.
 *
 * Appended to the syllable's own feature union, so they conjoin with the
 * segment predicates through the machinery that is already there. */
static rg_status append_syllable_shape(
    const rg_feature_constraint *const *source_features,
    const size_t *source_feature_counts,
    size_t start,
    size_t end,
    const rg_feature_constraint **union_out,
    size_t *union_count
) {
    rg_feature_constraint extra[3];
    size_t extra_count = 0;
    size_t nucleus = end;
    size_t i;
    int closed;
    int is_long = 0;
    rg_feature_constraint *grown;

    for (i = start; i < end; i++) {
        size_t c;
        for (c = 0; c < source_feature_counts[i]; c++) {
            const char *f = source_features[i][c].feature;
            if (strcmp(f, "vowel") == 0 || strcmp(f, "syllabic") == 0) {
                nucleus = i;
                break;
            }
        }
    }
    closed = !(nucleus != end && nucleus + 1 == end);
    extra[extra_count].feature = "syllable_shape";
    extra[extra_count].value = closed ? "closed" : "open";
    extra_count++;
    if (nucleus != end) {
        size_t c;
        for (c = 0; c < source_feature_counts[nucleus]; c++) {
            if (strcmp(source_features[nucleus][c].feature, "long") == 0) {
                is_long = 1;
            }
        }
        extra[extra_count].feature = "syllable_nucleus";
        extra[extra_count].value = is_long ? "long" : "short";
        extra_count++;
        /* Only where a nucleus was found. A syllable with none is one this
         * corpus does not let us weigh, and guessing would put every vowelless
         * form on the heavy side of every quantity rule in the report. */
        extra[extra_count].feature = "syllable_weight";
        extra[extra_count].value = (closed || is_long) ? "heavy" : "light";
        extra_count++;
    }
    grown = (rg_feature_constraint *)calloc(*union_count + extra_count, sizeof(*grown));
    if (grown == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < *union_count; i++) {
        grown[i] = (*union_out)[i];
    }
    for (i = 0; i < extra_count; i++) {
        grown[*union_count + i].feature = rg_strdup_internal(extra[i].feature);
        grown[*union_count + i].value = rg_strdup_internal(extra[i].value);
        if (grown[*union_count + i].feature == 0 || grown[*union_count + i].value == 0) {
            free(grown);
            return RG_ERR_OOM;
        }
    }
    rg_free_owned_internal(*union_out);
    *union_out = grown;
    *union_count += extra_count;
    return RG_OK;
}

rg_status syllable_data_build(
    const rg_context *ctx,
    const rg_form *form,
    const rg_feature_constraint *const *source_features,
    const size_t *source_feature_counts,
    syllable_data *out
) {
    size_t *breaks = 0;
    size_t break_count = 0;
    size_t *starts = 0;
    size_t syllable_count;
    size_t n;
    size_t i;
    size_t s;
    rg_status status;

    if (out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    if (form == 0 || form->segment_count == 0) {
        return RG_OK;
    }
    n = form->segment_count;

    status = rg_compute_syllable_breaks_internal(ctx, form, &breaks, &break_count, 0);
    if (status != RG_OK) {
        return status;
    }

    starts = (size_t *)calloc(break_count + 2, sizeof(*starts));
    if (starts == 0) {
        free(breaks);
        return RG_ERR_OOM;
    }
    starts[0] = 0;
    for (i = 0; i < break_count; i++) {
        starts[i + 1] = breaks[i] > n ? n : breaks[i];
    }
    starts[break_count + 1] = n;
    free(breaks);
    syllable_count = break_count + 1;

    out->segment_count = n;
    out->syllable_count = syllable_count;
    out->syllable_of = (size_t *)calloc(n, sizeof(*out->syllable_of));
    out->syllable_features = (const rg_feature_constraint **)calloc(syllable_count, sizeof(*out->syllable_features));
    out->syllable_feature_counts = (size_t *)calloc(syllable_count, sizeof(*out->syllable_feature_counts));
    out->same_syllable_excluding = (const rg_feature_constraint **)calloc(n, sizeof(*out->same_syllable_excluding));
    out->same_syllable_excluding_counts = (size_t *)calloc(n, sizeof(*out->same_syllable_excluding_counts));
    if (out->syllable_of == 0 || out->syllable_features == 0 || out->syllable_feature_counts == 0 ||
        out->same_syllable_excluding == 0 || out->same_syllable_excluding_counts == 0) {
        free(starts);
        syllable_data_clear(out);
        return RG_ERR_OOM;
    }

    s = 0;
    for (i = 0; i < n; i++) {
        while (s + 1 < syllable_count && i >= starts[s + 1]) {
            s++;
        }
        out->syllable_of[i] = s;
    }

    for (s = 0; s < syllable_count; s++) {
        status = context_feature_union_copy(
            source_features,
            source_feature_counts,
            starts[s],
            starts[s + 1],
            &out->syllable_features[s],
            &out->syllable_feature_counts[s]
        );
        if (status == RG_OK) {
            status = append_syllable_shape(source_features, source_feature_counts,
                                           starts[s], starts[s + 1],
                                           &out->syllable_features[s],
                                           &out->syllable_feature_counts[s]);
        }
        if (status != RG_OK) {
            free(starts);
            syllable_data_clear(out);
            return status;
        }
    }

    for (i = 0; i < n; i++) {
        size_t idx = out->syllable_of[i];
        status = context_feature_union_copy_excluding(
            source_features,
            source_feature_counts,
            starts[idx],
            starts[idx + 1],
            i,
            1,
            &out->same_syllable_excluding[i],
            &out->same_syllable_excluding_counts[i]
        );
        if (status != RG_OK) {
            free(starts);
            syllable_data_clear(out);
            return status;
        }
    }

    out->left_cumulative = (const rg_feature_constraint **)calloc(n + 1, sizeof(*out->left_cumulative));
    out->left_cumulative_counts = (size_t *)calloc(n + 1, sizeof(*out->left_cumulative_counts));
    out->right_cumulative = (const rg_feature_constraint **)calloc(n + 1, sizeof(*out->right_cumulative));
    out->right_cumulative_counts = (size_t *)calloc(n + 1, sizeof(*out->right_cumulative_counts));
    if (out->left_cumulative == 0 || out->left_cumulative_counts == 0 ||
        out->right_cumulative == 0 || out->right_cumulative_counts == 0) {
        free(starts);
        syllable_data_clear(out);
        return RG_ERR_OOM;
    }
    for (i = 0; i <= n; i++) {
        status = context_feature_union_copy(
            source_features, source_feature_counts, 0, i,
            &out->left_cumulative[i], &out->left_cumulative_counts[i]
        );
        if (status == RG_OK) {
            status = context_feature_union_copy(
                source_features, source_feature_counts, i, n,
                &out->right_cumulative[i], &out->right_cumulative_counts[i]
            );
        }
        if (status != RG_OK) {
            free(starts);
            syllable_data_clear(out);
            return status;
        }
    }

    out->preceding_at_distance = (const rg_distance_constraint **)calloc(n + 1, sizeof(*out->preceding_at_distance));
    out->preceding_at_distance_counts = (size_t *)calloc(n + 1, sizeof(*out->preceding_at_distance_counts));
    out->following_at_distance = (const rg_distance_constraint **)calloc(n + 1, sizeof(*out->following_at_distance));
    out->following_at_distance_counts = (size_t *)calloc(n + 1, sizeof(*out->following_at_distance_counts));
    out->stress = (const rg_feature_constraint **)calloc(n, sizeof(*out->stress));
    out->stress_counts = (size_t *)calloc(n, sizeof(*out->stress_counts));
    if (out->preceding_at_distance == 0 || out->preceding_at_distance_counts == 0 ||
        out->following_at_distance == 0 || out->following_at_distance_counts == 0 ||
        out->stress == 0 || out->stress_counts == 0) {
        free(starts);
        syllable_data_clear(out);
        return RG_ERR_OOM;
    }
    for (i = 0; i <= n; i++) {
        const rg_feature_constraint *pre2 = 0;
        const rg_feature_constraint *pre3 = 0;
        const rg_feature_constraint *fol2 = 0;
        const rg_feature_constraint *fol3 = 0;
        size_t pre2_count = 0;
        size_t pre3_count = 0;
        size_t fol2_count = 0;
        size_t fol3_count = 0;
        if (i >= 2) {
            pre2 = source_features[i - 2];
            pre2_count = source_feature_counts[i - 2];
        }
        if (i >= 3) {
            pre3 = source_features[i - 3];
            pre3_count = source_feature_counts[i - 3];
        }
        if (i + 1 < n) {
            fol2 = source_features[i + 1];
            fol2_count = source_feature_counts[i + 1];
        }
        if (i + 2 < n) {
            fol3 = source_features[i + 2];
            fol3_count = source_feature_counts[i + 2];
        }
        status = distance_context_copy_two(
            pre2, pre2_count, 2, pre3, pre3_count, 3,
            &out->preceding_at_distance[i], &out->preceding_at_distance_counts[i]
        );
        if (status == RG_OK) {
            status = distance_context_copy_two(
                fol2, fol2_count, 2, fol3, fol3_count, 3,
                &out->following_at_distance[i], &out->following_at_distance_counts[i]
            );
        }
        if (status != RG_OK) {
            free(starts);
            syllable_data_clear(out);
            return status;
        }
    }
    for (i = 0; i < n; i++) {
        status = context_copy_stress(form->segments[i].stress, &out->stress[i], &out->stress_counts[i]);
        if (status != RG_OK) {
            free(starts);
            syllable_data_clear(out);
            return status;
        }
    }

    free(starts);
    return RG_OK;
}

/* Fills a context whose every slot points into the form's precomputed arrays.
 * Nothing is allocated and nothing must be cleared: the result is valid only
 * while the syllable_data and the feature matrix live, and only for reading.
 * The DP scores millions of these, so building an owned copy per cell was the
 * single largest cost in training. */

