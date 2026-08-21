#include "search_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Where a segment sits in its own morpheme, and which morpheme that is.
 *
 * Boundaries are indices into the segment sequence: a break at i means a new
 * morpheme starts at i. A form with no boundaries reports neither value, so
 * the axis simply does not exist for corpora that do not carry them -- which
 * is most of them -- and the contrastive filter drops it.
 *
 * The names are borrowed from static storage and the index from a small table,
 * so a borrowed context owns nothing here. Words longer than the table are
 * reported as being in its last slot rather than not at all: a rule about the
 * ninth morpheme of a word is not one this is going to find. */
static const char *const morpheme_index_names[] = {
    "0", "1", "2", "3", "4", "5", "6", "7"
};

void morpheme_placement(
    const rg_form *form,
    size_t start,
    size_t end,
    const char **out_position,
    const char **out_index
) {
    size_t index = 0;
    size_t morpheme_start = 0;
    size_t morpheme_end = form->segment_count;
    size_t i;

    *out_position = 0;
    *out_index = 0;
    if (form->morpheme_break_count == 0 || form->morpheme_breaks == 0) {
        return;
    }
    for (i = 0; i < form->morpheme_break_count; i++) {
        size_t at;
        if (form->morpheme_breaks[i] < 0) {
            continue;
        }
        at = (size_t)form->morpheme_breaks[i];
        if (at <= start) {
            if (at > morpheme_start) {
                morpheme_start = at;
            }
            if (at > 0) {
                index++;
            }
        } else if (at < morpheme_end) {
            morpheme_end = at;
        }
    }
    if (index >= sizeof(morpheme_index_names) / sizeof(morpheme_index_names[0])) {
        index = sizeof(morpheme_index_names) / sizeof(morpheme_index_names[0]) - 1;
    }
    *out_index = morpheme_index_names[index];
    if (morpheme_start == start && end == morpheme_end) {
        *out_position = "only";
    } else if (morpheme_start == start) {
        *out_position = "initial";
    } else if (end == morpheme_end) {
        *out_position = "final";
    } else {
        *out_position = "internal";
    }
}

/* Two builders for one environment, and the difference between them is
 * ownership and nothing else.
 *
 * This one borrows: every slot points into the caller's precomputed per-form
 * arrays, nothing is allocated, and clearing the result would free memory the
 * form still owns -- which is why `position` is a literal here and a strdup in
 * the owning builder, and why a borrowed context must never be cleared. The DP
 * scores against this one, once per transition, which is why it may not
 * allocate. `build_link_context` owns what it returns and is what a published
 * link carries.
 *
 * They used to differ in substance as well. When syllable data was absent this
 * one returned after the immediate neighbours while the owning one filled the
 * distance, existential and stress slots by a longer route, so a link could
 * carry an environment the DP had never scored against. That case cannot arise:
 * syllable data is built for every form with segments, and a form without
 * segments produces no links. The longer route was removed rather than
 * mirrored, and all 63 corpora hash identically without it, which is the
 * evidence it was unreachable.
 *
 * The early return below is kept as the guard it is, not as a second
 * behaviour. */
void build_link_context_borrowed(
    const rg_form *source,
    const rg_feature_constraint *const *source_features,
    const size_t *source_feature_counts,
    const syllable_data *syllables,
    size_t source_start,
    size_t source_count,
    size_t target_count,
    rg_context_spec *out
) {
    size_t source_end = source_start + source_count;
    size_t n = source->segment_count;

    rg_context_spec_init_empty(out);
    if (source_start == 0) {
        out->position = "initial";
    } else if (source_end == n) {
        out->position = "final";
    } else {
        out->position = "medial";
    }
    morpheme_placement(source, source_start, source_end, &out->morphological, &out->morpheme_index);
    if (source_start > 0) {
        out->preceding = source_features[source_start - 1];
        out->preceding_count = source_feature_counts[source_start - 1];
    }
    if (source_end < n) {
        out->following = source_features[source_end];
        out->following_count = source_feature_counts[source_end];
    }
    if (syllables == 0 || syllables->segment_count == 0) {
        return;
    }
    out->preceding_at_distance = syllables->preceding_at_distance[source_start];
    out->preceding_at_distance_count = syllables->preceding_at_distance_counts[source_start];
    out->following_at_distance = syllables->following_at_distance[source_end];
    out->following_at_distance_count = syllables->following_at_distance_counts[source_end];
    out->somewhere_preceding = syllables->left_cumulative[source_start];
    out->somewhere_preceding_count = syllables->left_cumulative_counts[source_start];
    out->somewhere_following = syllables->right_cumulative[source_end];
    out->somewhere_following_count = syllables->right_cumulative_counts[source_end];

    /* Syllable and stress slots mirror the DP: only 1-to-1 links carry them. */
    if (source_count != 1 || target_count != 1 || source_start >= n) {
        return;
    }
    {
        size_t syllable_index = syllables->syllable_of[source_start];
        out->same_syllable = syllables->same_syllable_excluding[source_start];
        out->same_syllable_count = syllables->same_syllable_excluding_counts[source_start];
        if (syllable_index + 1 < syllables->syllable_count) {
            out->next_syllable = syllables->syllable_features[syllable_index + 1];
            out->next_syllable_count = syllables->syllable_feature_counts[syllable_index + 1];
        }
        if (syllable_index > 0) {
            out->previous_syllable = syllables->syllable_features[syllable_index - 1];
            out->previous_syllable_count = syllables->syllable_feature_counts[syllable_index - 1];
        }
    }
    if (syllables->syllable_role != 0) {
        out->syllable_role = syllables->syllable_role[source_start];
    }
    if (syllables->syllable_position != 0) {
        out->syllable_position = syllables->syllable_position[source_start];
    }
    out->self_stress = syllables->stress[source_start];
    out->self_stress_count = syllables->stress_counts[source_start];
    if (source_start > 0) {
        out->preceding_stress = syllables->stress[source_start - 1];
        out->preceding_stress_count = syllables->stress_counts[source_start - 1];
    }
    if (source_end < n) {
        out->following_stress = syllables->stress[source_end];
        out->following_stress_count = syllables->stress_counts[source_end];
    }
}

rg_status build_link_context(
    const rg_form *source,
    const rg_feature_constraint *const *source_features,
    const size_t *source_feature_counts,
    const syllable_data *syllables,
    size_t source_start,
    size_t source_count,
    size_t target_count,
    rg_context_spec *out
) {
    size_t source_end = source_start + source_count;
    rg_status status;
    rg_context_spec_init_empty(out);
    if (source_start == 0) {
        out->position = rg_strdup_internal("initial");
    } else if (source_end == source->segment_count) {
        out->position = rg_strdup_internal("final");
    } else {
        out->position = rg_strdup_internal("medial");
    }
    if (out->position == 0) {
        return RG_ERR_OOM;
    }
    {
        const char *placement = 0;
        const char *index = 0;
        morpheme_placement(source, source_start, source_end, &placement, &index);
        if (placement != 0) {
            out->morphological = rg_strdup_internal(placement);
            out->morpheme_index = rg_strdup_internal(index);
            if (out->morphological == 0 || out->morpheme_index == 0) {
                return RG_ERR_OOM;
            }
        }
    }
    if (source_start > 0) {
        status = context_copy_constraints(
            source_features[source_start - 1],
            source_feature_counts[source_start - 1],
            &out->preceding,
            &out->preceding_count
        );
        if (status != RG_OK) {
            rg_context_spec_clear_internal(out);
            return status;
        }
    }
    if (source_end < source->segment_count) {
        status = context_copy_constraints(
            source_features[source_end],
            source_feature_counts[source_end],
            &out->following,
            &out->following_count
        );
        if (status != RG_OK) {
            rg_context_spec_clear_internal(out);
            return status;
        }
    }
    {
        const rg_feature_constraint *pre2 = 0;
        const rg_feature_constraint *pre3 = 0;
        size_t pre2_count = 0;
        size_t pre3_count = 0;
        if (source_start >= 2) {
            pre2 = source_features[source_start - 2];
            pre2_count = source_feature_counts[source_start - 2];
        }
        if (source_start >= 3) {
            pre3 = source_features[source_start - 3];
            pre3_count = source_feature_counts[source_start - 3];
        }
        status = distance_context_copy_two(
            pre2,
            pre2_count,
            2,
            pre3,
            pre3_count,
            3,
            &out->preceding_at_distance,
            &out->preceding_at_distance_count
        );
        if (status != RG_OK) {
            rg_context_spec_clear_internal(out);
            return status;
        }
    }
    {
        const rg_feature_constraint *fol2 = 0;
        const rg_feature_constraint *fol3 = 0;
        size_t fol2_count = 0;
        size_t fol3_count = 0;
        if (source_end + 1 < source->segment_count) {
            fol2 = source_features[source_end + 1];
            fol2_count = source_feature_counts[source_end + 1];
        }
        if (source_end + 2 < source->segment_count) {
            fol3 = source_features[source_end + 2];
            fol3_count = source_feature_counts[source_end + 2];
        }
        status = distance_context_copy_two(
            fol2,
            fol2_count,
            2,
            fol3,
            fol3_count,
            3,
            &out->following_at_distance,
            &out->following_at_distance_count
        );
        if (status != RG_OK) {
            rg_context_spec_clear_internal(out);
            return status;
        }
    }
    /* Syllable data is built for every form that has segments, and a form with
     * none produces no links, so the precomputed cumulative arrays are always
     * there to copy from. The union fallback that stood here computed the same
     * answer the long way for a case that cannot arise -- and, being reachable
     * only in theory, was the half of the divergence that made this builder
     * disagree with the borrowed one the DP scores against. */
    status = context_copy_constraints(
        syllables->left_cumulative[source_start],
        syllables->left_cumulative_counts[source_start],
        &out->somewhere_preceding,
        &out->somewhere_preceding_count
    );
    if (status != RG_OK) {
        rg_context_spec_clear_internal(out);
        return status;
    }
    status = context_copy_constraints(
        syllables->right_cumulative[source_end],
        syllables->right_cumulative_counts[source_end],
        &out->somewhere_following,
        &out->somewhere_following_count
    );
    if (status != RG_OK) {
        rg_context_spec_clear_internal(out);
        return status;
    }
    /* Syllable-structural slots mirror the DP: only 1-to-1 links carry them. */
    if (source_count == 1 && target_count == 1 && syllables != 0 && syllables->segment_count > 0 &&
        source_start < syllables->segment_count) {
        size_t syllable_index = syllables->syllable_of[source_start];
        status = context_copy_constraints(
            syllables->same_syllable_excluding[source_start],
            syllables->same_syllable_excluding_counts[source_start],
            &out->same_syllable,
            &out->same_syllable_count
        );
        if (status != RG_OK) {
            rg_context_spec_clear_internal(out);
            return status;
        }
        if (syllable_index + 1 < syllables->syllable_count) {
            status = context_copy_constraints(
                syllables->syllable_features[syllable_index + 1],
                syllables->syllable_feature_counts[syllable_index + 1],
                &out->next_syllable,
                &out->next_syllable_count
            );
            if (status != RG_OK) {
                rg_context_spec_clear_internal(out);
                return status;
            }
        }
        if (syllable_index > 0) {
            status = context_copy_constraints(
                syllables->syllable_features[syllable_index - 1],
                syllables->syllable_feature_counts[syllable_index - 1],
                &out->previous_syllable,
                &out->previous_syllable_count
            );
            if (status != RG_OK) {
                rg_context_spec_clear_internal(out);
                return status;
            }
        }
        if (syllables->syllable_role != 0 && syllables->syllable_role[source_start] != 0) {
            out->syllable_role = rg_strdup_internal(syllables->syllable_role[source_start]);
            if (out->syllable_role == 0) {
                rg_context_spec_clear_internal(out);
                return RG_ERR_OOM;
            }
        }
        if (syllables->syllable_position != 0 && syllables->syllable_position[source_start] != 0) {
            out->syllable_position = rg_strdup_internal(syllables->syllable_position[source_start]);
            if (out->syllable_position == 0) {
                rg_context_spec_clear_internal(out);
                return RG_ERR_OOM;
            }
        }
    }
    if (source_count == 1 && target_count == 1) {
        status = context_copy_stress(source->segments[source_start].stress, &out->self_stress, &out->self_stress_count);
        if (status != RG_OK) {
            rg_context_spec_clear_internal(out);
            return status;
        }
        if (source_start > 0) {
            status = context_copy_stress(source->segments[source_start - 1].stress, &out->preceding_stress, &out->preceding_stress_count);
            if (status != RG_OK) {
                rg_context_spec_clear_internal(out);
                return status;
            }
        }
        if (source_end < source->segment_count) {
            status = context_copy_stress(source->segments[source_end].stress, &out->following_stress, &out->following_stress_count);
            if (status != RG_OK) {
                rg_context_spec_clear_internal(out);
                return status;
            }
        }
    }
    return RG_OK;
}

void rg_context_spec_array_free_internal(rg_context_spec *contexts, size_t count) {
    size_t i;
    if (contexts == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        rg_context_spec_clear_internal(&contexts[i]);
    }
    free(contexts);
}

rg_status rg_form_position_contexts_internal(
    const rg_context *ctx,
    const rg_form *form,
    rg_context_spec **out,
    size_t *out_count
) {
    const rg_feature_constraint **features = 0;
    size_t *feature_counts = 0;
    syllable_data syllables;
    rg_context_spec *contexts = 0;
    size_t i;
    rg_status status;

    if (ctx == 0 || form == 0 || out == 0 || out_count == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    *out_count = 0;
    if (form->segment_count == 0) {
        return RG_OK;
    }
    memset(&syllables, 0, sizeof(syllables));
    status = feature_matrix_build(ctx, form, &features, &feature_counts);
    if (status != RG_OK) {
        return status;
    }
    status = syllable_data_build(ctx, form, features, feature_counts, &syllables);
    if (status != RG_OK) {
        feature_matrix_clear(features, feature_counts, form->segment_count);
        return status;
    }
    contexts = (rg_context_spec *)calloc(form->segment_count, sizeof(*contexts));
    if (contexts == 0) {
        feature_matrix_clear(features, feature_counts, form->segment_count);
        syllable_data_clear(&syllables);
        return RG_ERR_OOM;
    }
    for (i = 0; i < form->segment_count; i++) {
        status = build_link_context(form, features, feature_counts, &syllables, i, 1, 1, &contexts[i]);
        if (status != RG_OK) {
            rg_context_spec_array_free_internal(contexts, i);
            feature_matrix_clear(features, feature_counts, form->segment_count);
            syllable_data_clear(&syllables);
            return status;
        }
    }
    feature_matrix_clear(features, feature_counts, form->segment_count);
    syllable_data_clear(&syllables);
    *out = contexts;
    *out_count = form->segment_count;
    return RG_OK;
}

rg_status link_from_slice(
    const rg_context *ctx,
    const rg_form *source,
    size_t source_start,
    size_t source_count,
    const rg_form *target,
    size_t target_start,
    size_t target_count,
    const rg_feature_constraint *const *source_features,
    const size_t *source_feature_counts,
    const syllable_data *syllables,
    rg_link *out
) {
    rg_status status;
    memset(out, 0, sizeof(*out));
    status = rg_link_copy_chunks_internal(
        out,
        source->segments + source_start,
        source_count,
        target->segments + target_start,
        target_count
    );
    if (status != RG_OK) {
        return status;
    }
    if (source_features != 0 && source_feature_counts != 0) {
        status = build_link_context(
            source,
            source_features,
            source_feature_counts,
            syllables,
            source_start,
            source_count,
            target_count,
            &out->context
        );
        if (status != RG_OK) {
            rg_link_clear_internal(out);
            return status;
        }
    }
    if (source_count == 1 && target_count == 1) {
        rg_feature_displacement *disp = 0;
        size_t disp_count = 0;
        status = rg_compute_displacement(ctx, source->segments[source_start], target->segments[target_start], &disp, &disp_count);
        if (status != RG_OK) {
            rg_link_clear_internal(out);
            return status;
        }
        out->feature_displacement = disp;
        out->feature_displacement_count = disp_count;
    }
    return RG_OK;
}

