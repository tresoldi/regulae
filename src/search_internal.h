#ifndef REGULAE_SEARCH_INTERNAL_H
#define REGULAE_SEARCH_INTERNAL_H

/* Shared between the alignment search's parts, and between them only.
 *
 * search.c ran to 1,900 lines over four jobs the DP needs but is not: what a
 * segment's features are, what syllable it sits in, what environment an
 * aligned position has, and what a scored alignment costs once
 * cross-dimensional rules are taken into account. Each is a file now, and the
 * DP itself is what remains.
 *
 * Not internal.h: no other module has any business with a syllable_data. */

#include "internal.h"

struct rg_alignment {
    rg_form source_form;
    rg_form target_form;
    rg_link *links;
    size_t link_count;
};

/* syllable_data holds the syllable-structural arrays a form's link contexts
 * need: which syllable each segment belongs to, the feature union of each
 * syllable, and the feature union of each segment's own syllable excluding
 * itself. Built once per alignment, mirroring Go's computeLongRangeData. */
typedef struct syllable_data {
    size_t segment_count;
    size_t syllable_count;
    size_t *syllable_of;
    const rg_feature_constraint **syllable_features;
    size_t *syllable_feature_counts;
    const rg_feature_constraint **same_syllable_excluding;
    size_t *same_syllable_excluding_counts;
    /* Cumulative feature unions over [0, i) and [i, n), so the existential
     * slots of a link context are a lookup rather than a rescan. Recomputing
     * them per DP cell dominated training time. */
    const rg_feature_constraint **left_cumulative;
    size_t *left_cumulative_counts;
    const rg_feature_constraint **right_cumulative;
    size_t *right_cumulative_counts;
    /* Distance-bounded slots at offsets 2 and 3, indexed by span start and
     * span end, and the one-element stress constraint of each position. With
     * these every slot of a link context is a precomputed array, which is what
     * lets the DP score against a borrowed context instead of building an
     * owned one per cell. */
    const rg_distance_constraint **preceding_at_distance;
    size_t *preceding_at_distance_counts;
    const rg_distance_constraint **following_at_distance;
    size_t *following_at_distance_counts;
    const rg_feature_constraint **stress;
    size_t *stress_counts;
    const char **syllable_role;
    const char **syllable_position;
} syllable_data;

/* ---- search.c: the alignment DP and the forms it copies */
rg_status copy_ints(const int *items, size_t count, const int **out);
rg_status form_copy(const rg_form *src, rg_form *out);
void form_clear(rg_form *form);

/* ---- search_context.c: the environment an aligned position sits in */
rg_status build_link_context( const rg_form *source, const rg_feature_constraint *const *source_features, const size_t *source_feature_counts, const syllable_data *syllables, size_t source_start, size_t source_count, size_t target_count, rg_context_spec *out );
rg_status link_from_slice( const rg_context *ctx, const rg_form *source, size_t source_start, size_t source_count, const rg_form *target, size_t target_start, size_t target_count, const rg_feature_constraint *const *source_features, const size_t *source_feature_counts, const syllable_data *syllables, rg_link *out );
void build_link_context_borrowed( const rg_form *source, const rg_feature_constraint *const *source_features, const size_t *source_feature_counts, const syllable_data *syllables, size_t source_start, size_t source_count, size_t target_count, rg_context_spec *out );
void morpheme_placement( const rg_form *form, size_t start, size_t end, const char **out_position, const char **out_index );

/* ---- search_cost.c: cross-dimensional adjustment to a scored alignment */
double cross_dimensional_link_adjustment( const rg_context *ctx, const rg_pairwise_model *model, const rg_form *source_form, size_t src_pos, const rg_form *target_form, size_t tgt_pos, size_t source_count, size_t target_count );
double cross_dimensional_alignment_adjustment( const rg_context *ctx, const rg_pairwise_model *model, const rg_alignment *alignment );

/* ---- search_features.c: a segment's features, and the constraint arrays built from them */
int feature_set_contains(const rg_feature_set *features, const char *feature);
rg_status context_copy_constraints( const rg_feature_constraint *src, size_t count, const rg_feature_constraint **out, size_t *out_count );
rg_status context_copy_stress( const char *stress, const rg_feature_constraint **out, size_t *out_count );
rg_status context_feature_union_copy( const rg_feature_constraint *const *source_features, const size_t *source_feature_counts, size_t start, size_t end, const rg_feature_constraint **out, size_t *out_count );
rg_status context_feature_union_copy_excluding( const rg_feature_constraint *const *source_features, const size_t *source_feature_counts, size_t start, size_t end, size_t exclude, int has_exclude, const rg_feature_constraint **out, size_t *out_count );
rg_status context_features_for_segment( const rg_context *ctx, const rg_segment *segment, const rg_feature_constraint **out, size_t *out_count );
rg_status distance_context_copy_two( const rg_feature_constraint *features_a, size_t feature_count_a, int offset_a, const rg_feature_constraint *features_b, size_t feature_count_b, int offset_b, const rg_distance_constraint **out, size_t *out_count );
rg_status feature_matrix_build( const rg_context *ctx, const rg_form *form, const rg_feature_constraint ***out_features, size_t **out_counts );
void feature_matrix_clear( const rg_feature_constraint **features, const size_t *feature_counts, size_t count );

/* ---- search_syllables.c: which syllable a segment is in, and what that syllable is */
rg_status syllable_data_build( const rg_context *ctx, const rg_form *form, const rg_feature_constraint *const *source_features, const size_t *source_feature_counts, syllable_data *out );
void syllable_data_clear(syllable_data *data);

#endif
