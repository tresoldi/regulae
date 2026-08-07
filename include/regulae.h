#ifndef REGULAE_H
#define REGULAE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(REGULAE_SHARED)
#  if defined(REGULAE_BUILDING_LIBRARY)
#    define RG_API __declspec(dllexport)
#  else
#    define RG_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define RG_API __attribute__((visibility("default")))
#else
#  define RG_API
#endif

#define RG_VERSION_MAJOR 0
#define RG_VERSION_MINOR 1
#define RG_VERSION_PATCH 0
#define RG_VERSION_STRING "0.1.0"
#define RG_ABI_VERSION 2
#define RG_DEFAULT_MAX_CHUNK_SIZE 3

typedef struct rg_context rg_context;
typedef struct rg_feature_set rg_feature_set;
typedef struct rg_alignment rg_alignment;
typedef struct rg_pairwise_model rg_pairwise_model;
typedef struct rg_multi_model rg_multi_model;
typedef struct rg_corpus rg_corpus;

typedef enum rg_status {
    RG_OK = 0,
    RG_ERR_INVALID_ARGUMENT,
    RG_ERR_IO,
    RG_ERR_PARSE,
    RG_ERR_MERKMAL,
    RG_ERR_UNKNOWN_GRAPHEME,
    RG_ERR_UNSUPPORTED_OPTION,
    RG_ERR_OOM
} rg_status;

typedef struct rg_bic_config {
    double delta_bic_threshold;
    int min_split_observations;
    int max_split_depth;
    int min_chunk_observations;
    double long_range_delta_bic_threshold;
    int long_range_min_split_observations;
    double long_range_min_dominant_fraction;
    int cross_dim_max_iterations;
    int cross_dim_min_rule_count;
    double cross_dim_min_rule_confidence;
    int multi_lect_bic_small_sample_correction;
    double multi_lect_min_commit_scale;
} rg_bic_config;

typedef struct rg_train_options {
    const char *feature_system;
    int max_chunk_size;
    double temperature;
    double concentration;
    int max_iter;
    double convergence_eps;
    double segment_weight;
    double displacement_weight;
    double tone_weight;
    double chunk_min_transparency;
    rg_bic_config bic;
    int bootstrap_n;
    int bootstrap_seed;
} rg_train_options;

typedef struct rg_segment {
    const char *grapheme;
    const char *tone;
    const char *length;
    const char *stress;
} rg_segment;

typedef struct rg_form {
    const char *lect_id;
    const rg_segment *segments;
    size_t segment_count;
    const int *syllable_breaks;
    size_t syllable_break_count;
    const int *morpheme_breaks;
    size_t morpheme_break_count;
} rg_form;

typedef struct rg_feature_constraint {
    const char *feature;
    const char *value;
} rg_feature_constraint;

typedef struct rg_distance_constraint {
    int offset;
    rg_feature_constraint constraint;
} rg_distance_constraint;

typedef struct rg_context_spec {
    const char *position;
    const rg_feature_constraint *preceding;
    size_t preceding_count;
    const rg_feature_constraint *following;
    size_t following_count;
    const char *morphological;
    const rg_distance_constraint *preceding_at_distance;
    size_t preceding_at_distance_count;
    const rg_distance_constraint *following_at_distance;
    size_t following_at_distance_count;
    const rg_feature_constraint *somewhere_preceding;
    size_t somewhere_preceding_count;
    const rg_feature_constraint *somewhere_following;
    size_t somewhere_following_count;
    const rg_feature_constraint *same_syllable;
    size_t same_syllable_count;
    const rg_feature_constraint *next_syllable;
    size_t next_syllable_count;
    const rg_feature_constraint *previous_syllable;
    size_t previous_syllable_count;
    const rg_feature_constraint *self_stress;
    size_t self_stress_count;
    const rg_feature_constraint *preceding_stress;
    size_t preceding_stress_count;
    const rg_feature_constraint *following_stress;
    size_t following_stress_count;
} rg_context_spec;

typedef struct rg_feature_displacement {
    const char *feature;
    const char *from_value;
    const char *to_value;
} rg_feature_displacement;

typedef struct rg_uncertainty_estimate {
    double estimate;
    double lower;
    double upper;
} rg_uncertainty_estimate;

typedef struct rg_link {
    const rg_segment *source;
    size_t source_count;
    const rg_segment *target;
    size_t target_count;
    rg_context_spec context;
    const rg_feature_displacement *feature_displacement;
    size_t feature_displacement_count;
    double confidence;
} rg_link;

typedef struct rg_form_pair {
    rg_form source;
    rg_form target;
    double weight;
} rg_form_pair;

typedef struct rg_cognate_form {
    const char *lect_id;
    rg_form form;
} rg_cognate_form;

typedef struct rg_cognate_set {
    const char *cognate_id;
    const rg_cognate_form *forms;
    size_t form_count;
    double confidence;
} rg_cognate_set;

typedef struct rg_segment_count_row {
    const char *source;
    const char *target;
    double count;
    double source_total;
    rg_uncertainty_estimate uncertainty;
} rg_segment_count_row;

/* One observed feature-displacement vector: the whole set of features gained
 * and lost between a source and target segment, counted as a unit. */
typedef struct rg_displacement_row {
    const rg_feature_displacement *items;
    size_t item_count;
    double count;
    double total;
    rg_uncertainty_estimate uncertainty;
} rg_displacement_row;

typedef struct rg_tonal_count_row {
    const char *source_tone;
    const char *target_tone;
    double count;
    double source_total;
    rg_uncertainty_estimate uncertainty;
} rg_tonal_count_row;

typedef struct rg_conditioned_segment_count_row {
    const char *source;
    const char *target;
    rg_context_spec context;
    double count;
    double source_total;
    rg_uncertainty_estimate uncertainty;
} rg_conditioned_segment_count_row;

typedef struct rg_chunk_row {
    const rg_segment *source;
    size_t source_count;
    const rg_segment *target;
    size_t target_count;
    double cost;
    double count;
    rg_uncertainty_estimate uncertainty;
} rg_chunk_row;

typedef struct rg_cross_dimensional_row {
    const char *source_feature;
    const char *source_value;
    const char *source_position;
    const char *target_dimension;
    const char *target_value;
    int target_position_offset;
    double count;
    double source_count;
    double confidence;
    rg_uncertainty_estimate uncertainty;
} rg_cross_dimensional_row;

typedef struct rg_multi_pair_model_row {
    const char *lect_a;
    const char *lect_b;
    const rg_pairwise_model *model;
} rg_multi_pair_model_row;

typedef struct rg_multi_class_row {
    int class_id;
    const char *const *lect_ids;
    const char *const *graphemes;
    const rg_context_spec *contexts;
    size_t segment_count;
    double count;
    double confidence;
    const char *const *supporting_cognates;
    size_t supporting_cognate_count;
    rg_uncertainty_estimate uncertainty;
} rg_multi_class_row;

typedef struct rg_multi_cross_dimensional_row {
    const char *source_lect;
    const char *target_lect;
    const char *source_feature;
    const char *source_value;
    const char *source_position;
    const char *target_dimension;
    const char *target_value;
    int target_position_offset;
    double count;
    double source_count;
    double confidence;
    rg_uncertainty_estimate uncertainty;
} rg_multi_cross_dimensional_row;

typedef struct rg_cognate_outlier_row {
    const char *cognate_id;
    int pair_count;
    double cost_per_segment;
    double z_score;
} rg_cognate_outlier_row;

RG_API const char *rg_version_string(void);
RG_API int rg_version_major(void);
RG_API int rg_version_minor(void);
RG_API int rg_version_patch(void);
RG_API uint32_t rg_abi_version(void);
RG_API const char *rg_status_string(rg_status status);
RG_API void rg_string_free(char *value);

RG_API void rg_bic_config_init_defaults(rg_bic_config *config);
RG_API void rg_train_options_init_defaults(rg_train_options *options);

RG_API rg_status rg_context_new_builtin(rg_context **out);
RG_API void rg_context_free(rg_context *ctx);
RG_API rg_status rg_context_use_system(rg_context *ctx, const char *system_name);
RG_API rg_status rg_context_system_name(const rg_context *ctx, const char **out);
RG_API rg_status rg_context_is_segment(const rg_context *ctx, const char *grapheme, int *out);
RG_API rg_status rg_context_segment_distance(
    const rg_context *ctx,
    const char *a,
    const char *b,
    double *out
);
RG_API rg_status rg_context_grapheme_features(
    const rg_context *ctx,
    const char *grapheme,
    rg_feature_set **out
);

RG_API rg_status rg_context_segment_word(
    const rg_context *ctx,
    const char *word,
    rg_segment **out,
    size_t *out_count
);
RG_API void rg_segments_free(rg_segment *segments, size_t count);

RG_API size_t rg_feature_set_size(const rg_feature_set *features);
RG_API const char *rg_feature_set_get(const rg_feature_set *features, size_t index);
RG_API void rg_feature_set_free(rg_feature_set *features);

RG_API void rg_context_spec_init_empty(rg_context_spec *context);
RG_API size_t rg_context_spec_constraint_count(const rg_context_spec *context);
RG_API rg_status rg_context_spec_is_subset(
    const rg_context_spec *subset,
    const rg_context_spec *other,
    int *out
);

RG_API rg_status rg_score_link(
    const rg_context *ctx,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    double *out
);
RG_API rg_status rg_score_link_with_model(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    const rg_train_options *options,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    double *out
);
RG_API rg_status rg_compute_displacement(
    const rg_context *ctx,
    rg_segment source,
    rg_segment target,
    rg_feature_displacement **out,
    size_t *out_count
);
RG_API void rg_feature_displacement_free(rg_feature_displacement *items, size_t count);

RG_API rg_status rg_compute_syllable_breaks(
    const rg_context *ctx,
    const rg_form *form,
    int **out,
    size_t *out_count
);
RG_API void rg_syllable_breaks_free(int *breaks);

RG_API rg_status rg_align_forms(
    const rg_context *ctx,
    const rg_form *source,
    const rg_form *target,
    int max_chunk_size,
    rg_alignment **out
);
RG_API rg_status rg_align_forms_with_model(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    const rg_train_options *options,
    const rg_form *source,
    const rg_form *target,
    int max_chunk_size,
    rg_alignment **out
);
RG_API void rg_alignment_free(rg_alignment *alignment);
RG_API size_t rg_alignment_link_count(const rg_alignment *alignment);
RG_API const rg_link *rg_alignment_link_at(const rg_alignment *alignment, size_t index);
RG_API rg_status rg_alignment_cost(const rg_context *ctx, const rg_alignment *alignment, double *out);
RG_API rg_status rg_alignment_cost_with_model(
    const rg_context *ctx,
    const rg_pairwise_model *model,
    const rg_train_options *options,
    const rg_alignment *alignment,
    double *out
);

RG_API rg_status rg_train_pairwise_segment_counts(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model **out
);
RG_API rg_status rg_train_pairwise(
    const rg_context *ctx,
    const rg_form_pair *pairs,
    size_t pair_count,
    const rg_train_options *options,
    rg_pairwise_model **out
);
RG_API void rg_pairwise_model_free(rg_pairwise_model *model);
RG_API size_t rg_pairwise_model_segment_count_row_count(const rg_pairwise_model *model);
RG_API const rg_segment_count_row *rg_pairwise_model_segment_count_row_at(
    const rg_pairwise_model *model,
    size_t index
);
RG_API size_t rg_pairwise_model_displacement_row_count(const rg_pairwise_model *model);
RG_API const rg_displacement_row *rg_pairwise_model_displacement_row_at(
    const rg_pairwise_model *model,
    size_t index
);
RG_API size_t rg_pairwise_model_tonal_count_row_count(const rg_pairwise_model *model);
RG_API const rg_tonal_count_row *rg_pairwise_model_tonal_count_row_at(
    const rg_pairwise_model *model,
    size_t index
);
RG_API size_t rg_pairwise_model_conditioned_segment_count_row_count(const rg_pairwise_model *model);
RG_API const rg_conditioned_segment_count_row *rg_pairwise_model_conditioned_segment_count_row_at(
    const rg_pairwise_model *model,
    size_t index
);
RG_API size_t rg_pairwise_model_chunk_row_count(const rg_pairwise_model *model);
RG_API const rg_chunk_row *rg_pairwise_model_chunk_row_at(
    const rg_pairwise_model *model,
    size_t index
);
RG_API size_t rg_pairwise_model_cross_dimensional_row_count(const rg_pairwise_model *model);
RG_API const rg_cross_dimensional_row *rg_pairwise_model_cross_dimensional_row_at(
    const rg_pairwise_model *model,
    size_t index
);

typedef struct rg_tsv_load_options {
    const char *cognate_id_column;
    const char *lect_id_column;
    const char *segments_column;
    const char *alignment_column;
    const char *confidence_column;
} rg_tsv_load_options;

/* Wide-format TSV: one row per cognate, one column per lect, cells holding
 * whole unsegmented words. This is the shape linguistic data is actually
 * written in; the generic long-format loader wants pre-segmented input.
 * A "<lect>_breaks" column supplies morpheme boundary indices for that lect. */
typedef struct rg_wide_load_options {
    const char *cognate_id_column;
    const char *confidence_column;
    const char *const *lect_columns;
    size_t lect_column_count;
} rg_wide_load_options;

typedef struct rg_gled_load_options {
    const char *family;
    const char *const *doculects;
    size_t doculect_count;
    int min_lects;
} rg_gled_load_options;

typedef struct rg_arcaverborum_load_options {
    const char *dataset;
    const char *family;
    const char *const *language_ids;
    size_t language_id_count;
    int min_lects;
    /* Zero selects the delimiter from the file extension when loading a path,
     * and comma when parsing text. */
    char delimiter;
} rg_arcaverborum_load_options;

RG_API rg_status rg_corpus_load_tsv(
    const char *path,
    const rg_tsv_load_options *options,
    rg_corpus **out
);
RG_API rg_status rg_corpus_load_wide_tsv(
    const rg_context *ctx,
    const char *path,
    const rg_wide_load_options *options,
    rg_corpus **out
);
RG_API rg_status rg_corpus_load_gled(
    const char *path,
    const rg_gled_load_options *options,
    rg_corpus **out
);
RG_API rg_status rg_corpus_load_arcaverborum(
    const char *path,
    const rg_arcaverborum_load_options *options,
    rg_corpus **out
);
RG_API rg_status rg_corpus_from_pairs(
    const rg_form_pair *pairs,
    size_t pair_count,
    const char *lect_a,
    const char *lect_b,
    const char *cognate_id_prefix,
    rg_corpus **out
);
/* Parse variants take the corpus as text rather than a path, so a caller that
 * already holds the data needs no temporary file. The WebAssembly build links
 * without a filesystem and uses these exclusively. */
RG_API rg_status rg_corpus_parse_tsv(
    const char *text,
    const rg_tsv_load_options *options,
    rg_corpus **out
);
RG_API rg_status rg_corpus_parse_wide_tsv(
    const rg_context *ctx,
    const char *text,
    const rg_wide_load_options *options,
    rg_corpus **out
);
RG_API rg_status rg_corpus_parse_gled(
    const char *text,
    const rg_gled_load_options *options,
    rg_corpus **out
);
RG_API rg_status rg_corpus_parse_arcaverborum(
    const char *text,
    const rg_arcaverborum_load_options *options,
    rg_corpus **out
);

RG_API void rg_corpus_free(rg_corpus *corpus);
RG_API size_t rg_corpus_cognate_count(const rg_corpus *corpus);
RG_API const rg_cognate_set *rg_corpus_cognates(const rg_corpus *corpus);
RG_API const rg_cognate_set *rg_corpus_cognate_at(const rg_corpus *corpus, size_t index);

/* Formatting helpers. The strings these produce are diagnostics meant to be
 * read; iterate the model accessors for anything a program depends on. Every
 * returned string is caller-owned and freed with rg_string_free. */
typedef struct rg_format_model_options {
    int top_segments;
    int top_displacements;
    double min_count;
    int top_chunks;
    int top_classes;
} rg_format_model_options;

RG_API void rg_format_model_options_init_defaults(rg_format_model_options *options);
RG_API char *rg_format_segments(const rg_segment *segments, size_t count);
RG_API char *rg_format_link(const rg_link *link);
RG_API char *rg_format_alignment(const rg_alignment *alignment);
RG_API char *rg_format_pairwise_model(
    const rg_pairwise_model *model,
    const rg_format_model_options *options
);
RG_API char *rg_format_multi_model(
    const rg_multi_model *model,
    const rg_format_model_options *options
);
RG_API char *rg_describe_multi_class(
    const rg_multi_model *model,
    const char *lect_id,
    const char *grapheme
);

RG_API rg_status rg_train_model(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    rg_multi_model **out
);
RG_API void rg_multi_model_free(rg_multi_model *model);
RG_API size_t rg_multi_model_lect_count(const rg_multi_model *model);
RG_API const char *rg_multi_model_lect_at(const rg_multi_model *model, size_t index);
RG_API size_t rg_multi_model_pair_model_count(const rg_multi_model *model);
RG_API const rg_multi_pair_model_row *rg_multi_model_pair_model_at(
    const rg_multi_model *model,
    size_t index
);
RG_API size_t rg_multi_model_unconditioned_class_count(const rg_multi_model *model);
RG_API const rg_multi_class_row *rg_multi_model_unconditioned_class_at(
    const rg_multi_model *model,
    size_t index
);
RG_API size_t rg_multi_model_conditioned_class_count(const rg_multi_model *model);
RG_API const rg_multi_class_row *rg_multi_model_conditioned_class_at(
    const rg_multi_model *model,
    size_t index
);
RG_API size_t rg_multi_model_cross_dimensional_row_count(const rg_multi_model *model);
RG_API const rg_multi_cross_dimensional_row *rg_multi_model_cross_dimensional_row_at(
    const rg_multi_model *model,
    size_t index
);
RG_API rg_status rg_find_cognate_outliers(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_multi_model *model,
    const rg_train_options *options,
    int top_k,
    int max_chunk_size,
    rg_cognate_outlier_row **out,
    size_t *out_count
);
RG_API void rg_cognate_outlier_rows_free(rg_cognate_outlier_row *rows, size_t count);

#ifdef __cplusplus
}
#endif

#endif
