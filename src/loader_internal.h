#ifndef REGULAE_LOADER_INTERNAL_H
#define REGULAE_LOADER_INTERNAL_H

/* Shared between the loaders, and between them only.
 *
 * loaders.c ran to 2,000 lines carrying four unrelated jobs: reading a
 * delimited table, turning cells into segments and forms, accumulating and
 * publishing a corpus, and four format readers that use all three. The file
 * already marked those seams with banner comments; this makes them modules.
 *
 * Not internal.h, for the same reason src/model_internal.h is not: no other
 * module has any business with a loader_row. */

#include "internal.h"

typedef struct loader_field {
    char *value;
} loader_field;

typedef struct loader_row {
    char **fields;
    size_t field_count;
} loader_row;

typedef struct loader_table {
    char **header;
    size_t column_count;
    loader_row *rows;
    size_t row_count;
} loader_table;

typedef struct loader_form {
    char *lect_id;
    rg_segment *segments;
    size_t segment_count;
    int *morpheme_breaks;
    size_t morpheme_break_count;
    int *syllable_breaks;
    size_t syllable_break_count;
} loader_form;

typedef struct loader_cognate {
    char *cognate_id;
    loader_form *forms;
    size_t form_count;
    size_t form_cap;
    double confidence;
    int has_confidence;
    long alignment_length;
} loader_cognate;

struct rg_corpus {
    loader_cognate *cognates;
    size_t count;
    size_t cap;
    rg_cognate_set *view;
    size_t view_count;
    size_t doublet_set_count;
    size_t doublet_expansion_count;
    rg_cognate_form **view_forms;
    size_t view_form_group_count;
};


/* ---- loader_table.c: reading a delimited table -------------------------------*/
rg_status read_table_source(const char *path, const char *text, char delim, loader_table *out);
void loader_table_clear(loader_table *table);
void loader_row_clear(loader_row *row);
const char *cell(const loader_row *row, long index);
long column_index(const loader_table *table, const char *name);
char *trim_copy(const char *value);
long alignment_token_count(const char *raw);
int string_list_contains(const char *const *items, size_t count, const char *value);
void loader_clear_diagnosis(rg_load_diagnosis *diagnosis);
void loader_fail(rg_load_diagnosis *diagnosis, size_t line, const char *message);

/* ---- loader_forms.c: cells into segments and forms ---------------------------*/
/* Which per-segment dimension a companion column carries. Length is the one
 * that arrived last: a corpus can also write it into the grapheme as `aː`,
 * which merkmal reads as its own segment with the `long` feature, and that is
 * the right shape where length is contrastive. The dimension is for corpora
 * that treat it as something happening *to* a vowel -- compensatory
 * lengthening -- where making the long vowel a different segment from the
 * short one splits every correspondence it takes part in. */
#define RG_DIMENSION_TONE 0
#define RG_DIMENSION_STRESS 1
#define RG_DIMENSION_LENGTH 2
rg_status parse_segments( const char *raw, int track_boundaries, rg_segment **out_segments, size_t *out_count, int **out_breaks, size_t *out_break_count );
rg_status parse_break_indices(const char *raw, int **out, size_t *out_count);
rg_status attach_dimension( const char *raw, rg_segment *segments, size_t segment_count, int dimension );
void loader_form_clear(loader_form *form);
rg_status loader_form_from(const rg_form *src, const char *lect_id, loader_form *out);

/* ---- loader_corpus.c: accumulation and publication ---------------------------*/
loader_cognate *corpus_ensure_cognate(rg_corpus *corpus, const char *cognate_id);
loader_cognate *corpus_append_cognate(rg_corpus *corpus, const char *base_id);
rg_status cognate_append_form(loader_cognate *cognate, loader_form *form);
rg_status corpus_publish(rg_corpus *corpus, int min_lects);
void loader_cognate_clear(loader_cognate *cognate);

/* ---- the four readers, called by the public entry points in loaders.c --------*/
rg_status load_tsv( const char *path, const char *text, const rg_tsv_load_options *options, rg_corpus **out, rg_load_diagnosis *diagnosis );
rg_status load_wide_tsv( const rg_context *ctx, const char *path, const char *text, const rg_wide_load_options *options, rg_corpus **out, rg_load_diagnosis *diagnosis );
rg_status load_gled( const char *path, const char *text, const rg_gled_load_options *options, rg_corpus **out, rg_load_diagnosis *diagnosis );
rg_status load_arcaverborum( const char *path, const char *text, const rg_arcaverborum_load_options *options, rg_corpus **out, rg_load_diagnosis *diagnosis );

#endif
