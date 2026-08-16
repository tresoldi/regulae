#include "loader_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- wide-format TSV ---------------------------------------------------- */

static int has_suffix(const char *value, const char *suffix) {
    size_t v = strlen(value);
    size_t s = strlen(suffix);
    return v >= s && strcmp(value + v - s, suffix) == 0;
}

/* Parses a "<lect>_breaks" cell: comma-separated morpheme boundary indices,
 * with "-" or an empty cell meaning none. */
rg_status parse_break_indices(const char *raw, int **out, size_t *out_count) {
    int *breaks = 0;
    size_t count = 0;
    size_t cap = 0;
    const char *p = raw == 0 ? "" : raw;

    *out = 0;
    *out_count = 0;
    if (p[0] == '\0' || strcmp(p, "-") == 0) {
        return RG_OK;
    }
    while (*p != '\0') {
        char *endptr = 0;
        long value;
        while (*p == ' ' || *p == ',' || *p == '\t') {
            p++;
        }
        if (*p == '\0') {
            break;
        }
        value = strtol(p, &endptr, 10);
        if (endptr == p) {
            free(breaks);
            return RG_ERR_PARSE;
        }
        p = endptr;
        if (value <= 0) {
            /* A boundary at or before the first segment marks nothing. */
            continue;
        }
        if (count == cap) {
            size_t next_cap = cap == 0 ? 4 : cap * 2;
            int *next = (int *)realloc(breaks, next_cap * sizeof(*next));
            if (next == 0) {
                free(breaks);
                return RG_ERR_OOM;
            }
            breaks = next;
            cap = next_cap;
        }
        breaks[count++] = (int)value;
    }
    *out = breaks;
    *out_count = count;
    return RG_OK;
}

rg_status load_wide_tsv(
    const rg_context *ctx,
    const char *path,
    const char *text,
    const rg_wide_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
) {
    loader_table table;
    rg_corpus *corpus = 0;
    rg_wide_load_options opts;
    long id_col = -1;
    long confidence_col = -1;
    long etymon_group_col = -1;
    long source_group_col = -1;
    long *lect_cols = 0;
    long *break_cols = 0;
    long *syllable_cols = 0;
    long *length_cols = 0;
    long *tone_cols = 0;
    long *stress_cols = 0;
    size_t lect_count = 0;
    size_t c;
    size_t r;
    rg_status status;

    if (ctx == 0 || (path == 0 && text == 0) || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    memset(&opts, 0, sizeof(opts));
    if (options != 0) {
        opts = *options;
    }
    status = read_table_source(path, text, '\t', &table);
    if (status != RG_OK) {
        return status;
    }
    if (table.column_count == 0) {
        loader_table_clear(&table);
        return RG_ERR_PARSE;
    }
    /* The cognate id defaults to the first column, which is how every corpus
     * in this repository is written. */
    id_col = opts.cognate_id_column == 0 ? 0 : column_index(&table, opts.cognate_id_column);
    if (id_col < 0) {
        loader_fail(diagnosis, 1,
                    "no gloss column: the wide format needs one identifier column then one column per lect");
        loader_table_clear(&table);
        return RG_ERR_PARSE;
    }
    confidence_col = column_index(&table, opts.confidence_column == 0 ? "confidence" : opts.confidence_column);
    etymon_group_col = column_index(&table, opts.etymon_group_column == 0
                                    ? "etymon_group" : opts.etymon_group_column);
    source_group_col = column_index(&table, opts.source_group_column == 0
                                    ? "source_group" : opts.source_group_column);

    lect_cols = (long *)calloc(table.column_count, sizeof(*lect_cols));
    break_cols = (long *)calloc(table.column_count, sizeof(*break_cols));
    syllable_cols = (long *)calloc(table.column_count, sizeof(*syllable_cols));
    length_cols = (long *)calloc(table.column_count, sizeof(*length_cols));
    tone_cols = (long *)calloc(table.column_count, sizeof(*tone_cols));
    stress_cols = (long *)calloc(table.column_count, sizeof(*stress_cols));
    if (lect_cols == 0 || break_cols == 0 || tone_cols == 0 || stress_cols == 0) {
        free(lect_cols);
        free(break_cols);
        free(syllable_cols);
        free(length_cols);
        free(tone_cols);
        free(stress_cols);
        loader_table_clear(&table);
        return RG_ERR_OOM;
    }
    if (opts.lect_column_count > 0 && opts.lect_columns != 0) {
        for (c = 0; c < opts.lect_column_count; c++) {
            long index = column_index(&table, opts.lect_columns[c]);
            if (index < 0) {
                free(lect_cols);
                free(break_cols);
                free(syllable_cols);
                free(length_cols);
                free(tone_cols);
                free(stress_cols);
                loader_table_clear(&table);
                return RG_ERR_PARSE;
            }
            lect_cols[lect_count++] = index;
        }
    } else {
        /* Every column is a lect except the id, the confidence, and the
         * companion "_breaks" and "_tone" columns. */
        for (c = 0; c < table.column_count; c++) {
            if ((long)c == id_col || (long)c == confidence_col ||
                (long)c == etymon_group_col || (long)c == source_group_col) {
                continue;
            }
            if (has_suffix(table.header[c], "_breaks") ||
                has_suffix(table.header[c], "_syllables") ||
                has_suffix(table.header[c], "_tone") ||
                has_suffix(table.header[c], "_stress") ||
                has_suffix(table.header[c], "_length")) {
                continue;
            }
            lect_cols[lect_count++] = (long)c;
        }
    }
    if (lect_count == 0) {
        free(lect_cols);
        free(break_cols);
        free(syllable_cols);
        free(length_cols);
        free(tone_cols);
        free(stress_cols);
        loader_table_clear(&table);
        return RG_ERR_PARSE;
    }
    for (c = 0; c < lect_count; c++) {
        char companion[256];
        snprintf(companion, sizeof(companion), "%s_breaks", table.header[lect_cols[c]]);
        break_cols[c] = column_index(&table, companion);
        snprintf(companion, sizeof(companion), "%s_tone", table.header[lect_cols[c]]);
        tone_cols[c] = column_index(&table, companion);
        snprintf(companion, sizeof(companion), "%s_stress", table.header[lect_cols[c]]);
        stress_cols[c] = column_index(&table, companion);
        snprintf(companion, sizeof(companion), "%s_syllables", table.header[lect_cols[c]]);
        syllable_cols[c] = column_index(&table, companion);
        snprintf(companion, sizeof(companion), "%s_length", table.header[lect_cols[c]]);
        length_cols[c] = column_index(&table, companion);
    }

    corpus = (rg_corpus *)calloc(1, sizeof(*corpus));
    if (corpus == 0) {
        free(lect_cols);
        free(break_cols);
        free(syllable_cols);
        free(length_cols);
        free(tone_cols);
        free(stress_cols);
        loader_table_clear(&table);
        return RG_ERR_OOM;
    }

    for (r = 0; r < table.row_count && status == RG_OK; r++) {
        const loader_row *row = &table.rows[r];
        char *cognate_id = trim_copy(cell(row, id_col));
        loader_cognate *cognate;

        if (cognate_id == 0) {
            status = RG_ERR_OOM;
            break;
        }
        if (cognate_id[0] == '\0') {
            free(cognate_id);
            continue;
        }
        cognate = corpus_append_cognate(corpus, cognate_id);
        free(cognate_id);
        if (cognate == 0) {
            status = RG_ERR_OOM;
            break;
        }
        {
            char *etymon_group = etymon_group_col < 0 ? 0 : trim_copy(cell(row, etymon_group_col));
            char *source_group = source_group_col < 0 ? 0 : trim_copy(cell(row, source_group_col));
            if ((etymon_group_col >= 0 && etymon_group == 0) ||
                (source_group_col >= 0 && source_group == 0)) {
                free(etymon_group);
                free(source_group);
                status = RG_ERR_OOM;
                break;
            }
            status = cognate_set_groups(cognate, etymon_group, source_group);
            free(etymon_group);
            free(source_group);
            if (status != RG_OK) {
                break;
            }
        }
        for (c = 0; c < lect_count && status == RG_OK; c++) {
            const char *lect_id = table.header[lect_cols[c]];
            char *word = trim_copy(cell(row, lect_cols[c]));
            loader_form form;

            if (word == 0) {
                status = RG_ERR_OOM;
                break;
            }
            if (word[0] == '\0' || strcmp(word, "-") == 0) {
                free(word);
                continue;
            }
            memset(&form, 0, sizeof(form));
            status = rg_context_segment_word(ctx, word, &form.segments, &form.segment_count);
            free(word);
            if (status != RG_OK) {
                break;
            }
            if (form.segment_count == 0) {
                loader_form_clear(&form);
                continue;
            }
            if (break_cols[c] >= 0) {
                status = parse_break_indices(cell(row, break_cols[c]), &form.morpheme_breaks, &form.morpheme_break_count);
            }
            if (status == RG_OK && length_cols[c] >= 0) {
                status = attach_dimension(cell(row, length_cols[c]), form.segments,
                                          form.segment_count, RG_DIMENSION_LENGTH);
            }
            if (status == RG_OK && syllable_cols[c] >= 0) {
                status = parse_break_indices(cell(row, syllable_cols[c]), &form.syllable_breaks, &form.syllable_break_count);
                if (status != RG_OK) {
                    loader_form_clear(&form);
                    break;
                }
            }
            if (tone_cols[c] >= 0) {
                status = attach_dimension(cell(row, tone_cols[c]), form.segments, form.segment_count, RG_DIMENSION_TONE);
                if (status != RG_OK) {
                    loader_form_clear(&form);
                    break;
                }
            }
            if (stress_cols[c] >= 0) {
                status = attach_dimension(cell(row, stress_cols[c]), form.segments, form.segment_count, RG_DIMENSION_STRESS);
                if (status != RG_OK) {
                    loader_form_clear(&form);
                    break;
                }
            }
            form.lect_id = rg_strdup_internal(lect_id);
            if (form.lect_id == 0) {
                loader_form_clear(&form);
                status = RG_ERR_OOM;
                break;
            }
            status = cognate_append_form(cognate, &form);
            if (status != RG_OK) {
                loader_form_clear(&form);
                break;
            }
        }
        if (status == RG_OK && confidence_col >= 0) {
            char *raw = trim_copy(cell(row, confidence_col));
            if (raw == 0) {
                status = RG_ERR_OOM;
                break;
            }
            if (raw[0] != '\0' && strcmp(raw, "-") != 0) {
                char *endptr = 0;
                double value = strtod(raw, &endptr);
                if (endptr == raw || *endptr != '\0') {
                    free(raw);
                    status = RG_ERR_PARSE;
                    break;
                }
                if (!cognate->has_confidence || value < cognate->confidence) {
                    cognate->confidence = value;
                    cognate->has_confidence = 1;
                }
            }
            free(raw);
        }
    }

    free(lect_cols);
    free(break_cols);
    free(syllable_cols);
    free(length_cols);
    free(tone_cols);
    free(stress_cols);
    loader_table_clear(&table);
    if (status == RG_OK) {
        /* A wide row with a single filled cell has nothing to align against. */
        status = corpus_publish(corpus, 2);
    }
    if (status != RG_OK) {
        rg_corpus_free(corpus);
        return status;
    }
    *out = corpus;
    return RG_OK;
}
