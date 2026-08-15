#include "loader_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- arcaverborum ------------------------------------------------------- */

rg_status load_arcaverborum(
    const char *path,
    const char *text,
    const rg_arcaverborum_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
) {
    loader_table table;
    rg_corpus *corpus = 0;
    long language_col;
    long segments_col;
    long cognacy_col;
    long dataset_col;
    long family_col;
    long alignment_col;
    int min_lects = 2;
    char delim = ',';
    size_t path_len;
    size_t r;
    rg_status status;

    if ((path == 0 && text == 0) || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    if (options != 0 && options->min_lects != 0) {
        min_lects = options->min_lects;
    }
    /* The delimiter is taken from the file extension, which a string has none
     * of; callers parsing text set it explicitly, defaulting to comma. */
    if (options != 0 && options->delimiter != 0) {
        delim = options->delimiter;
    } else if (path != 0) {
        path_len = strlen(path);
        if ((path_len >= 4 && strcmp(path + path_len - 4, ".tsv") == 0) ||
            (path_len >= 4 && strcmp(path + path_len - 4, ".txt") == 0)) {
            delim = '\t';
        }
    }
    status = read_table_source(path, text, delim, &table);
    if (status != RG_OK) {
        return status;
    }
    if (table.column_count == 0) {
        loader_fail(diagnosis, 1, "the file has no columns");
        loader_table_clear(&table);
        return RG_ERR_PARSE;
    }
    language_col = column_index(&table, "Language_ID");
    segments_col = column_index(&table, "Segments");
    cognacy_col = column_index(&table, "Cognacy");
    dataset_col = column_index(&table, "Dataset");
    family_col = column_index(&table, "Family");
    alignment_col = column_index(&table, "Alignment");
    if (language_col < 0 || segments_col < 0 || cognacy_col < 0) {
        loader_fail(diagnosis, 1,
                    "not an Arca Verborum table: it needs Language_ID, Segments and Cognacy columns");
        loader_table_clear(&table);
        return RG_ERR_PARSE;
    }
    corpus = (rg_corpus *)calloc(1, sizeof(*corpus));
    if (corpus == 0) {
        loader_table_clear(&table);
        return RG_ERR_OOM;
    }
    for (r = 0; r < table.row_count; r++) {
        const loader_row *row = &table.rows[r];
        const char *lect_id;
        const char *raw_cognacy;
        char *cognate_id;
        const char *semicolon;
        loader_cognate *cognate;
        loader_form form;

        if (options != 0 && options->dataset != 0 && options->dataset[0] != '\0' &&
            (dataset_col < 0 || strcmp(cell(row, dataset_col), options->dataset) != 0)) {
            continue;
        }
        if (options != 0 && options->family != 0 && options->family[0] != '\0' &&
            (family_col < 0 || strcmp(cell(row, family_col), options->family) != 0)) {
            continue;
        }
        lect_id = cell(row, language_col);
        if (lect_id[0] == '\0') {
            continue;
        }
        if (options != 0 && options->language_id_count > 0 &&
            !string_list_contains(options->language_ids, options->language_id_count, lect_id)) {
            continue;
        }
        raw_cognacy = cell(row, cognacy_col);
        if (raw_cognacy[0] == '\0' || strcmp(raw_cognacy, "<NA>") == 0 || strcmp(raw_cognacy, "NA") == 0) {
            continue;
        }
        semicolon = strchr(raw_cognacy, ';');
        if (semicolon != 0) {
            char *first = (char *)malloc((size_t)(semicolon - raw_cognacy) + 1);
            if (first == 0) {
                status = RG_ERR_OOM;
                break;
            }
            memcpy(first, raw_cognacy, (size_t)(semicolon - raw_cognacy));
            first[semicolon - raw_cognacy] = '\0';
            cognate_id = trim_copy(first);
            free(first);
        } else {
            cognate_id = trim_copy(raw_cognacy);
        }
        if (cognate_id == 0) {
            status = RG_ERR_OOM;
            break;
        }
        if (cognate_id[0] == '\0') {
            free(cognate_id);
            continue;
        }
        memset(&form, 0, sizeof(form));
        status = parse_segments(
            cell(row, segments_col),
            1,
            &form.segments,
            &form.segment_count,
            &form.morpheme_breaks,
            &form.morpheme_break_count
        );
        if (status != RG_OK) {
            free(cognate_id);
            break;
        }
        if (form.segment_count == 0) {
            loader_form_clear(&form);
            free(cognate_id);
            continue;
        }
        cognate = corpus_ensure_cognate(corpus, cognate_id);
        free(cognate_id);
        if (cognate == 0) {
            loader_form_clear(&form);
            status = RG_ERR_OOM;
            break;
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
        if (alignment_col >= 0) {
            const char *raw = cell(row, alignment_col);
            if (raw[0] != '\0' && strcmp(raw, "<NA>") != 0 && strcmp(raw, "NA") != 0) {
                long length = alignment_token_count(raw);
                if (length > 0) {
                    if (cognate->alignment_length >= 0 && cognate->alignment_length != length) {
                        cognate->alignment_length = -2;
                    } else if (cognate->alignment_length != -2) {
                        cognate->alignment_length = length;
                    }
                }
            }
        }
    }
    loader_table_clear(&table);
    if (status == RG_OK) {
        status = corpus_publish(corpus, min_lects);
    }
    if (status != RG_OK) {
        rg_corpus_free(corpus);
        return status;
    }
    *out = corpus;
    return RG_OK;
}

