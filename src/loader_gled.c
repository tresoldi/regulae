#include "loader_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- GLED --------------------------------------------------------------- */

rg_status load_gled(
    const char *path,
    const char *text,
    const rg_gled_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
) {
    loader_table table;
    rg_corpus *corpus = 0;
    long doculect_col;
    long family_col;
    long ipa_col;
    long cogset_col;
    long alignment_col;
    int min_lects = 2;
    size_t r;
    rg_status status;

    if ((path == 0 && text == 0) || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    if (options != 0 && options->min_lects != 0) {
        min_lects = options->min_lects;
    }
    status = read_table_source(path, text, '\t', &table);
    if (status != RG_OK) {
        return status;
    }
    doculect_col = column_index(&table, "DOCULECT");
    family_col = column_index(&table, "FAMILY");
    ipa_col = column_index(&table, "IPA");
    cogset_col = column_index(&table, "COGSET");
    alignment_col = column_index(&table, "ALIGNMENT");
    if (doculect_col < 0 || family_col < 0 || ipa_col < 0 || cogset_col < 0) {
        loader_fail(diagnosis, 1,
                    "not a GLED table: it needs DOCULECT, FAMILY, IPA and COGSET columns");
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
        char *cognate_id;
        loader_cognate *cognate;
        loader_form form;

        if (options != 0 && options->family != 0 && options->family[0] != '\0' &&
            strcmp(cell(row, family_col), options->family) != 0) {
            continue;
        }
        lect_id = cell(row, doculect_col);
        if (options != 0 && options->doculect_count > 0 &&
            !string_list_contains(options->doculects, options->doculect_count, lect_id)) {
            continue;
        }
        cognate_id = trim_copy(cell(row, cogset_col));
        if (cognate_id == 0) {
            status = RG_ERR_OOM;
            break;
        }
        if (cognate_id[0] == '\0') {
            free(cognate_id);
            continue;
        }
        memset(&form, 0, sizeof(form));
        status = parse_segments(cell(row, ipa_col), 0, &form.segments, &form.segment_count, 0, 0);
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
            long length = alignment_token_count(cell(row, alignment_col));
            if (length > 0) {
                if (cognate->alignment_length >= 0 && cognate->alignment_length != length) {
                    /* Inconsistent hints are discarded, not fatal. */
                    cognate->alignment_length = -2;
                } else if (cognate->alignment_length != -2) {
                    cognate->alignment_length = length;
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

