#include "loader_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- pairwise corpora --------------------------------------------------- */

rg_status loader_form_from(const rg_form *src, const char *lect_id, loader_form *out) {
    size_t i;
    memset(out, 0, sizeof(*out));
    out->lect_id = rg_strdup_internal(lect_id);
    if (out->lect_id == 0) {
        return RG_ERR_OOM;
    }
    if (src->segment_count > 0) {
        out->segments = (rg_segment *)calloc(src->segment_count, sizeof(*out->segments));
        if (out->segments == 0) {
            loader_form_clear(out);
            return RG_ERR_OOM;
        }
        for (i = 0; i < src->segment_count; i++) {
            out->segments[i].grapheme = rg_strdup_internal(
                src->segments[i].grapheme == 0 ? "" : src->segments[i].grapheme);
            if (out->segments[i].grapheme == 0) {
                out->segment_count = i + 1;
                loader_form_clear(out);
                return RG_ERR_OOM;
            }
            /* Suprasegmentals are part of the segment's identity, and dropping
             * them here silently untoned -- and later unstressed -- every
             * corpus built from form pairs. */
            if (src->segments[i].tone != 0 && src->segments[i].tone[0] != '\0') {
                out->segments[i].tone = rg_strdup_internal(src->segments[i].tone);
                if (out->segments[i].tone == 0) {
                    out->segment_count = i + 1;
                    loader_form_clear(out);
                    return RG_ERR_OOM;
                }
            }
            if (src->segments[i].stress != 0 && src->segments[i].stress[0] != '\0') {
                out->segments[i].stress = rg_strdup_internal(src->segments[i].stress);
                if (out->segments[i].stress == 0) {
                    out->segment_count = i + 1;
                    loader_form_clear(out);
                    return RG_ERR_OOM;
                }
            }
            if (src->segments[i].length != 0 && src->segments[i].length[0] != '\0') {
                out->segments[i].length = rg_strdup_internal(src->segments[i].length);
                if (out->segments[i].length == 0) {
                    out->segment_count = i + 1;
                    loader_form_clear(out);
                    return RG_ERR_OOM;
                }
            }
        }
        out->segment_count = src->segment_count;
    }
    if (src->syllable_break_count > 0 && src->syllable_breaks != 0) {
        out->syllable_breaks = (int *)calloc(src->syllable_break_count, sizeof(*out->syllable_breaks));
        if (out->syllable_breaks == 0) {
            /* Every other failure here clears the partly-built form before
             * returning, and this one did not: the lect id and every segment
             * copied so far leaked. The caller cannot clean up after it,
             * because a failed loader_form_from is documented as leaving
             * nothing. clang-analyzer found it once the split put this
             * function and its caller in view of each other. */
            loader_form_clear(out);
            return RG_ERR_OOM;
        }
        memcpy(out->syllable_breaks, src->syllable_breaks, src->syllable_break_count * sizeof(int));
        out->syllable_break_count = src->syllable_break_count;
    }
    if (src->morpheme_break_count > 0 && src->morpheme_breaks != 0) {
        out->morpheme_breaks = (int *)calloc(src->morpheme_break_count, sizeof(*out->morpheme_breaks));
        if (out->morpheme_breaks == 0) {
            loader_form_clear(out);
            return RG_ERR_OOM;
        }
        memcpy(out->morpheme_breaks, src->morpheme_breaks, src->morpheme_break_count * sizeof(int));
        out->morpheme_break_count = src->morpheme_break_count;
    }
    return RG_OK;
}

/* Lifts a directed pairwise corpus into cognate sets, so a two-lect study can
 * go through the same multi-lect entry point as everything else. Cognate ids
 * are the prefix plus the pair's index, keeping input order recoverable. */
rg_status rg_corpus_from_pairs(
    const rg_form_pair *pairs,
    size_t pair_count,
    const char *lect_a,
    const char *lect_b,
    const char *cognate_id_prefix,
    rg_corpus **out
) {
    rg_corpus *corpus;
    size_t i;
    rg_status status = RG_OK;

    if (out == 0 || (pair_count > 0 && pairs == 0) || lect_a == 0 || lect_b == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    if (lect_a[0] == '\0' || lect_b[0] == '\0' || strcmp(lect_a, lect_b) == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    if (cognate_id_prefix == 0 || cognate_id_prefix[0] == '\0') {
        cognate_id_prefix = "pair";
    }
    corpus = (rg_corpus *)calloc(1, sizeof(*corpus));
    if (corpus == 0) {
        return RG_ERR_OOM;
    }
    for (i = 0; i < pair_count && status == RG_OK; i++) {
        char cognate_id[256];
        loader_cognate *cognate;
        loader_form source;
        loader_form target;

        snprintf(cognate_id, sizeof(cognate_id), "%s.%05lu", cognate_id_prefix, (unsigned long)i);
        cognate = corpus_ensure_cognate(corpus, cognate_id);
        if (cognate == 0) {
            status = RG_ERR_OOM;
            break;
        }
        status = loader_form_from(&pairs[i].source, lect_a, &source);
        if (status != RG_OK) {
            break;
        }
        status = loader_form_from(&pairs[i].target, lect_b, &target);
        if (status != RG_OK) {
            loader_form_clear(&source);
            break;
        }
        status = cognate_append_form(cognate, &source);
        if (status != RG_OK) {
            loader_form_clear(&source);
            loader_form_clear(&target);
            break;
        }
        /* The cognate owns it now. Saying so costs one memset and is what
         * makes the handover visible -- to a reader, and to the analyzer,
         * which otherwise reads the transfer as a leak and had to be told
         * not to with a NOLINT. */
        memset(&source, 0, sizeof(source));
        status = cognate_append_form(cognate, &target);
        if (status != RG_OK) {
            loader_form_clear(&target);
            break;
        }
        memset(&target, 0, sizeof(target));
    }
    if (status == RG_OK) {
        status = corpus_publish(corpus, 1);
    }
    if (status != RG_OK) {
        rg_corpus_free(corpus);
        return status;
    }
    *out = corpus;
    return RG_OK;
}

/* ---- generic TSV -------------------------------------------------------- */

rg_status load_tsv(
    const char *path,
    const char *text,
    const rg_tsv_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
) {
    loader_table table;
    rg_corpus *corpus = 0;
    rg_tsv_load_options opts;
    long cognate_col;
    long lect_col;
    long segments_col;
    long alignment_col = -1;
    long confidence_col = -1;
    long tone_col = -1;
    long breaks_col = -1;
    long syllables_col = -1;
    long length_col = -1;
    long stress_col = -1;
    size_t r;
    rg_status status;

    if ((path == 0 && text == 0) || out == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    memset(&opts, 0, sizeof(opts));
    if (options != 0) {
        opts = *options;
    }
    if (opts.cognate_id_column == 0) {
        opts.cognate_id_column = "cognate_id";
    }
    if (opts.lect_id_column == 0) {
        opts.lect_id_column = "lect_id";
    }
    if (opts.segments_column == 0) {
        opts.segments_column = "segments";
    }
    if (opts.tone_column == 0) {
        opts.tone_column = "tone";
    }
    if (opts.stress_column == 0) {
        opts.stress_column = "stress";
    }

    status = read_table_source(path, text, '\t', &table);
    if (status != RG_OK) {
        return status;
    }
    if (table.column_count == 0) {
        loader_fail(diagnosis, 1, "the file has no columns");
        loader_table_clear(&table);
        return RG_ERR_PARSE;
    }
    cognate_col = column_index(&table, opts.cognate_id_column);
    lect_col = column_index(&table, opts.lect_id_column);
    segments_col = column_index(&table, opts.segments_column);
    if (cognate_col < 0 || lect_col < 0 || segments_col < 0) {
        loader_fail(diagnosis, 1, cognate_col < 0
                    ? "no cognate_id column"
                    : (lect_col < 0 ? "no lect_id column" : "no segments column"));
        loader_table_clear(&table);
        return RG_ERR_PARSE;
    }
    if (opts.alignment_column != 0) {
        alignment_col = column_index(&table, opts.alignment_column);
    }
    if (opts.confidence_column != 0) {
        confidence_col = column_index(&table, opts.confidence_column);
    }
    tone_col = column_index(&table, opts.tone_column);
    /* Morpheme boundaries, as indices into the segment sequence, the same
     * shape the wide loader's <lect>_breaks column carries. Without a column
     * for them the long format could describe a morphologically conditioned
     * change but not supply the boundaries that condition it. */
    breaks_col = column_index(&table, opts.morpheme_breaks_column == 0
                              ? "breaks" : opts.morpheme_breaks_column);
    syllables_col = column_index(&table, opts.syllable_breaks_column == 0
                                 ? "syllables" : opts.syllable_breaks_column);
    length_col = column_index(&table, opts.length_column == 0 ? "length" : opts.length_column);
    stress_col = column_index(&table, opts.stress_column);

    corpus = (rg_corpus *)calloc(1, sizeof(*corpus));
    if (corpus == 0) {
        loader_table_clear(&table);
        return RG_ERR_OOM;
    }

    for (r = 0; r < table.row_count; r++) {
        const loader_row *row = &table.rows[r];
        char *cognate_id = trim_copy(cell(row, cognate_col));
        char *lect_id = trim_copy(cell(row, lect_col));
        loader_cognate *cognate;
        loader_form form;

        memset(&form, 0, sizeof(form));

        if (cognate_id == 0 || lect_id == 0) {
            status = RG_ERR_OOM;
            goto row_done;
        }
        /* An unlabelled row is skipped rather than refused: a wordlist with a
         * blank line in it is not a broken file. status stays RG_OK, so the
         * loop continues. */
        if (cognate_id[0] == '\0' || lect_id[0] == '\0') {
            goto row_done;
        }
        status = parse_segments(cell(row, segments_col), 0, &form.segments, &form.segment_count, 0, 0);
        if (status != RG_OK) {
            goto row_done;
        }
        if (form.segment_count == 0) {
            goto row_done;
        }
        if (stress_col >= 0) {
            status = attach_dimension(cell(row, stress_col), form.segments, form.segment_count, RG_DIMENSION_STRESS);
            if (status != RG_OK) {
                goto row_done;
            }
        }
        if (tone_col >= 0) {
            status = attach_dimension(cell(row, tone_col), form.segments, form.segment_count, RG_DIMENSION_TONE);
            if (status != RG_OK) {
                goto row_done;
            }
        }
        if (breaks_col >= 0) {
            status = parse_break_indices(cell(row, breaks_col), &form.morpheme_breaks,
                                         &form.morpheme_break_count);
            if (status != RG_OK) {
                goto row_done;
            }
        }
        if (length_col >= 0) {
            status = attach_dimension(cell(row, length_col), form.segments, form.segment_count,
                                      RG_DIMENSION_LENGTH);
            if (status != RG_OK) {
                goto row_done;
            }
        }
        if (syllables_col >= 0) {
            status = parse_break_indices(cell(row, syllables_col), &form.syllable_breaks,
                                         &form.syllable_break_count);
            if (status != RG_OK) {
                goto row_done;
            }
        }
        cognate = corpus_ensure_cognate(corpus, cognate_id);
        if (cognate == 0) {
            status = RG_ERR_OOM;
            goto row_done;
        }
        /* The form takes the lect id, and then the cognate takes the form.
         * Each handover is a clearing of the local, so the cleanup below frees
         * exactly what this iteration still owns and nothing that has moved
         * on. */
        form.lect_id = lect_id;
        lect_id = 0;
        status = cognate_append_form(cognate, &form);
        if (status != RG_OK) {
            goto row_done;
        }
        memset(&form, 0, sizeof(form));

        if (alignment_col >= 0) {
            long length = alignment_token_count(cell(row, alignment_col));
            if (length > 0) {
                if (cognate->alignment_length >= 0 && cognate->alignment_length != length) {
                    status = RG_ERR_PARSE;
                    goto row_done;
                }
                cognate->alignment_length = length;
            }
        }
        if (confidence_col >= 0) {
            char *raw = trim_copy(cell(row, confidence_col));
            if (raw == 0) {
                status = RG_ERR_OOM;
                goto row_done;
            }
            if (raw[0] != '\0') {
                char *endptr = 0;
                double value = strtod(raw, &endptr);
                if (endptr == raw || (endptr != 0 && *endptr != '\0')) {
                    free(raw);
                    status = RG_ERR_PARSE;
                    goto row_done;
                }
                /* A cognate set takes the lowest confidence any of its rows
                 * reports, so a single doubtful reflex downweights the set. */
                if (!cognate->has_confidence || value < cognate->confidence) {
                    cognate->confidence = value;
                    cognate->has_confidence = 1;
                }
            }
            free(raw);
        }

    row_done:
        /* One cleanup for the iteration, whichever way it ended. There were
         * eight of these, written out separately, and one of them -- the
         * stress column's -- was missing the two frees the other seven had.
         * The fuzzer found it as a leak on a refused row; this is why it was
         * possible to write. */
        loader_form_clear(&form);
        free(cognate_id);
        free(lect_id);
        if (status != RG_OK) {
            break;
        }
    }
    loader_table_clear(&table);
    if (status == RG_OK) {
        status = corpus_publish(corpus, 1);
    }
    if (status != RG_OK) {
        rg_corpus_free(corpus);
        return status;
    }
    *out = corpus;
    return RG_OK;
}

