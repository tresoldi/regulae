#include "internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Loaders for external cognate-set data formats: a generic TSV loader plus
 * format-specific loaders for GLED and arcaverborum (Lexibank-derived) data.
 * Each produces an rg_corpus owning every cognate set, so downstream training
 * is format-agnostic. Mirrors loaders.go.
 *
 * Alignment hint columns are parsed for validation but not retained: the Go
 * reference documents CognateSet.Alignments as reserved and never consumes it
 * during training. */

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
    rg_cognate_form **view_forms;
    size_t view_form_group_count;
};

/* ---- delimited reading -------------------------------------------------- */

static void loader_table_clear(loader_table *table) {
    size_t i;
    size_t j;
    if (table == 0) {
        return;
    }
    for (i = 0; i < table->column_count; i++) {
        free(table->header[i]);
    }
    free(table->header);
    for (i = 0; i < table->row_count; i++) {
        for (j = 0; j < table->rows[i].field_count; j++) {
            free(table->rows[i].fields[j]);
        }
        free(table->rows[i].fields);
    }
    free(table->rows);
    memset(table, 0, sizeof(*table));
}

static rg_status read_whole_file(const char *path, char **out, size_t *out_len) {
    FILE *fh;
    char *buffer = 0;
    size_t len = 0;
    size_t cap = 0;

    *out = 0;
    *out_len = 0;
    fh = fopen(path, "rb");
    if (fh == 0) {
        return RG_ERR_IO;
    }
    for (;;) {
        size_t got;
        if (len + 65536 + 1 > cap) {
            size_t next_cap = cap == 0 ? 131072 : cap * 2;
            char *next = (char *)realloc(buffer, next_cap);
            if (next == 0) {
                free(buffer);
                fclose(fh);
                return RG_ERR_OOM;
            }
            buffer = next;
            cap = next_cap;
        }
        got = fread(buffer + len, 1, 65536, fh);
        len += got;
        if (got < 65536) {
            break;
        }
    }
    if (ferror(fh)) {
        free(buffer);
        fclose(fh);
        return RG_ERR_IO;
    }
    fclose(fh);
    if (buffer == 0) {
        buffer = (char *)malloc(1);
        if (buffer == 0) {
            return RG_ERR_OOM;
        }
    }
    buffer[len] = '\0';
    *out = buffer;
    *out_len = len;
    return RG_OK;
}

static rg_status row_append_field(loader_row *row, size_t *cap, const char *start, size_t length) {
    char *value;
    if (row->field_count == *cap) {
        size_t next_cap = *cap == 0 ? 8 : *cap * 2;
        char **next = (char **)realloc(row->fields, next_cap * sizeof(*next));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        row->fields = next;
        *cap = next_cap;
    }
    value = (char *)malloc(length + 1);
    if (value == 0) {
        return RG_ERR_OOM;
    }
    memcpy(value, start, length);
    value[length] = '\0';
    row->fields[row->field_count] = value;
    row->field_count++;
    return RG_OK;
}

/* Parses one record starting at *cursor. Handles quoted fields with doubled
 * quotes, and is lenient about stray quotes inside unquoted fields, matching
 * the Go reader's LazyQuotes setting. */
static rg_status parse_record(const char **cursor, const char *end, char delim, loader_row *row, int *have_row) {
    char *scratch = 0;
    size_t scratch_cap = 0;
    size_t scratch_len = 0;
    size_t field_cap = 0;
    const char *p = *cursor;
    rg_status status = RG_OK;

    memset(row, 0, sizeof(*row));
    *have_row = 0;
    if (p >= end) {
        return RG_OK;
    }
    *have_row = 1;
    for (;;) {
        int in_quotes = 0;
        scratch_len = 0;
        if (p < end && *p == '"') {
            in_quotes = 1;
            p++;
        }
        for (;;) {
            char ch;
            if (p >= end) {
                break;
            }
            ch = *p;
            if (in_quotes) {
                if (ch == '"') {
                    if (p + 1 < end && p[1] == '"') {
                        p += 2;
                        ch = '"';
                    } else {
                        p++;
                        in_quotes = 0;
                        continue;
                    }
                } else {
                    p++;
                }
            } else {
                if (ch == delim || ch == '\n' || ch == '\r') {
                    break;
                }
                p++;
            }
            if (scratch_len + 1 > scratch_cap) {
                size_t next_cap = scratch_cap == 0 ? 64 : scratch_cap * 2;
                char *next = (char *)realloc(scratch, next_cap);
                if (next == 0) {
                    free(scratch);
                    return RG_ERR_OOM;
                }
                scratch = next;
                scratch_cap = next_cap;
            }
            scratch[scratch_len++] = ch;
        }
        status = row_append_field(row, &field_cap, scratch == 0 ? "" : scratch, scratch_len);
        if (status != RG_OK) {
            free(scratch);
            return status;
        }
        if (p < end && *p == delim) {
            p++;
            continue;
        }
        break;
    }
    if (p < end && *p == '\r') {
        p++;
    }
    if (p < end && *p == '\n') {
        p++;
    }
    free(scratch);
    *cursor = p;
    return RG_OK;
}

/* Parses an already-loaded buffer. Kept separate from the file wrapper so a
 * corpus held in memory needs no temporary file, which is what the WebAssembly
 * build requires: it links without a filesystem. */
static rg_status read_table_from_buffer(const char *data, size_t len, char delim, loader_table *out) {
    const char *cursor;
    const char *end;
    size_t row_cap = 0;
    rg_status status = RG_OK;

    memset(out, 0, sizeof(*out));
    if (data == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    cursor = data;
    end = data + len;

    {
        loader_row header_row;
        int have_row = 0;
        status = parse_record(&cursor, end, delim, &header_row, &have_row);
        if (status != RG_OK || !have_row) {
            return status;
        }
        out->header = header_row.fields;
        out->column_count = header_row.field_count;
    }
    while (cursor < end) {
        loader_row row;
        int have_row = 0;
        status = parse_record(&cursor, end, delim, &row, &have_row);
        if (status != RG_OK) {
            loader_table_clear(out);
            return status;
        }
        if (!have_row) {
            break;
        }
        /* Skip blank lines, which parse as a single empty field. */
        if (row.field_count == 1 && row.fields[0][0] == '\0') {
            free(row.fields[0]);
            free(row.fields);
            continue;
        }
        if (out->row_count == row_cap) {
            size_t next_cap = row_cap == 0 ? 64 : row_cap * 2;
            loader_row *next = (loader_row *)realloc(out->rows, next_cap * sizeof(*next));
            if (next == 0) {
                size_t i;
                for (i = 0; i < row.field_count; i++) {
                    free(row.fields[i]);
                }
                free(row.fields);
                loader_table_clear(out);
                return RG_ERR_OOM;
            }
            out->rows = next;
            row_cap = next_cap;
        }
        out->rows[out->row_count] = row;
        out->row_count++;
    }
    return RG_OK;
}

static rg_status read_table(const char *path, char delim, loader_table *out) {
    char *data = 0;
    size_t len = 0;
    rg_status status;

    memset(out, 0, sizeof(*out));
    status = read_whole_file(path, &data, &len);
    if (status != RG_OK) {
        return status;
    }
    status = read_table_from_buffer(data, len, delim, out);
    free(data);
    return status;
}

/* One of path or text is set; the other is null. */
static rg_status read_table_source(const char *path, const char *text, char delim, loader_table *out) {
    if (text != 0) {
        return read_table_from_buffer(text, strlen(text), delim, out);
    }
    return read_table(path, delim, out);
}

static long column_index(const loader_table *table, const char *name) {
    size_t i;
    if (name == 0) {
        return -1;
    }
    for (i = 0; i < table->column_count; i++) {
        if (strcmp(table->header[i], name) == 0) {
            return (long)i;
        }
    }
    return -1;
}

static const char *cell(const loader_row *row, long index) {
    if (index < 0 || (size_t)index >= row->field_count) {
        return "";
    }
    return row->fields[index];
}

static char *trim_copy(const char *value) {
    size_t start = 0;
    size_t end;
    char *out;
    if (value == 0) {
        value = "";
    }
    end = strlen(value);
    while (start < end && (value[start] == ' ' || value[start] == '\t')) {
        start++;
    }
    while (end > start && (value[end - 1] == ' ' || value[end - 1] == '\t')) {
        end--;
    }
    out = (char *)malloc(end - start + 1);
    if (out == 0) {
        return 0;
    }
    memcpy(out, value + start, end - start);
    out[end - start] = '\0';
    return out;
}

/* ---- segment parsing ---------------------------------------------------- */

/* Splits a whitespace-separated segment cell. "-" gap markers are dropped and,
 * when track_boundaries is set, "+" tokens record a morpheme boundary at the
 * current position instead of producing a segment. */
static rg_status parse_segments(
    const char *raw,
    int track_boundaries,
    rg_segment **out_segments,
    size_t *out_count,
    int **out_breaks,
    size_t *out_break_count
) {
    rg_segment *segments = 0;
    size_t count = 0;
    size_t cap = 0;
    int *breaks = 0;
    size_t break_count = 0;
    size_t break_cap = 0;
    const char *p = raw == 0 ? "" : raw;

    *out_segments = 0;
    *out_count = 0;
    if (out_breaks != 0) {
        *out_breaks = 0;
        *out_break_count = 0;
    }
    for (;;) {
        const char *start;
        size_t length;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
            p++;
        }
        if (*p == '\0') {
            break;
        }
        start = p;
        while (*p != '\0' && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') {
            p++;
        }
        length = (size_t)(p - start);
        if (length == 1 && start[0] == '-') {
            continue;
        }
        if (length == 1 && start[0] == '+') {
            if (!track_boundaries || count == 0) {
                continue;
            }
            if (break_count == break_cap) {
                size_t next_cap = break_cap == 0 ? 4 : break_cap * 2;
                int *next = (int *)realloc(breaks, next_cap * sizeof(*next));
                if (next == 0) {
                    goto fail;
                }
                breaks = next;
                break_cap = next_cap;
            }
            breaks[break_count++] = (int)count;
            continue;
        }
        if (count == cap) {
            size_t next_cap = cap == 0 ? 8 : cap * 2;
            rg_segment *next = (rg_segment *)realloc(segments, next_cap * sizeof(*next));
            if (next == 0) {
                goto fail;
            }
            segments = next;
            cap = next_cap;
        }
        memset(&segments[count], 0, sizeof(segments[count]));
        {
            char *grapheme = (char *)malloc(length + 1);
            if (grapheme == 0) {
                goto fail;
            }
            memcpy(grapheme, start, length);
            grapheme[length] = '\0';
            segments[count].grapheme = grapheme;
        }
        count++;
    }
    *out_segments = segments;
    *out_count = count;
    if (out_breaks != 0) {
        *out_breaks = breaks;
        *out_break_count = break_count;
    } else {
        free(breaks);
    }
    return RG_OK;

fail:
    {
        size_t i;
        for (i = 0; i < count; i++) {
            free((char *)segments[i].grapheme);
        }
    }
    free(segments);
    free(breaks);
    return RG_ERR_OOM;
}

/* Attaches a tone cell's whitespace-separated values to already-parsed
 * segments, by position. A value replaces whatever the word itself carried, so
 * an explicit column wins over tone written into the transcription. "-" and
 * empty tokens leave the segment as segmentation found it, so a tone-bearing
 * corpus can mark its consonants without inventing a tone for them, and a
 * corpus that annotates only some segments does not erase the rest.
 * Rejects a cell whose token count disagrees with the segment count:
 * silently truncating would tone the wrong vowels, and a corpus that annotates
 * tone at all is annotating it deliberately. */
static rg_status attach_tones(const char *raw, rg_segment *segments, size_t segment_count) {
    const char *p = raw == 0 ? "" : raw;
    size_t index = 0;
    for (;;) {
        const char *start;
        size_t length;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
            p++;
        }
        if (*p == '\0') {
            break;
        }
        start = p;
        while (*p != '\0' && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') {
            p++;
        }
        length = (size_t)(p - start);
        if (index >= segment_count) {
            return RG_ERR_PARSE;
        }
        if (length == 1 && start[0] == '-') {
            index++;
            continue;
        }
        {
            char *tone = (char *)malloc(length + 1);
            if (tone == 0) {
                return RG_ERR_OOM;
            }
            memcpy(tone, start, length);
            tone[length] = '\0';
            free((char *)segments[index].tone);
            segments[index].tone = tone;
        }
        index++;
    }
    if (index != 0 && index != segment_count) {
        return RG_ERR_PARSE;
    }
    return RG_OK;
}

/* Counts the tokens of an alignment cell, where "-" marks a gap. Only the
 * length is meaningful to the loaders, which use it to reject cognate sets
 * whose alignment hints disagree across lects. */
static long alignment_token_count(const char *raw) {
    long count = 0;
    const char *p = raw == 0 ? "" : raw;
    for (;;) {
        while (*p == ' ' || *p == '\t') {
            p++;
        }
        if (*p == '\0') {
            break;
        }
        while (*p != '\0' && *p != ' ' && *p != '\t') {
            p++;
        }
        count++;
    }
    return count;
}

/* ---- accumulation ------------------------------------------------------- */

static void loader_form_clear(loader_form *form) {
    size_t i;
    if (form == 0) {
        return;
    }
    free(form->lect_id);
    for (i = 0; i < form->segment_count; i++) {
        free((char *)form->segments[i].grapheme);
        free((char *)form->segments[i].tone);
    }
    free(form->segments);
    free(form->morpheme_breaks);
    memset(form, 0, sizeof(*form));
}

static void loader_cognate_clear(loader_cognate *cognate) {
    size_t i;
    if (cognate == 0) {
        return;
    }
    free(cognate->cognate_id);
    for (i = 0; i < cognate->form_count; i++) {
        loader_form_clear(&cognate->forms[i]);
    }
    free(cognate->forms);
    memset(cognate, 0, sizeof(*cognate));
}

/* Appends a new cognate set, disambiguating an id that is already taken. In
 * wide format each row is its own cognate set and the id column is a gloss
 * that may legitimately repeat: two rows glossed "die" are two cognate sets,
 * not one, and merging them would silently drop data. */
static loader_cognate *corpus_append_cognate(rg_corpus *corpus, const char *base_id) {
    char candidate[512];
    size_t attempt = 1;
    size_t i;

    snprintf(candidate, sizeof(candidate), "%s", base_id);
    for (;;) {
        int taken = 0;
        for (i = 0; i < corpus->count; i++) {
            if (strcmp(corpus->cognates[i].cognate_id, candidate) == 0) {
                taken = 1;
                break;
            }
        }
        if (!taken) {
            break;
        }
        attempt++;
        snprintf(candidate, sizeof(candidate), "%s.%lu", base_id, (unsigned long)attempt);
    }
    if (corpus->count == corpus->cap) {
        size_t next_cap = corpus->cap == 0 ? 64 : corpus->cap * 2;
        loader_cognate *next = (loader_cognate *)realloc(corpus->cognates, next_cap * sizeof(*next));
        if (next == 0) {
            return 0;
        }
        corpus->cognates = next;
        corpus->cap = next_cap;
    }
    memset(&corpus->cognates[corpus->count], 0, sizeof(corpus->cognates[corpus->count]));
    corpus->cognates[corpus->count].cognate_id = rg_strdup_internal(candidate);
    if (corpus->cognates[corpus->count].cognate_id == 0) {
        return 0;
    }
    corpus->cognates[corpus->count].confidence = 1.0;
    corpus->cognates[corpus->count].alignment_length = -1;
    corpus->count++;
    return &corpus->cognates[corpus->count - 1];
}

static loader_cognate *corpus_ensure_cognate(rg_corpus *corpus, const char *cognate_id) {
    size_t i;
    for (i = 0; i < corpus->count; i++) {
        if (strcmp(corpus->cognates[i].cognate_id, cognate_id) == 0) {
            return &corpus->cognates[i];
        }
    }
    if (corpus->count == corpus->cap) {
        size_t next_cap = corpus->cap == 0 ? 64 : corpus->cap * 2;
        loader_cognate *next = (loader_cognate *)realloc(corpus->cognates, next_cap * sizeof(*next));
        if (next == 0) {
            return 0;
        }
        corpus->cognates = next;
        corpus->cap = next_cap;
    }
    memset(&corpus->cognates[corpus->count], 0, sizeof(corpus->cognates[corpus->count]));
    corpus->cognates[corpus->count].cognate_id = rg_strdup_internal(cognate_id);
    if (corpus->cognates[corpus->count].cognate_id == 0) {
        return 0;
    }
    corpus->cognates[corpus->count].confidence = 1.0;
    corpus->cognates[corpus->count].alignment_length = -1;
    corpus->count++;
    return &corpus->cognates[corpus->count - 1];
}

static loader_form *cognate_find_form(loader_cognate *cognate, const char *lect_id) {
    size_t i;
    for (i = 0; i < cognate->form_count; i++) {
        if (strcmp(cognate->forms[i].lect_id, lect_id) == 0) {
            return &cognate->forms[i];
        }
    }
    return 0;
}

static rg_status cognate_append_form(loader_cognate *cognate, loader_form *form) {
    if (cognate->form_count == cognate->form_cap) {
        size_t next_cap = cognate->form_cap == 0 ? 4 : cognate->form_cap * 2;
        loader_form *next = (loader_form *)realloc(cognate->forms, next_cap * sizeof(*next));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        cognate->forms = next;
        cognate->form_cap = next_cap;
    }
    cognate->forms[cognate->form_count] = *form;
    cognate->form_count++;
    return RG_OK;
}

static int string_list_contains(const char *const *items, size_t count, const char *value) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (strcmp(items[i], value) == 0) {
            return 1;
        }
    }
    return 0;
}

/* Publishes borrowed rg_cognate_set views over the accumulated data, dropping
 * cognate sets with fewer than min_lects forms. */
static rg_status corpus_publish(rg_corpus *corpus, int min_lects) {
    size_t kept = 0;
    size_t i;

    for (i = 0; i < corpus->count; i++) {
        if (min_lects > 0 && corpus->cognates[i].form_count < (size_t)min_lects) {
            continue;
        }
        kept++;
    }
    corpus->view = (rg_cognate_set *)calloc(kept == 0 ? 1 : kept, sizeof(*corpus->view));
    corpus->view_forms = (rg_cognate_form **)calloc(kept == 0 ? 1 : kept, sizeof(*corpus->view_forms));
    if (corpus->view == 0 || corpus->view_forms == 0) {
        return RG_ERR_OOM;
    }
    corpus->view_form_group_count = kept;
    kept = 0;
    for (i = 0; i < corpus->count; i++) {
        loader_cognate *cognate = &corpus->cognates[i];
        rg_cognate_form *forms;
        size_t f;
        if (min_lects > 0 && cognate->form_count < (size_t)min_lects) {
            continue;
        }
        forms = (rg_cognate_form *)calloc(cognate->form_count == 0 ? 1 : cognate->form_count, sizeof(*forms));
        if (forms == 0) {
            return RG_ERR_OOM;
        }
        corpus->view_forms[kept] = forms;
        for (f = 0; f < cognate->form_count; f++) {
            forms[f].lect_id = cognate->forms[f].lect_id;
            forms[f].form.lect_id = cognate->forms[f].lect_id;
            forms[f].form.segments = cognate->forms[f].segments;
            forms[f].form.segment_count = cognate->forms[f].segment_count;
            forms[f].form.morpheme_breaks = cognate->forms[f].morpheme_breaks;
            forms[f].form.morpheme_break_count = cognate->forms[f].morpheme_break_count;
        }
        corpus->view[kept].cognate_id = cognate->cognate_id;
        corpus->view[kept].forms = forms;
        corpus->view[kept].form_count = cognate->form_count;
        corpus->view[kept].confidence = cognate->confidence;
        kept++;
    }
    corpus->view_count = kept;
    return RG_OK;
}

void rg_corpus_free(rg_corpus *corpus) {
    size_t i;
    if (corpus == 0) {
        return;
    }
    for (i = 0; i < corpus->count; i++) {
        loader_cognate_clear(&corpus->cognates[i]);
    }
    free(corpus->cognates);
    for (i = 0; i < corpus->view_form_group_count; i++) {
        free(corpus->view_forms[i]);
    }
    free(corpus->view_forms);
    free(corpus->view);
    free(corpus);
}

size_t rg_corpus_cognate_count(const rg_corpus *corpus) {
    return corpus == 0 ? 0 : corpus->view_count;
}

const rg_cognate_set *rg_corpus_cognates(const rg_corpus *corpus) {
    return corpus == 0 ? 0 : corpus->view;
}

const rg_cognate_set *rg_corpus_cognate_at(const rg_corpus *corpus, size_t index) {
    if (corpus == 0 || index >= corpus->view_count) {
        return 0;
    }
    return &corpus->view[index];
}

/* ---- wide-format TSV ---------------------------------------------------- */

static int has_suffix(const char *value, const char *suffix) {
    size_t v = strlen(value);
    size_t s = strlen(suffix);
    return v >= s && strcmp(value + v - s, suffix) == 0;
}

/* Parses a "<lect>_breaks" cell: comma-separated morpheme boundary indices,
 * with "-" or an empty cell meaning none. */
static rg_status parse_break_indices(const char *raw, int **out, size_t *out_count) {
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

static rg_status load_wide_tsv(
    const rg_context *ctx,
    const char *path,
    const char *text,
    const rg_wide_load_options *options,
    rg_corpus **out
) {
    loader_table table;
    rg_corpus *corpus = 0;
    rg_wide_load_options opts;
    long id_col = -1;
    long confidence_col = -1;
    long *lect_cols = 0;
    long *break_cols = 0;
    long *tone_cols = 0;
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
        loader_table_clear(&table);
        return RG_ERR_PARSE;
    }
    confidence_col = column_index(&table, opts.confidence_column == 0 ? "confidence" : opts.confidence_column);

    lect_cols = (long *)calloc(table.column_count, sizeof(*lect_cols));
    break_cols = (long *)calloc(table.column_count, sizeof(*break_cols));
    tone_cols = (long *)calloc(table.column_count, sizeof(*tone_cols));
    if (lect_cols == 0 || break_cols == 0 || tone_cols == 0) {
        free(lect_cols);
        free(break_cols);
        free(tone_cols);
        loader_table_clear(&table);
        return RG_ERR_OOM;
    }
    if (opts.lect_column_count > 0 && opts.lect_columns != 0) {
        for (c = 0; c < opts.lect_column_count; c++) {
            long index = column_index(&table, opts.lect_columns[c]);
            if (index < 0) {
                free(lect_cols);
                free(break_cols);
                free(tone_cols);
                loader_table_clear(&table);
                return RG_ERR_PARSE;
            }
            lect_cols[lect_count++] = index;
        }
    } else {
        /* Every column is a lect except the id, the confidence, and the
         * companion "_breaks" and "_tone" columns. */
        for (c = 0; c < table.column_count; c++) {
            if ((long)c == id_col || (long)c == confidence_col) {
                continue;
            }
            if (has_suffix(table.header[c], "_breaks") || has_suffix(table.header[c], "_tone")) {
                continue;
            }
            lect_cols[lect_count++] = (long)c;
        }
    }
    if (lect_count == 0) {
        free(lect_cols);
        free(break_cols);
        free(tone_cols);
        loader_table_clear(&table);
        return RG_ERR_PARSE;
    }
    for (c = 0; c < lect_count; c++) {
        char companion[256];
        snprintf(companion, sizeof(companion), "%s_breaks", table.header[lect_cols[c]]);
        break_cols[c] = column_index(&table, companion);
        snprintf(companion, sizeof(companion), "%s_tone", table.header[lect_cols[c]]);
        tone_cols[c] = column_index(&table, companion);
    }

    corpus = (rg_corpus *)calloc(1, sizeof(*corpus));
    if (corpus == 0) {
        free(lect_cols);
        free(break_cols);
        free(tone_cols);
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
            if (cognate_find_form(cognate, lect_id) != 0) {
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
                if (status != RG_OK) {
                    loader_form_clear(&form);
                    break;
                }
            }
            if (tone_cols[c] >= 0) {
                status = attach_tones(cell(row, tone_cols[c]), form.segments, form.segment_count);
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
    free(tone_cols);
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

/* ---- pairwise corpora --------------------------------------------------- */

static rg_status loader_form_from(const rg_form *src, const char *lect_id, loader_form *out) {
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
             * them here silently untoned every corpus built from form pairs. */
            if (src->segments[i].tone != 0 && src->segments[i].tone[0] != '\0') {
                out->segments[i].tone = rg_strdup_internal(src->segments[i].tone);
                if (out->segments[i].tone == 0) {
                    out->segment_count = i + 1;
                    loader_form_clear(out);
                    return RG_ERR_OOM;
                }
            }
        }
        out->segment_count = src->segment_count;
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
        status = cognate_append_form(cognate, &target);
        if (status != RG_OK) {
            loader_form_clear(&target);
            break;
        }
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

static rg_status load_tsv(const char *path, const char *text, const rg_tsv_load_options *options, rg_corpus **out) {
    loader_table table;
    rg_corpus *corpus = 0;
    rg_tsv_load_options opts;
    long cognate_col;
    long lect_col;
    long segments_col;
    long alignment_col = -1;
    long confidence_col = -1;
    long tone_col = -1;
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

    status = read_table_source(path, text, '\t', &table);
    if (status != RG_OK) {
        return status;
    }
    if (table.column_count == 0) {
        loader_table_clear(&table);
        return RG_ERR_PARSE;
    }
    cognate_col = column_index(&table, opts.cognate_id_column);
    lect_col = column_index(&table, opts.lect_id_column);
    segments_col = column_index(&table, opts.segments_column);
    if (cognate_col < 0 || lect_col < 0 || segments_col < 0) {
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

        if (cognate_id == 0 || lect_id == 0) {
            free(cognate_id);
            free(lect_id);
            status = RG_ERR_OOM;
            break;
        }
        if (cognate_id[0] == '\0' || lect_id[0] == '\0') {
            free(cognate_id);
            free(lect_id);
            continue;
        }
        memset(&form, 0, sizeof(form));
        status = parse_segments(cell(row, segments_col), 0, &form.segments, &form.segment_count, 0, 0);
        if (status != RG_OK) {
            free(cognate_id);
            free(lect_id);
            break;
        }
        if (form.segment_count == 0) {
            loader_form_clear(&form);
            free(cognate_id);
            free(lect_id);
            continue;
        }
        if (tone_col >= 0) {
            status = attach_tones(cell(row, tone_col), form.segments, form.segment_count);
            if (status != RG_OK) {
                loader_form_clear(&form);
                free(cognate_id);
                free(lect_id);
                break;
            }
        }
        cognate = corpus_ensure_cognate(corpus, cognate_id);
        free(cognate_id);
        if (cognate == 0) {
            loader_form_clear(&form);
            free(lect_id);
            status = RG_ERR_OOM;
            break;
        }
        if (cognate_find_form(cognate, lect_id) != 0) {
            /* The generic loader treats a repeated (cognate, lect) row as a
             * data error rather than silently keeping the first reflex. */
            loader_form_clear(&form);
            free(lect_id);
            status = RG_ERR_PARSE;
            break;
        }
        form.lect_id = lect_id;
        status = cognate_append_form(cognate, &form);
        if (status != RG_OK) {
            loader_form_clear(&form);
            break;
        }
        if (alignment_col >= 0) {
            long length = alignment_token_count(cell(row, alignment_col));
            if (length > 0) {
                if (cognate->alignment_length >= 0 && cognate->alignment_length != length) {
                    status = RG_ERR_PARSE;
                    break;
                }
                cognate->alignment_length = length;
            }
        }
        if (confidence_col >= 0) {
            char *raw = trim_copy(cell(row, confidence_col));
            if (raw == 0) {
                status = RG_ERR_OOM;
                break;
            }
            if (raw[0] != '\0') {
                char *endptr = 0;
                double value = strtod(raw, &endptr);
                if (endptr == raw || (endptr != 0 && *endptr != '\0')) {
                    free(raw);
                    status = RG_ERR_PARSE;
                    break;
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

/* ---- GLED --------------------------------------------------------------- */

static rg_status load_gled(const char *path, const char *text, const rg_gled_load_options *options, rg_corpus **out) {
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
        if (cognate_find_form(cognate, lect_id) != 0) {
            /* Keep the first reflex per (lect, cognate). */
            loader_form_clear(&form);
            continue;
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

/* ---- arcaverborum ------------------------------------------------------- */

static rg_status load_arcaverborum(
    const char *path,
    const char *text,
    const rg_arcaverborum_load_options *options,
    rg_corpus **out
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
        if (cognate_find_form(cognate, lect_id) != 0) {
            loader_form_clear(&form);
            continue;
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

/* ---- public entry points ------------------------------------------------ */

rg_status rg_corpus_load_tsv(const char *path, const rg_tsv_load_options *options, rg_corpus **out) {
    return load_tsv(path, 0, options, out);
}

rg_status rg_corpus_parse_tsv(const char *text, const rg_tsv_load_options *options, rg_corpus **out) {
    return load_tsv(0, text, options, out);
}

rg_status rg_corpus_load_wide_tsv(
    const rg_context *ctx,
    const char *path,
    const rg_wide_load_options *options,
    rg_corpus **out
) {
    return load_wide_tsv(ctx, path, 0, options, out);
}

rg_status rg_corpus_parse_wide_tsv(
    const rg_context *ctx,
    const char *text,
    const rg_wide_load_options *options,
    rg_corpus **out
) {
    return load_wide_tsv(ctx, 0, text, options, out);
}

rg_status rg_corpus_load_gled(const char *path, const rg_gled_load_options *options, rg_corpus **out) {
    return load_gled(path, 0, options, out);
}

rg_status rg_corpus_parse_gled(const char *text, const rg_gled_load_options *options, rg_corpus **out) {
    return load_gled(0, text, options, out);
}

rg_status rg_corpus_load_arcaverborum(
    const char *path,
    const rg_arcaverborum_load_options *options,
    rg_corpus **out
) {
    return load_arcaverborum(path, 0, options, out);
}

rg_status rg_corpus_parse_arcaverborum(
    const char *text,
    const rg_arcaverborum_load_options *options,
    rg_corpus **out
) {
    return load_arcaverborum(0, text, options, out);
}
