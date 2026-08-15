#include "loader_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- delimited reading -------------------------------------------------- */

void loader_table_clear(loader_table *table) {
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

/* parse_record appends fields as it goes and can fail after some of them are
 * built -- on an allocation failure, or on a malformed record. It frees its own
 * scratch buffer and leaves the partly-filled row to its caller, who has to
 * release it. Neither caller did, which is a leak on the out-of-memory path;
 * clang-analyzer found it, and no fuzzer would have, since fuzzing does not
 * produce allocation failures. */
void loader_row_clear(loader_row *row) {
    size_t i;
    if (row == 0) {
        return;
    }
    for (i = 0; i < row->field_count; i++) {
        free(row->fields[i]);
    }
    free(row->fields);
    memset(row, 0, sizeof(*row));
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
            loader_row_clear(&header_row);
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
            loader_row_clear(&row);
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
rg_status read_table_source(const char *path, const char *text, char delim, loader_table *out) {
    if (text != 0) {
        return read_table_from_buffer(text, strlen(text), delim, out);
    }
    return read_table(path, delim, out);
}

long column_index(const loader_table *table, const char *name) {
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

const char *cell(const loader_row *row, long index) {
    if (index < 0 || (size_t)index >= row->field_count) {
        return "";
    }
    return row->fields[index];
}

char *trim_copy(const char *value) {
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

