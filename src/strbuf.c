#include "strbuf.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void builder_init(string_builder *builder) {
    memset(builder, 0, sizeof(*builder));
}

static void builder_reserve(string_builder *builder, size_t extra) {
    size_t needed = builder->length + extra + 1;
    char *next;
    size_t capacity;
    if (builder->failed || needed <= builder->capacity) {
        return;
    }
    capacity = builder->capacity == 0 ? 256 : builder->capacity;
    while (capacity < needed) {
        capacity *= 2;
    }
    next = (char *)realloc(builder->data, capacity);
    if (next == 0) {
        builder->failed = 1;
        return;
    }
    builder->data = next;
    builder->capacity = capacity;
}

void builder_append(string_builder *builder, const char *text) {
    size_t length;
    if (builder->failed || text == 0) {
        return;
    }
    length = strlen(text);
    builder_reserve(builder, length);
    if (builder->failed) {
        return;
    }
    memcpy(builder->data + builder->length, text, length);
    builder->length += length;
    builder->data[builder->length] = '\0';
}

void builder_appendf(string_builder *builder, const char *format, ...) {
    char scratch[512];
    va_list args;
    int written;
    if (builder->failed) {
        return;
    }
    va_start(args, format);
    written = vsnprintf(scratch, sizeof(scratch), format, args);
    va_end(args);
    if (written < 0) {
        builder->failed = 1;
        return;
    }
    builder_append(builder, scratch);
}

char *builder_finish(string_builder *builder) {
    if (builder->failed) {
        free(builder->data);
        return 0;
    }
    if (builder->data == 0) {
        builder->data = (char *)calloc(1, 1);
    }
    return builder->data;
}
