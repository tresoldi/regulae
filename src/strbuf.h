#ifndef REGULAE_STRBUF_H
#define REGULAE_STRBUF_H

#include <stddef.h>

/* A growing string, and the failure discipline every renderer in this library
 * relies on.
 *
 * It was private to format.c until the notation renderer needed one too. The
 * alternative was a second copy of the same forty lines, which is how the
 * eighteen environment slots came to be enumerated in fourteen places.
 *
 * Allocation failure is recorded rather than reported: every append after a
 * failure is a no-op, so a caller may write a whole report without checking a
 * single return value, and `builder_finish` is the one place that says whether
 * any of it worked. It returns 0 on failure, having freed what was built.
 */
typedef struct string_builder {
    char *data;
    size_t length;
    size_t capacity;
    int failed;
} string_builder;

void builder_init(string_builder *builder);
void builder_append(string_builder *builder, const char *text);

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 2, 3)))
#endif
void builder_appendf(string_builder *builder, const char *format, ...);

/* The built string, ownership transferred, or 0 if any append failed. An empty
 * builder yields an empty string rather than 0, so "nothing to say" and "out of
 * memory" stay distinguishable. */
char *builder_finish(string_builder *builder);

#endif
