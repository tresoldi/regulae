#include "regulae.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* WebAssembly adapter: a small JSON-in, JSON-out surface over the C ABI.
 *
 * The library is reached only through the parse-from-string loaders, so this
 * links with -sFILESYSTEM=0 and carries no filesystem shim. Every entry point
 * returns a caller-owned string that JavaScript must hand back to
 * regulae_free. */

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

/* The four functions -sEXPORTED_FUNCTIONS names. Declared here so the compiler
 * can check each definition against the shape the page calls it with, and so
 * -Wmissing-prototypes has something to match: a definition with no prior
 * declaration is how a wasm export and its JavaScript caller drift apart. */
EMSCRIPTEN_KEEPALIVE char *regulae_train_json(
    const char *corpus_text,
    const char *format,
    const char *options_json
);
EMSCRIPTEN_KEEPALIVE char *regulae_segment_json(const char *word);
EMSCRIPTEN_KEEPALIVE const char *regulae_version(void);
EMSCRIPTEN_KEEPALIVE void regulae_free(char *text);

#ifdef __EMSCRIPTEN__
/* Progress crosses into JavaScript. Returning a truthy value from
 * Module.onProgress cancels the run, which surfaces to the caller as an
 * ok:false payload with status "cancelled". */
/* EM_JS's documented spelling ends in a semicolon, which after expansion is a
 * stray one at file scope -- ISO C forbids it and -Wpedantic says so. The macro
 * is emscripten's, so the suppression is scoped to the one construct rather
 * than to the file. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wextra-semi"
EM_JS(int, regulae_js_progress, (const char *stage, int completed, int total), {
    if (typeof Module !== "undefined" && typeof Module.onProgress === "function") {
        return Module.onProgress(UTF8ToString(stage), completed, total) ? 1 : 0;
    }
    return 0;
});
#pragma clang diagnostic pop
#else
static int regulae_js_progress(const char *stage, int completed, int total) {
    (void)stage;
    (void)completed;
    (void)total;
    return 0;
}
#endif

static int progress_bridge(const char *stage, size_t completed, size_t total, void *user_data) {
    (void)user_data;
    return regulae_js_progress(stage, (int)completed, (int)total);
}

/* The context owns merkmal's built-in registry and its caches. It is built
 * once and kept: constructing it per call would re-resolve every grapheme. */
static rg_context *shared_context(void) {
    static rg_context *ctx = 0;
    if (ctx == 0 && rg_context_new_builtin(&ctx) != RG_OK) {
        return 0;
    }
    return ctx;
}

static char *error_payload(rg_status status, const char *detail) {
    char *text = rg_error_to_json(status, detail);
    if (text != 0) {
        return text;
    }
    /* Out of memory while reporting out of memory; hand back a literal so the
     * caller still receives valid JSON. */
    {
        static const char fallback[] = "{\"ok\":false,\"status\":\"out of memory\"}";
        char *copy = (char *)malloc(sizeof(fallback));
        if (copy != 0) {
            memcpy(copy, fallback, sizeof(fallback));
        }
        return copy;
    }
}

/* Trains on a pasted corpus and returns the model as JSON.
 *
 * format is "wide", "tsv", "gled" or "arcaverborum"; a null or empty value
 * means "wide", which is how corpora are actually written. options_json may be
 * null. Alignments and outliers are included, since the caller needs them to
 * render and computing them costs little next to training. */
EMSCRIPTEN_KEEPALIVE
char *regulae_train_json(const char *corpus_text, const char *format, const char *options_json) {
    rg_context *ctx = shared_context();
    rg_corpus *corpus = 0;
    rg_multi_model *model = 0;
    rg_train_options options;
    char detail[256];
    char *text;
    rg_status status;

    if (ctx == 0) {
        return error_payload(RG_ERR_OOM, "could not create the feature context");
    }
    if (corpus_text == 0 || corpus_text[0] == '\0') {
        return error_payload(RG_ERR_INVALID_ARGUMENT, "the corpus is empty");
    }

    status = rg_train_options_from_json(options_json, &options, detail, sizeof(detail));
    if (status != RG_OK) {
        return error_payload(status, detail);
    }
    options.progress = progress_bridge;
    options.progress_user_data = 0;

    if (format == 0 || format[0] == '\0' || strcmp(format, "wide") == 0) {
        status = rg_corpus_parse_wide_tsv(ctx, corpus_text, 0, &corpus);
    } else if (strcmp(format, "tsv") == 0) {
        rg_tsv_load_options tsv_options;
        memset(&tsv_options, 0, sizeof(tsv_options));
        tsv_options.confidence_column = "confidence";
        status = rg_corpus_parse_tsv(corpus_text, &tsv_options, &corpus);
    } else if (strcmp(format, "gled") == 0) {
        status = rg_corpus_parse_gled(corpus_text, 0, &corpus);
    } else if (strcmp(format, "arcaverborum") == 0) {
        status = rg_corpus_parse_arcaverborum(corpus_text, 0, &corpus);
    } else {
        return error_payload(RG_ERR_UNSUPPORTED_OPTION, "unknown corpus format");
    }
    if (status != RG_OK) {
        char detail_buffer[256];
        const char *grapheme = 0;
        const char *system = 0;
        if (status == RG_ERR_UNKNOWN_GRAPHEME) {
            rg_context_last_error(ctx, &grapheme, &system);
        }
        if (grapheme != 0) {
            snprintf(detail_buffer, sizeof(detail_buffer),
                     "grapheme \"%s\" is not in the \"%s\" feature system",
                     grapheme, system == 0 ? "" : system);
            return error_payload(status, detail_buffer);
        }
        return error_payload(status, "the corpus could not be read");
    }
    if (rg_corpus_cognate_count(corpus) == 0) {
        rg_corpus_free(corpus);
        return error_payload(RG_ERR_PARSE, "no cognate set had two or more lects");
    }

    status = rg_train_model(ctx, rg_corpus_cognates(corpus), rg_corpus_cognate_count(corpus),
                            &options, &model);
    if (status != RG_OK) {
        char detail_buffer[256];
        const char *train_detail = 0;
        if (status == RG_ERR_UNKNOWN_GRAPHEME) {
            const char *grapheme = 0;
            const char *system = 0;
            rg_context_last_error(ctx, &grapheme, &system);
            if (grapheme != 0) {
                snprintf(detail_buffer, sizeof(detail_buffer),
                         "grapheme \"%s\" is not in the \"%s\" feature system",
                         grapheme, system == 0 ? "" : system);
                train_detail = detail_buffer;
            }
        }
        rg_corpus_free(corpus);
        return error_payload(status, train_detail);
    }

    text = rg_model_to_json(ctx, model, rg_corpus_cognates(corpus),
                            rg_corpus_cognate_count(corpus), &options, 1, 1);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
    if (text == 0) {
        return error_payload(RG_ERR_OOM, "could not render the model");
    }
    return text;
}

/* Segments one written word, so a caller can show how input will be read
 * before committing to a full run. */
EMSCRIPTEN_KEEPALIVE
char *regulae_segment_json(const char *word) {
    rg_context *ctx = shared_context();
    rg_segment *segments = 0;
    size_t count = 0;
    char *text;

    if (ctx == 0) {
        return error_payload(RG_ERR_OOM, "could not create the feature context");
    }
    if (rg_context_segment_word(ctx, word == 0 ? "" : word, &segments, &count) != RG_OK) {
        const char *grapheme = 0;
        const char *system = 0;
        char detail_buffer[256];
        rg_context_last_error(ctx, &grapheme, &system);
        if (grapheme != 0) {
            snprintf(detail_buffer, sizeof(detail_buffer),
                     "grapheme \"%s\" is not in the \"%s\" feature system",
                     grapheme, system == 0 ? "" : system);
            return error_payload(RG_ERR_UNKNOWN_GRAPHEME, detail_buffer);
        }
        return error_payload(RG_ERR_UNKNOWN_GRAPHEME, "the word could not be segmented");
    }
    text = rg_segments_to_json(segments, count);
    rg_segments_free(segments, count);
    if (text == 0) {
        return error_payload(RG_ERR_OOM, 0);
    }
    return text;
}

EMSCRIPTEN_KEEPALIVE
const char *regulae_version(void) {
    return rg_version_string();
}

EMSCRIPTEN_KEEPALIVE
void regulae_free(char *text) {
    rg_string_free(text);
}
