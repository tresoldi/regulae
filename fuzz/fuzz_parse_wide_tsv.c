#include "regulae.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

/* The wide loader, which is the one that segments. It takes a context, so this
 * target reaches merkmal's segmenter with arbitrary bytes -- the deepest the
 * loaders go into another library.
 *
 * The context is built once and kept. Building one per input would spend the
 * whole run re-resolving merkmal's registry, and the context carries no state
 * from one parse to the next that the fuzzer is interested in, beyond the
 * grapheme caches it exists to hold. */
static rg_context *shared_context(void) {
    static rg_context *ctx = 0;
    if (ctx == 0 && rg_context_new_builtin(&ctx) != RG_OK) {
        return 0;
    }
    return ctx;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    rg_context *ctx = shared_context();
    char *text;
    rg_corpus *corpus = 0;
    rg_load_diagnosis diagnosis;

    if (ctx == 0) {
        return 0;
    }
    text = (char *)malloc(size + 1);
    if (text == 0) {
        return 0;
    }
    memcpy(text, data, size);
    text[size] = '\0';

    if (rg_corpus_parse_wide_tsv(ctx, text, 0, &corpus, &diagnosis) == RG_OK) {
        rg_corpus_free(corpus);
    }
    free(text);
    return 0;
}
