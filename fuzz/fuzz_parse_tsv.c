#include "regulae.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

/* The generic long-format TSV loader, over arbitrary bytes.
 *
 * The parse entry points take a NUL-terminated string, so the fuzzer's buffer
 * is copied and terminated. An embedded NUL therefore truncates the input,
 * which is exactly what a caller handing over a C string would see, and is the
 * boundary being tested rather than a limitation of the harness. */
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    char *text;
    rg_corpus *corpus = 0;
    rg_load_diagnosis diagnosis;

    text = (char *)malloc(size + 1);
    if (text == 0) {
        return 0;
    }
    memcpy(text, data, size);
    text[size] = '\0';

    if (rg_corpus_parse_tsv(text, 0, &corpus, &diagnosis) == RG_OK) {
        /* Walking the result is part of the target: a corpus that parses and
         * then cannot be read is still a defect. */
        size_t i;
        for (i = 0; i < rg_corpus_cognate_count(corpus); i++) {
            const rg_cognate_set *set = rg_corpus_cognate_at(corpus, i);
            size_t f;
            for (f = 0; set != 0 && f < set->form_count; f++) {
                (void)set->forms[f].form.segment_count;
            }
        }
        rg_corpus_free(corpus);
    }
    free(text);
    return 0;
}
