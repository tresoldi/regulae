#include "regulae.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

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

    if (rg_corpus_parse_gled(text, 0, &corpus, &diagnosis) == RG_OK) {
        rg_corpus_free(corpus);
    }
    free(text);
    return 0;
}
