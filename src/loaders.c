#include "loader_internal.h"

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


/* ---- public entry points ------------------------------------------------ */

rg_status rg_corpus_load_tsv(
    const char *path,
    const rg_tsv_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
) {
    loader_clear_diagnosis(diagnosis);
    return load_tsv(path, 0, options, out, diagnosis);
}

rg_status rg_corpus_parse_tsv(
    const char *text,
    const rg_tsv_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
) {
    loader_clear_diagnosis(diagnosis);
    return load_tsv(0, text, options, out, diagnosis);
}

rg_status rg_corpus_load_wide_tsv(
    const rg_context *ctx,
    const char *path,
    const rg_wide_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
) {
    loader_clear_diagnosis(diagnosis);
    return load_wide_tsv(ctx, path, 0, options, out, diagnosis);
}

rg_status rg_corpus_parse_wide_tsv(
    const rg_context *ctx,
    const char *text,
    const rg_wide_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
) {
    loader_clear_diagnosis(diagnosis);
    return load_wide_tsv(ctx, 0, text, options, out, diagnosis);
}

rg_status rg_corpus_load_gled(
    const char *path,
    const rg_gled_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
) {
    loader_clear_diagnosis(diagnosis);
    return load_gled(path, 0, options, out, diagnosis);
}

rg_status rg_corpus_parse_gled(
    const char *text,
    const rg_gled_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
) {
    loader_clear_diagnosis(diagnosis);
    return load_gled(0, text, options, out, diagnosis);
}

rg_status rg_corpus_load_arcaverborum(
    const char *path,
    const rg_arcaverborum_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
) {
    loader_clear_diagnosis(diagnosis);
    return load_arcaverborum(path, 0, options, out, diagnosis);
}

rg_status rg_corpus_parse_arcaverborum(
    const char *text,
    const rg_arcaverborum_load_options *options,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
) {
    loader_clear_diagnosis(diagnosis);
    return load_arcaverborum(0, text, options, out, diagnosis);
}
