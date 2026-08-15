#include "regulae.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The regulae command-line front end. `train --summary` and `outliers` emit a
 * compact, deterministic line format meant to be diffed. It was shaped for the
 * parity harness and kept after it, because a format two implementations could
 * be compared through is also the one a user can grep and a test can pin. */

#define MAX_PARTS 64

static int usage(void) {
    printf("Usage: regulae <command> [options]\n");
    printf("\n");
    printf("Commands:\n");
    printf("  train <file>      train a multi-lect model and print a summary\n");
    printf("  outliers <file>   rank cognate sets by alignment cost\n");
    printf("  align <file>      print the alignment of every lect pair per cognate\n");
    printf("  check <file>      report every grapheme the feature system cannot read\n");
    printf("  version           print version\n");
    printf("  help              print this help\n");
    printf("\n");
    printf("Options:\n");
    printf("  --format <tsv|wide|gled|arcaverborum>\n");
    printf("                                     input format (default tsv);\n");
    printf("                                     'wide' is one row per cognate,\n");
    printf("                                     one column per lect, whole words\n");
    printf("  --summary                          machine-readable output (default)\n");
    printf("  --top-k <n>                        limit outlier rows (default all)\n");
    printf("  --model                            align under the trained model\n");
    printf("  --pairwise                         dump per-pair learned tables\n");
    printf("  --human                            human-readable model summary\n");
    printf("  --json                             machine-readable model, with\n");
    printf("                                     alignments and outliers\n");
    printf("  --permutations <n>                 calibrate the fit against <n>\n");
    printf("                                     trainings on shuffled pairings.\n");
    printf("                                     Costs one training run each,\n");
    printf("                                     and is the only way to read\n");
    printf("                                     whether the corpus has signal\n");
    printf("  --tune-search                      set the search charge from that\n");
    printf("                                     baseline instead of the default.\n");
    printf("                                     Buys precision with recall\n");
    return 0;
}

/* strdup is not in C99, and the CLI is built with -std=c99. */
static char *dup_string(const char *value) {
    size_t n = strlen(value);
    char *out = (char *)malloc(n + 1);
    if (out != 0) {
        memcpy(out, value, n + 1);
    }
    return out;
}

static int string_cmp(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

/* Appends "feature:value" for every constraint, sorted, joined by ';'. */
static void append_constraints(
    char *buffer,
    size_t size,
    const rg_feature_constraint *items,
    size_t count
) {
    char *parts[MAX_PARTS];
    size_t i;
    size_t used = 0;
    size_t offset = 0;

    buffer[0] = '\0';
    for (i = 0; i < count && used < MAX_PARTS; i++) {
        char part[128];
        snprintf(part, sizeof(part), "%s:%s",
                 items[i].feature == 0 ? "" : items[i].feature,
                 items[i].value == 0 ? "" : items[i].value);
        parts[used] = dup_string(part);
        if (parts[used] == 0) {
            break;
        }
        used++;
    }
    qsort(parts, used, sizeof(*parts), string_cmp);
    for (i = 0; i < used; i++) {
        int written = snprintf(buffer + offset, size - offset, "%s%s", i > 0 ? ";" : "", parts[i]);
        if (written > 0 && (size_t)written < size - offset) {
            offset += (size_t)written;
        }
        free(parts[i]);
    }
}

static void append_distances(
    char *buffer,
    size_t size,
    const rg_distance_constraint *items,
    size_t count
) {
    char *parts[MAX_PARTS];
    size_t i;
    size_t used = 0;
    size_t offset = 0;

    buffer[0] = '\0';
    for (i = 0; i < count && used < MAX_PARTS; i++) {
        char part[160];
        snprintf(part, sizeof(part), "%d@%s:%s",
                 items[i].offset,
                 items[i].constraint.feature == 0 ? "" : items[i].constraint.feature,
                 items[i].constraint.value == 0 ? "" : items[i].constraint.value);
        parts[used] = dup_string(part);
        if (parts[used] == 0) {
            break;
        }
        used++;
    }
    qsort(parts, used, sizeof(*parts), string_cmp);
    for (i = 0; i < used; i++) {
        int written = snprintf(buffer + offset, size - offset, "%s%s", i > 0 ? ";" : "", parts[i]);
        if (written > 0 && (size_t)written < size - offset) {
            offset += (size_t)written;
        }
        free(parts[i]);
    }
}

/* Canonical rendering of a context: fixed slot order, sorted constraint lists,
 * empty slots omitted. Must stay byte-identical to the Go dumper's contextKey. */
static void context_key(const rg_context_spec *context, char *out, size_t size) {
    char slot[1024];
    size_t offset = 0;
    int first = 1;

    out[0] = '\0';
    if (context == 0) {
        snprintf(out, size, "-");
        return;
    }

#define EMIT(name, text)                                                                  \
    do {                                                                                  \
        if ((text)[0] != '\0') {                                                          \
            int written = snprintf(out + offset, size - offset, "%s%s=%s",                \
                                   first ? "" : ",", (name), (text));                     \
            if (written > 0 && (size_t)written < size - offset) {                         \
                offset += (size_t)written;                                                \
            }                                                                             \
            first = 0;                                                                    \
        }                                                                                 \
    } while (0)

    EMIT("pos", context->position == 0 ? "" : context->position);
    EMIT("morph", context->morphological == 0 ? "" : context->morphological);
    append_constraints(slot, sizeof(slot), context->preceding, context->preceding_count);
    EMIT("pre", slot);
    append_constraints(slot, sizeof(slot), context->following, context->following_count);
    EMIT("fol", slot);
    append_distances(slot, sizeof(slot), context->preceding_at_distance, context->preceding_at_distance_count);
    EMIT("preat", slot);
    append_distances(slot, sizeof(slot), context->following_at_distance, context->following_at_distance_count);
    EMIT("folat", slot);
    append_constraints(slot, sizeof(slot), context->somewhere_preceding, context->somewhere_preceding_count);
    EMIT("swpre", slot);
    append_constraints(slot, sizeof(slot), context->somewhere_following, context->somewhere_following_count);
    EMIT("swfol", slot);
    append_constraints(slot, sizeof(slot), context->same_syllable, context->same_syllable_count);
    EMIT("samesyl", slot);
    append_constraints(slot, sizeof(slot), context->next_syllable, context->next_syllable_count);
    EMIT("nextsyl", slot);
    append_constraints(slot, sizeof(slot), context->previous_syllable, context->previous_syllable_count);
    EMIT("prevsyl", slot);
    append_constraints(slot, sizeof(slot), context->self_stress, context->self_stress_count);
    EMIT("selfstress", slot);
    append_constraints(slot, sizeof(slot), context->preceding_stress, context->preceding_stress_count);
    EMIT("prestress", slot);
    append_constraints(slot, sizeof(slot), context->following_stress, context->following_stress_count);
    EMIT("folstress", slot);

#undef EMIT

    if (first) {
        snprintf(out, size, "-");
    }
}

static void print_class(const rg_multi_class_row *class_row, const char *label, int with_contexts) {
    size_t i;
    printf("%s\t%d\t", label, class_row->class_id);
    for (i = 0; i < class_row->segment_count; i++) {
        printf("%s%s:%s", i > 0 ? "|" : "", class_row->lect_ids[i], class_row->graphemes[i]);
    }
    printf("\t%.6f\t%.6f\t", class_row->count, class_row->confidence);
    if (with_contexts) {
        for (i = 0; i < class_row->segment_count; i++) {
            char key[2048];
            context_key(class_row->contexts == 0 ? 0 : &class_row->contexts[i], key, sizeof(key));
            printf("%s%s=%s", i > 0 ? "|" : "", class_row->lect_ids[i], key);
        }
    } else {
        for (i = 0; i < class_row->supporting_cognate_count; i++) {
            printf("%s%s", i > 0 ? "," : "", class_row->supporting_cognates[i]);
        }
    }
    printf("\n");
}

static void print_summary(const rg_multi_model *model) {
    size_t i;
    printf("LECTS\t");
    for (i = 0; i < rg_multi_model_lect_count(model); i++) {
        printf("%s%s", i > 0 ? " " : "", rg_multi_model_lect_at(model, i));
    }
    printf("\n");
    for (i = 0; i < rg_multi_model_unconditioned_class_count(model); i++) {
        print_class(rg_multi_model_unconditioned_class_at(model, i), "UNCOND", 0);
    }
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        print_class(rg_multi_model_conditioned_class_at(model, i), "COND", 1);
    }
    for (i = 0; i < rg_multi_model_cross_dimensional_row_count(model); i++) {
        const rg_multi_cross_dimensional_row *row = rg_multi_model_cross_dimensional_row_at(model, i);
        printf("XDIM\t%s>%s\t%s=%s@%s\t%s=%s@%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
               row->source_lect, row->target_lect,
               row->source_feature, row->source_value, row->source_position,
               row->target_dimension, row->target_value, row->target_position_offset,
               row->count, row->source_count, row->confidence,
               row->contrast_count, row->contrast_source_count, row->contrast_confidence,
               row->delta_bic);
    }
}

static rg_status load_corpus(const char *path, const char *format, rg_corpus **out);

/* Dumps the per-pair learned tables, one row per line, for diffing. */
static void print_pairwise(const rg_multi_model *model) {
    size_t p;
    for (p = 0; p < rg_multi_model_pair_model_count(model); p++) {
        const rg_multi_pair_model_row *row = rg_multi_model_pair_model_at(model, p);
        const rg_pairwise_model *pm = row->model;
        size_t i;
        size_t k;
        for (i = 0; i < rg_pairwise_model_segment_count_row_count(pm); i++) {
            const rg_segment_count_row *seg = rg_pairwise_model_segment_count_row_at(pm, i);
            printf("SEG\t%s>%s\t%s\t%s\t-\t%.6f\n", row->lect_a, row->lect_b, seg->source, seg->target, seg->count);
        }
        for (i = 0; i < rg_pairwise_model_conditioned_segment_count_row_count(pm); i++) {
            const rg_conditioned_segment_count_row *seg = rg_pairwise_model_conditioned_segment_count_row_at(pm, i);
            char key[2048];
            context_key(&seg->context, key, sizeof(key));
            /* Which form's environment the rule names. Two rows can carry the
             * same context and mean different things: one says the source
             * looked like that, the other the target. */
            printf("SEG\t%s>%s\t%s\t%s\t%s%s\t%.6f\n", row->lect_a, row->lect_b,
                   seg->source, seg->target,
                   seg->context_is_target ? "@target " : "", key, seg->count);
        }
        for (i = 0; i < rg_pairwise_model_chunk_row_count(pm); i++) {
            const rg_chunk_row *chunk = rg_pairwise_model_chunk_row_at(pm, i);
            printf("CHUNK\t%s>%s\t", row->lect_a, row->lect_b);
            for (k = 0; k < chunk->source_count; k++) {
                printf("%s", chunk->source[k].grapheme);
            }
            printf("\t");
            for (k = 0; k < chunk->target_count; k++) {
                printf("%s", chunk->target[k].grapheme);
            }
            printf("\t%.6f\t%.6f\n", chunk->cost, chunk->count);
        }
        for (i = 0; i < rg_pairwise_model_tonal_count_row_count(pm); i++) {
            const rg_tonal_count_row *tone = rg_pairwise_model_tonal_count_row_at(pm, i);
            printf("TONE\t%s>%s\t%s>%s\t%.6f\n", row->lect_a, row->lect_b,
                   tone->source_tone, tone->target_tone, tone->count);
        }
    }
}

static rg_status load_corpus_with_context(
    const rg_context *ctx,
    const char *path,
    const char *format,
    rg_corpus **out
) {
    if (format != 0 && strcmp(format, "wide") == 0) {
        return rg_corpus_load_wide_tsv(ctx, path, 0, out);
    }
    return load_corpus(path, format, out);
}

static rg_status load_corpus(const char *path, const char *format, rg_corpus **out) {
    if (format == 0 || strcmp(format, "tsv") == 0) {
        rg_tsv_load_options options;
        memset(&options, 0, sizeof(options));
        options.confidence_column = "confidence";
        return rg_corpus_load_tsv(path, &options, out);
    }
    if (strcmp(format, "gled") == 0) {
        return rg_corpus_load_gled(path, 0, out);
    }
    if (strcmp(format, "arcaverborum") == 0) {
        return rg_corpus_load_arcaverborum(path, 0, out);
    }
    return RG_ERR_UNSUPPORTED_OPTION;
}

/* Names the grapheme when one is to blame, and the character within it that
 * broke, which is usually the repair. */
static void report_failure(const rg_context *ctx, const char *what, rg_status status) {
    const char *grapheme = 0;
    const char *system = 0;
    rg_grapheme_diagnosis diagnosis;
    int diagnosed = 0;

    if ((status == RG_ERR_UNKNOWN_GRAPHEME || status == RG_ERR_SOURCE_MARKER ||
         status == RG_ERR_PARSE) && ctx != 0) {
        rg_context_last_error(ctx, &grapheme, &system);
        diagnosed = rg_context_last_diagnosis(ctx, &diagnosis);
    }
    if (grapheme == 0) {
        fprintf(stderr, "regulae: %s: %s\n", what, rg_status_string(status));
        return;
    }
    if (status == RG_ERR_SOURCE_MARKER) {
        fprintf(stderr,
                "regulae: %s: \"%s\" is CLDF/CLTS markup, not a transcribed sound. "
                "The gap is in the source data, not in the feature system.\n",
                what, grapheme);
        return;
    }
    fprintf(stderr,
            "regulae: %s: unknown grapheme \"%s\" in feature system \"%s\". "
            "Either the grapheme is a typo, or the feature system does not cover it.\n",
            what, grapheme, system == 0 ? "" : system);
    if (diagnosed && diagnosis.offending[0] != '\0' && diagnosis.valid_prefix_bytes > 0) {
        fprintf(stderr,
                "regulae: %.*s resolves; \"%s\" at byte %zu does not.\n",
                (int)diagnosis.valid_prefix_bytes, grapheme,
                diagnosis.offending, diagnosis.offending_offset);
    }
}

static int fail(const char *what, rg_status status) {
    report_failure(0, what, status);
    return 1;
}


/* ---- check ------------------------------------------------------------- */

typedef struct grapheme_report {
    char *grapheme;
    rg_status status;
    size_t count;
    char *first_context;
} grapheme_report;

static int grapheme_report_cmp(const void *a, const void *b) {
    const grapheme_report *ra = (const grapheme_report *)a;
    const grapheme_report *rb = (const grapheme_report *)b;
    if (ra->count != rb->count) {
        return ra->count > rb->count ? -1 : 1;
    }
    return strcmp(ra->grapheme, rb->grapheme);
}

static void report_grapheme(
    grapheme_report **items,
    size_t *count,
    size_t *cap,
    const char *grapheme,
    rg_status status,
    const char *where
) {
    size_t i;
    for (i = 0; i < *count; i++) {
        if (strcmp((*items)[i].grapheme, grapheme) == 0) {
            (*items)[i].count++;
            return;
        }
    }
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 16 : *cap * 2;
        grapheme_report *next = (grapheme_report *)realloc(*items, next_cap * sizeof(*next));
        if (next == 0) {
            return;
        }
        *items = next;
        *cap = next_cap;
    }
    (*items)[*count].grapheme = dup_string(grapheme);
    (*items)[*count].status = status;
    (*items)[*count].count = 1;
    (*items)[*count].first_context = where == 0 ? 0 : dup_string(where);
    (*count)++;
}

/* Reports every grapheme a corpus contains that the feature system cannot
 * read, rather than stopping at the first one.
 *
 * Training refuses an unreadable grapheme, and should: silently skipping input
 * would train a model on a corpus the user did not supply. But that makes
 * reading an unfamiliar dataset a fix-one-rerun loop, and real datasets carry
 * whole families of unreadable tokens at once -- source markup, cover symbols
 * from a reconstruction, a systematic diacritic the feature system lacks. Those
 * are one decision each, not one per occurrence, and a user cannot make them
 * without seeing the whole list. */
static int command_check(const char *path, const char *format) {
    rg_context *ctx = 0;
    FILE *handle;
    char line[8192];
    grapheme_report *items = 0;
    size_t count = 0;
    size_t cap = 0;
    size_t forms = 0;
    size_t bad_forms = 0;
    size_t i;
    long segments_column = -1;
    int is_word[64];
    int wide = format != 0 && strcmp(format, "wide") == 0;
    rg_status status;

    memset(is_word, 0, sizeof(is_word));
    status = rg_context_new_builtin(&ctx);
    if (status != RG_OK) {
        return fail("context", status);
    }
    handle = fopen(path, "r");
    if (handle == 0) {
        rg_context_free(ctx);
        return fail("reading", RG_ERR_IO);
    }
    if (fgets(line, sizeof(line), handle) == 0) {
        fclose(handle);
        rg_context_free(ctx);
        fprintf(stderr, "regulae: check: %s is empty\n", path);
        return 2;
    }
    {
        char *tok;
        long col = 0;
        line[strcspn(line, "\r\n")] = '\0';
        for (tok = strtok(line, "\t"); tok != 0 && col < 64; tok = strtok(0, "\t")) {
            size_t n = strlen(tok);
            if (strcmp(tok, "segments") == 0) {
                segments_column = col;
            }
            /* A companion column holds boundaries, tone or a weight, not a
             * word. Reading one as a form reports its digits as graphemes. */
            is_word[col] = col > 0 &&
                strcmp(tok, "confidence") != 0 &&
                !(n > 7 && strcmp(tok + n - 7, "_breaks") == 0) &&
                !(n > 5 && strcmp(tok + n - 5, "_tone") == 0);
            col++;
        }
        if (!wide && segments_column < 0) {
            fclose(handle);
            rg_context_free(ctx);
            fprintf(stderr, "regulae: check: no \"segments\" column; use --format wide\n");
            return 2;
        }
    }
    while (fgets(line, sizeof(line), handle) != 0) {
        char *fields[64];
        size_t field_count = 0;
        char *tok;
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') {
            continue;
        }
        for (tok = strtok(line, "\t"); tok != 0 && field_count < 64; tok = strtok(0, "\t")) {
            fields[field_count++] = tok;
        }
        if (wide) {
            size_t f;
            /* Every cell but the first is a candidate word; a cell the
             * segmenter refuses is reported against the word it came from. */
            for (f = 1; f < field_count; f++) {
                rg_segment *segments = 0;
                size_t n = 0;
                if (!is_word[f] || fields[f][0] == '\0' || strcmp(fields[f], "-") == 0) {
                    continue;
                }
                forms++;
                if (rg_context_segment_word(ctx, fields[f], &segments, &n) != RG_OK) {
                    const char *grapheme = 0;
                    rg_grapheme_diagnosis diagnosis;
                    rg_context_last_error(ctx, &grapheme, 0);
                    diagnosis.status = RG_ERR_UNKNOWN_GRAPHEME;
                    rg_context_last_diagnosis(ctx, &diagnosis);
                    bad_forms++;
                    report_grapheme(&items, &count, &cap,
                                    grapheme == 0 ? fields[f] : grapheme,
                                    diagnosis.status, fields[f]);
                    continue;
                }
                for (i = 0; i < n; i++) {
                    rg_grapheme_diagnosis diagnosis;
                    if (rg_context_diagnose(ctx, segments[i].grapheme, &diagnosis) == RG_OK &&
                        diagnosis.status != RG_OK) {
                        bad_forms++;
                        report_grapheme(&items, &count, &cap, segments[i].grapheme,
                                        diagnosis.status, fields[f]);
                        break;
                    }
                }
                rg_segments_free(segments, n);
            }
        } else if ((size_t)segments_column < field_count) {
            char *cell = fields[segments_column];
            char *segment;
            int reported = 0;
            forms++;
            for (segment = strtok(cell, " "); segment != 0; segment = strtok(0, " ")) {
                rg_grapheme_diagnosis diagnosis;
                if (rg_context_diagnose(ctx, segment, &diagnosis) == RG_OK &&
                    diagnosis.status != RG_OK) {
                    if (!reported) {
                        bad_forms++;
                        reported = 1;
                    }
                    report_grapheme(&items, &count, &cap, segment, diagnosis.status, 0);
                }
            }
        }
    }
    fclose(handle);

    if (count > 1) {
        qsort(items, count, sizeof(*items), grapheme_report_cmp);
    }
    printf("forms\t%lu\nunreadable\t%lu\ngraphemes\t%lu\n",
           (unsigned long)forms, (unsigned long)bad_forms, (unsigned long)count);
    for (i = 0; i < count; i++) {
        printf("GRAPHEME\t%s\t%s\t%lu",
               items[i].grapheme, rg_status_string(items[i].status),
               (unsigned long)items[i].count);
        if (items[i].first_context != 0) {
            printf("\t%s", items[i].first_context);
        }
        printf("\n");
        free(items[i].grapheme);
        free(items[i].first_context);
    }
    free(items);
    rg_context_free(ctx);
    return count == 0 ? 0 : 1;
}

static int command_train(const char *path, const char *format, int pairwise, int human, int json, int permutations, int tune_search) {
    rg_context *ctx = 0;
    rg_corpus *corpus = 0;
    rg_multi_model *model = 0;
    rg_train_options options;
    rg_status status;

    status = rg_context_new_builtin(&ctx);
    if (status != RG_OK) {
        return fail("creating context", status);
    }
    status = load_corpus_with_context(ctx, path, format, &corpus);
    if (status != RG_OK) {
        report_failure(ctx, "loading corpus", status);
        rg_context_free(ctx);
        return 1;
    }
    rg_train_options_init_defaults(&options);
    options.permutation_count = permutations;
    options.tune_search_penalty = tune_search;
    status = rg_train_model(ctx, rg_corpus_cognates(corpus), rg_corpus_cognate_count(corpus), &options, &model);
    if (status != RG_OK) {
        report_failure(ctx, "training", status);
        rg_corpus_free(corpus);
        rg_context_free(ctx);
        return 1;
    }
    if (json) {
        char *text = rg_model_to_json(ctx, model,
                                      rg_corpus_cognates(corpus), rg_corpus_cognate_count(corpus),
                                      &options, 1, 1);
        if (text == 0) {
            rg_multi_model_free(model);
            rg_corpus_free(corpus);
            rg_context_free(ctx);
            return fail("rendering json", RG_ERR_OOM);
        }
        puts(text);
        rg_string_free(text);
    } else if (human) {
        char *text = rg_format_multi_model(model, 0);
        if (text != 0) {
            fputs(text, stdout);
            rg_string_free(text);
        }
    } else if (pairwise) {
        print_pairwise(model);
    } else {
        print_summary(model);
    }
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
    rg_context_free(ctx);
    return 0;
}

static int command_outliers(const char *path, const char *format, int top_k) {
    rg_context *ctx = 0;
    rg_corpus *corpus = 0;
    rg_multi_model *model = 0;
    rg_cognate_outlier_row *rows = 0;
    size_t row_count = 0;
    rg_train_options options;
    size_t i;
    rg_status status;

    status = rg_context_new_builtin(&ctx);
    if (status != RG_OK) {
        return fail("creating context", status);
    }
    status = load_corpus_with_context(ctx, path, format, &corpus);
    if (status != RG_OK) {
        report_failure(ctx, "loading corpus", status);
        rg_context_free(ctx);
        return 1;
    }
    rg_train_options_init_defaults(&options);
    status = rg_train_model(ctx, rg_corpus_cognates(corpus), rg_corpus_cognate_count(corpus), &options, &model);
    if (status != RG_OK) {
        report_failure(ctx, "training", status);
        rg_corpus_free(corpus);
        rg_context_free(ctx);
        return 1;
    }
    status = rg_find_cognate_outliers(
        ctx,
        rg_corpus_cognates(corpus),
        rg_corpus_cognate_count(corpus),
        model,
        &options,
        top_k,
        0,
        &rows,
        &row_count
    );
    if (status != RG_OK) {
        report_failure(ctx, "finding outliers", status);
        rg_multi_model_free(model);
        rg_corpus_free(corpus);
        rg_context_free(ctx);
        return 1;
    }
    for (i = 0; i < row_count; i++) {
        printf("OUTLIER\t%s\t%d\t%.6f\t%.6f\n",
               rows[i].cognate_id, rows[i].pair_count, rows[i].cost_per_segment, rows[i].z_score);
    }
    rg_cognate_outlier_rows_free(rows, row_count);
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
    rg_context_free(ctx);
    return 0;
}

/* Aligns each cognate's lect pairs in ascending lect-id order under the trained
 * pairwise model, which is the same view of the data reconciliation sees. */
static int command_align_with_model(const char *path, const char *format) {
    rg_context *ctx = 0;
    rg_corpus *corpus = 0;
    rg_multi_model *model = 0;
    rg_train_options options;
    size_t c;
    rg_status status;

    status = rg_context_new_builtin(&ctx);
    if (status != RG_OK) {
        return fail("creating context", status);
    }
    status = load_corpus_with_context(ctx, path, format, &corpus);
    if (status != RG_OK) {
        report_failure(ctx, "loading corpus", status);
        rg_context_free(ctx);
        return 1;
    }
    rg_train_options_init_defaults(&options);
    status = rg_train_model(ctx, rg_corpus_cognates(corpus), rg_corpus_cognate_count(corpus), &options, &model);
    if (status != RG_OK) {
        report_failure(ctx, "training", status);
        rg_corpus_free(corpus);
        rg_context_free(ctx);
        return 1;
    }
    for (c = 0; c < rg_corpus_cognate_count(corpus); c++) {
        const rg_cognate_set *cognate = rg_corpus_cognate_at(corpus, c);
        size_t order[64];
        size_t order_count = 0;
        size_t i;
        for (i = 0; i < cognate->form_count && order_count < 64; i++) {
            size_t insert_at = order_count;
            while (insert_at > 0 &&
                   strcmp(cognate->forms[order[insert_at - 1]].lect_id, cognate->forms[i].lect_id) > 0) {
                insert_at--;
            }
            if (insert_at < order_count) {
                memmove(&order[insert_at + 1], &order[insert_at], (order_count - insert_at) * sizeof(*order));
            }
            order[insert_at] = i;
            order_count++;
        }
        for (i = 0; i < order_count; i++) {
            size_t j;
            for (j = i + 1; j < order_count; j++) {
                const rg_pairwise_model *pair_model = 0;
                const char *lect_a = cognate->forms[order[i]].lect_id;
                const char *lect_b = cognate->forms[order[j]].lect_id;
                rg_alignment *alignment = 0;
                size_t link_i;
                size_t p;
                for (p = 0; p < rg_multi_model_pair_model_count(model); p++) {
                    const rg_multi_pair_model_row *row = rg_multi_model_pair_model_at(model, p);
                    if ((strcmp(row->lect_a, lect_a) == 0 && strcmp(row->lect_b, lect_b) == 0) ||
                        (strcmp(row->lect_a, lect_b) == 0 && strcmp(row->lect_b, lect_a) == 0)) {
                        pair_model = row->model;
                        break;
                    }
                }
                if (pair_model == 0) {
                    continue;
                }
                status = rg_align_forms_with_model(
                    ctx, pair_model, &options,
                    &cognate->forms[order[i]].form, &cognate->forms[order[j]].form, 3, &alignment);
                if (status != RG_OK) {
                    report_failure(ctx, "aligning", status);
                    rg_multi_model_free(model);
                    rg_corpus_free(corpus);
                    rg_context_free(ctx);
                    return 1;
                }
                {
                    double mcost = 0.0;
                    rg_alignment_cost_with_model(ctx, pair_model, &options, alignment, &mcost);
                    printf("MALIGN\t%s\t%s>%s\t%.12f\t", cognate->cognate_id, lect_a, lect_b, mcost);
                }
                for (link_i = 0; link_i < rg_alignment_link_count(alignment); link_i++) {
                    const rg_link *link = rg_alignment_link_at(alignment, link_i);
                    size_t k;
                    printf("%s", link_i > 0 ? " " : "");
                    for (k = 0; k < link->source_count; k++) {
                        printf("%s", link->source[k].grapheme);
                    }
                    printf("/");
                    for (k = 0; k < link->target_count; k++) {
                        printf("%s", link->target[k].grapheme);
                    }
                }
                printf("\n");
                rg_alignment_free(alignment);
            }
        }
    }
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
    rg_context_free(ctx);
    return 0;
}

static int command_align(const char *path, const char *format) {
    rg_context *ctx = 0;
    rg_corpus *corpus = 0;
    size_t c;
    rg_status status;

    status = rg_context_new_builtin(&ctx);
    if (status != RG_OK) {
        return fail("creating context", status);
    }
    status = load_corpus_with_context(ctx, path, format, &corpus);
    if (status != RG_OK) {
        report_failure(ctx, "loading corpus", status);
        rg_context_free(ctx);
        return 1;
    }
    for (c = 0; c < rg_corpus_cognate_count(corpus); c++) {
        const rg_cognate_set *cognate = rg_corpus_cognate_at(corpus, c);
        size_t i;
        for (i = 0; i < cognate->form_count; i++) {
            size_t j;
            for (j = i + 1; j < cognate->form_count; j++) {
                rg_alignment *alignment = 0;
                size_t link_i;
                double cost = 0.0;
                status = rg_align_forms(ctx, &cognate->forms[i].form, &cognate->forms[j].form, 0, &alignment);
                if (status != RG_OK) {
                    report_failure(ctx, "aligning", status);
                    rg_corpus_free(corpus);
                    rg_context_free(ctx);
                    return 1;
                }
                rg_alignment_cost(ctx, alignment, &cost);
                printf("ALIGN\t%s\t%s>%s\t%.6f\t", cognate->cognate_id,
                       cognate->forms[i].lect_id, cognate->forms[j].lect_id, cost);
                for (link_i = 0; link_i < rg_alignment_link_count(alignment); link_i++) {
                    const rg_link *link = rg_alignment_link_at(alignment, link_i);
                    size_t k;
                    printf("%s", link_i > 0 ? " " : "");
                    for (k = 0; k < link->source_count; k++) {
                        printf("%s", link->source[k].grapheme);
                    }
                    printf("/");
                    for (k = 0; k < link->target_count; k++) {
                        printf("%s", link->target[k].grapheme);
                    }
                }
                printf("\n");
                rg_alignment_free(alignment);
            }
        }
    }
    rg_corpus_free(corpus);
    rg_context_free(ctx);
    return 0;
}

int main(int argc, char **argv) {
    const char *format = "tsv";
    const char *path = 0;
    int top_k = 0;
    int use_model = 0;
    int pairwise = 0;
    int human = 0;
    int json = 0;
    int permutations = 0;
    int tune_search = 0;
    int i;

    if (argc < 2 || strcmp(argv[1], "help") == 0 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        return usage();
    }
    if (strcmp(argv[1], "version") == 0 || strcmp(argv[1], "--version") == 0) {
        printf("regulae %s\n", rg_version_string());
        return 0;
    }
    for (i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            format = argv[++i];
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--permutations") == 0 && i + 1 < argc) {
            permutations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tune-search") == 0) {
            tune_search = 1;
        } else if (strcmp(argv[i], "--json") == 0) {
            json = 1;
        } else if (strcmp(argv[i], "--human") == 0) {
            human = 1;
        } else if (strcmp(argv[i], "--pairwise") == 0) {
            pairwise = 1;
        } else if (strcmp(argv[i], "--model") == 0) {
            use_model = 1;
        } else if (strcmp(argv[i], "--summary") == 0) {
            /* The summary form is the only output today; accepted so scripts
             * can be explicit about what they depend on. */
            continue;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "regulae: unknown option: %s\n", argv[i]);
            return 2;
        } else if (path == 0) {
            path = argv[i];
        }
    }
    if (path == 0) {
        fprintf(stderr, "regulae: %s requires an input file\n", argv[1]);
        return 2;
    }
    if (strcmp(argv[1], "train") == 0) {
        return command_train(path, format, pairwise, human, json, permutations, tune_search);
    }
    if (strcmp(argv[1], "outliers") == 0) {
        return command_outliers(path, format, top_k);
    }
    if (strcmp(argv[1], "align") == 0) {
        return use_model ? command_align_with_model(path, format) : command_align(path, format);
    }
    if (strcmp(argv[1], "check") == 0) {
        return command_check(path, format);
    }
    fprintf(stderr, "regulae: unknown command: %s\n", argv[1]);
    usage();
    return 2;
}
