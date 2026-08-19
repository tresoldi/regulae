#include "regulae.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The regulae command-line front end. `train --summary` and `outliers` emit a
 * compact, deterministic line format meant to be diffed. A format stable
 * enough to diff cleanly is also the one a user can grep and a test can pin. */

#define MAX_PARTS 64

static char *dup_string(const char *value) {
    size_t length = strlen(value) + 1;
    char *copy = (char *)malloc(length);
    if (copy != 0) {
        memcpy(copy, value, length);
    }
    return copy;
}

static int parse_split_scorer(const char *value, rg_split_scorer *out) {
    if (strcmp(value, "corrected-bic") == 0 || strcmp(value, "corrected_bic") == 0) {
        *out = RG_SPLIT_SCORER_CORRECTED_BIC;
    } else if (strcmp(value, "nml") == 0 || strcmp(value, "multinomial-nml") == 0) {
        *out = RG_SPLIT_SCORER_MULTINOMIAL_NML;
    } else if (strcmp(value, "dirichlet") == 0 || strcmp(value, "dirichlet-marginal") == 0) {
        *out = RG_SPLIT_SCORER_DIRICHLET_MARGINAL;
    } else {
        return 0;
    }
    return 1;
}

static int usage(void) {
    printf("Usage: regulae <command> [options]\n");
    printf("\n");
    printf("Commands:\n");
    printf("  train <file>      train a multi-lect model and print a summary\n");
    printf("  outliers <file>   rank cognate sets by alignment cost\n");
    printf("  align <file>      print the alignment of every lect pair per cognate\n");
    printf("  check <file>      report what a corpus costs before training on it:\n");
    printf("                    unreadable graphemes, transcription drift,\n");
    printf("                    which annotations it carries, and which\n");
    printf("                    correspondences no environment accounts for\n");
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
    printf("  --permutation-seed <n>             reproducible shuffle seed\n");
    printf("  --predictive-folds <n>             dependency-group-held-out folds;\n");
    printf("                                     selects rules inside training only\n");
    printf("  --predictive-seed <n>              reproducible group-fold seed\n");
    printf("                                     trainings on shuffled pairings.\n");
    printf("                                     Costs one training run each,\n");
    printf("                                     and is the only way to read\n");
    printf("                                     whether the corpus has signal\n");
    printf("  --tune-search                      set the search charge from that\n");
    printf("                                     baseline instead of the default.\n");
    printf("                                     Buys precision with recall\n");
    printf("  --feature-system <name>            merkmal feature system for training\n");
    printf("  --scorer <corrected-bic|nml|dirichlet>\n");
    printf("                                     conditioned-split criterion\n");
    printf("  --split-prior <mass>               symmetric Dirichlet total mass\n");
    printf("  --search-gamma <value>             adaptive-search charge multiplier\n");
    printf("  --split-threshold <value>          immediate split score threshold\n");
    printf("  --long-split-threshold <value>     long-range split score threshold\n");
    printf("  --no-multilect-small-sample        remove the legacy BIC-only addition\n");
    printf("  --multilect-small-sample           enable it for compatibility experiments\n");
    return 0;
}

/* strdup is not in C99, and the CLI is built with -std=c99. */
static rg_status load_corpus(
    const char *path,
    const char *format,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
);

static rg_status load_corpus_with_context(
    const rg_context *ctx,
    const char *path,
    const char *format,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
) {
    /* Every loader clears this on entry, but an unknown format returns without
     * reaching one, and report_load_failure then reads an uninitialised
     * message. Found by clang-analyzer, which is the only thing that would
     * have: the path needs a format the CLI does not recognise. */
    if (diagnosis != 0) {
        diagnosis->line = 0;
        diagnosis->message[0] = '\0';
    }
    if (format != 0 && strcmp(format, "wide") == 0) {
        return rg_corpus_load_wide_tsv(ctx, path, 0, out, diagnosis);
    }
    return load_corpus(path, format, out, diagnosis);
}

static rg_status load_corpus(
    const char *path,
    const char *format,
    rg_corpus **out,
    rg_load_diagnosis *diagnosis
) {
    if (format == 0 || strcmp(format, "tsv") == 0) {
        rg_tsv_load_options options;
        memset(&options, 0, sizeof(options));
        options.confidence_column = "confidence";
        return rg_corpus_load_tsv(path, &options, out, diagnosis);
    }
    if (strcmp(format, "gled") == 0) {
        return rg_corpus_load_gled(path, 0, out, diagnosis);
    }
    if (strcmp(format, "arcaverborum") == 0) {
        return rg_corpus_load_arcaverborum(path, 0, out, diagnosis);
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

/* Below this the pair shows too little systematic correspondence for a
 * per-correspondence residue to mean anything: on `restraint/chance`, two
 * wordlists with no history between them, H(target|source) is 2.47 bits and
 * every correspondence in it is "unexplained" -- truthfully, and uselessly,
 * because there was nothing to explain. Real pairs sit far below: Latin and
 * Spanish without vowel quantity 1.10, Grimm 0.55, the same Latin/Spanish with
 * quantity and stress 0.38, Latin rhotacism 0.13. Two bits is "more than two
 * equally likely reflexes per source on average", which is a description of
 * the corpus rather than a tuned cut. */
#define RG_RESIDUE_MAX_PAIR_ENTROPY 2.0

/* A correspondence at least as uncertain as a fair coin. Also not tuned: one
 * bit is the point where a source segment stops having a reflex and starts
 * having a choice of them. */
#define RG_RESIDUE_MIN_ENTROPY 1.0

/* The evidence floor, so a correspondence is not called unexplained on the
 * strength of a handful of words. */
#define RG_RESIDUE_MIN_OBSERVATIONS 8.0

/* How predictable one lect's segment is from the other's, in bits, and which
 * individual correspondences carry that uncertainty without any environment
 * accounting for it.
 *
 * This is the question a comparativist has before any of the others: is my
 * data good enough to start? A correspondence with high entropy and no
 * committed environment is either a change conditioned by something the
 * transcription does not write -- vowel length, stress, tone, a morpheme
 * boundary -- or one conditioned by nothing, and the tool cannot tell those
 * apart. Saying which correspondences are in that position is what it can do,
 * and it is how Verner got to the accent.
 *
 * Both directions are reported because they are different questions: what
 * Latin /e/ becomes, and what a Spanish /e/ comes from. */
static void report_pair_residue(const rg_multi_pair_model_row *pair, size_t *unexplained) {
    const rg_segment_count_row *segments;
    const rg_conditioned_segment_count_row *conditioned;
    size_t segment_count = 0;
    size_t conditioned_count = 0;
    size_t i;
    size_t j;
    double corpus_total = 0.0;
    double corpus_entropy = 0.0;

    segments = rg_pairwise_model_segment_counts(pair->model, &segment_count);
    conditioned = rg_pairwise_model_conditioned_segment_counts(pair->model, &conditioned_count);
    if (segments == 0 || segment_count == 0) {
        return;
    }

    /* The table is sorted by source, so each source's rows are contiguous and
     * one pass covers both the per-source entropy and the corpus mean. */
    for (i = 0; i < segment_count;) {
        double total = 0.0;
        double entropy = 0.0;
        for (j = i; j < segment_count && strcmp(segments[j].source, segments[i].source) == 0; j++) {
            total += segments[j].count;
        }
        for (; i < j; i++) {
            double share = total > 0.0 ? segments[i].count / total : 0.0;
            if (share > 0.0) {
                entropy -= share * (log(share) / log(2.0));
            }
        }
        corpus_total += total;
        corpus_entropy += total * entropy;
    }
    if (corpus_total <= 0.0) {
        return;
    }
    corpus_entropy /= corpus_total;
    printf("PAIR\t%s\t%s\t%.3f\n", pair->lect_a, pair->lect_b, corpus_entropy);
    if (corpus_entropy >= RG_RESIDUE_MAX_PAIR_ENTROPY) {
        return;
    }

    for (i = 0; i < segment_count;) {
        double total = 0.0;
        double entropy = 0.0;
        double explained = 0.0;
        size_t start = i;
        for (j = i; j < segment_count && strcmp(segments[j].source, segments[i].source) == 0; j++) {
            total += segments[j].count;
        }
        for (; i < j; i++) {
            double share = total > 0.0 ? segments[i].count / total : 0.0;
            if (share > 0.0) {
                entropy -= share * (log(share) / log(2.0));
            }
        }
        if (entropy < RG_RESIDUE_MIN_ENTROPY || total < RG_RESIDUE_MIN_OBSERVATIONS) {
            continue;
        }
        /* Any committed environment at all disqualifies it: the claim is that
         * nothing in the vocabulary reduced this uncertainty, not that nothing
         * reduced all of it. */
        for (j = 0; j < conditioned_count; j++) {
            if (strcmp(conditioned[j].source, segments[start].source) == 0) {
                explained += conditioned[j].count;
            }
        }
        if (explained > 0.0) {
            continue;
        }
        printf("RESIDUE\t%s\t%s\t%s\t%.2f\t%.0f", pair->lect_a, pair->lect_b,
               segments[start].source, entropy, total);
        for (j = start; j < i; j++) {
            printf("%c%s", j == start ? '\t' : ' ', segments[j].target);
        }
        printf("\n");
        (*unexplained)++;
    }
}

/* Which annotations the corpus carries alongside its graphemes: the tone,
 * length and stress a segment is marked with, and the morpheme boundaries a
 * form is divided by. A dimension nothing carries is a dimension no rule can
 * be stated over, and a user reading "no conditioning found" deserves to know
 * whether the axis it would have been found on is absent from their file
 * rather than from the language.
 *
 * Annotations, not features, and the distinction is load-bearing. A
 * suprasegmental written into the grapheme is a feature of that grapheme and
 * the search reads it as one: `eː` carries `long:+` and can be conditioned on
 * without appearing here at all. What this counts is the separate channel --
 * a `length` column, a CLDF tone token, a stress diacritic the loader lifts
 * off the vowel -- because that is the channel a corpus can silently lack.
 * Reporting `length 0` for a corpus full of `eː` would send a user to add
 * something they already have. */
static size_t report_annotations(const rg_cognate_set *sets, size_t set_count) {
    static const char *const names[] = { "tone", "length", "stress", "boundaries" };
    size_t carried[4] = { 0, 0, 0, 0 };
    size_t i;
    size_t f;
    size_t s;
    size_t present = 0;

    for (i = 0; i < set_count; i++) {
        for (f = 0; f < sets[i].form_count; f++) {
            const rg_form *form = &sets[i].forms[f].form;
            carried[3] += form->morpheme_break_count > 0 ? 1 : 0;
            for (s = 0; s < form->segment_count; s++) {
                carried[0] += form->segments[s].tone != 0 && form->segments[s].tone[0] != '\0';
                carried[1] += form->segments[s].length != 0 && form->segments[s].length[0] != '\0';
                carried[2] += form->segments[s].stress != 0 && form->segments[s].stress[0] != '\0';
            }
        }
    }
    for (i = 0; i < 4; i++) {
        present += carried[i] > 0 ? 1 : 0;
    }
    printf("annotations\t%lu\n", (unsigned long)present);
    for (i = 0; i < 4; i++) {
        printf("ANNOTATION\t%s\t%lu\n", names[i], (unsigned long)carried[i]);
    }
    return present;
}

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

    /* Everything above is about graphemes that will not resolve. This is about
     * graphemes that resolve perfectly and mean the corpus was assembled from
     * two sources that disagree about where a segment ends -- which is invisible
     * to every other check there is, and produces confident, well-supported,
     * entirely false correspondences. Reported after the refusals because a
     * corpus that will not load has a more urgent problem. */
    {
        rg_corpus *corpus = 0;
        rg_load_diagnosis diagnosis;
        if (load_corpus_with_context(ctx, path, format, &corpus, &diagnosis) == RG_OK) {
            rg_transcription_drift_row *drift = 0;
            size_t drift_count = 0;
            if (rg_find_transcription_drift(ctx, rg_corpus_cognate_at(corpus, 0),
                                            rg_corpus_cognate_count(corpus),
                                            &drift, &drift_count) == RG_OK) {
                printf("drift\t%lu\n", (unsigned long)drift_count);
                for (i = 0; i < drift_count; i++) {
                    printf("DRIFT\t%s\t%s\t%s\t%s\t%lu/%lu\n",
                           drift[i].lect, drift[i].other_lect, drift[i].grapheme,
                           drift[i].written_as,
                           (unsigned long)drift[i].corroborated,
                           (unsigned long)drift[i].forms);
                }
                rg_transcription_drift_rows_free(drift, drift_count);
            }

            /* Everything above is about the corpus as written. This is about
             * what can be learned from it, and it costs a training run --
             * which is the honest price: the question "is my data good enough
             * to start" cannot be answered without trying. */
            {
                size_t annotated = report_annotations(rg_corpus_cognate_at(corpus, 0),
                                                      rg_corpus_cognate_count(corpus));
                rg_multi_model *model = 0;
                rg_train_options options;
                rg_train_options_init_defaults(&options);
                if (rg_train_model(ctx, rg_corpus_cognate_at(corpus, 0),
                                   rg_corpus_cognate_count(corpus), &options, &model) == RG_OK) {
                    size_t pairs = rg_multi_model_pair_model_count(model);
                    size_t unexplained = 0;
                    size_t p;
                    for (p = 0; p < pairs; p++) {
                        const rg_multi_pair_model_row *pair = rg_multi_model_pair_model_at(model, p);
                        if (pair != 0) {
                            report_pair_residue(pair, &unexplained);
                        }
                    }
                    printf("unexplained\t%lu\n", (unsigned long)unexplained);
                    /* The two numbers are only useful together, and combining
                     * them is the tool's job rather than the reader's: an
                     * unexplained correspondence is a different problem when
                     * the corpus had an axis to try and did not use it than
                     * when it never carried one. */
                    if (unexplained > 0 && annotated == 0) {
                        printf("NOTE\t%lu correspondence%s carr%s uncertainty that no environment "
                               "accounts for, and the corpus writes no tone, length, stress or "
                               "morpheme boundary. A rule cannot be stated over an axis the file "
                               "does not carry.\n",
                               (unsigned long)unexplained,
                               unexplained == 1 ? "" : "s",
                               unexplained == 1 ? "ies" : "y");
                    }
                    rg_multi_model_free(model);
                }
            }
            rg_corpus_free(corpus);
        }
    }
    rg_context_free(ctx);
    return count == 0 ? 0 : 1;
}

/* A corpus that will not load is the first thing a new user meets, and "parse
 * error" names neither the line nor the reason. */
static void report_load_failure(
    rg_context *ctx,
    rg_status status,
    const rg_load_diagnosis *diagnosis
) {
    if (diagnosis != 0 && diagnosis->message[0] != '\0') {
        fprintf(stderr, "regulae: line %lu: %s\n",
                (unsigned long)diagnosis->line, diagnosis->message);
        return;
    }
    report_failure(ctx, "loading corpus", status);
}

static int command_train(const char *path, const char *format, int pairwise, int human,
                         int json, int permutations, int tune_search,
                         int permutation_seed, const char *feature_system,
                         rg_split_scorer scorer, double split_prior,
                         double search_gamma, double split_threshold,
                         double long_split_threshold, int small_sample,
                         int predictive_folds, int predictive_seed) {
    rg_load_diagnosis load_diagnosis;
    rg_context *ctx = 0;
    rg_corpus *corpus = 0;
    rg_multi_model *model = 0;
    rg_train_options options;
    rg_status status;

    status = rg_context_new_builtin(&ctx);
    if (status != RG_OK) {
        return fail("creating context", status);
    }
    if (feature_system != 0) {
        status = rg_context_use_system(ctx, feature_system);
        if (status != RG_OK) {
            rg_context_free(ctx);
            return fail("selecting feature system", status);
        }
    }
    status = load_corpus_with_context(ctx, path, format, &corpus, &load_diagnosis);
    if (status != RG_OK) {
        report_load_failure(ctx, status, &load_diagnosis);
        rg_context_free(ctx);
        return 1;
    }
    if (rg_corpus_doublet_set_count(corpus) > 0) {
        fprintf(stderr, "regulae: %lu cognate sets carry a doublet; the corpus reads as %lu extra "
                "sets, each with its share of the confidence\n",
                (unsigned long)rg_corpus_doublet_set_count(corpus),
                (unsigned long)rg_corpus_doublet_expansion_count(corpus));
    }
    rg_train_options_init_defaults(&options);
    options.permutation_count = permutations;
    if (permutation_seed >= 0) {
        options.permutation_seed = permutation_seed;
    }
    options.tune_search_penalty = tune_search;
    options.predictive_folds = predictive_folds;
    if (predictive_seed >= 0) {
        options.predictive_seed = predictive_seed;
    }
    options.bic.split_scorer = scorer;
    if (split_prior > 0.0) {
        options.bic.split_prior_concentration = split_prior;
    }
    if (search_gamma >= 0.0) {
        options.bic.search_penalty_gamma = search_gamma;
    }
    if (isfinite(split_threshold)) {
        options.bic.delta_bic_threshold = split_threshold;
        options.bic.cross_dim_delta_bic_threshold = split_threshold;
    }
    if (isfinite(long_split_threshold)) {
        options.bic.long_range_delta_bic_threshold = long_split_threshold;
    }
    if (small_sample >= 0) {
        options.bic.multi_lect_bic_small_sample_correction = small_sample != 0;
    }
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
                                      &options, true, true);
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
    } else {
        char *text = pairwise ? rg_format_pairwise_tables(model)
                              : rg_format_multi_model_summary(model);
        if (text == 0) {
            rg_multi_model_free(model);
            rg_corpus_free(corpus);
            rg_context_free(ctx);
            return fail("rendering summary", RG_ERR_OOM);
        }
        fputs(text, stdout);
        rg_string_free(text);
    }
    rg_multi_model_free(model);
    rg_corpus_free(corpus);
    rg_context_free(ctx);
    return 0;
}

static int command_outliers(const char *path, const char *format, int top_k) {
    rg_load_diagnosis load_diagnosis;
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
    status = load_corpus_with_context(ctx, path, format, &corpus, &load_diagnosis);
    if (status != RG_OK) {
        report_load_failure(ctx, status, &load_diagnosis);
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
        printf("OUTLIER\t%s\t%d\t%.6f\t%.6f\t%.6f\n",
               rows[i].cognate_id, rows[i].pair_count, rows[i].cost_per_segment,
               rows[i].z_score, rows[i].confidence);
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
    rg_load_diagnosis load_diagnosis;
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
    status = load_corpus_with_context(ctx, path, format, &corpus, &load_diagnosis);
    if (status != RG_OK) {
        report_load_failure(ctx, status, &load_diagnosis);
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
    rg_load_diagnosis load_diagnosis;
    rg_context *ctx = 0;
    rg_corpus *corpus = 0;
    size_t c;
    rg_status status;

    status = rg_context_new_builtin(&ctx);
    if (status != RG_OK) {
        return fail("creating context", status);
    }
    status = load_corpus_with_context(ctx, path, format, &corpus, &load_diagnosis);
    if (status != RG_OK) {
        report_load_failure(ctx, status, &load_diagnosis);
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
    int permutation_seed = -1;
    int tune_search = 0;
    int predictive_folds = 0;
    int predictive_seed = -1;
    const char *feature_system = 0;
    rg_split_scorer scorer = RG_SPLIT_SCORER_CORRECTED_BIC;
    double split_prior = -1.0;
    double search_gamma = -1.0;
    double split_threshold = NAN;
    double long_split_threshold = NAN;
    int small_sample = -1;
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
        } else if (strcmp(argv[i], "--permutation-seed") == 0 && i + 1 < argc) {
            permutation_seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--predictive-folds") == 0 && i + 1 < argc) {
            predictive_folds = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--predictive-seed") == 0 && i + 1 < argc) {
            predictive_seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--feature-system") == 0 && i + 1 < argc) {
            feature_system = argv[++i];
        } else if (strcmp(argv[i], "--scorer") == 0 && i + 1 < argc) {
            if (!parse_split_scorer(argv[++i], &scorer)) {
                fprintf(stderr, "regulae: unknown split scorer: %s\n", argv[i]);
                return 2;
            }
        } else if (strcmp(argv[i], "--split-prior") == 0 && i + 1 < argc) {
            split_prior = strtod(argv[++i], 0);
        } else if (strcmp(argv[i], "--search-gamma") == 0 && i + 1 < argc) {
            search_gamma = strtod(argv[++i], 0);
        } else if (strcmp(argv[i], "--split-threshold") == 0 && i + 1 < argc) {
            split_threshold = strtod(argv[++i], 0);
        } else if (strcmp(argv[i], "--long-split-threshold") == 0 && i + 1 < argc) {
            long_split_threshold = strtod(argv[++i], 0);
        } else if (strcmp(argv[i], "--no-multilect-small-sample") == 0) {
            small_sample = 0;
        } else if (strcmp(argv[i], "--multilect-small-sample") == 0) {
            small_sample = 1;
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
        return command_train(path, format, pairwise, human, json, permutations,
                             tune_search, permutation_seed, feature_system,
                             scorer, split_prior, search_gamma, split_threshold,
                             long_split_threshold, small_sample,
                             predictive_folds, predictive_seed);
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
