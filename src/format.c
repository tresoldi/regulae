#include "internal.h"
#include "environment.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Human-readable renderings of segments, links, alignments and models. These
 * are diagnostics: the layout is meant to be read, not parsed. Machine
 * consumers should iterate the model accessors, or the CLI's summary output,
 * both of which are stable contracts. Mirrors format.go. */

#define RG_EMPTY_CHUNK_SYMBOL "\xe2\x88\x85"

typedef struct string_builder {
    char *data;
    size_t length;
    size_t capacity;
    int failed;
} string_builder;

static void builder_init(string_builder *builder) {
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

static void builder_append(string_builder *builder, const char *text) {
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

static void builder_appendf(string_builder *builder, const char *format, ...);

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 2, 3)))
#endif
static void builder_appendf(string_builder *builder, const char *format, ...) {
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

static char *builder_finish(string_builder *builder) {
    if (builder->failed) {
        free(builder->data);
        return 0;
    }
    if (builder->data == 0) {
        builder->data = (char *)calloc(1, 1);
    }
    return builder->data;
}

/* Whole counts print without a decimal point; anything else gets one place. */
static void append_count(string_builder *builder, double count) {
    if (count == (double)(long)count) {
        builder_appendf(builder, "%ld", (long)count);
    } else {
        builder_appendf(builder, "%.1f", count);
    }
}

static const char *score_label(rg_split_scorer scorer) {
    switch (scorer) {
    case RG_SPLIT_SCORER_CORRECTED_BIC:
        return "dBIC";
    case RG_SPLIT_SCORER_MULTINOMIAL_NML:
        return "dNML";
    case RG_SPLIT_SCORER_DIRICHLET_MARGINAL:
        return "dDir";
    default:
        return "score";
    }
}

static void append_segments(string_builder *builder, const rg_segment *segments, size_t count) {
    size_t i;
    if (count == 0) {
        builder_append(builder, RG_EMPTY_CHUNK_SYMBOL);
        return;
    }
    for (i = 0; i < count; i++) {
        int annotated = 0;
        if (i > 0) {
            builder_append(builder, " ");
        }
        builder_append(builder, segments[i].grapheme == 0 ? "" : segments[i].grapheme);
        if (segments[i].tone != 0 && segments[i].tone[0] != '\0') {
            builder_appendf(builder, "%sT=%s", annotated ? "," : "[", segments[i].tone);
            annotated = 1;
        }
        if (segments[i].length != 0 && segments[i].length[0] != '\0') {
            builder_appendf(builder, "%sL=%s", annotated ? "," : "[", segments[i].length);
            annotated = 1;
        }
        if (segments[i].stress != 0 && segments[i].stress[0] != '\0') {
            builder_appendf(builder, "%sS=%s", annotated ? "," : "[", segments[i].stress);
            annotated = 1;
        }
        if (annotated) {
            builder_append(builder, "]");
        }
    }
}

static void append_constraint_bracket(
    string_builder *builder,
    const char *label,
    const rg_feature_constraint *items,
    size_t count
) {
    size_t i;
    if (count == 0) {
        return;
    }
    builder_appendf(builder, " %s[", label);
    for (i = 0; i < count; i++) {
        builder_appendf(builder, "%s%s:%s", i > 0 ? "," : "", items[i].feature, items[i].value);
    }
    builder_append(builder, "]");
}

static void append_context(string_builder *builder, const rg_context_spec *context) {
    if (context == 0 || rg_context_spec_constraint_count(context) == 0) {
        return;
    }
    if (context->position != 0 && context->position[0] != '\0') {
        builder_appendf(builder, " @%s", context->position);
    }
    if (context->morpheme_index != 0 && context->morpheme_index[0] != '\0') {
        builder_appendf(builder, " morph#%s", context->morpheme_index);
    }
    if (context->morphological != 0 && context->morphological[0] != '\0') {
        builder_appendf(builder, " morph=%s", context->morphological);
    }
    if (context->syllable_role != 0 && context->syllable_role[0] != '\0') {
        builder_appendf(builder, " syl=%s", context->syllable_role);
    }
    if (context->syllable_position != 0 && context->syllable_position[0] != '\0') {
        builder_appendf(builder, " syl@%s", context->syllable_position);
    }
    /* Every feature slot, then every distance slot, each under the short label
     * the slot list carries. */
#define SLOT(name, label, key) \
    append_constraint_bracket(builder, label, context->name, context->name##_count);
    RG_ENV_FEATURE_SLOTS(SLOT)
#undef SLOT
    {
        size_t i;
#define DISTANCE_SLOT(name, label, key)                                  \
        for (i = 0; i < context->name##_count; i++) {               \
            builder_appendf(builder, " " label "%d[%s:%s]",         \
                            context->name[i].offset,                \
                            context->name[i].constraint.feature,    \
                            context->name[i].constraint.value);     \
        }
        RG_ENV_DISTANCE_SLOTS(DISTANCE_SLOT)
#undef DISTANCE_SLOT
    }
}

char *rg_format_segments(const rg_segment *segments, size_t count) {
    string_builder builder;
    builder_init(&builder);
    append_segments(&builder, segments, count);
    return builder_finish(&builder);
}

char *rg_format_link(const rg_link *link) {
    string_builder builder;
    if (link == 0) {
        return 0;
    }
    builder_init(&builder);
    append_segments(&builder, link->source, link->source_count);
    builder_append(&builder, " ~ ");
    append_segments(&builder, link->target, link->target_count);
    return builder_finish(&builder);
}

char *rg_format_alignment(const rg_alignment *alignment) {
    string_builder builder;
    size_t i;
    if (alignment == 0) {
        return 0;
    }
    builder_init(&builder);
    for (i = 0; i < rg_alignment_link_count(alignment); i++) {
        const rg_link *link = rg_alignment_link_at(alignment, i);
        builder_appendf(&builder, "  %2lu  ", (unsigned long)i);
        append_segments(&builder, link->source, link->source_count);
        builder_append(&builder, " ~ ");
        append_segments(&builder, link->target, link->target_count);
        append_context(&builder, &link->context);
        builder_append(&builder, "\n");
    }
    if (rg_alignment_link_count(alignment) == 0) {
        builder_append(&builder, "  (no links)\n");
    }
    return builder_finish(&builder);
}

/* The indices of `count` rows in the order discovery settled them.
 *
 * Insertion sort, so rows committed by one decision keep the order the table
 * publishes them in; qsort is not stable and would hand that to the
 * implementation. Written out twice here, once per table, until the rows had a
 * named evidence field to reach it through.
 *
 * Returns 0 on allocation failure, which the caller treats as "publish in table
 * order" rather than as a reason to fail a report. */
static size_t *decision_order_of(
    size_t count,
    int (*decision_index_of)(const void *rows, size_t index),
    const void *rows
) {
    size_t *order = (size_t *)calloc(count == 0 ? 1 : count, sizeof(*order));
    size_t i;
    if (order == 0) {
        return 0;
    }
    for (i = 0; i < count; i++) {
        size_t insert_at = i;
        int here = decision_index_of(rows, i);
        while (insert_at > 0 && decision_index_of(rows, order[insert_at - 1]) > here) {
            order[insert_at] = order[insert_at - 1];
            insert_at--;
        }
        order[insert_at] = i;
    }
    return order;
}

static int conditioned_class_decision_index(const void *rows, size_t index) {
    return ((const rg_multi_class_row *)rows)[index].evidence.decision_index;
}

static int multi_cross_dimensional_decision_index(const void *rows, size_t index) {
    return ((const rg_multi_cross_dimensional_row *)rows)[index].rule.evidence.decision_index;
}

void rg_format_model_options_init_defaults(rg_format_model_options *options) {
    if (options == 0) {
        return;
    }
    options->top_segments = 15;
    options->top_displacements = 5;
    options->min_count = 1.0;
    options->top_chunks = 10;
    options->top_classes = 20;
}

static void resolve_format_options(const rg_format_model_options *options, rg_format_model_options *out) {
    rg_format_model_options defaults;
    rg_format_model_options_init_defaults(&defaults);
    *out = options == 0 ? defaults : *options;
    if (out->top_segments <= 0) {
        out->top_segments = defaults.top_segments;
    }
    if (out->top_displacements <= 0) {
        out->top_displacements = defaults.top_displacements;
    }
    if (out->min_count <= 0.0) {
        out->min_count = defaults.min_count;
    }
    if (out->top_chunks <= 0) {
        out->top_chunks = defaults.top_chunks;
    }
    if (out->top_classes <= 0) {
        out->top_classes = defaults.top_classes;
    }
}

/* Row indices ordered by count, descending; ties keep table order, which is
 * itself deterministic. */
static size_t *order_by_count_desc(const double *counts, size_t count) {
    size_t *order = (size_t *)calloc(count == 0 ? 1 : count, sizeof(*order));
    size_t i;
    if (order == 0) {
        return 0;
    }
    for (i = 0; i < count; i++) {
        order[i] = i;
    }
    for (i = 1; i < count; i++) {
        size_t key = order[i];
        size_t j = i;
        while (j > 0 && counts[order[j - 1]] < counts[key]) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = key;
    }
    return order;
}

char *rg_format_pairwise_model(const rg_pairwise_model *model, const rg_format_model_options *options) {
    string_builder builder;
    rg_format_model_options opts;
    double *counts = 0;
    size_t *order = 0;
    size_t i;
    size_t shown;
    const rg_segment_count_row *segment_rows;
    const rg_conditioned_segment_count_row *conditioned_rows;
    const rg_chunk_row *chunk_rows;
    size_t segment_total = 0;
    size_t conditioned_total = 0;
    size_t chunk_total = 0;
    size_t xdim_total = 0;
    size_t xdim_published = 0;
    size_t tonal_total = 0;
    size_t null_total = 0;

    if (model == 0) {
        return 0;
    }
    resolve_format_options(options, &opts);
    builder_init(&builder);
    builder_append(&builder, "============================================================\n");
    builder_append(&builder, "PairwiseModel\n");
    builder_append(&builder, "============================================================\n");
    segment_rows = rg_pairwise_model_segment_counts(model, &segment_total);
    conditioned_rows = rg_pairwise_model_conditioned_segment_counts(model, &conditioned_total);
    chunk_rows = rg_pairwise_model_chunks(model, &chunk_total);
    {
        const rg_cross_dimensional_row *xdim =
            rg_pairwise_model_cross_dimensional_rows(model, &xdim_total);
        for (i = 0; i < xdim_total; i++) {
            if (rg_cross_dim_row_publishable_internal(&xdim[i])) {
                xdim_published++;
            }
        }
    }
    rg_pairwise_model_tonal_counts(model, &tonal_total);
    rg_pairwise_model_null_correspondences(model, &null_total);
    builder_appendf(&builder, "segment correspondences: %lu\n",
                    (unsigned long)(segment_total + null_total));
    builder_appendf(&builder, "conditioned entries:     %lu\n", (unsigned long)conditioned_total);
    builder_appendf(&builder, "promoted chunks:         %lu\n", (unsigned long)chunk_total);
    builder_appendf(&builder, "cross-dimensional rules: %lu\n", (unsigned long)xdim_published);
    builder_appendf(&builder, "tonal correspondences:   %lu\n\n", (unsigned long)tonal_total);

    /* One section: a correspondence to ∅ (a loss `g ~ ∅` or epenthesis `∅ ~ g`)
     * is a segment correspondence like any other, ranked by count alongside the
     * 1-to-1 rows. */
    builder_appendf(&builder, "--- Top %d segment correspondences ---\n", opts.top_segments);
    {
        const rg_segment_count_row *null_rows =
            rg_pairwise_model_null_correspondences(model, &null_total);
        size_t row_count = segment_total + null_total;
        const rg_segment_count_row **all =
            (const rg_segment_count_row **)calloc(row_count == 0 ? 1 : row_count, sizeof(*all));
        counts = (double *)calloc(row_count == 0 ? 1 : row_count, sizeof(*counts));
        if (all == 0 || counts == 0) {
            free((void *)all);
            free(counts);
            builder.failed = 1;
            return builder_finish(&builder);
        }
        for (i = 0; i < segment_total; i++) {
            all[i] = &segment_rows[i];
        }
        for (i = 0; i < null_total; i++) {
            all[segment_total + i] = &null_rows[i];
        }
        for (i = 0; i < row_count; i++) {
            counts[i] = all[i]->count;
        }
        order = order_by_count_desc(counts, row_count);
        if (order == 0) {
            free((void *)all);
            free(counts);
            builder.failed = 1;
            return builder_finish(&builder);
        }
        shown = 0;
        for (i = 0; i < row_count && shown < (size_t)opts.top_segments; i++) {
            const rg_segment_count_row *row = all[order[i]];
            if (row->count < opts.min_count) {
                continue;
            }
            builder_append(&builder, "  count=");
            append_count(&builder, row->count);
            builder_appendf(&builder, "  %s ~ %s\n", row->source, row->target);
            shown++;
        }
        if (shown == 0) {
            builder_append(&builder, "  (none)\n");
        }
        free((void *)all);
        free(counts);
        free(order);
    }

    builder_appendf(&builder, "\n--- Top %d promoted chunks ---\n", opts.top_chunks);
    shown = 0;
    for (i = 0; i < chunk_total && shown < (size_t)opts.top_chunks; i++) {
        const rg_chunk_row *row = &chunk_rows[i];
        builder_append(&builder, "  cost=");
        builder_appendf(&builder, "%7.3f  ", row->cost);
        append_segments(&builder, row->source, row->source_count);
        builder_append(&builder, " ~ ");
        append_segments(&builder, row->target, row->target_count);
        if (row->reordering) {
            builder_append(&builder, "   [reordering]");
        }
        builder_append(&builder, "\n");
        shown++;
    }
    if (shown == 0) {
        builder_append(&builder, "  (none)\n");
    }

    builder_append(&builder, "\n--- Conditioned entries ---\n");
    shown = 0;
    for (i = 0; i < conditioned_total; i++) {
        const rg_conditioned_segment_count_row *row = &conditioned_rows[i];
        builder_append(&builder, "  count=");
        append_count(&builder, row->count);
        builder_append(&builder, "/");
        append_count(&builder, row->source_total);
        builder_append(&builder, "  elsewhere=");
        append_count(&builder, row->contrast_count);
        builder_append(&builder, "/");
        append_count(&builder, row->contrast_total);
        builder_appendf(&builder, "  %s=%.1f  [%.2f,%.2f]%s  %s ~ %s",
                        score_label(row->evidence.scorer), row->evidence.delta_score,
                        row->uncertainty.lower, row->uncertainty.upper,
                        row->uncertainty.post_selection ? "*" : "",
                        row->source, row->target);
        append_context(&builder, &row->context);
        if (row->evidence.predictive.status != RG_PREDICTIVE_UNMEASURED) {
            builder_appendf(&builder, "  predictive=%s gain=%+.3f n=%lu",
                            rg_predictive_status_string(row->evidence.predictive.status),
                            row->evidence.predictive.log_loss_gain,
                            (unsigned long)row->evidence.predictive.conditioned.observation_count);
        }
        builder_append(&builder, "\n");
        shown++;
    }
    if (shown == 0) {
        builder_append(&builder, "  (none)\n");
    }
    return builder_finish(&builder);
}

/* Whether every lect in the class shows the same grapheme: a retention rather
 * than a change. */
static int class_is_identity(const rg_multi_class_row *row) {
    size_t i;
    if (row->segment_count == 0) {
        return 0;
    }
    for (i = 1; i < row->segment_count; i++) {
        if (strcmp(row->graphemes[i], row->graphemes[0]) != 0) {
            return 0;
        }
    }
    return 1;
}

/* The class a contrast id names. Conditioned first, because a split's
 * complement is usually itself a committed row, and the tables are small
 * enough that a scan is cheaper than an index nothing else needs. */
static const rg_multi_class_row *class_by_id(
    const rg_multi_class_row *uncond,
    size_t uncond_total,
    const rg_multi_class_row *cond,
    size_t cond_total,
    int id
) {
    size_t i;
    if (id < 0) {
        return 0;
    }
    for (i = 0; i < cond_total; i++) {
        if (cond[i].class_id == id) {
            return &cond[i];
        }
    }
    for (i = 0; i < uncond_total; i++) {
        if (uncond[i].class_id == id) {
            return &uncond[i];
        }
    }
    return 0;
}

static void append_class_segments(string_builder *builder, const rg_multi_class_row *row) {
    size_t i;
    for (i = 0; i < row->segment_count; i++) {
        builder_appendf(builder, "%s%s:%s", i > 0 ? " ~ " : "", row->lect_ids[i], row->graphemes[i]);
        /* The suprasegmentals are part of the outcome, so a tone correspondence
         * reads on the row: `north:a[⁵⁵]`. Bracketed only when carried. */
        if (row->suprasegmentals != 0) {
            const rg_suprasegmentals *s = &row->suprasegmentals[i];
            if (s->tone[0] != '\0' || s->length[0] != '\0' || s->stress[0] != '\0') {
                int first = 1;
                builder_appendf(builder, "[");
                if (s->tone[0] != '\0') {
                    builder_appendf(builder, "%s", s->tone);
                    first = 0;
                }
                if (s->length[0] != '\0') {
                    builder_appendf(builder, "%slen:%s", first ? "" : ",", s->length);
                    first = 0;
                }
                if (s->stress[0] != '\0') {
                    builder_appendf(builder, "%sstr:%s", first ? "" : ",", s->stress);
                }
                builder_appendf(builder, "]");
            }
        }
    }
}

static void append_class_contexts(string_builder *builder, const rg_multi_class_row *row) {
    size_t i;
    if (row->contexts == 0) {
        return;
    }
    for (i = 0; i < row->segment_count; i++) {
        if (rg_context_spec_constraint_count(&row->contexts[i]) == 0) {
            continue;
        }
        builder_appendf(builder, "  [%s", row->lect_ids[i]);
        append_context(builder, &row->contexts[i]);
        builder_append(builder, "]");
    }
}

/* A committed split whose elsewhere count exceeds its in-environment count,
 * or that is thin and covers little of its class, is printed but does not
 * lead the report: the reader should meet the split that carries the corpus
 * first. */
static int class_is_weak(const rg_multi_class_row *row) {
    if (row->contrast_count > row->count) {
        return 1;
    }
    if (row->count < 8 && row->confidence < 0.25) {
        return 1;
    }
    return 0;
}

static void append_conditioned_row(
    string_builder *builder,
    const rg_multi_class_row *row,
    const rg_multi_class_row *uncond_rows,
    size_t uncond_total,
    const rg_multi_class_row *cond_rows,
    size_t cond_total
) {
    builder_append(builder, "  count=");
    append_count(builder, row->count);
    /* The pivot's other reflex out of the environment, and where to read it.
     * This is the comparison the split was scored on; `elsewhere` below is
     * the same reflex out of the environment, ~0 whenever the rule is real,
     * and shown second so the two are not confused. */
    if (row->contrast_class_id >= 0) {
        builder_append(builder, " vs ");
        append_count(builder, row->contrast_alternative_count);
        builder_appendf(builder, " as #%d", row->contrast_class_id);
    }
    builder_append(builder, " elsewhere=");
    append_count(builder, row->contrast_count);
    builder_appendf(builder, " sets=%lu", (unsigned long)row->supporting_cognate_count);
    builder_appendf(builder, "  #%d", row->evidence.decision_index);
    if (row->evidence.standing == RG_RULE_STANDING_ABOVE_NOISE) {
        builder_append(builder, " STANDS");
    } else if (row->evidence.standing == RG_RULE_STANDING_UNMEASURED) {
        /* No permutations ran, so nothing about standing is claimed; say so
         * rather than let the row read as measured. */
        builder_append(builder, " unmeasured");
    } else {
        builder_append(builder, " within-noise");
    }
    builder_appendf(builder, " cov=%.2f %s=%.1f margin=%.2f [%.2f,%.2f]%s  ",
                    row->confidence, score_label(row->evidence.scorer),
                    row->evidence.delta_score, row->evidence.search_margin,
                    row->uncertainty.lower, row->uncertainty.upper,
                    row->uncertainty.post_selection ? "*" : "");
    if (row->environment_alternatives > 0) {
        builder_append(builder, "TIED ");
    }
    append_class_segments(builder, row);
    if (row->environment_alternatives > 0) {
        builder_appendf(builder, "  [environment not identifiable: %d other%s carve%s it the same]",
                        row->environment_alternatives,
                        row->environment_alternatives == 1 ? "" : "s",
                        row->environment_alternatives == 1 ? "s" : "");
    }
    append_class_contexts(builder, row);
    if (row->evidence.predictive.status != RG_PREDICTIVE_UNMEASURED) {
        builder_appendf(builder, " predictive=%s gain=%+.3f n=%lu",
                        rg_predictive_status_string(row->evidence.predictive.status),
                        row->evidence.predictive.log_loss_gain,
                        (unsigned long)row->evidence.predictive.conditioned.observation_count);
    }
    /* Two in five committed rules are X ~ X. Most are the retention side
     * of a real split, with the change sitting in the contrast class, and
     * "p ~ p before a vowel" is a null statement to read: a reader has to
     * notice the graphemes are the same and then chase an id to find the
     * event. Name which half of the pair is the event instead. */
    if (class_is_identity(row)) {
        const rg_multi_class_row *contrast = class_by_id(
            uncond_rows, uncond_total, cond_rows, cond_total, row->contrast_class_id);
        if (contrast != 0 && !class_is_identity(contrast)) {
            builder_append(builder, "  [unchanged here; the change is #");
            builder_appendf(builder, "%d]", contrast->class_id);
        }
    }
    builder_append(builder, "\n");
}

char *rg_format_multi_model(const rg_multi_model *model, const rg_format_model_options *options) {
    string_builder builder;
    const char *const *lect_names;
    const rg_multi_class_row *uncond_rows;
    const rg_multi_class_row *cond_rows;
    const rg_multi_cross_dimensional_row *xdim_rows;
    size_t lect_total = 0;
    size_t uncond_total = 0;
    size_t cond_total = 0;
    size_t xdim_total = 0;
    rg_format_model_options opts;
    size_t *decision_order = 0;
    size_t i;
    size_t total;

    if (model == 0) {
        return 0;
    }
    resolve_format_options(options, &opts);
    builder_init(&builder);
    builder_append(&builder, "============================================================\n");
    builder_append(&builder, "MultiLectModel\n");
    builder_append(&builder, "============================================================\n");
    lect_names = rg_multi_model_lects(model, &lect_total);
    uncond_rows = rg_multi_model_unconditioned_classes(model, &uncond_total);
    cond_rows = rg_multi_model_conditioned_classes(model, &cond_total);
    xdim_rows = rg_multi_model_cross_dimensional_rows(model, &xdim_total);
    {
        size_t xp = 0;
        for (i = 0; i < xdim_total; i++) {
            if (rg_cross_dim_row_publishable_internal(&xdim_rows[i].rule)) {
                xp++;
            }
        }
        builder_appendf(&builder, "lects (%lu): ", (unsigned long)lect_total);
        for (i = 0; i < lect_total; i++) {
            builder_appendf(&builder, "%s%s", i > 0 ? ", " : "", lect_names[i]);
        }
        builder_append(&builder, "\n");
        builder_appendf(&builder, "pairwise models:     %lu\n", (unsigned long)rg_multi_model_pair_model_count(model));
        builder_appendf(&builder, "unconditioned cls:   %lu\n", (unsigned long)uncond_total);
        builder_appendf(&builder, "conditioned cls:     %lu\n", (unsigned long)cond_total);
        builder_appendf(&builder, "cross-dimensional:   %lu\n", (unsigned long)xp);
    }
    {
        const rg_corpus_fit *fit = rg_multi_model_fit(model);
        builder_appendf(&builder, "cost/segment:        %.4f over %lu sets\n",
                        fit->cost_per_segment, (unsigned long)fit->scored_set_count);
        builder_appendf(&builder,
                        "observation units:   %lu etymon, %lu source; bootstrap=%s (%lu units)\n",
                        (unsigned long)fit->etymon_group_count,
                        (unsigned long)fit->source_group_count,
                        rg_observation_unit_string(fit->bootstrap_unit),
                        (unsigned long)fit->bootstrap_effective_unit_count);
        if (fit->sets_without_etymon_group > 0 || fit->sets_without_source_group > 0) {
            builder_appendf(&builder,
                            "  missing group labels are treated as separate cognate sets"
                            " (etymon %lu, source %lu)\n",
                            (unsigned long)fit->sets_without_etymon_group,
                            (unsigned long)fit->sets_without_source_group);
        }
        /* Printed only where it is worth a second look. A mean says nothing
         * about shape, and 4 is where the corpora in this repository that are
         * one thing stop and the ones that are two begin. Not a threshold in
         * the code -- the number is always published in the API -- but a
         * report that mentioned it every time would train the reader to skip
         * the line. */
        if (fit->cost_split_separation > 4.0) {
            builder_appendf(&builder,
                            "  alignment cost is bimodal: %.0f%% of the sets align notably worse,\n"
                            "  %.1f standard deviations from the rest. Causes include a regular\n"
                            "  structural split (a diphthong against a monophthong, a transposition\n"
                            "  against none), a lexical stratum, or bad cognates. `outliers --model`\n"
                            "  lists the expensive sets first; which of these it is, the\n"
                            "  distributions cannot say.\n",
                            100.0 * fit->cost_split_fraction, fit->cost_split_separation);
        }
        if (fit->inferred_nucleus_form_count > 0) {
            builder_appendf(&builder,
                            "syllable nuclei:     %lu of %lu forms had none of their own\n"
                            "  and were given one; a syllable-conditioned rule on those rests\n"
                            "  on a guess. Supply a syllables column to decide it yourself.\n",
                            (unsigned long)fit->inferred_nucleus_form_count,
                            (unsigned long)fit->syllabified_form_count);
        }
        if (fit->permutation_count > 0) {
            builder_appendf(&builder,
                            "pairing-shuffle null: %.4f +/- %.4f over %lu shuffles, z = %.1f\n",
                            fit->null_cost_per_segment_mean, fit->null_cost_per_segment_sd,
                            (unsigned long)fit->permutation_count, fit->cost_per_segment_z);
            builder_appendf(&builder,
                            "  the same shuffles give %.1f unconditioned and %.1f conditioned classes\n",
                            fit->null_unconditioned_class_mean, fit->null_conditioned_class_mean);
            builder_append(&builder,
                           "context-permuted null: each conditioned class is judged against its own\n"
                           "  pivot's environment shuffled against its outcome, correspondences held\n"
                           "  intact; a class at or under its pivot's p95 was findable with no\n"
                           "  environment to find\n");
            builder_appendf(&builder,
                            "verdict:             %lu of %lu conditioned rules stand above their pivot's null\n",
                            (unsigned long)fit->rules_above_noise,
                            (unsigned long)fit->rules_measured);
            /* Counted per lect pair, so not added to the line above: a rule
             * visible in every pair of a four-lect corpus is six here and one
             * there. Shown because the line above can read 0 of 4 on a corpus
             * whose one standing rule is in this table. */
            builder_appendf(&builder,
                            "                     %lu of %lu per-pair conditioned"
                            " correspondences above the pairing-shuffle level,\n"
                            "                     counted per pair\n",
                            (unsigned long)fit->pairwise_rules_above_noise,
                            (unsigned long)fit->pairwise_rules_measured);
        } else {
            builder_append(&builder,
                           "shuffled baseline:   not run (--permutations <n>)\n"
                           "  Class counts are not evidence of relatedness: shuffling the pairings\n"
                           "  in a corpus removes every correspondence and raises them. cost/segment\n"
                           "  is the number that falls, and the baseline is what makes it readable.\n");
        }
        /* The divisors for the per-pair line: pairs are correlated views of
         * one corpus, and a duplicate lect adds views without adding
         * evidence. */
        builder_appendf(&builder,
                        "pair opportunities:  %lu pairs over %lu lects"
                        ", %lu duplicate, %lu missing forms\n",
                        (unsigned long)fit->pair_count, (unsigned long)fit->lect_count,
                        (unsigned long)fit->duplicate_lect_count,
                        (unsigned long)fit->missing_form_count);
        if (fit->predictive.status == RG_PREDICTIVE_UNMEASURED) {
            builder_append(&builder,
                           "predictive evidence: not run (--predictive-folds <n>)\n");
        } else if (fit->predictive.status == RG_PREDICTIVE_DESCRIPTIVE_ONLY) {
            builder_appendf(&builder,
                            "predictive evidence: descriptive only; %lu dependency groups cannot\n"
                            "  support the requested grouped validation without leakage\n",
                            (unsigned long)fit->predictive_group_count);
        } else {
            builder_appendf(&builder,
                            "predictive evidence: %s over %lu folds and %lu reflexes;\n"
                            "  held-out log loss %.4f conditioned vs %.4f unconditioned"
                            " (gain %+.4f), top-%d %.1f%%, abstention %.1f%%\n",
                            rg_predictive_status_string(fit->predictive.status),
                            (unsigned long)fit->predictive.folds,
                            (unsigned long)fit->predictive.conditioned.observation_count,
                            fit->predictive.conditioned.log_loss,
                            fit->predictive.unconditioned.log_loss,
                            fit->predictive.log_loss_gain, fit->predictive_top_k,
                            100.0 * fit->predictive.conditioned.top_k_coverage,
                            100.0 * fit->predictive.conditioned.abstention_rate);
            if (fit->predictive_leave_one_lect_out_conditioned.observation_count > 0) {
                builder_appendf(&builder,
                                "  leave-one-lect-out: %lu reflexes, log loss %.4f"
                                " vs %.4f unconditioned\n",
                                (unsigned long)fit->predictive_leave_one_lect_out_conditioned.observation_count,
                                fit->predictive_leave_one_lect_out_conditioned.log_loss,
                                fit->predictive_leave_one_lect_out_unconditioned.log_loss);
            }
        }
        builder_append(&builder, "\n");
    }

    total = uncond_total;
    builder_appendf(&builder, "--- Top %d unconditioned classes ---\n", opts.top_classes);
    if (total == 0) {
        builder_append(&builder, "  (none)\n");
    }
    for (i = 0; i < total && i < (size_t)opts.top_classes; i++) {
        const rg_multi_class_row *row = &uncond_rows[i];
        builder_appendf(&builder, "  [%lu-way] count=", (unsigned long)row->segment_count);
        append_count(&builder, row->count);
        /* Beside count, never instead of it. count is aligned positions; a
         * correspondence is a claim about recurrence across the lexicon, and
         * these differ whenever one word realises the row twice. */
        builder_appendf(&builder, " sets=%lu  ", (unsigned long)row->supporting_cognate_count);
        append_class_segments(&builder, row);
        builder_append(&builder, "\n");
    }
    if (total > (size_t)opts.top_classes) {
        builder_appendf(&builder, "  ... (%lu more)\n", (unsigned long)(total - (size_t)opts.top_classes));
    }

    total = cond_total;
    decision_order = decision_order_of(total, conditioned_class_decision_index, cond_rows);
    if (decision_order == 0) {
        free(builder_finish(&builder));
        return 0;
    }
    /* In the order they were decided, not by size. Discovery is greedy and each
     * rule is committed against what the earlier ones left unexplained, so this
     * is a decision list: a later rule refines, or applies within, what an
     * earlier one did not settle. Sorted by count it reads as a set of
     * unrelated facts, some of them weak for no visible reason. */
    builder_appendf(&builder, "\n--- Conditioned classes, in the order they were decided"
                              " (first %d) ---\n", opts.top_classes);
    if (total == 0) {
        builder_append(&builder, "  (none - class-level discovery committed no splits)\n");
    }
    {
        size_t shown = 0;
        size_t complement_total = 0;
        size_t weak_total = 0;
        /* Changes lead. Retentions (every lect shows the same grapheme) are
         * the complement of a real split, and thin or elsewhere-heavy rows
         * are not peers of the split that carries the corpus; both print
         * under their own headings below. */
        for (i = 0; i < total && shown < (size_t)opts.top_classes; i++) {
            const rg_multi_class_row *row = &cond_rows[decision_order[i]];
            if (class_is_identity(row)) {
                complement_total++;
                continue;
            }
            if (class_is_weak(row)) {
                weak_total++;
                continue;
            }
            append_conditioned_row(&builder, row,
                                   uncond_rows, uncond_total, cond_rows, cond_total);
            shown++;
        }
        if (total > 0 && shown == 0) {
            builder_append(&builder, "  (none - every committed row is a complement or below the reporting floor)\n");
        }
        for (i = 0; i < total; i++) {
            const rg_multi_class_row *row = &cond_rows[decision_order[i]];
            if (!class_is_identity(row)) {
                continue;
            }
            if (complement_total > 0) {
                builder_append(&builder, "\n--- Complements (unchanged in the environment) ---\n");
                complement_total = 0;
            }
            append_conditioned_row(&builder, row,
                                   uncond_rows, uncond_total, cond_rows, cond_total);
        }
        for (i = 0; i < total; i++) {
            const rg_multi_class_row *row = &cond_rows[decision_order[i]];
            if (class_is_identity(row) || !class_is_weak(row)) {
                continue;
            }
            if (weak_total > 0) {
                builder_append(&builder, "\n--- Weak conditioned classes (elsewhere-heavy or thin) ---\n");
                weak_total = 0;
            }
            append_conditioned_row(&builder, row,
                                   uncond_rows, uncond_total, cond_rows, cond_total);
        }
    }
    free(decision_order);
    decision_order = 0;

    /* Spans of more than one segment are a first-class result: kt answering
     * tʃ, or two segments trading places, is one fact, not several segment
     * correspondences. The pairwise report has always shown these; the
     * default report reaches them through the pair models. */
    {
        size_t pair_total = rg_multi_model_pair_model_count(model);
        size_t p;
        size_t any_chunks = 0;
        builder_append(&builder, "\n--- Multi-segment correspondences ---\n");
        for (p = 0; p < pair_total; p++) {
            const rg_multi_pair_model_row *pair = rg_multi_model_pair_model_at(model, p);
            size_t chunk_total = 0;
            size_t ci;
            size_t shown = 0;
            const rg_chunk_row *chunks;
            if (pair == 0) {
                continue;
            }
            chunks = rg_pairwise_model_chunks(pair->model, &chunk_total);
            for (ci = 0; ci < chunk_total && shown < (size_t)opts.top_chunks; ci++) {
                const rg_chunk_row *chunk = &chunks[ci];
                builder_appendf(&builder, "  %s>%s  ", pair->lect_a, pair->lect_b);
                append_segments(&builder, chunk->source, chunk->source_count);
                builder_append(&builder, " ~ ");
                append_segments(&builder, chunk->target, chunk->target_count);
                builder_append(&builder, "  count=");
                append_count(&builder, chunk->count);
                if (chunk->reordering) {
                    builder_append(&builder, "  [reordering]");
                }
                builder_append(&builder, "\n");
                shown++;
                any_chunks++;
            }
        }
        if (any_chunks == 0) {
            builder_append(&builder, "  (none)\n");
        }
    }

    total = xdim_total;
    {
        size_t event_total = 0;
        const rg_proposed_event_row *events = rg_multi_model_proposed_events(model, &event_total);
        size_t e;
        builder_append(&builder, "\n--- Classes that look like one change ---\n");
        if (event_total == 0) {
            builder_append(&builder, "  (none)\n");
        }
        for (e = 0; e < event_total; e++) {
            const rg_proposed_event_row *event = &events[e];
            size_t m;
            builder_append(&builder, "  count=");
            append_count(&builder, event->count);
            builder_appendf(&builder, " sets=%lu over %lu class%s  margin=%.2f  ",
                            (unsigned long)event->supporting_cognate_count,
                            (unsigned long)event->class_id_count,
                            event->class_id_count == 1 ? "" : "es",
                            event->search_margin);
            for (m = 0; m < event->member_count; m++) {
                const rg_event_member *member = &event->members[m];
                size_t g;
                builder_appendf(&builder, "%s%s:{", m == 0 ? "" : " ~ ", member->lect_id);
                for (g = 0; g < member->grapheme_count; g++) {
                    builder_appendf(&builder, "%s%s", g == 0 ? "" : ",", member->graphemes[g]);
                }
                builder_append(&builder, "}");
                /* The suprasegmentals every member agrees on, bracketed as a
                 * class row writes them: without this a tone shift over two
                 * vowels reads as `{a,e} ~ {a,e}`. */
                if (member->suprasegmentals != 0) {
                    const rg_suprasegmentals *s = member->suprasegmentals;
                    int first_bit = 1;
                    builder_append(&builder, "[");
                    if (s->tone[0] != '\0') {
                        builder_appendf(&builder, "%s", s->tone);
                        first_bit = 0;
                    }
                    if (s->length[0] != '\0') {
                        builder_appendf(&builder, "%slen:%s", first_bit ? "" : ",", s->length);
                        first_bit = 0;
                    }
                    if (s->stress[0] != '\0') {
                        builder_appendf(&builder, "%sstr:%s", first_bit ? "" : ",", s->stress);
                    }
                    builder_append(&builder, "]");
                }
                for (g = 0; g < member->class_feature_count; g++) {
                    builder_appendf(&builder, "%s%s", g == 0 ? "=[" : " & ",
                                    member->class_features[g]);
                }
                if (member->class_feature_count > 0) {
                    builder_append(&builder, "]");
                }
            }
            /* The environment every member states, per lect, in the same
             * spelling a conditioned class row uses. A change without its
             * conditioning is a different claim from the one that was found. */
            for (m = 0; m < event->member_count; m++) {
                const rg_event_member *member = &event->members[m];
                if (rg_context_spec_constraint_count(&member->context) == 0) {
                    continue;
                }
                builder_appendf(&builder, "  %s:", member->lect_id);
                append_context(&builder, &member->context);
            }
            /* An event grouped by its outcome has members that differ in
             * environment -- that is the axis -- so an empty environment is
             * not a finding about conditioning and must not read as one. */
            if (event->axis == RG_EVENT_AXIS_OUTCOME) {
                builder_appendf(&builder, "  one change, %lu environments (see members)",
                                (unsigned long)event->class_id_count);
            }
            if (event->environment_alternatives > 0) {
                builder_appendf(&builder,
                                "  [%d rival conditioner%s: the corpus cannot tell this "
                                "environment from another]",
                                event->environment_alternatives,
                                event->environment_alternatives == 1 ? "" : "s");
            }
            /* A set no feature picks out may still be the set a change applied
             * to; saying so is not the same as doubting the grouping. */
            if (!event->featurally_definable) {
                builder_append(&builder, "  [no feature names this set in this corpus]");
            }
            if (event->shared_displacement_count > 0) {
                size_t d;
                builder_append(&builder, "  \xce\x94:");
                for (d = 0; d < event->shared_displacement_count; d++) {
                    const rg_feature_displacement *fd = &event->shared_displacement[d];
                    if (strcmp(fd->from_value, "absent") == 0 &&
                        strcmp(fd->to_value, "present") == 0) {
                        builder_appendf(&builder, " +%s(%s)",
                                        fd->feature, event->members[1].lect_id);
                    } else if (strcmp(fd->from_value, "present") == 0 &&
                               strcmp(fd->to_value, "absent") == 0) {
                        builder_appendf(&builder, " +%s(%s)",
                                        fd->feature, event->members[0].lect_id);
                    } else {
                        builder_appendf(&builder, " %s:%s\xe2\x86\x92%s",
                                        fd->feature, fd->from_value, fd->to_value);
                    }
                }
            }
            builder_append(&builder, "\n");
        }
        if (event_total > 0) {
            builder_append(&builder,
                "  Each member class is published above and stays there. Whether one\n"
                "  pooled rule beats the several is not decided here.\n");
        }
    }

    builder_append(&builder, "\n--- Cross-dimensional rules, in the order they were decided ---\n");
    {
        size_t publishable = 0;
        for (i = 0; i < total; i++) {
            if (rg_cross_dim_row_publishable_internal(&xdim_rows[i].rule)) {
                publishable++;
            }
        }
        if (publishable == 0) {
            builder_append(&builder, "  (none)\n");
        }
    }
    decision_order = decision_order_of(total, multi_cross_dimensional_decision_index, xdim_rows);
    if (decision_order == 0) {
        free(builder_finish(&builder));
        return 0;
    }
    for (i = 0; i < total; i++) {
        const rg_multi_cross_dimensional_row *row = &xdim_rows[decision_order[i]];
        const char *env_lect;
        const char *other_lect;
        if (!rg_cross_dim_row_publishable_internal(&row->rule)) {
            continue;
        }
        env_lect = row->rule.context_is_target ? row->target_lect : row->source_lect;
        other_lect = row->rule.context_is_target ? row->source_lect : row->target_lect;
        if (row->rule.dimension_from_environment) {
            /* Lect-internal: one lect's onset conditions its own tone. Written
             * `lect (self)` so it does not read as a cross-lect prediction. */
            builder_appendf(&builder, "  #%d %s (self) ", row->rule.evidence.decision_index, env_lect);
        } else {
            builder_appendf(&builder, "  #%d %s>%s ", row->rule.evidence.decision_index,
                            env_lect, other_lect);
        }
        append_context(&builder, &row->rule.environment);
        builder_appendf(&builder, " -> %s=%s@%+d  count=",
                        row->rule.dimension, row->rule.value, row->rule.position_offset);
        append_count(&builder, row->rule.count);
        builder_appendf(&builder, " conf=%.2f vs %.2f elsewhere %s=%.1f%s",
                        row->rule.confidence, row->rule.contrast_confidence,
                        score_label(row->rule.evidence.scorer), row->rule.evidence.delta_score,
                        row->rule.evidence.standing == RG_RULE_STANDING_UNMEASURED ? ""
                            : (row->rule.evidence.standing == RG_RULE_STANDING_ABOVE_NOISE ? "  STANDS" : "  within-noise"));
        if (row->rule.evidence.predictive.status != RG_PREDICTIVE_UNMEASURED) {
            builder_appendf(&builder, " predictive=%s gain=%+.3f n=%lu",
                            rg_predictive_status_string(row->rule.evidence.predictive.status),
                            row->rule.evidence.predictive.log_loss_gain,
                            (unsigned long)row->rule.evidence.predictive.conditioned.observation_count);
        }
        if (row->rule.environment_alternatives > 0) {
            /* The corpus cannot tell this environment from another segment's
             * feature; the named conditioner is one of several it supports. */
            builder_appendf(&builder, "  [environment not identifiable: %d other%s carve%s it the same]",
                            row->rule.environment_alternatives,
                            row->rule.environment_alternatives == 1 ? "" : "s",
                            row->rule.environment_alternatives == 1 ? "s" : "");
        }
        builder_append(&builder, "\n");
    }
    free(decision_order);
    return builder_finish(&builder);
}

char *rg_describe_multi_class(const rg_multi_model *model, const char *lect_id, const char *grapheme) {
    string_builder builder;
    const rg_multi_class_row *uncond_rows;
    const rg_multi_class_row *cond_rows;
    size_t uncond_total = 0;
    size_t cond_total = 0;
    size_t i;
    size_t shown;

    if (model == 0 || lect_id == 0 || grapheme == 0) {
        return 0;
    }
    builder_init(&builder);
    builder_appendf(&builder, "Classes with %s:%s\n", lect_id, grapheme);
    builder_append(&builder, "==================================================\n");

    uncond_rows = rg_multi_model_unconditioned_classes(model, &uncond_total);
    cond_rows = rg_multi_model_conditioned_classes(model, &cond_total);
    shown = 0;
    for (i = 0; i < uncond_total; i++) {
        const rg_multi_class_row *row = &uncond_rows[i];
        size_t j;
        for (j = 0; j < row->segment_count; j++) {
            if (strcmp(row->lect_ids[j], lect_id) == 0 && strcmp(row->graphemes[j], grapheme) == 0) {
                if (shown == 0) {
                    builder_append(&builder, "Unconditioned entries\n");
                }
                builder_append(&builder, "  count=");
                append_count(&builder, row->count);
                builder_append(&builder, "  ");
                append_class_segments(&builder, row);
                builder_append(&builder, "\n");
                shown++;
                break;
            }
        }
    }
    if (shown == 0) {
        builder_append(&builder, "Unconditioned entries\n  (none)\n");
    }

    builder_append(&builder, "\nConditioned entries\n");
    shown = 0;
    for (i = 0; i < cond_total; i++) {
        const rg_multi_class_row *row = &cond_rows[i];
        size_t j;
        for (j = 0; j < row->segment_count; j++) {
            if (strcmp(row->lect_ids[j], lect_id) == 0 && strcmp(row->graphemes[j], grapheme) == 0) {
                builder_append(&builder, "  count=");
                append_count(&builder, row->count);
                builder_append(&builder, " elsewhere=");
                append_count(&builder, row->contrast_count);
                builder_appendf(&builder, " cov=%.2f %s=%.1f  ", row->confidence,
                                score_label(row->evidence.scorer), row->evidence.delta_score);
                append_class_segments(&builder, row);
                append_class_contexts(&builder, row);
                builder_append(&builder, "\n");
                shown++;
                break;
            }
        }
    }
    if (shown == 0) {
        builder_append(&builder, "  (none)\n");
    }
    return builder_finish(&builder);
}

char *rg_format_drift(const rg_transcription_drift_row *rows, size_t count) {
    string_builder builder;
    size_t i;
    builder_init(&builder);
    for (i = 0; i < count; i++) {
        builder_appendf(&builder, "DRIFT\t%s\t%s\t%s\t%s\t%lu/%lu\n",
                        rows[i].lect, rows[i].other_lect, rows[i].grapheme,
                        rows[i].written_as,
                        (unsigned long)rows[i].corroborated,
                        (unsigned long)rows[i].forms);
    }
    if (count > 0) {
        builder_appendf(&builder,
                        "  %lu transcription-drift row%s: one lect writes one grapheme where the\n"
                        "  other writes its pieces. `regulae check` reports this before training;\n"
                        "  what follows trains over it as if it were a correspondence.\n",
                        (unsigned long)count, count == 1 ? "" : "s");
    }
    return builder_finish(&builder);
}

/* ---- The machine-readable summary -------------------------------------
 *
 * The CLI's default output, and a stable contract: one tab-separated line per
 * class and per cross-dimensional rule. It lived in cmd/regulae-c/main.c, which
 * made the CLI a fourth renderer of the same tables alongside this file's human
 * format, json.c and the CLI's own --pairwise dump -- and it had drifted. Its
 * environment key enumerated thirteen of the eighteen slots and omitted `self`
 * and `morpheme_index`, so two environments differing only there rendered to
 * the same key, in a format meant to be parsed.
 *
 * The key is not `append_context`'s rendering: it sorts each constraint list and
 * joins with ';' so the same environment always spells the same, which a
 * machine reader needs and a human one does not. Two renderings, one slot list.
 */

#define RG_SUMMARY_MAX_PARTS 32

static int summary_part_cmp(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static void summary_join_sorted(char *buffer, size_t size, char **parts, size_t used) {
    size_t offset = 0;
    size_t i;
    qsort(parts, used, sizeof(*parts), summary_part_cmp);
    for (i = 0; i < used; i++) {
        int written = snprintf(buffer + offset, size - offset, "%s%s", i > 0 ? ";" : "", parts[i]);
        if (written > 0 && (size_t)written < size - offset) {
            offset += (size_t)written;
        }
        free(parts[i]);
    }
}

static void summary_constraints(
    char *buffer,
    size_t size,
    const rg_feature_constraint *items,
    size_t count
) {
    char *parts[RG_SUMMARY_MAX_PARTS];
    size_t i;
    size_t used = 0;
    buffer[0] = '\0';
    for (i = 0; i < count && used < RG_SUMMARY_MAX_PARTS; i++) {
        char part[128];
        snprintf(part, sizeof(part), "%s:%s",
                 items[i].feature == 0 ? "" : items[i].feature,
                 items[i].value == 0 ? "" : items[i].value);
        parts[used] = rg_strdup_internal(part);
        if (parts[used] == 0) {
            break;
        }
        used++;
    }
    summary_join_sorted(buffer, size, parts, used);
}

static void summary_distances(
    char *buffer,
    size_t size,
    const rg_distance_constraint *items,
    size_t count
) {
    char *parts[RG_SUMMARY_MAX_PARTS];
    size_t i;
    size_t used = 0;
    buffer[0] = '\0';
    for (i = 0; i < count && used < RG_SUMMARY_MAX_PARTS; i++) {
        char part[160];
        snprintf(part, sizeof(part), "%d@%s:%s",
                 items[i].offset,
                 items[i].constraint.feature == 0 ? "" : items[i].constraint.feature,
                 items[i].constraint.value == 0 ? "" : items[i].constraint.value);
        parts[used] = rg_strdup_internal(part);
        if (parts[used] == 0) {
            break;
        }
        used++;
    }
    summary_join_sorted(buffer, size, parts, used);
}

/* Canonical rendering of an environment: slot-list order, sorted constraint
 * lists, empty slots omitted. */
static void summary_context_key(const rg_context_spec *context, char *out, size_t size) {
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

#define STRING_SLOT(name, key) EMIT(key, context->name == 0 ? "" : context->name);
    RG_ENV_STRING_SLOTS(STRING_SLOT)
#undef STRING_SLOT

#define FEATURE_SLOT(name, label, key)                                          \
    summary_constraints(slot, sizeof(slot), context->name, context->name##_count); \
    EMIT(key, slot);
#define DISTANCE_SLOT(name, label, key)                                         \
    summary_distances(slot, sizeof(slot), context->name, context->name##_count); \
    EMIT(key, slot);
    RG_ENV_SLOTS(FEATURE_SLOT, DISTANCE_SLOT)
#undef FEATURE_SLOT
#undef DISTANCE_SLOT

#undef EMIT

    if (first) {
        snprintf(out, size, "-");
    }
}

static void summary_class(
    string_builder *builder,
    const rg_multi_class_row *class_row,
    const char *label,
    int with_contexts
) {
    size_t i;
    builder_appendf(builder, "%s\t%d\t", label, class_row->class_id);
    for (i = 0; i < class_row->segment_count; i++) {
        builder_appendf(builder, "%s%s:%s", i > 0 ? "|" : "",
                        class_row->lect_ids[i], class_row->graphemes[i]);
    }
    builder_appendf(builder, "\t%.6f\t%.6f\t", class_row->count, class_row->confidence);
    /* One column each, always. This was a single column holding the supporting
     * cognates on an unconditioned row and the environment on a conditioned
     * one, so the only rows that state an environment were the only rows whose
     * evidence a consumer could not reach -- and a column whose meaning depends
     * on the row is not a column. Empty where a row has nothing to put in it. */
    for (i = 0; i < class_row->supporting_cognate_count; i++) {
        builder_appendf(builder, "%s%s", i > 0 ? "," : "", class_row->supporting_cognates[i]);
    }
    builder_append(builder, "\t");
    if (with_contexts) {
        for (i = 0; i < class_row->segment_count; i++) {
            char key[2048];
            summary_context_key(class_row->contexts == 0 ? 0 : &class_row->contexts[i],
                                key, sizeof(key));
            builder_appendf(builder, "%s%s=%s", i > 0 ? "|" : "", class_row->lect_ids[i], key);
        }
    }
    /* The contrast link, so a machine consumer of the summary reaches the same
     * comparison the human report shows: the class holding the pivot's other
     * reflex, and its mass out of the environment. -1 / 0 where there is none. */
    builder_appendf(builder, "\t%d\t%.6f", class_row->contrast_class_id,
                    class_row->contrast_alternative_count);
    builder_appendf(builder, "\t%d", class_row->environment_alternatives);
    builder_appendf(builder, "\t%s\t%.6f\n",
                    rg_predictive_status_string(class_row->evidence.predictive.status),
                    class_row->evidence.predictive.log_loss_gain);
}

char *rg_format_multi_model_summary(const rg_multi_model *model) {
    string_builder builder;
    size_t i;
    size_t total = 0;
    const char *const *lect_names;
    const rg_multi_class_row *class_rows;
    const rg_multi_cross_dimensional_row *xdim_rows;

    if (model == 0) {
        return 0;
    }
    builder_init(&builder);
    lect_names = rg_multi_model_lects(model, &total);
    builder_append(&builder, "LECTS\t");
    for (i = 0; i < total; i++) {
        builder_appendf(&builder, "%s%s", i > 0 ? " " : "", lect_names[i]);
    }
    builder_append(&builder, "\n");
    class_rows = rg_multi_model_unconditioned_classes(model, &total);
    for (i = 0; i < total; i++) {
        summary_class(&builder, &class_rows[i], "UNCOND", 0);
    }
    class_rows = rg_multi_model_conditioned_classes(model, &total);
    for (i = 0; i < total; i++) {
        summary_class(&builder, &class_rows[i], "COND", 1);
    }
    xdim_rows = rg_multi_model_cross_dimensional_rows(model, &total);
    for (i = 0; i < total; i++) {
        const rg_multi_cross_dimensional_row *row = &xdim_rows[i];
        char environment[2048];
        if (!rg_cross_dim_row_publishable_internal(&row->rule)) {
            continue;
        }
        summary_context_key(&row->rule.environment, environment, sizeof(environment));
        {
        const char *env_lect = row->rule.context_is_target ? row->target_lect : row->source_lect;
        const char *cond_lect = row->rule.dimension_from_environment
            ? env_lect
            : (row->rule.context_is_target ? row->source_lect : row->target_lect);
        builder_appendf(&builder,
                        "XDIM\t%s>%s\t%s\t%s=%s@%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%d\t%.6f\n",
                        env_lect, cond_lect,
                        environment,
                        row->rule.dimension, row->rule.value,
                        row->rule.position_offset,
                        row->rule.count, row->rule.source_count, row->rule.confidence,
                        row->rule.contrast_count, row->rule.contrast_source_count,
                        row->rule.contrast_confidence,
                        row->rule.environment_alternatives,
                        row->rule.evidence.delta_bic);
        }
    }
    {
        const rg_corpus_fit *fit = rg_multi_model_fit(model);
        builder_appendf(&builder,
                        "PREDICT\t%s\t%s\t%lu\t%lu\t%lu\t%.6f\t%.6f\t%.6f"
                        "\t%.6f\t%.6f\t%.6f\t%.6f\t%lu\t%.6f\t%.6f\n",
                        rg_predictive_status_string(fit->predictive.status),
                        rg_observation_unit_string(fit->predictive.observation_unit),
                        (unsigned long)fit->predictive.folds,
                        (unsigned long)fit->predictive_group_count,
                        (unsigned long)fit->predictive.conditioned.observation_count,
                        fit->predictive.conditioned.log_loss,
                        fit->predictive.unconditioned.log_loss,
                        fit->predictive.log_loss_gain,
                        fit->predictive.conditioned.top1_coverage,
                        fit->predictive.conditioned.top_k_coverage,
                        fit->predictive.conditioned.calibration_error,
                        fit->predictive.conditioned.abstention_rate,
                        (unsigned long)fit->predictive_leave_one_lect_out_conditioned.observation_count,
                        fit->predictive_leave_one_lect_out_conditioned.log_loss,
                        fit->predictive_leave_one_lect_out_unconditioned.log_loss);
    }
    return builder_finish(&builder);
}

char *rg_format_pairwise_tables(const rg_multi_model *model) {
    string_builder builder;
    size_t p;

    if (model == 0) {
        return 0;
    }
    builder_init(&builder);
    for (p = 0; p < rg_multi_model_pair_model_count(model); p++) {
        const rg_multi_pair_model_row *row = rg_multi_model_pair_model_at(model, p);
        const rg_pairwise_model *pm = row->model;
        const rg_segment_count_row *seg_rows;
        const rg_conditioned_segment_count_row *cond_rows;
        const rg_chunk_row *chunk_rows;
        const rg_tonal_count_row *tone_rows;
        size_t n = 0;
        size_t i;
        size_t k;
        seg_rows = rg_pairwise_model_segment_counts(pm, &n);
        for (i = 0; i < n; i++) {
            const rg_segment_count_row *seg = &seg_rows[i];
            builder_appendf(&builder, "SEG\t%s>%s\t%s\t%s\t-\t%.6f\t[%.4f,%.4f]\t%s\n",
                            row->lect_a, row->lect_b,
                            seg->source, seg->target, seg->count,
                            seg->uncertainty.lower, seg->uncertainty.upper,
                            rg_uncertainty_method_string(seg->uncertainty.method));
        }
        cond_rows = rg_pairwise_model_conditioned_segment_counts(pm, &n);
        for (i = 0; i < n; i++) {
            const rg_conditioned_segment_count_row *seg = &cond_rows[i];
            char key[2048];
            summary_context_key(&seg->context, key, sizeof(key));
            /* Which form's environment the rule names. Two rows can carry the
             * same context and mean different things: one says the source
             * looked like that, the other the target. */
            builder_appendf(&builder, "SEG\t%s>%s\t%s\t%s\t%s%s\t%.6f\t[%.4f,%.4f]\t%s%s"
                            "\t%s\t%lu\t%.6f\n",
                            row->lect_a, row->lect_b,
                            seg->source, seg->target,
                            seg->context_is_target ? "@target " : "", key, seg->count,
                            seg->uncertainty.lower, seg->uncertainty.upper,
                            rg_uncertainty_method_string(seg->uncertainty.method),
                            seg->uncertainty.post_selection ? "/post-selection" : "",
                            rg_predictive_status_string(seg->evidence.predictive.status),
                            (unsigned long)seg->evidence.predictive.conditioned.observation_count,
                            seg->evidence.predictive.log_loss_gain);
        }
        chunk_rows = rg_pairwise_model_chunks(pm, &n);
        for (i = 0; i < n; i++) {
            const rg_chunk_row *chunk = &chunk_rows[i];
            builder_appendf(&builder, "CHUNK\t%s>%s\t", row->lect_a, row->lect_b);
            for (k = 0; k < chunk->source_count; k++) {
                builder_append(&builder, chunk->source[k].grapheme);
            }
            builder_append(&builder, "\t");
            for (k = 0; k < chunk->target_count; k++) {
                builder_append(&builder, chunk->target[k].grapheme);
            }
            builder_appendf(&builder, "\t%.6f\t%.6f\t%s\n", chunk->cost, chunk->count,
                            chunk->reordering ? "reordering" : "-");
        }
        tone_rows = rg_pairwise_model_tonal_counts(pm, &n);
        for (i = 0; i < n; i++) {
            const rg_tonal_count_row *tone = &tone_rows[i];
            builder_appendf(&builder, "TONE\t%s>%s\t%s>%s\t%.6f\n", row->lect_a, row->lect_b,
                            tone->source_tone, tone->target_tone, tone->count);
        }
        {
            /* Correspondences to ∅ ride the same SEG rows as the 1-to-1 table,
             * with the gap grapheme in the source (a loss) or target (an
             * epenthesis) column. */
            const rg_segment_count_row *null_rows =
                rg_pairwise_model_null_correspondences(pm, &n);
            for (i = 0; i < n; i++) {
                const rg_segment_count_row *seg = &null_rows[i];
                builder_appendf(&builder, "SEG\t%s>%s\t%s\t%s\t-\t%.6f\t[%.4f,%.4f]\t%s\n",
                                row->lect_a, row->lect_b,
                                seg->source, seg->target, seg->count,
                                seg->uncertainty.lower, seg->uncertainty.upper,
                                rg_uncertainty_method_string(seg->uncertainty.method));
            }
        }
    }
    return builder_finish(&builder);
}
