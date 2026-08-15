#include "internal.h"

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
    append_constraint_bracket(builder, "pre", context->preceding, context->preceding_count);
    append_constraint_bracket(builder, "fol", context->following, context->following_count);
    append_constraint_bracket(builder, "somewhere-pre", context->somewhere_preceding, context->somewhere_preceding_count);
    append_constraint_bracket(builder, "somewhere-fol", context->somewhere_following, context->somewhere_following_count);
    append_constraint_bracket(builder, "same-syl", context->same_syllable, context->same_syllable_count);
    append_constraint_bracket(builder, "next-syl", context->next_syllable, context->next_syllable_count);
    append_constraint_bracket(builder, "prev-syl", context->previous_syllable, context->previous_syllable_count);
    append_constraint_bracket(builder, "self-stress", context->self_stress, context->self_stress_count);
    append_constraint_bracket(builder, "pre-stress", context->preceding_stress, context->preceding_stress_count);
    append_constraint_bracket(builder, "fol-stress", context->following_stress, context->following_stress_count);
    {
        size_t i;
        for (i = 0; i < context->preceding_at_distance_count; i++) {
            builder_appendf(builder, " pre@%d[%s:%s]",
                            context->preceding_at_distance[i].offset,
                            context->preceding_at_distance[i].constraint.feature,
                            context->preceding_at_distance[i].constraint.value);
        }
        for (i = 0; i < context->following_at_distance_count; i++) {
            builder_appendf(builder, " fol@%d[%s:%s]",
                            context->following_at_distance[i].offset,
                            context->following_at_distance[i].constraint.feature,
                            context->following_at_distance[i].constraint.value);
        }
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

    if (model == 0) {
        return 0;
    }
    resolve_format_options(options, &opts);
    builder_init(&builder);
    builder_append(&builder, "============================================================\n");
    builder_append(&builder, "PairwiseModel\n");
    builder_append(&builder, "============================================================\n");
    builder_appendf(&builder, "segment correspondences: %lu\n",
                    (unsigned long)rg_pairwise_model_segment_count_row_count(model));
    builder_appendf(&builder, "conditioned entries:     %lu\n",
                    (unsigned long)rg_pairwise_model_conditioned_segment_count_row_count(model));
    builder_appendf(&builder, "promoted chunks:         %lu\n",
                    (unsigned long)rg_pairwise_model_chunk_row_count(model));
    builder_appendf(&builder, "cross-dimensional rules: %lu\n",
                    (unsigned long)rg_pairwise_model_cross_dimensional_row_count(model));
    builder_appendf(&builder, "tonal correspondences:   %lu\n\n",
                    (unsigned long)rg_pairwise_model_tonal_count_row_count(model));

    builder_appendf(&builder, "--- Top %d segment correspondences ---\n", opts.top_segments);
    {
        size_t row_count = rg_pairwise_model_segment_count_row_count(model);
        counts = (double *)calloc(row_count == 0 ? 1 : row_count, sizeof(*counts));
        if (counts == 0) {
            builder.failed = 1;
            return builder_finish(&builder);
        }
        for (i = 0; i < row_count; i++) {
            counts[i] = rg_pairwise_model_segment_count_row_at(model, i)->count;
        }
        order = order_by_count_desc(counts, row_count);
        if (order == 0) {
            free(counts);
            builder.failed = 1;
            return builder_finish(&builder);
        }
        shown = 0;
        for (i = 0; i < row_count && shown < (size_t)opts.top_segments; i++) {
            const rg_segment_count_row *row = rg_pairwise_model_segment_count_row_at(model, order[i]);
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
        free(counts);
        free(order);
    }

    builder_appendf(&builder, "\n--- Top %d promoted chunks ---\n", opts.top_chunks);
    shown = 0;
    for (i = 0; i < rg_pairwise_model_chunk_row_count(model) && shown < (size_t)opts.top_chunks; i++) {
        const rg_chunk_row *row = rg_pairwise_model_chunk_row_at(model, i);
        builder_append(&builder, "  cost=");
        builder_appendf(&builder, "%7.3f  ", row->cost);
        append_segments(&builder, row->source, row->source_count);
        builder_append(&builder, " ~ ");
        append_segments(&builder, row->target, row->target_count);
        builder_append(&builder, "\n");
        shown++;
    }
    if (shown == 0) {
        builder_append(&builder, "  (none)\n");
    }

    builder_append(&builder, "\n--- Conditioned entries ---\n");
    shown = 0;
    for (i = 0; i < rg_pairwise_model_conditioned_segment_count_row_count(model); i++) {
        const rg_conditioned_segment_count_row *row =
            rg_pairwise_model_conditioned_segment_count_row_at(model, i);
        builder_append(&builder, "  count=");
        append_count(&builder, row->count);
        builder_append(&builder, "/");
        append_count(&builder, row->source_total);
        builder_append(&builder, "  elsewhere=");
        append_count(&builder, row->contrast_count);
        builder_append(&builder, "/");
        append_count(&builder, row->contrast_total);
        builder_appendf(&builder, "  dBIC=%.1f  %s ~ %s", row->delta_bic, row->source, row->target);
        append_context(&builder, &row->context);
        builder_append(&builder, "\n");
        shown++;
    }
    if (shown == 0) {
        builder_append(&builder, "  (none)\n");
    }
    return builder_finish(&builder);
}

static void append_class_segments(string_builder *builder, const rg_multi_class_row *row) {
    size_t i;
    for (i = 0; i < row->segment_count; i++) {
        builder_appendf(builder, "%s%s:%s", i > 0 ? " ~ " : "", row->lect_ids[i], row->graphemes[i]);
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

char *rg_format_multi_model(const rg_multi_model *model, const rg_format_model_options *options) {
    string_builder builder;
    rg_format_model_options opts;
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
    builder_appendf(&builder, "lects (%lu): ", (unsigned long)rg_multi_model_lect_count(model));
    for (i = 0; i < rg_multi_model_lect_count(model); i++) {
        builder_appendf(&builder, "%s%s", i > 0 ? ", " : "", rg_multi_model_lect_at(model, i));
    }
    builder_append(&builder, "\n");
    builder_appendf(&builder, "pairwise models:     %lu\n", (unsigned long)rg_multi_model_pair_model_count(model));
    builder_appendf(&builder, "unconditioned cls:   %lu\n", (unsigned long)rg_multi_model_unconditioned_class_count(model));
    builder_appendf(&builder, "conditioned cls:     %lu\n", (unsigned long)rg_multi_model_conditioned_class_count(model));
    builder_appendf(&builder, "cross-dimensional:   %lu\n", (unsigned long)rg_multi_model_cross_dimensional_row_count(model));
    {
        const rg_corpus_fit *fit = rg_multi_model_fit(model);
        builder_appendf(&builder, "cost/segment:        %.4f over %lu sets\n",
                        fit->cost_per_segment, (unsigned long)fit->scored_set_count);
        if (fit->permutation_count > 0) {
            builder_appendf(&builder,
                            "shuffled baseline:   %.4f +/- %.4f over %lu shuffles, z = %.1f\n",
                            fit->null_cost_per_segment_mean, fit->null_cost_per_segment_sd,
                            (unsigned long)fit->permutation_count, fit->cost_per_segment_z);
            builder_appendf(&builder,
                            "  the same shuffles give %.1f unconditioned and %.1f conditioned classes\n",
                            fit->null_unconditioned_class_mean, fit->null_conditioned_class_mean);
            builder_appendf(&builder,
                            "  noise reaches search margin %.2f (p%.0f); a rule at or under that\n"
                            "  was findable in data with no correspondences left in it\n",
                            fit->null_search_margin, fit->null_search_margin_quantile * 100.0);
        } else {
            builder_append(&builder,
                           "shuffled baseline:   not run (--permutations <n>)\n"
                           "  Class counts are not evidence of relatedness: shuffling the pairings\n"
                           "  in a corpus removes every correspondence and raises them. cost/segment\n"
                           "  is the number that falls, and the baseline is what makes it readable.\n");
        }
        builder_append(&builder, "\n");
    }

    total = rg_multi_model_unconditioned_class_count(model);
    builder_appendf(&builder, "--- Top %d unconditioned classes ---\n", opts.top_classes);
    if (total == 0) {
        builder_append(&builder, "  (none)\n");
    }
    for (i = 0; i < total && i < (size_t)opts.top_classes; i++) {
        const rg_multi_class_row *row = rg_multi_model_unconditioned_class_at(model, i);
        builder_appendf(&builder, "  [%lu-way] count=", (unsigned long)row->segment_count);
        append_count(&builder, row->count);
        builder_append(&builder, "  ");
        append_class_segments(&builder, row);
        builder_append(&builder, "\n");
    }
    if (total > (size_t)opts.top_classes) {
        builder_appendf(&builder, "  ... (%lu more)\n", (unsigned long)(total - (size_t)opts.top_classes));
    }

    total = rg_multi_model_conditioned_class_count(model);
    builder_appendf(&builder, "\n--- Top %d conditioned classes ---\n", opts.top_classes);
    if (total == 0) {
        builder_append(&builder, "  (none - class-level discovery committed no splits)\n");
    }
    for (i = 0; i < total && i < (size_t)opts.top_classes; i++) {
        const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
        builder_append(&builder, "  count=");
        append_count(&builder, row->count);
        builder_append(&builder, " elsewhere=");
        append_count(&builder, row->contrast_count);
        builder_appendf(&builder, " cov=%.2f dBIC=%.1f margin=%.2f  ",
                        row->confidence, row->delta_bic, row->search_margin);
        append_class_segments(&builder, row);
        append_class_contexts(&builder, row);
        builder_append(&builder, "\n");
    }
    if (total > (size_t)opts.top_classes) {
        builder_appendf(&builder, "  ... (%lu more)\n", (unsigned long)(total - (size_t)opts.top_classes));
    }

    total = rg_multi_model_cross_dimensional_row_count(model);
    builder_append(&builder, "\n--- Cross-dimensional rules ---\n");
    if (total == 0) {
        builder_append(&builder, "  (none)\n");
    }
    for (i = 0; i < total; i++) {
        const rg_multi_cross_dimensional_row *row = rg_multi_model_cross_dimensional_row_at(model, i);
        builder_appendf(&builder, "  %s>%s  %s=%s@%s -> %s=%s@%+d  count=",
                        row->source_lect, row->target_lect,
                        row->source_feature, row->source_value, row->source_position,
                        row->target_dimension, row->target_value, row->target_position_offset);
        append_count(&builder, row->count);
        builder_appendf(&builder, " conf=%.2f vs %.2f elsewhere\n",
                        row->confidence, row->contrast_confidence);
    }
    return builder_finish(&builder);
}

char *rg_describe_multi_class(const rg_multi_model *model, const char *lect_id, const char *grapheme) {
    string_builder builder;
    size_t i;
    size_t shown;

    if (model == 0 || lect_id == 0 || grapheme == 0) {
        return 0;
    }
    builder_init(&builder);
    builder_appendf(&builder, "Classes with %s:%s\n", lect_id, grapheme);
    builder_append(&builder, "==================================================\n");

    shown = 0;
    for (i = 0; i < rg_multi_model_unconditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_unconditioned_class_at(model, i);
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
    for (i = 0; i < rg_multi_model_conditioned_class_count(model); i++) {
        const rg_multi_class_row *row = rg_multi_model_conditioned_class_at(model, i);
        size_t j;
        for (j = 0; j < row->segment_count; j++) {
            if (strcmp(row->lect_ids[j], lect_id) == 0 && strcmp(row->graphemes[j], grapheme) == 0) {
                builder_append(&builder, "  count=");
                append_count(&builder, row->count);
                builder_append(&builder, " elsewhere=");
                append_count(&builder, row->contrast_count);
                builder_appendf(&builder, " cov=%.2f dBIC=%.1f  ", row->confidence, row->delta_bic);
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
