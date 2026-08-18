#ifndef REGULAE_MULTILECT_INTERNAL_H
#define REGULAE_MULTILECT_INTERNAL_H

/* Shared between the multi-lect stages, and between them only.
 *
 * multilect.c ran to 3,500 lines over four responsibilities that do not
 * overlap much: training each lect pair, reconciling those pairwise results
 * into position classes, discovering conditioned classes across lects, and
 * measuring how well the whole thing fits the corpus it was trained on. They
 * are one file each now, and this is what they had been sharing implicitly by
 * sitting in the same translation unit.
 *
 * Not internal.h, for the same reason src/model_internal.h is not: that header
 * is the surface between modules, and no other module has any business with a
 * pivot bucket or a permutation baseline. */

#include "internal.h"

#include <string.h>

/* What a run of shuffled corpora reached, held by rg_train_model and filled
 * by multilect_fit.c. */
typedef struct permutation_baseline permutation_baseline;
struct permutation_baseline {
    size_t runs;
    double cost_mean;
    double cost_sd;
    double unconditioned_mean;
    double conditioned_mean;
    double search_margin;
    double search_margin_quantile;
};

/* multilect_fit.c: how well the model fits, and what the same training does
 * to a corpus with its correspondences shuffled out. */
rg_status compute_corpus_fit(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    const permutation_baseline *baseline,
    rg_multi_model *model
);
rg_status run_permutation_baseline(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    const char *const *lect_ids,
    size_t lect_count,
    permutation_baseline *out
);
rg_status bootstrap_class_intervals(
    rg_multi_model *model,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options
);

/* Reconciliation helpers the fit measurement reads back. */
const rg_pairwise_model *pair_model_for(const rg_multi_model *model, const char *lect_a, const char *lect_b);
rg_status present_lects_sorted(
    const rg_cognate_set *cognate,
    const rg_multi_model *model,
    const rg_form **forms,
    size_t **out_order,
    size_t *out_count
);

/* One reconciled position tuple: which lects, which graphemes, which
 * environments. Produced by multilect_reconcile.c, consumed by class
 * discovery, released by the model's teardown. */
typedef struct reconciled_observation {
    size_t cognate_index;
    char **lects;
    char **graphemes;
    size_t *positions;
    size_t *lect_indices;
    size_t segment_count;
    double weight;
    /* Which unconditioned bucket this landed in, before the buckets are
     * sorted into class order. */
    size_t bucket_index;
} reconciled_observation;

/* Small shared helpers over the model's own storage. */
void string_array_clear(char **items, size_t count);
double cognate_weight(const rg_cognate_set *cognate);
const rg_form *form_for_lect(const rg_cognate_set *cognate, const char *lect_id);

/* True for the deletion sentinel a class carries for a lect that lost the
 * segment. A gap has no phonological position, so the conditioning search steps
 * over it -- see the pivot loop in multilect_classes.c. */
static inline int grapheme_is_gap(const char *grapheme) {
    return grapheme != 0 && strcmp(grapheme, RG_GAP_GRAPHEME) == 0;
}
void reconciled_observation_clear(reconciled_observation *obs);
void reconciled_observations_free(reconciled_observation *items, size_t count);

/* multilect_reconcile.c: pairwise alignments into position classes. */
rg_status aggregate_position_classes(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    rg_multi_model *model,
    reconciled_observation **out_observations,
    size_t *out_observation_count
);

rg_status class_position_add_id(rg_class_position *position, int class_id);

/* multilect_classes.c: conditioned classes discovered across lects. */
rg_status multi_lect_context_discovery(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    rg_multi_model *model,
    const reconciled_observation *observations,
    size_t observation_count
);

/* multilect_pairs.c: one pairwise model per ordered lect pair, and the
 * cross-dimensional rows lifted out of them. */
rg_status train_pair_models(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    rg_progress_state *progress,
    rg_multi_model *model
);
rg_status lift_cross_dimensional_rows(rg_multi_model *model);

/* events.c: conditioned classes that look like one change, grouped. Runs after
 * class discovery and changes nothing it reads. */
rg_status rg_propose_events_internal(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    rg_multi_model *model
);
void rg_proposed_events_free_internal(rg_proposed_event_row *rows, size_t count);

/* predictive.c: selection-nested validation over dependency components. */
rg_status rg_predictive_evaluate_internal(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    const rg_train_options *options,
    rg_multi_model *model
);

#endif
