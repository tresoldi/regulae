#include "regulae.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* M5 asks whether the same supplied surface relationship is recovered when the
 * metadata changes. These are the probes that have to hold in every build, not
 * only when scripts/evaluate_m5.py is run: the panel measures the whole space,
 * and this file keeps the commitments that a refactor could quietly lose.
 *
 * None of it is a claim about inheritance or about which lect is ancestral. */

typedef struct corpus_view {
    rg_corpus *corpus;
    rg_cognate_set *sets;
    rg_cognate_form *forms;
    size_t set_count;
} corpus_view;

/* The corpus with every lect id replaced. The published rows borrow the ids the
 * caller supplies, so `names` has to outlive the model. */
static void corpus_view_open(
    corpus_view *view,
    const char *path,
    const char *const *from,
    const char *const *to,
    size_t name_count
) {
    size_t total = 0;
    size_t i;
    size_t next = 0;
    assert(rg_corpus_load_tsv(path, 0, &view->corpus, 0) == RG_OK);
    view->set_count = rg_corpus_cognate_count(view->corpus);
    for (i = 0; i < view->set_count; i++) {
        total += rg_corpus_cognate_at(view->corpus, i)->form_count;
    }
    view->sets = (rg_cognate_set *)calloc(view->set_count, sizeof(*view->sets));
    view->forms = (rg_cognate_form *)calloc(total == 0 ? 1 : total, sizeof(*view->forms));
    assert(view->sets != 0 && view->forms != 0);
    for (i = 0; i < view->set_count; i++) {
        const rg_cognate_set *source = rg_corpus_cognate_at(view->corpus, i);
        size_t f;
        view->sets[i] = *source;
        view->sets[i].forms = &view->forms[next];
        for (f = 0; f < source->form_count; f++) {
            size_t n;
            view->forms[next] = source->forms[f];
            for (n = 0; n < name_count; n++) {
                if (strcmp(source->forms[f].lect_id, from[n]) == 0) {
                    view->forms[next].lect_id = to[n];
                    break;
                }
            }
            next++;
        }
    }
}

static void corpus_view_reverse_rows(corpus_view *view) {
    size_t i;
    for (i = 0; i < view->set_count / 2; i++) {
        rg_cognate_set swap = view->sets[i];
        view->sets[i] = view->sets[view->set_count - 1 - i];
        view->sets[view->set_count - 1 - i] = swap;
    }
}

static void corpus_view_close(corpus_view *view) {
    free(view->sets);
    free(view->forms);
    rg_corpus_free(view->corpus);
}

static rg_multi_model *train_view(rg_context *ctx, const corpus_view *view) {
    rg_multi_model *model = 0;
    rg_train_options options;
    rg_train_options_init_defaults(&options);
    assert(rg_train_model(ctx, view->sets, view->set_count, &options, &model) == RG_OK);
    return model;
}

static int compare_strings(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static const char *renamed_lect(
    const char *lect,
    const char *const *from,
    const char *const *to,
    size_t name_count
) {
    size_t i;
    for (i = 0; i < name_count; i++) {
        if (strcmp(lect, from[i]) == 0) {
            return to[i];
        }
    }
    return lect;
}

/* Every class as one string, sorted, with the probe's renaming undone. Class
 * ids and table order are permitted to differ; which lects carry which
 * graphemes is not. */
static char **class_keys(
    const rg_multi_model *model,
    const char *const *from,
    const char *const *to,
    size_t name_count,
    size_t *out_count
) {
    const rg_multi_class_row *rows;
    size_t count = 0;
    size_t i;
    char **keys;
    rows = rg_multi_model_unconditioned_classes(model, &count);
    keys = (char **)calloc(count == 0 ? 1 : count, sizeof(*keys));
    assert(keys != 0);
    for (i = 0; i < count; i++) {
        char buffer[1024];
        char parts[16][64];
        char *ordered[16];
        size_t used = 0;
        size_t s;
        assert(rows[i].segment_count <= 16);
        /* Sorted by the mapped lect id: the model holds lects in ascending id
         * order, so a renaming that reverses that order also reverses the
         * order the segments appear in, which is not a different class. */
        for (s = 0; s < rows[i].segment_count; s++) {
            int written = snprintf(parts[s], sizeof(parts[s]), "%s=%s;",
                                   renamed_lect(rows[i].lect_ids[s], from, to, name_count),
                                   rows[i].graphemes[s]);
            assert(written > 0 && (size_t)written < sizeof(parts[s]));
            ordered[s] = parts[s];
        }
        qsort(ordered, rows[i].segment_count, sizeof(*ordered), compare_strings);
        buffer[0] = '\0';
        for (s = 0; s < rows[i].segment_count; s++) {
            int written = snprintf(buffer + used, sizeof(buffer) - used, "%s", ordered[s]);
            assert(written > 0 && (size_t)written < sizeof(buffer) - used);
            used += (size_t)written;
        }
        keys[i] = (char *)malloc(used + 1);
        assert(keys[i] != 0);
        memcpy(keys[i], buffer, used + 1);
    }
    qsort(keys, count, sizeof(*keys), compare_strings);
    *out_count = count;
    return keys;
}

static void assert_same_classes(
    const rg_multi_model *left,
    const rg_multi_model *right,
    const char *const *from,
    const char *const *to,
    size_t name_count
) {
    size_t left_count = 0;
    size_t right_count = 0;
    char **left_keys = class_keys(left, 0, 0, 0, &left_count);
    char **right_keys = class_keys(right, from, to, name_count, &right_count);
    size_t i;
    assert(left_count == right_count);
    for (i = 0; i < left_count; i++) {
        assert(strcmp(left_keys[i], right_keys[i]) == 0);
        free(left_keys[i]);
        free(right_keys[i]);
    }
    free(left_keys);
    free(right_keys);
}

#define M5_FIXTURE(name) REGULAE_SOURCE_DIR "/testdata/evaluation/m5/generated/" name ".tsv"

static const char *const THREE_LECTS[] = {"aa_one", "mm_two", "zz_three"};
/* Reversing the order the ids sort in is the probe: it is what decides which
 * lect is the source of each pair, and a quarter of the published classes once
 * moved when it changed. */
static const char *const THREE_REVERSED[] = {"zz_one", "mm_two", "aa_three"};

static void test_renaming_lects_keeps_every_class(rg_context *ctx) {
    corpus_view plain;
    corpus_view renamed;
    rg_multi_model *left;
    rg_multi_model *right;
    corpus_view_open(&plain, M5_FIXTURE("multilect_three"), THREE_LECTS, THREE_LECTS, 3);
    corpus_view_open(&renamed, M5_FIXTURE("multilect_three"), THREE_LECTS, THREE_REVERSED, 3);
    left = train_view(ctx, &plain);
    right = train_view(ctx, &renamed);
    /* The renaming is a bijection, so undoing it on one side is enough. */
    assert_same_classes(left, right, THREE_REVERSED, THREE_LECTS, 3);
    rg_multi_model_free(left);
    rg_multi_model_free(right);
    corpus_view_close(&plain);
    corpus_view_close(&renamed);
}

static void test_row_order_is_not_evidence(rg_context *ctx) {
    corpus_view plain;
    corpus_view reversed;
    rg_multi_model *left;
    rg_multi_model *right;
    corpus_view_open(&plain, M5_FIXTURE("multilect_three"), THREE_LECTS, THREE_LECTS, 3);
    corpus_view_open(&reversed, M5_FIXTURE("multilect_three"), THREE_LECTS, THREE_LECTS, 3);
    corpus_view_reverse_rows(&reversed);
    left = train_view(ctx, &plain);
    right = train_view(ctx, &reversed);
    assert_same_classes(left, right, 0, 0, 0);
    rg_multi_model_free(left);
    rg_multi_model_free(right);
    corpus_view_close(&plain);
    corpus_view_close(&reversed);
}

/* The association is true of the pair, not of the lect that sorts first.
 *
 * Cross-dimensional discovery searched one orientation only until M5, so on
 * these two corpora -- the same relationship with the conditioning lect sorting
 * first and second -- one published the rule and the other published nothing.
 * The tone bearer merged the voicing contrast, so its own form predicts nothing
 * and the environment can only be stated over the other lect. */
static void test_cross_dimensional_association_survives_lect_order(rg_context *ctx) {
    const char *const corpora[2] = {
        M5_FIXTURE("xdim_conditioner_first"),
        M5_FIXTURE("xdim_conditioner_second")
    };
    const char *const conditioners[2] = {"aa_conditioner", "zz_conditioner"};
    size_t which;
    for (which = 0; which < 2; which++) {
        corpus_view view;
        rg_multi_model *model;
        const rg_multi_cross_dimensional_row *rows;
        size_t count = 0;
        size_t i;
        int raised_high = 0;
        int raised_low = 0;
        corpus_view_open(&view, corpora[which], conditioners, conditioners, 1);
        model = train_view(ctx, &view);
        rows = rg_multi_model_cross_dimensional_rows(model, &count);
        for (i = 0; i < count; i++) {
            const rg_cross_dimensional_row *rule = &rows[i].rule;
            const char *environment_lect =
                rule->context_is_target ? rows[i].target_lect : rows[i].source_lect;
            if (strcmp(environment_lect, conditioners[which]) != 0) {
                continue;
            }
            if (rule->environment.preceding_count != 1 ||
                strcmp(rule->environment.preceding[0].feature, "voiced") != 0) {
                continue;
            }
            assert(strcmp(rule->dimension, "tone") == 0);
            if (strcmp(rule->environment.preceding[0].value, "+") == 0 &&
                strcmp(rule->value, "4") == 0) {
                raised_high = 1;
            }
            if (strcmp(rule->environment.preceding[0].value, "-") == 0 &&
                strcmp(rule->value, "1") == 0) {
                raised_low = 1;
            }
        }
        assert(raised_high);
        assert(raised_low);
        rg_multi_model_free(model);
        corpus_view_close(&view);
    }
}

/* A lect with no environment to state and no dimension to condition publishes
 * nothing, in either orientation. Without this the probe above would pass on a
 * search that simply commits everything. */
static void test_absent_association_is_not_invented(rg_context *ctx) {
    corpus_view view;
    rg_multi_model *model;
    size_t count = 0;
    const char *const names[] = {"aa_conditioner"};
    corpus_view_open(&view, M5_FIXTURE("xdim_neither_orientation"), names, names, 1);
    model = train_view(ctx, &view);
    rg_multi_model_cross_dimensional_rows(model, &count);
    assert(count == 0);
    rg_multi_model_free(model);
    corpus_view_close(&view);
}

/* Both lects carry the dimension and each one's value goes with the other's, so
 * the association holds whichever lect the environment is stated over. It is
 * published once per lect: which lect carries the environment is part of the
 * claim, and merging the two would erase it. */
static void test_symmetric_association_is_stated_over_both_lects(rg_context *ctx) {
    corpus_view view;
    rg_multi_model *model;
    const rg_multi_cross_dimensional_row *rows;
    size_t count = 0;
    size_t i;
    int over_first = 0;
    int over_second = 0;
    const char *const names[] = {"aa_tone"};
    corpus_view_open(&view, M5_FIXTURE("xdim_both_orientations"), names, names, 1);
    model = train_view(ctx, &view);
    rows = rg_multi_model_cross_dimensional_rows(model, &count);
    assert(count > 0);
    for (i = 0; i < count; i++) {
        if (rows[i].rule.context_is_target) {
            over_second = 1;
        } else {
            over_first = 1;
        }
    }
    assert(over_first && over_second);
    rg_multi_model_free(model);
    corpus_view_close(&view);
}

/* A duplicated lect is counted as one, and the pairs it duplicates keep their
 * conditioned correspondences. Six pairs of four lects are six correlated views
 * of one corpus, and the fit has to say so. */
static void test_duplicate_lect_adds_pairs_and_no_evidence(rg_context *ctx) {
    corpus_view plain;
    corpus_view twinned;
    rg_multi_model *left;
    rg_multi_model *right;
    const rg_corpus_fit *before;
    const rg_corpus_fit *after;
    size_t i;
    size_t left_rows = 0;
    size_t right_rows = 0;

    corpus_view_open(&plain, M5_FIXTURE("segmental_conditioned"), THREE_LECTS, THREE_LECTS, 0);
    corpus_view_open(&twinned, M5_FIXTURE("segmental_conditioned"), THREE_LECTS, THREE_LECTS, 0);
    /* Every form of the first lect again, under a name of its own. */
    {
        rg_cognate_form *extended = (rg_cognate_form *)calloc(
            twinned.set_count * 3, sizeof(*extended));
        size_t next = 0;
        assert(extended != 0);
        for (i = 0; i < twinned.set_count; i++) {
            size_t f;
            size_t start = next;
            for (f = 0; f < twinned.sets[i].form_count; f++) {
                extended[next++] = twinned.sets[i].forms[f];
            }
            extended[next] = twinned.sets[i].forms[0];
            extended[next].lect_id = "aa_old_twin";
            next++;
            twinned.sets[i].forms = &extended[start];
            twinned.sets[i].form_count += 1;
        }
        free(twinned.forms);
        twinned.forms = extended;
    }

    left = train_view(ctx, &plain);
    right = train_view(ctx, &twinned);
    before = rg_multi_model_fit(left);
    after = rg_multi_model_fit(right);
    assert(before->lect_count == 2 && before->pair_count == 1);
    assert(before->duplicate_lect_count == 0);
    assert(after->lect_count == 3 && after->pair_count == 3);
    assert(after->duplicate_lect_count == 1);
    assert(after->missing_form_count == 0);

    /* The pair that was there before is unchanged. */
    for (i = 0; i < rg_multi_model_pair_model_count(left); i++) {
        const rg_multi_pair_model_row *pair = rg_multi_model_pair_model_at(left, i);
        rg_pairwise_model_conditioned_segment_counts(pair->model, &left_rows);
    }
    for (i = 0; i < rg_multi_model_pair_model_count(right); i++) {
        const rg_multi_pair_model_row *pair = rg_multi_model_pair_model_at(right, i);
        size_t rows = 0;
        if (strcmp(pair->lect_a, "aa_old") != 0 || strcmp(pair->lect_b, "zz_new") != 0) {
            continue;
        }
        rg_pairwise_model_conditioned_segment_counts(pair->model, &rows);
        right_rows = rows;
    }
    assert(left_rows == right_rows);

    rg_multi_model_free(left);
    rg_multi_model_free(right);
    corpus_view_close(&plain);
    corpus_view_close(&twinned);
}

int main(void) {
    rg_context *ctx = 0;
    assert(rg_context_new_builtin(&ctx) == RG_OK);
    test_renaming_lects_keeps_every_class(ctx);
    test_row_order_is_not_evidence(ctx);
    test_cross_dimensional_association_survives_lect_order(ctx);
    test_absent_association_is_not_invented(ctx);
    test_symmetric_association_is_stated_over_both_lects(ctx);
    test_duplicate_lect_adds_pairs_and_no_evidence(ctx);
    rg_context_free(ctx);
    printf("test_evaluation_m5: ok\n");
    return 0;
}
