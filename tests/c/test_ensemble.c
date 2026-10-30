/* Ensembles (docs/ensembles.md): members, weights, determinism, refusals and
 * the regulae_package export. */
#include "regulae.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SEG 8
#define SETS 16

typedef struct word {
    rg_segment segs[MAX_SEG];
    size_t n;
} word;

static char *dup(const char *s) {
    char *out = (char *)malloc(strlen(s) + 1);
    assert(out != 0);
    strcpy(out, s);
    return out;
}

static word make(const char *text) {
    word w;
    char *copy = dup(text);
    char *tok = strtok(copy, " ");
    memset(&w, 0, sizeof(w));
    while (tok != 0) {
        assert(w.n < MAX_SEG);
        w.segs[w.n++].grapheme = dup(tok);
        tok = strtok(0, " ");
    }
    free(copy);
    return w;
}

int main(void) {
    static const char *corpus[SETS][2] = {
        {"a k t a", "a t t a"}, {"o k t o", "o t t o"}, {"a k t u", "a t t u"},
        {"p a t a", "p a d a"}, {"k a p a", "k a b a"}, {"t a k a", "t a g a"},
        {"a p t a", "a t t a"}, {"m a k t o", "m a t t o"}, {"n o k t e", "n o t t e"},
        {"s a p o", "s a b o"}, {"l u p a", "l u b a"}, {"p o t e", "p o d e"},
        {"t o p a", "t o b a"}, {"k u t a", "k u d a"}, {"p i t u", "p i d u"},
        {"m a t a", "m a d a"},
    };
    word words[SETS][2];
    rg_cognate_form forms[SETS][2];
    rg_cognate_set sets[SETS];
    char ids[SETS][8];
    rg_context *ctx = 0;
    rg_train_options options;
    rg_ensemble_options eopts;
    rg_ensemble *a = 0;
    rg_ensemble *b = 0;
    char *ja;
    char *jb;
    size_t i;
    size_t m;

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    for (i = 0; i < SETS; i++) {
        size_t l;
        for (l = 0; l < 2; l++) {
            words[i][l] = make(corpus[i][l]);
            memset(&forms[i][l], 0, sizeof(forms[i][l]));
            forms[i][l].lect_id = l == 0 ? "old" : "new";
            forms[i][l].form.lect_id = forms[i][l].lect_id;
            forms[i][l].form.segments = words[i][l].segs;
            forms[i][l].form.segment_count = words[i][l].n;
        }
        snprintf(ids[i], sizeof(ids[i]), "c%02zu", i);
        memset(&sets[i], 0, sizeof(sets[i]));
        sets[i].cognate_id = ids[i];
        sets[i].forms = forms[i];
        sets[i].form_count = 2;
        sets[i].confidence = 1.0;
    }
    rg_train_options_init_defaults(&options);
    rg_ensemble_options_init_defaults(&eopts);
    eopts.member_count = 4;
    eopts.alignment_temperature = 0.25;

    assert(rg_train_ensemble(ctx, sets, SETS, &options, &eopts, &a) == RG_OK);
    assert(rg_ensemble_member_count(a) == 4);
    for (m = 0; m < 4; m++) {
        size_t count = 0;
        const double *w = rg_ensemble_member_set_weights(a, m, &count);
        double sum = 0.0;
        assert(count == SETS && w != 0);
        for (i = 0; i < count; i++) {
            assert(w[i] > 0.0);
            sum += w[i];
        }
        /* Mean 1: each set is its own observation group here. */
        assert(fabs(sum / (double)count - 1.0) < 1e-9);
        assert(rg_ensemble_member_at(a, m) != 0);
    }
    assert(rg_ensemble_member_at(a, 4) == 0);

    /* One seed, one ensemble: the export is byte-identical. */
    assert(rg_train_ensemble(ctx, sets, SETS, &options, &eopts, &b) == RG_OK);
    ja = rg_ensemble_to_json(a, 1);
    jb = rg_ensemble_to_json(b, 1);
    assert(ja != 0 && jb != 0);
    assert(strcmp(ja, jb) == 0);
    assert(strstr(ja, "\"export_kind\":\"regulae_package\"") != 0);
    assert(strstr(ja, "\"approx_hint\":\"C1\"") != 0);
    assert(strstr(ja, "\"partition\"") != 0);
    assert(strstr(ja, "\"conditioned_correspondences\"") != 0);
    rg_string_free(ja);
    rg_string_free(jb);
    rg_ensemble_free(b);
    b = 0;

    /* Another seed, other weights. */
    eopts.seed += 1;
    assert(rg_train_ensemble(ctx, sets, SETS, &options, &eopts, &b) == RG_OK);
    {
        size_t count = 0;
        const double *wa = rg_ensemble_member_set_weights(a, 0, &count);
        const double *wb = rg_ensemble_member_set_weights(b, 0, &count);
        assert(wa[0] != wb[0]);
    }
    rg_ensemble_free(b);
    b = 0;

    /* Refusals. */
    options.bic.split_scorer = RG_SPLIT_SCORER_MULTINOMIAL_NML;
    assert(rg_train_ensemble(ctx, sets, SETS, &options, &eopts, &b) == RG_ERR_UNSUPPORTED_OPTION);
    rg_train_options_init_defaults(&options);
    sets[0].confidence = 1.5;
    assert(rg_train_ensemble(ctx, sets, SETS, &options, &eopts, &b) == RG_ERR_INVALID_ARGUMENT);
    sets[0].confidence = 1.0;
    eopts.member_count = 0;
    assert(rg_train_ensemble(ctx, sets, SETS, &options, &eopts, &b) == RG_ERR_INVALID_ARGUMENT);
    /* And the public single-model path still refuses confidence above 1. */
    {
        rg_multi_model *model = 0;
        sets[0].confidence = 1.5;
        assert(rg_train_model(ctx, sets, SETS, &options, &model) == RG_ERR_INVALID_ARGUMENT);
        sets[0].confidence = 1.0;
    }

    /* Soft cognacy: every set's two forms at weight 0.9, a decoy third form
     * of the new lect at 0.3, and a cross-meaning edge at 0.05. */
    {
        rg_soft_form soft[SETS * 3];
        rg_cognacy_edge edges[SETS * 2 + 1];
        char form_ids[SETS * 3][12];
        char meanings[SETS][8];
        word decoys[SETS];
        size_t edge_count = 0;
        rg_ensemble *s1 = 0;
        rg_ensemble *s2 = 0;
        char *t1;
        char *t2;
        size_t distinct_counts = 0;
        size_t last = (size_t)-1;
        for (i = 0; i < SETS; i++) {
            size_t l;
            snprintf(meanings[i], sizeof(meanings[i]), "m%02zu", i);
            decoys[i] = make("s u n a");
            for (l = 0; l < 3; l++) {
                rg_soft_form *f = &soft[i * 3 + l];
                memset(f, 0, sizeof(*f));
                snprintf(form_ids[i * 3 + l], sizeof(form_ids[0]), "f%02zu_%zu", i, l);
                f->form_id = form_ids[i * 3 + l];
                f->lect_id = l == 0 ? "old" : "new";
                f->meaning = meanings[i];
                f->form.lect_id = f->lect_id;
                if (l < 2) {
                    f->form.segments = words[i][l].segs;
                    f->form.segment_count = words[i][l].n;
                } else {
                    f->form.segments = decoys[i].segs;
                    f->form.segment_count = decoys[i].n;
                }
            }
            edges[edge_count].a = i * 3;
            edges[edge_count].b = i * 3 + 1;
            edges[edge_count++].weight = 0.9;
            edges[edge_count].a = i * 3;
            edges[edge_count].b = i * 3 + 2;
            edges[edge_count++].weight = 0.3;
        }
        edges[edge_count].a = 0;
        edges[edge_count].b = 4;
        edges[edge_count++].weight = 0.05;

        rg_ensemble_options_init_defaults(&eopts);
        eopts.member_count = 6;
        eopts.alignment_temperature = 0.0;
        assert(rg_train_ensemble_soft(ctx, soft, SETS * 3, edges, edge_count, &options, &eopts, &s1) == RG_OK);
        assert(rg_ensemble_member_count(s1) == 6);
        for (m = 0; m < 6; m++) {
            size_t count = 0;
            rg_ensemble_member_set_weights(s1, m, &count);
            assert(count > 0 && count <= SETS);
            if (count != last) {
                distinct_counts++;
                last = count;
            }
        }
        /* Sampled partitions differ between members. */
        assert(distinct_counts >= 2);
        assert(rg_train_ensemble_soft(ctx, soft, SETS * 3, edges, edge_count, &options, &eopts, &s2) == RG_OK);
        t1 = rg_ensemble_to_json(s1, 1);
        t2 = rg_ensemble_to_json(s2, 1);
        assert(t1 != 0 && t2 != 0 && strcmp(t1, t2) == 0);
        assert(strstr(t1, "\"condition_on\":\"sampled_partitions\"") != 0);
        assert(strstr(t1, "\"cognate_sets\"") != 0);
        rg_string_free(t1);
        rg_string_free(t2);
        rg_ensemble_free(s1);
        rg_ensemble_free(s2);
        /* An edge weight outside [0, 1] is refused. */
        edges[0].weight = 1.2;
        assert(rg_train_ensemble_soft(ctx, soft, SETS * 3, edges, edge_count, &options, &eopts, &s1) == RG_ERR_INVALID_ARGUMENT);
    }

    rg_ensemble_free(a);
    rg_context_free(ctx);
    printf("ensemble: ok\n");
    return 0;
}
