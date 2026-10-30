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

    rg_ensemble_free(a);
    rg_context_free(ctx);
    printf("ensemble: ok\n");
    return 0;
}
