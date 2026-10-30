/* Lost-trigger traces (docs/ensembles.md): the law of palatals in miniature.
 *
 * Greek and Latin keep *e; Sanskrit palatalises *k before *e and *i and then
 * merges *e (with *o and *a) into back ɑ. The conditioned class k ~ k ~ c
 * before a front vowel, read on a Greek pivot, must find the trigger at every
 * Latin position and at only the *i half of the Sanskrit ones. */
#include "regulae.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SEG 8
#define PER_VOWEL 6
#define SETS (5 * PER_VOWEL)
#define LECTS 3

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

static const char *sanskrit_vowel(char v) {
    return v == 'i' ? "i" : (v == 'u' ? "u" : "ɑ");
}

int main(void) {
    static const char *lects[LECTS] = {"greek", "latin", "sanskrit"};
    static const char *onsets[] = {"p", "t", "m", "n", "s", "l"};
    static const char *codas[] = {"a", "o", "u"};
    const char *vowels = "eioau";
    word words[SETS][LECTS];
    rg_cognate_form forms[SETS][LECTS];
    rg_cognate_set sets[SETS];
    char ids[SETS][8];
    rg_context *ctx = 0;
    rg_train_options options;
    rg_multi_model *model = 0;
    const rg_lost_trigger_row *rows;
    size_t row_count = 0;
    size_t i;
    int found_sanskrit = 0;
    int found_latin = 0;
    char *json;

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    for (i = 0; i < SETS; i++) {
        char v = vowels[i / PER_VOWEL];
        const char *onset = onsets[i % 6];
        const char *coda = codas[(i / 2) % 3];
        char kept[64];
        char merged[64];
        size_t l;
        snprintf(kept, sizeof(kept), "%s a k %c t %s", onset, v, coda);
        snprintf(merged, sizeof(merged), "%s a %s %s t %s", onset,
                 (v == 'e' || v == 'i') ? "c" : "k", sanskrit_vowel(v), coda);
        for (l = 0; l < LECTS; l++) {
            words[i][l] = make(l < 2 ? kept : merged);
            memset(&forms[i][l], 0, sizeof(forms[i][l]));
            forms[i][l].lect_id = lects[l];
            forms[i][l].form.lect_id = lects[l];
            forms[i][l].form.segments = words[i][l].segs;
            forms[i][l].form.segment_count = words[i][l].n;
        }
        snprintf(ids[i], sizeof(ids[i]), "w%02zu", i);
        memset(&sets[i], 0, sizeof(sets[i]));
        sets[i].cognate_id = ids[i];
        sets[i].forms = forms[i];
        sets[i].form_count = LECTS;
        sets[i].confidence = 1.0;
    }
    rg_train_options_init_defaults(&options);
    assert(rg_train_model(ctx, sets, SETS, &options, &model) == RG_OK);

    rows = rg_multi_model_lost_triggers(model, &row_count);
    assert(row_count > 0 && rows != 0);
    for (i = 0; i < row_count; i++) {
        const rg_lost_trigger_row *row = &rows[i];
        const rg_context_spec *env = row->environment;
        int front = env->following_count == 1 && strcmp(env->following[0].feature, "front") == 0 &&
                    strcmp(env->following[0].value, "+") == 0;
        assert(row->holds >= 0.0 && row->holds <= row->tested + 1e-9);
        assert(row->tested_cognate_count <= SETS);
        if (!front || strcmp(row->pivot_lect, "greek") != 0) {
            continue;
        }
        if (strcmp(row->sister_lect, "sanskrit") == 0 && strcmp(row->sister_grapheme, "c") == 0) {
            /* ke and ki: the trigger survives in ci, not in cɑ. */
            assert(fabs(row->tested - 2.0 * PER_VOWEL) < 1e-9);
            assert(fabs(row->holds / row->tested - 0.5) < 1e-9);
            assert(row->tested_cognate_count == 2 * PER_VOWEL);
            found_sanskrit = 1;
        }
        if (strcmp(row->sister_lect, "latin") == 0) {
            assert(fabs(row->holds - row->tested) < 1e-9);
            found_latin = 1;
        }
    }
    assert(found_sanskrit && found_latin);

    json = rg_model_to_json(ctx, model, sets, SETS, &options, false, false);
    assert(json != 0);
    assert(strstr(json, "\"lost_triggers\"") != 0);
    assert(strstr(json, "\"holds_rate\"") != 0);
    rg_string_free(json);

    /* The ensemble matches traces across members and summarises them. */
    {
        rg_ensemble_options eopts;
        rg_ensemble *ensemble = 0;
        rg_ensemble_options_init_defaults(&eopts);
        eopts.member_count = 3;
        eopts.alignment_temperature = 0.0;
        assert(rg_train_ensemble(ctx, sets, SETS, &options, &eopts, &ensemble) == RG_OK);
        json = rg_ensemble_to_json(ensemble, 0);
        assert(json != 0);
        assert(strstr(json, "\"lost_triggers\"") != 0);
        assert(strstr(json, "\"sister_lect\":\"sanskrit\"") != 0);
        rg_string_free(json);
        rg_ensemble_free(ensemble);
    }

    rg_multi_model_free(model);
    rg_context_free(ctx);
    printf("lost_triggers: ok\n");
    return 0;
}
