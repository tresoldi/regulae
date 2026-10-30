/* Partial cognacy (docs/ensembles.md): one cognate id per morpheme.
 *
 * testdata/corpora/partial_cognacy.tsv has compounds whose second morpheme
 * begins with p, voiced to b in lect b after the first morpheme's vowel, and
 * simplex words whose p is word-initial and stays p. The second morphemes form
 * their own sets. The voicing is only findable if a span's environment is read
 * from its whole word: alone, the span's p has no left neighbour and no
 * morpheme boundary, and cannot be told from the simplex p. */
#include "regulae.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const rg_cognate_set *find_set(const rg_corpus *corpus, const char *id) {
    size_t i;
    for (i = 0; i < rg_corpus_cognate_count(corpus); i++) {
        const rg_cognate_set *set = rg_corpus_cognate_at(corpus, i);
        if (strcmp(set->cognate_id, id) == 0) {
            return set;
        }
    }
    return 0;
}

int main(void) {
    rg_context *ctx = 0;
    rg_corpus *corpus = 0;
    rg_load_diagnosis diagnosis;
    rg_train_options options;
    rg_multi_model *model = 0;
    const rg_cognate_set *set;
    const rg_multi_class_row *rows;
    size_t row_count = 0;
    size_t i;
    int found = 0;

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/corpora/partial_cognacy.tsv", 0,
                              &corpus, &diagnosis) == RG_OK);
    /* 12 first morphemes, 12 second morphemes, 12 simplex words. */
    assert(rg_corpus_cognate_count(corpus) == 36);
    set = find_set(corpus, "S00");
    assert(set != 0 && set->form_count == 2);
    for (i = 0; i < set->form_count; i++) {
        const rg_form *form = &set->forms[i].form;
        assert(form->word != 0);
        assert(form->word_offset == 2);
        assert(form->segment_count == 4);
        assert(form->word->segment_count == 6);
        assert(strcmp(form->segments[0].grapheme, form->word->segments[2].grapheme) == 0);
    }
    set = find_set(corpus, "W00");
    assert(set != 0 && set->forms[0].form.word == 0);

    rg_train_options_init_defaults(&options);
    assert(rg_train_model(ctx, rg_corpus_cognates(corpus), rg_corpus_cognate_count(corpus),
                          &options, &model) == RG_OK);
    rows = rg_multi_model_conditioned_classes(model, &row_count);
    for (i = 0; i < row_count; i++) {
        const rg_multi_class_row *row = &rows[i];
        size_t s;
        int p_in_a = 0;
        int b_in_b = 0;
        int conditioned = 0;
        for (s = 0; s < row->segment_count; s++) {
            if (strcmp(row->lect_ids[s], "a") == 0 && strcmp(row->graphemes[s], "p") == 0) {
                p_in_a = 1;
                conditioned = rg_context_spec_constraint_count(&row->contexts[s]) > 0;
            }
            if (strcmp(row->lect_ids[s], "b") == 0 && strcmp(row->graphemes[s], "b") == 0) {
                b_in_b = 1;
            }
        }
        if (p_in_a && b_in_b && conditioned) {
            size_t r;
            int preceding_vowel = 0;
            const rg_context_spec *env = 0;
            for (s = 0; s < row->segment_count; s++) {
                if (rg_context_spec_constraint_count(&row->contexts[s]) > 0) {
                    env = &row->contexts[s];
                }
            }
            /* Whichever reading is committed, the boundary-crossing one is
             * either it or among its rivals. */
            for (r = 0; r < env->preceding_count; r++) {
                if (strcmp(env->preceding[r].feature, "vowel") == 0) {
                    preceding_vowel = 1;
                }
            }
            for (r = 0; r < row->environment_rival_count; r++) {
                const rg_environment_rival *rival = &row->environment_rivals[r];
                if (rival->slot != 0 && strcmp(rival->slot, "preceding") == 0 &&
                    strcmp(rival->feature, "vowel") == 0) {
                    preceding_vowel = 1;
                }
            }
            assert(preceding_vowel);
            found = 1;
        }
    }
    assert(found);
    rg_multi_model_free(model);

    /* A span that is not its word's segments is refused. */
    {
        rg_segment word_segments[3];
        rg_segment span_segments[1];
        rg_form word;
        rg_form span;
        rg_form other;
        rg_alignment *alignment = 0;
        memset(word_segments, 0, sizeof(word_segments));
        memset(span_segments, 0, sizeof(span_segments));
        word_segments[0].grapheme = "t";
        word_segments[1].grapheme = "a";
        word_segments[2].grapheme = "p";
        span_segments[0].grapheme = "k";
        memset(&word, 0, sizeof(word));
        word.lect_id = "a";
        word.segments = word_segments;
        word.segment_count = 3;
        span = word;
        span.segments = span_segments;
        span.segment_count = 1;
        span.word = &word;
        span.word_offset = 2;
        other = word;
        assert(rg_align_forms(ctx, &span, &other, 0, &alignment) == RG_ERR_INVALID_ARGUMENT);
        span_segments[0].grapheme = "p";
        assert(rg_align_forms(ctx, &span, &other, 0, &alignment) == RG_OK);
        rg_alignment_free(alignment);
        alignment = 0;
        span.word_offset = 3;
        assert(rg_align_forms(ctx, &span, &other, 0, &alignment) == RG_ERR_INVALID_ARGUMENT);
    }

    /* Two ids for a word with no breaks: refused, with the line. */
    {
        rg_corpus *bad = 0;
        const char *text =
            "cognate_id\tlect_id\tsegments\tbreaks\tmorpheme_cognates\n"
            "\ta\tt a p a\t\tX Y\n";
        assert(rg_corpus_parse_tsv(text, 0, &bad, &diagnosis) == RG_ERR_PARSE);
        assert(diagnosis.line == 2);
        assert(bad == 0);
    }

    rg_corpus_free(corpus);
    rg_context_free(ctx);
    printf("partial_cognacy: ok\n");
    return 0;
}
