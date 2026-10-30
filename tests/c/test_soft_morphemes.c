/* Soft cognacy between morphemes (docs/ensembles.md): a pairs table with
 * morph_a and morph_b, as cognator writes it, turns the corpus's nodes into
 * morpheme spans `form_id#k` that read their environments from the whole
 * word. */
#include "regulae.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    static const char *firsts[] = {"t a", "m a", "n a", "s a", "l a", "r a"};
    static const char *vowels[] = {"a", "o", "u", "e"};
    char forms_text[8192];
    char pairs_text[8192];
    size_t i;
    size_t node_count = 0;
    size_t edge_count = 0;
    const rg_soft_form *nodes;
    const rg_cognacy_edge *edges;
    rg_soft_corpus *corpus = 0;
    rg_load_diagnosis diagnosis;
    rg_context *ctx = 0;
    rg_train_options options;
    rg_ensemble_options eopts;
    rg_ensemble *ensemble = 0;
    char *json;
    int spans_seen = 0;

    strcpy(forms_text, "form_id\tlect_id\tmeaning\tsegments\tbreaks\n");
    strcpy(pairs_text, "form_a\tform_b\tweight\tmorph_a\tmorph_b\n");
    for (i = 0; i < 12; i++) {
        char row[256];
        const char *f = firsts[i % 6];
        const char *v = vowels[i % 4];
        const char *w = vowels[(i / 4) % 4];
        snprintf(row, sizeof(row), "a%02lu\ta\tm%02lu\t%s p %s t %s\t2\n", (unsigned long)i,
                 (unsigned long)i, f, v, w);
        strcat(forms_text, row);
        snprintf(row, sizeof(row), "b%02lu\tb\tm%02lu\t%s b %s t %s\t2\n", (unsigned long)i,
                 (unsigned long)i, f, v, w);
        strcat(forms_text, row);
        /* The second morphemes are cognate; the first ones less surely. */
        snprintf(row, sizeof(row), "a%02lu\tb%02lu\t0.9\t1\t1\na%02lu\tb%02lu\t0.6\t0\t0\n",
                 (unsigned long)i, (unsigned long)i, (unsigned long)i, (unsigned long)i);
        strcat(pairs_text, row);
    }
    /* A simplex word: no breaks, one node #0. */
    strcat(forms_text, "a99\ta\tm99\tp a t a\t\nb99\tb\tm99\tp a t a\t\n");
    strcat(pairs_text, "a99\tb99\t0.9\t0\t0\n");

    assert(rg_soft_corpus_load(0, forms_text, 0, pairs_text, &corpus, &diagnosis) == RG_OK);
    nodes = rg_soft_corpus_forms(corpus, &node_count);
    edges = rg_soft_corpus_edges(corpus, &edge_count);
    assert(node_count == 12 * 2 * 2 + 2);
    assert(edge_count == 12 * 2 + 1);
    for (i = 0; i < node_count; i++) {
        if (strcmp(nodes[i].form_id, "a03#1") == 0) {
            assert(nodes[i].form.word != 0);
            assert(nodes[i].form.word_offset == 2);
            assert(nodes[i].form.segment_count == 4);
            assert(strcmp(nodes[i].form.segments[0].grapheme, "p") == 0);
            spans_seen++;
        }
        if (strcmp(nodes[i].form_id, "a99#0") == 0) {
            assert(nodes[i].form.word == 0);
            assert(nodes[i].form.segment_count == 4);
            spans_seen++;
        }
    }
    assert(spans_seen == 2);
    for (i = 0; i < edge_count; i++) {
        const char *x = nodes[edges[i].a].form_id;
        const char *y = nodes[edges[i].b].form_id;
        /* Morpheme k of one form meets morpheme k of the other. */
        assert(strcmp(strchr(x, '#'), strchr(y, '#')) == 0);
    }

    assert(rg_context_new_builtin(&ctx) == RG_OK);
    rg_train_options_init_defaults(&options);
    rg_ensemble_options_init_defaults(&eopts);
    eopts.member_count = 3;
    eopts.alignment_temperature = 0.0;
    assert(rg_train_ensemble_soft(ctx, nodes, node_count, edges, edge_count, &options, &eopts,
                                  &ensemble) == RG_OK);
    json = rg_ensemble_to_json(ensemble, 0);
    assert(json != 0 && strstr(json, "#1") != 0);
    rg_string_free(json);
    rg_ensemble_free(ensemble);
    rg_soft_corpus_free(corpus);

    /* An index past the form's morphemes is refused, with the line. */
    corpus = 0;
    assert(rg_soft_corpus_load(0, "form_id\tlect_id\tmeaning\tsegments\tbreaks\nx\ta\tm\tt a p a\t2\n"
                                  "y\tb\tm\tt a b a\t2\n",
                               0, "form_a\tform_b\tweight\tmorph_a\tmorph_b\nx\ty\t0.5\t2\t0\n",
                               &corpus, &diagnosis) == RG_ERR_PARSE);
    assert(diagnosis.line == 2 && corpus == 0);
    /* One morpheme column without the other is refused. */
    assert(rg_soft_corpus_load(0, "form_id\tlect_id\tmeaning\tsegments\nx\ta\tm\tt a\n",
                               0, "form_a\tform_b\tweight\tmorph_a\nx\tx\t0.5\t0\n",
                               &corpus, &diagnosis) == RG_ERR_PARSE);

    rg_context_free(ctx);
    printf("soft_morphemes: ok\n");
    return 0;
}
