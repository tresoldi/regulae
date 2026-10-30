#include "loader_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A soft-cognacy corpus (docs/ensembles.md): one row per form, and one row per
 * pairwise cognacy weight. Forms are read with the TSV loader's own cell
 * parsers, so segments, tone, stress and breaks mean what they mean there.
 *
 *   forms:  form_id  lect_id  meaning  segments  [tone] [stress] [breaks]
 *   pairs:  form_a   form_b   weight  [morph_a  morph_b]
 *
 * With morph_a and morph_b (0-based morpheme indices over each form's breaks,
 * as cognator writes them), the pairs are between morphemes and the corpus's
 * nodes are morpheme spans `form_id#k`, each pointing at its whole word for
 * environments (rg_form.word). A form without breaks is one morpheme, #0. */

struct rg_soft_corpus {
    rg_soft_form *forms;
    loader_form *owned;
    char **ids;
    char **meanings;
    size_t count;
    rg_cognacy_edge *edges;
    size_t edge_count;
    /* Morpheme mode: the whole words the spans point into (stable storage),
     * and the node ids' owned strings. */
    rg_form *words;
    size_t word_count;
    char **node_ids;
    size_t node_count;
    rg_soft_form *nodes;
};

void rg_soft_corpus_free(rg_soft_corpus *corpus) {
    size_t i;
    if (corpus == 0) {
        return;
    }
    for (i = 0; i < corpus->count; i++) {
        loader_form_clear(&corpus->owned[i]);
        free(corpus->ids[i]);
        free(corpus->meanings[i]);
    }
    free(corpus->forms);
    free(corpus->owned);
    free(corpus->ids);
    free(corpus->meanings);
    free(corpus->edges);
    for (i = 0; i < corpus->node_count; i++) {
        free(corpus->node_ids[i]);
    }
    free(corpus->node_ids);
    free(corpus->nodes);
    free(corpus->words);
    free(corpus);
}

/* The morphemes of a word: [0, b1), [b1, b2), ... over its sorted breaks. */
static size_t morpheme_bounds(const rg_form *form, size_t *bounds, size_t cap) {
    size_t count = 0;
    size_t i;
    size_t j;
    bounds[count++] = 0;
    for (i = 0; i < form->morpheme_break_count && count + 1 < cap; i++) {
        int at = form->morpheme_breaks[i];
        size_t pos;
        if (at <= 0 || (size_t)at >= form->segment_count) {
            continue;
        }
        pos = (size_t)at;
        for (j = 1; j < count && bounds[j] != pos; j++) {
        }
        if (j == count) {
            bounds[count++] = pos;
        }
    }
    for (i = 1; i < count; i++) {
        for (j = i; j > 1 && bounds[j - 1] > bounds[j]; j--) {
            size_t t = bounds[j];
            bounds[j] = bounds[j - 1];
            bounds[j - 1] = t;
        }
    }
    bounds[count] = form->segment_count;
    return count;
}

/* Expands every form into its morpheme nodes. `first[f]` receives the index
 * of form f's first node and `spans[f]` its morpheme count. */
static rg_status expand_morphemes(rg_soft_corpus *corpus, size_t *first, size_t *spans) {
    size_t total = 0;
    size_t f;
    size_t bounds[66];
    corpus->words = (rg_form *)calloc(corpus->count + 1, sizeof(*corpus->words));
    if (corpus->words == 0) {
        return RG_ERR_OOM;
    }
    for (f = 0; f < corpus->count; f++) {
        corpus->words[f] = corpus->forms[f].form;
        spans[f] = morpheme_bounds(&corpus->words[f], bounds, 65);
        first[f] = total;
        total += spans[f];
    }
    corpus->word_count = corpus->count;
    corpus->nodes = (rg_soft_form *)calloc(total + 1, sizeof(*corpus->nodes));
    corpus->node_ids = (char **)calloc(total + 1, sizeof(*corpus->node_ids));
    if (corpus->nodes == 0 || corpus->node_ids == 0) {
        return RG_ERR_OOM;
    }
    for (f = 0; f < corpus->count; f++) {
        size_t k;
        const rg_form *word = &corpus->words[f];
        morpheme_bounds(word, bounds, 65);
        for (k = 0; k < spans[f]; k++) {
            rg_soft_form *node = &corpus->nodes[first[f] + k];
            size_t len = strlen(corpus->ids[f]) + 24;
            char *id = (char *)malloc(len);
            if (id == 0) {
                return RG_ERR_OOM;
            }
            snprintf(id, len, "%s#%lu", corpus->ids[f], (unsigned long)k);
            corpus->node_ids[first[f] + k] = id;
            corpus->node_count = first[f] + k + 1;
            node->form_id = id;
            node->lect_id = corpus->forms[f].lect_id;
            node->meaning = corpus->forms[f].meaning;
            node->form.lect_id = word->lect_id;
            node->form.segments = word->segments + bounds[k];
            node->form.segment_count = bounds[k + 1] - bounds[k];
            if (spans[f] > 1) {
                node->form.word = word;
                node->form.word_offset = bounds[k];
            } else {
                node->form = *word;
            }
        }
    }
    return RG_OK;
}

static long find_form(const rg_soft_corpus *corpus, const char *id) {
    size_t i;
    for (i = 0; i < corpus->count; i++) {
        if (strcmp(corpus->ids[i], id) == 0) {
            return (long)i;
        }
    }
    return -1;
}

rg_status rg_soft_corpus_load(
    const char *forms_path,
    const char *forms_text,
    const char *pairs_path,
    const char *pairs_text,
    rg_soft_corpus **out,
    rg_load_diagnosis *diagnosis
) {
    loader_table forms;
    loader_table pairs;
    rg_soft_corpus *corpus;
    long id_col, lect_col, meaning_col, seg_col, tone_col, stress_col, breaks_col;
    long a_col, b_col, w_col, ma_col, mb_col;
    size_t *first = 0;
    size_t *spans = 0;
    size_t r;
    rg_status status;

    if (out == 0 || (forms_path == 0 && forms_text == 0) || (pairs_path == 0 && pairs_text == 0)) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    loader_clear_diagnosis(diagnosis);
    memset(&forms, 0, sizeof(forms));
    memset(&pairs, 0, sizeof(pairs));
    status = read_table_source(forms_path, forms_text, '\t', &forms);
    if (status != RG_OK) {
        return status;
    }
    status = read_table_source(pairs_path, pairs_text, '\t', &pairs);
    if (status != RG_OK) {
        loader_table_clear(&forms);
        return status;
    }
    id_col = column_index(&forms, "form_id");
    lect_col = column_index(&forms, "lect_id");
    meaning_col = column_index(&forms, "meaning");
    seg_col = column_index(&forms, "segments");
    tone_col = column_index(&forms, "tone");
    stress_col = column_index(&forms, "stress");
    breaks_col = column_index(&forms, "breaks");
    a_col = column_index(&pairs, "form_a");
    b_col = column_index(&pairs, "form_b");
    w_col = column_index(&pairs, "weight");
    ma_col = column_index(&pairs, "morph_a");
    mb_col = column_index(&pairs, "morph_b");
    if ((ma_col < 0) != (mb_col < 0)) {
        loader_fail(diagnosis, 1, "pairs need both morph_a and morph_b, or neither");
        loader_table_clear(&forms);
        loader_table_clear(&pairs);
        return RG_ERR_PARSE;
    }
    if (id_col < 0 || lect_col < 0 || meaning_col < 0 || seg_col < 0) {
        loader_fail(diagnosis, 1, "forms need form_id, lect_id, meaning and segments columns");
        loader_table_clear(&forms);
        loader_table_clear(&pairs);
        return RG_ERR_PARSE;
    }
    if (a_col < 0 || b_col < 0 || w_col < 0) {
        loader_fail(diagnosis, 1, "pairs need form_a, form_b and weight columns");
        loader_table_clear(&forms);
        loader_table_clear(&pairs);
        return RG_ERR_PARSE;
    }
    corpus = (rg_soft_corpus *)calloc(1, sizeof(*corpus));
    if (corpus != 0) {
        corpus->forms = (rg_soft_form *)calloc(forms.row_count + 1, sizeof(*corpus->forms));
        corpus->owned = (loader_form *)calloc(forms.row_count + 1, sizeof(*corpus->owned));
        corpus->ids = (char **)calloc(forms.row_count + 1, sizeof(*corpus->ids));
        corpus->meanings = (char **)calloc(forms.row_count + 1, sizeof(*corpus->meanings));
        corpus->edges = (rg_cognacy_edge *)calloc(pairs.row_count + 1, sizeof(*corpus->edges));
    }
    if (corpus == 0 || corpus->forms == 0 || corpus->owned == 0 || corpus->ids == 0 ||
        corpus->meanings == 0 || corpus->edges == 0) {
        rg_soft_corpus_free(corpus);
        loader_table_clear(&forms);
        loader_table_clear(&pairs);
        return RG_ERR_OOM;
    }
    status = RG_OK;
    for (r = 0; r < forms.row_count && status == RG_OK; r++) {
        const loader_row *row = &forms.rows[r];
        loader_form *form = &corpus->owned[corpus->count];
        char *id = trim_copy(cell(row, id_col));
        char *lect = trim_copy(cell(row, lect_col));
        char *meaning = trim_copy(cell(row, meaning_col));
        if (id == 0 || lect == 0 || meaning == 0) {
            free(id);
            free(lect);
            free(meaning);
            status = RG_ERR_OOM;
            break;
        }
        if (id[0] == '\0' || lect[0] == '\0') {
            free(id);
            free(lect);
            free(meaning);
            continue;
        }
        if (find_form(corpus, id) >= 0) {
            loader_fail(diagnosis, r + 2, "duplicate form_id");
            free(id);
            free(lect);
            free(meaning);
            status = RG_ERR_PARSE;
            break;
        }
        form->lect_id = lect;
        status = parse_segments(cell(row, seg_col), 0, &form->segments, &form->segment_count, 0, 0);
        if (status == RG_OK && tone_col >= 0) {
            status = attach_dimension(cell(row, tone_col), form->segments, form->segment_count, RG_DIMENSION_TONE);
        }
        if (status == RG_OK && stress_col >= 0) {
            status = attach_dimension(cell(row, stress_col), form->segments, form->segment_count, RG_DIMENSION_STRESS);
        }
        if (status == RG_OK && breaks_col >= 0) {
            status = parse_break_indices(cell(row, breaks_col), &form->morpheme_breaks, &form->morpheme_break_count);
        }
        if (status != RG_OK || form->segment_count == 0) {
            loader_form_clear(form);
            free(id);
            free(meaning);
            if (status != RG_OK) {
                loader_fail(diagnosis, r + 2, "unreadable form");
            }
            continue;
        }
        corpus->ids[corpus->count] = id;
        corpus->meanings[corpus->count] = meaning;
        {
            rg_soft_form *soft = &corpus->forms[corpus->count];
            soft->form_id = id;
            soft->lect_id = form->lect_id;
            soft->meaning = meaning;
            soft->form.lect_id = form->lect_id;
            soft->form.segments = form->segments;
            soft->form.segment_count = form->segment_count;
            soft->form.morpheme_breaks = form->morpheme_breaks;
            soft->form.morpheme_break_count = form->morpheme_break_count;
        }
        corpus->count++;
    }
    if (status == RG_OK && ma_col >= 0) {
        first = (size_t *)calloc(corpus->count + 1, sizeof(*first));
        spans = (size_t *)calloc(corpus->count + 1, sizeof(*spans));
        status = first == 0 || spans == 0 ? RG_ERR_OOM : expand_morphemes(corpus, first, spans);
    }
    for (r = 0; r < pairs.row_count && status == RG_OK; r++) {
        const loader_row *row = &pairs.rows[r];
        const char *a = cell(row, a_col);
        const char *b = cell(row, b_col);
        const char *w = cell(row, w_col);
        char *end = 0;
        char *ta = trim_copy(a);
        char *tb = trim_copy(b);
        long ia;
        long ib;
        double weight;
        if (ta == 0 || tb == 0) {
            free(ta);
            free(tb);
            status = RG_ERR_OOM;
            break;
        }
        ia = find_form(corpus, ta);
        ib = find_form(corpus, tb);
        free(ta);
        free(tb);
        weight = w != 0 ? strtod(w, &end) : NAN;
        if (ia < 0 || ib < 0 || end == w || !isfinite(weight) || weight < 0.0 || weight > 1.0) {
            loader_fail(diagnosis, r + 2, ia < 0 || ib < 0 ? "pair names an unknown form"
                                                          : "weight must be in [0, 1]");
            status = RG_ERR_PARSE;
            break;
        }
        if (ma_col >= 0) {
            /* Morpheme pairs: move both ends to their morpheme nodes. */
            char *ea = 0;
            char *eb = 0;
            const char *ra = cell(row, ma_col);
            const char *rb = cell(row, mb_col);
            long ka = ra != 0 ? strtol(ra, &ea, 10) : -1;
            long kb = rb != 0 ? strtol(rb, &eb, 10) : -1;
            if (ea == ra || eb == rb || ka < 0 || kb < 0 ||
                (size_t)ka >= spans[ia] || (size_t)kb >= spans[ib]) {
                loader_fail(diagnosis, r + 2, "morph_a/morph_b must index a morpheme of the form's breaks");
                status = RG_ERR_PARSE;
                break;
            }
            ia = (long)(first[ia] + (size_t)ka);
            ib = (long)(first[ib] + (size_t)kb);
        }
        if (ia == ib) {
            continue;
        }
        corpus->edges[corpus->edge_count].a = (size_t)ia;
        corpus->edges[corpus->edge_count].b = (size_t)ib;
        corpus->edges[corpus->edge_count].weight = weight;
        corpus->edge_count++;
    }
    loader_table_clear(&forms);
    loader_table_clear(&pairs);
    free(first);
    free(spans);
    if (status != RG_OK) {
        rg_soft_corpus_free(corpus);
        return status;
    }
    *out = corpus;
    return RG_OK;
}

const rg_soft_form *rg_soft_corpus_forms(const rg_soft_corpus *corpus, size_t *count) {
    if (count != 0) {
        *count = corpus == 0 ? 0 : (corpus->nodes != 0 ? corpus->node_count : corpus->count);
    }
    if (corpus == 0) {
        return 0;
    }
    return corpus->nodes != 0 ? corpus->nodes : corpus->forms;
}

const rg_cognacy_edge *rg_soft_corpus_edges(const rg_soft_corpus *corpus, size_t *count) {
    if (count != 0) {
        *count = corpus == 0 ? 0 : corpus->edge_count;
    }
    return corpus == 0 ? 0 : corpus->edges;
}
