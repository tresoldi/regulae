#include "loader_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* A soft-cognacy corpus (docs/ensembles.md): one row per form, and one row per
 * pairwise cognacy weight. Forms are read with the TSV loader's own cell
 * parsers, so segments, tone, stress and breaks mean what they mean there.
 *
 *   forms:  form_id  lect_id  meaning  segments  [tone] [stress] [breaks]
 *   pairs:  form_a   form_b   weight
 */

struct rg_soft_corpus {
    rg_soft_form *forms;
    loader_form *owned;
    char **ids;
    char **meanings;
    size_t count;
    rg_cognacy_edge *edges;
    size_t edge_count;
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
    free(corpus);
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
    long a_col, b_col, w_col;
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
    if (status != RG_OK) {
        rg_soft_corpus_free(corpus);
        return status;
    }
    *out = corpus;
    return RG_OK;
}

const rg_soft_form *rg_soft_corpus_forms(const rg_soft_corpus *corpus, size_t *count) {
    if (count != 0) {
        *count = corpus == 0 ? 0 : corpus->count;
    }
    return corpus == 0 ? 0 : corpus->forms;
}

const rg_cognacy_edge *rg_soft_corpus_edges(const rg_soft_corpus *corpus, size_t *count) {
    if (count != 0) {
        *count = corpus == 0 ? 0 : corpus->edge_count;
    }
    return corpus == 0 ? 0 : corpus->edges;
}
