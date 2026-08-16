#include "loader_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- accumulation ------------------------------------------------------- */

void loader_form_clear(loader_form *form) {
    size_t i;
    if (form == 0) {
        return;
    }
    free(form->lect_id);
    for (i = 0; i < form->segment_count; i++) {
        rg_free_owned_internal(form->segments[i].grapheme);
        rg_free_owned_internal(form->segments[i].tone);
        rg_free_owned_internal(form->segments[i].stress);
        rg_free_owned_internal(form->segments[i].length);
    }
    free(form->segments);
    free(form->morpheme_breaks);
    free(form->syllable_breaks);
    memset(form, 0, sizeof(*form));
}

void loader_cognate_clear(loader_cognate *cognate) {
    size_t i;
    if (cognate == 0) {
        return;
    }
    free(cognate->cognate_id);
    free(cognate->etymon_group);
    free(cognate->source_group);
    for (i = 0; i < cognate->form_count; i++) {
        loader_form_clear(&cognate->forms[i]);
    }
    free(cognate->forms);
    memset(cognate, 0, sizeof(*cognate));
}

/* Appends a new cognate set, disambiguating an id that is already taken. In
 * wide format each row is its own cognate set and the id column is a gloss
 * that may legitimately repeat: two rows glossed "die" are two cognate sets,
 * not one, and merging them would silently drop data. */
loader_cognate *corpus_append_cognate(rg_corpus *corpus, const char *base_id) {
    char candidate[512];
    size_t attempt = 1;
    size_t i;

    snprintf(candidate, sizeof(candidate), "%s", base_id);
    for (;;) {
        int taken = 0;
        for (i = 0; i < corpus->count; i++) {
            if (strcmp(corpus->cognates[i].cognate_id, candidate) == 0) {
                taken = 1;
                break;
            }
        }
        if (!taken) {
            break;
        }
        attempt++;
        snprintf(candidate, sizeof(candidate), "%s.%lu", base_id, (unsigned long)attempt);
    }
    if (corpus->count == corpus->cap) {
        size_t next_cap = corpus->cap == 0 ? 64 : corpus->cap * 2;
        loader_cognate *next = (loader_cognate *)realloc(corpus->cognates, next_cap * sizeof(*next));
        if (next == 0) {
            return 0;
        }
        corpus->cognates = next;
        corpus->cap = next_cap;
    }
    memset(&corpus->cognates[corpus->count], 0, sizeof(corpus->cognates[corpus->count]));
    corpus->cognates[corpus->count].cognate_id = rg_strdup_internal(candidate);
    if (corpus->cognates[corpus->count].cognate_id == 0) {
        return 0;
    }
    corpus->cognates[corpus->count].confidence = 1.0;
    corpus->cognates[corpus->count].alignment_length = -1;
    corpus->count++;
    return &corpus->cognates[corpus->count - 1];
}

loader_cognate *corpus_ensure_cognate(rg_corpus *corpus, const char *cognate_id) {
    size_t i;
    for (i = 0; i < corpus->count; i++) {
        if (strcmp(corpus->cognates[i].cognate_id, cognate_id) == 0) {
            return &corpus->cognates[i];
        }
    }
    if (corpus->count == corpus->cap) {
        size_t next_cap = corpus->cap == 0 ? 64 : corpus->cap * 2;
        loader_cognate *next = (loader_cognate *)realloc(corpus->cognates, next_cap * sizeof(*next));
        if (next == 0) {
            return 0;
        }
        corpus->cognates = next;
        corpus->cap = next_cap;
    }
    memset(&corpus->cognates[corpus->count], 0, sizeof(corpus->cognates[corpus->count]));
    corpus->cognates[corpus->count].cognate_id = rg_strdup_internal(cognate_id);
    if (corpus->cognates[corpus->count].cognate_id == 0) {
        return 0;
    }
    corpus->cognates[corpus->count].confidence = 1.0;
    corpus->cognates[corpus->count].alignment_length = -1;
    corpus->count++;
    return &corpus->cognates[corpus->count - 1];
}


rg_status cognate_append_form(loader_cognate *cognate, loader_form *form) {
    if (cognate->form_count == cognate->form_cap) {
        size_t next_cap = cognate->form_cap == 0 ? 4 : cognate->form_cap * 2;
        loader_form *next = (loader_form *)realloc(cognate->forms, next_cap * sizeof(*next));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        cognate->forms = next;
        cognate->form_cap = next_cap;
    }
    cognate->forms[cognate->form_count] = *form;
    cognate->form_count++;
    return RG_OK;
}

static rg_status set_group(char **owned, const char *value) {
    if (value == 0 || value[0] == '\0' || strcmp(value, "-") == 0) {
        return RG_OK;
    }
    if (*owned != 0) {
        return strcmp(*owned, value) == 0 ? RG_OK : RG_ERR_PARSE;
    }
    *owned = rg_strdup_internal(value);
    return *owned == 0 ? RG_ERR_OOM : RG_OK;
}

rg_status cognate_set_groups(loader_cognate *cognate, const char *etymon_group,
                             const char *source_group) {
    rg_status status;
    if (cognate == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    status = set_group(&cognate->etymon_group, etymon_group);
    if (status != RG_OK) {
        return status;
    }
    return set_group(&cognate->source_group, source_group);
}

int string_list_contains(const char *const *items, size_t count, const char *value) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (strcmp(items[i], value) == 0) {
            return 1;
        }
    }
    return 0;
}

/* Publishes borrowed rg_cognate_set views over the accumulated data, dropping
 * cognate sets with fewer than min_lects forms. */
/* How many ways one cognate may be expanded. A doublet in one lect gives two,
 * doublets in two lects give four; beyond this the corpus is saying something
 * the split cannot usefully represent, and the extras are dropped and counted
 * rather than multiplied out. */
#define RG_MAX_DOUBLET_EXPANSION 8

/* The distinct lects of a cognate, in first-seen order, with the forms each
 * one contributes. */
typedef struct lect_group {
    const char *lect_id;
    size_t form_indices[RG_MAX_DOUBLET_EXPANSION];
    size_t form_count;
} lect_group;

static size_t group_forms_by_lect(const loader_cognate *cognate, lect_group *groups, size_t cap) {
    size_t count = 0;
    size_t f;
    for (f = 0; f < cognate->form_count; f++) {
        size_t g;
        int placed = 0;
        for (g = 0; g < count; g++) {
            if (strcmp(groups[g].lect_id, cognate->forms[f].lect_id) == 0) {
                if (groups[g].form_count < RG_MAX_DOUBLET_EXPANSION) {
                    groups[g].form_indices[groups[g].form_count++] = f;
                }
                placed = 1;
                break;
            }
        }
        if (!placed && count < cap) {
            groups[count].lect_id = cognate->forms[f].lect_id;
            groups[count].form_indices[0] = f;
            groups[count].form_count = 1;
            count++;
        }
    }
    return count;
}

/* Publishes the loader's cognates as the corpus view, expanding doublets.
 *
 * A lect with two reflexes in one cognate set is a doublet, and it is a fact
 * about the language rather than an error in the file -- 3.2% of cognate-set
 * members across the Lexibank datasets with expert judgements, and near 10% in
 * some Austronesian ones. Keeping the first invents a correspondence by
 * picking one arbitrarily; refusing the set throws the evidence away. So the
 * set becomes one set per combination of reflexes, each carrying its share of
 * the original confidence, and both reflexes are counted once between them.
 *
 * Every loader publishes through here, which is the point: this was four
 * different undocumented behaviours -- hard error, keep-first, skip, and a
 * silent drop in the API -- for one situation. */
rg_status corpus_publish(rg_corpus *corpus, int min_lects) {
    size_t kept = 0;
    size_t i;

    for (i = 0; i < corpus->count; i++) {
        lect_group groups[64];
        size_t group_count;
        size_t combinations = 1;
        size_t g;
        if (corpus->cognates[i].form_count == 0) {
            continue;
        }
        group_count = group_forms_by_lect(&corpus->cognates[i], groups, 64);
        if (min_lects > 0 && group_count < (size_t)min_lects) {
            continue;
        }
        for (g = 0; g < group_count; g++) {
            combinations *= groups[g].form_count;
            if (combinations >= RG_MAX_DOUBLET_EXPANSION) {
                combinations = RG_MAX_DOUBLET_EXPANSION;
                break;
            }
        }
        kept += combinations;
    }
    corpus->view = (rg_cognate_set *)calloc(kept == 0 ? 1 : kept, sizeof(*corpus->view));
    corpus->view_forms = (rg_cognate_form **)calloc(kept == 0 ? 1 : kept, sizeof(*corpus->view_forms));
    if (corpus->view == 0 || corpus->view_forms == 0) {
        return RG_ERR_OOM;
    }
    corpus->view_form_group_count = kept;
    kept = 0;
    for (i = 0; i < corpus->count; i++) {
        loader_cognate *cognate = &corpus->cognates[i];
        lect_group groups[64];
        size_t group_count;
        size_t combinations = 1;
        size_t g;
        size_t combo;
        if (cognate->form_count == 0) {
            continue;
        }
        group_count = group_forms_by_lect(cognate, groups, 64);
        if (min_lects > 0 && group_count < (size_t)min_lects) {
            continue;
        }
        for (g = 0; g < group_count; g++) {
            combinations *= groups[g].form_count;
            if (combinations >= RG_MAX_DOUBLET_EXPANSION) {
                combinations = RG_MAX_DOUBLET_EXPANSION;
                break;
            }
        }
        if (combinations > 1) {
            corpus->doublet_set_count++;
            corpus->doublet_expansion_count += combinations - 1;
        }
        for (combo = 0; combo < combinations; combo++) {
            rg_cognate_form *forms = (rg_cognate_form *)calloc(group_count == 0 ? 1 : group_count, sizeof(*forms));
            size_t remainder = combo;
            if (forms == 0) {
                return RG_ERR_OOM;
            }
            corpus->view_forms[kept] = forms;
            for (g = 0; g < group_count; g++) {
                size_t pick = groups[g].form_count > 1 ? remainder % groups[g].form_count : 0;
                const loader_form *source = &cognate->forms[groups[g].form_indices[pick]];
                if (groups[g].form_count > 1) {
                    remainder /= groups[g].form_count;
                }
                forms[g].lect_id = source->lect_id;
                forms[g].form.lect_id = source->lect_id;
                forms[g].form.segments = source->segments;
                forms[g].form.segment_count = source->segment_count;
                forms[g].form.morpheme_breaks = source->morpheme_breaks;
                forms[g].form.morpheme_break_count = source->morpheme_break_count;
                forms[g].form.syllable_breaks = source->syllable_breaks;
                forms[g].form.syllable_break_count = source->syllable_break_count;
            }
            corpus->view[kept].cognate_id = cognate->cognate_id;
            corpus->view[kept].etymon_group = cognate->etymon_group;
            corpus->view[kept].source_group = cognate->source_group;
            corpus->view[kept].forms = forms;
            corpus->view[kept].form_count = group_count;
            /* Each reading carries its share, so a doublet is not two votes. */
            corpus->view[kept].confidence = cognate->confidence / (double)combinations;
            kept++;
        }
    }
    corpus->view_count = kept;
    return RG_OK;
}

size_t rg_corpus_doublet_set_count(const rg_corpus *corpus) {
    return corpus == 0 ? 0 : corpus->doublet_set_count;
}

size_t rg_corpus_doublet_expansion_count(const rg_corpus *corpus) {
    return corpus == 0 ? 0 : corpus->doublet_expansion_count;
}

/* The diagnosis belongs to the caller's struct, so two loads on two threads
 * cannot overwrite each other's and a successful one cannot be read as the
 * previous failure. Both were true of the process-wide buffer this replaced:
 * the load_* entry points cleared it and the parse_* ones did not.
 *
 * Null means the caller did not ask, which is not an error. */
void loader_clear_diagnosis(rg_load_diagnosis *diagnosis) {
    if (diagnosis == 0) {
        return;
    }
    diagnosis->line = 0;
    diagnosis->message[0] = '\0';
}

void loader_fail(rg_load_diagnosis *diagnosis, size_t line, const char *message) {
    size_t i = 0;
    if (diagnosis == 0) {
        return;
    }
    diagnosis->line = line;
    while (message[i] != '\0' && i + 1 < sizeof(diagnosis->message)) {
        diagnosis->message[i] = message[i];
        i++;
    }
    diagnosis->message[i] = '\0';
}

void rg_corpus_free(rg_corpus *corpus) {
    size_t i;
    if (corpus == 0) {
        return;
    }
    for (i = 0; i < corpus->count; i++) {
        loader_cognate_clear(&corpus->cognates[i]);
    }
    free(corpus->cognates);
    for (i = 0; i < corpus->view_form_group_count; i++) {
        free(corpus->view_forms[i]);
    }
    free(corpus->view_forms);
    free(corpus->view);
    free(corpus);
}

size_t rg_corpus_cognate_count(const rg_corpus *corpus) {
    return corpus == 0 ? 0 : corpus->view_count;
}

const rg_cognate_set *rg_corpus_cognates(const rg_corpus *corpus) {
    return corpus == 0 ? 0 : corpus->view;
}

const rg_cognate_set *rg_corpus_cognate_at(const rg_corpus *corpus, size_t index) {
    if (corpus == 0 || index >= corpus->view_count) {
        return 0;
    }
    return &corpus->view[index];
}
