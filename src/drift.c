#include "internal.h"

#include <merkmal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Whether two lects in one corpus are transcribed by sources that disagree
 * about where a segment ends.
 *
 * This is the commonest way a comparative dataset goes wrong and the least
 * visible. One source writes the affricate as `tʃ` and the other as `t ʃ`; one
 * writes aspiration on the stop and the other as a following /h/; one writes a
 * long vowel with a length mark and the other by doubling it. Every grapheme
 * involved is valid IPA, everything resolves, nothing refuses to load -- and
 * what comes out of training is a family of clean, well-supported
 * correspondences that read as deaffrication, loss of aspiration and loss of
 * vowel length. None of those changes happened.
 *
 * The shuffled baseline cannot help here and it is worth being clear why. A
 * baseline separates a pattern from chance, and this pattern is not chance: it
 * is perfectly systematic, which is exactly what a sound law is. On
 * `testdata/diagnostics/drift.tsv` the corpus sits twenty-six standard
 * deviations below its own shuffles and both invented conditioned rules stand
 * above the noise floor.
 *
 * So the test has to be about the *writing*, and it is one question asked three
 * ways: take a grapheme one lect uses and the other never does, work out what
 * the other lect would have to write instead, and go and see whether it writes
 * it. The last step is what keeps this from being a similarity heuristic. Two
 * genuinely different languages, one of which lost its affricates, have exactly
 * the inventory asymmetry that drift has; what they do not have is the same
 * cognate sets showing the pieces in the same order.
 *
 * It reports and never refuses. A corpus can honestly contain both a language
 * with affricates and one without, and this cannot tell that apart from drift
 * on its own -- only the person who assembled it can. What it can do is say
 * where to look, which is the difference between finding out now and finding
 * out from a reviewer. */

/* Modifier letters that a source writing the same sound apart would spell with
 * an ordinary letter, and which ordinary letter it would use.
 *
 * Deliberately short. Each of these four is a convention that two real
 * transcription traditions differ over, and each has an unambiguous
 * plain-letter spelling that a source actually writes. Pharyngealisation and
 * the other secondary articulations are left out for the opposite reason: no
 * tradition writes `tˤ` as a sequence of two ordinary letters, so a pair for
 * it would report drift wherever a corpus had a genuine contrast. */
static const struct {
    const char *modifier;
    const char *spelled;
} modifier_pairs[] = {
    {"\xca\xb0", "h"},        /* ʰ  aspiration */
    {"\xca\xb7", "w"},        /* ʷ  labialisation */
    {"\xca\xb2", "j"},        /* ʲ  palatalisation */
    {"\xe2\x81\xbf", "n"}     /* ⁿ  nasal release */
};
#define MODIFIER_PAIR_COUNT (sizeof(modifier_pairs) / sizeof(modifier_pairs[0]))

#define LENGTH_MARK "\xcb\x90"   /* ː */

/* The corpus, indexed so the scan below is proportional to what it looks at
 * rather than to the corpus squared.
 *
 * The obvious shape -- for every ordered lect pair, for every grapheme, walk
 * the whole corpus -- is fine on the two-lect wordlist somebody assembled by
 * hand and hopeless on a published one. BDPA has 538 lects, 752 graphemes and
 * 750 cognate sets: a quarter of a million pairs each rescanning fifty
 * thousand forms. It ran for minutes and was still going.
 *
 * Three indexes turn it into a scan proportional to the evidence:
 *
 *   `ids`      every grapheme in the corpus, sorted, so a grapheme is a small
 *              integer and "does this lect use it" is one bit
 *   `uses`     one bitset per lect over those ids
 *   `sites`    for each (lect, grapheme), the cognate sets where it occurs,
 *              so corroboration walks ten sets and not seven hundred
 *
 * Which leaves the pair loop itself, and that stays quadratic: the report names
 * two lects because two lects is what a reader acts on. What it no longer does
 * inside the loop is touch the corpus. */

typedef struct drift_accumulator {
    rg_transcription_drift_row *rows;
    size_t count;
    size_t cap;
} drift_accumulator;

typedef struct grapheme_table {
    char **items;
    size_t count;
    size_t cap;
} grapheme_table;

typedef struct site_list {
    size_t *cognates;
    size_t count;
    size_t cap;
} site_list;

typedef struct lect_index {
    char *lect;
    unsigned char *uses;          /* bitset over grapheme_table ids */
    const rg_form **form_of;      /* cognate index -> this lect's form, or 0 */
    site_list *sites;             /* per grapheme id */
} lect_index;

typedef struct corpus_index {
    grapheme_table graphemes;
    lect_index *lects;
    size_t lect_count;
    size_t cognate_count;
} corpus_index;

static int grapheme_table_find(const grapheme_table *table, const char *grapheme, size_t *out) {
    size_t low = 0;
    size_t high = table->count;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        int c = strcmp(table->items[mid], grapheme);
        if (c == 0) {
            *out = mid;
            return 1;
        }
        if (c < 0) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    *out = low;
    return 0;
}

static rg_status grapheme_table_intern(grapheme_table *table, const char *grapheme, size_t *out) {
    size_t at;
    if (grapheme_table_find(table, grapheme, &at)) {
        *out = at;
        return RG_OK;
    }
    if (table->count == table->cap) {
        size_t next_cap = table->cap == 0 ? 64 : table->cap * 2;
        char **next = (char **)realloc(table->items, next_cap * sizeof(*next));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        table->items = next;
        table->cap = next_cap;
    }
    memmove(&table->items[at + 1], &table->items[at], (table->count - at) * sizeof(*table->items));
    table->items[at] = rg_strdup_internal(grapheme);
    if (table->items[at] == 0) {
        return RG_ERR_OOM;
    }
    table->count++;
    *out = at;
    return RG_OK;
}

static void corpus_index_clear(corpus_index *index) {
    size_t i;
    size_t g;
    for (i = 0; i < index->lect_count; i++) {
        free(index->lects[i].lect);
        free(index->lects[i].uses);
        free(index->lects[i].form_of);
        if (index->lects[i].sites != 0) {
            for (g = 0; g < index->graphemes.count; g++) {
                free(index->lects[i].sites[g].cognates);
            }
            free(index->lects[i].sites);
        }
    }
    free(index->lects);
    for (g = 0; g < index->graphemes.count; g++) {
        free(index->graphemes.items[g]);
    }
    free(index->graphemes.items);
    memset(index, 0, sizeof(*index));
}

static int lect_uses(const lect_index *lect, size_t id) {
    return (lect->uses[id / 8] >> (id % 8)) & 1;
}

static rg_status site_list_add(site_list *list, size_t cognate) {
    if (list->count > 0 && list->cognates[list->count - 1] == cognate) {
        return RG_OK;
    }
    if (list->count == list->cap) {
        size_t next_cap = list->cap == 0 ? 4 : list->cap * 2;
        size_t *next = (size_t *)realloc(list->cognates, next_cap * sizeof(*next));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        list->cognates = next;
        list->cap = next_cap;
    }
    list->cognates[list->count++] = cognate;
    return RG_OK;
}

static rg_status build_corpus_index(
    const rg_cognate_set *cognates,
    size_t cognate_count,
    corpus_index *index
) {
    size_t c;
    size_t f;
    size_t s;
    size_t i;
    size_t bytes;

    memset(index, 0, sizeof(*index));
    index->cognate_count = cognate_count;

    /* Pass one: every lect and every grapheme, so the table is fixed before
     * anything is sized against it. */
    for (c = 0; c < cognate_count; c++) {
        for (f = 0; f < cognates[c].form_count; f++) {
            const rg_cognate_form *form = &cognates[c].forms[f];
            size_t index_of = index->lect_count;
            for (i = 0; i < index->lect_count; i++) {
                if (strcmp(index->lects[i].lect, form->lect_id) == 0) {
                    index_of = i;
                    break;
                }
            }
            if (index_of == index->lect_count) {
                lect_index *next = (lect_index *)realloc(
                    index->lects, (index->lect_count + 1) * sizeof(*next));
                if (next == 0) {
                    corpus_index_clear(index);
                    return RG_ERR_OOM;
                }
                index->lects = next;
                memset(&index->lects[index->lect_count], 0, sizeof(index->lects[0]));
                index->lects[index->lect_count].lect = rg_strdup_internal(form->lect_id);
                if (index->lects[index->lect_count].lect == 0) {
                    corpus_index_clear(index);
                    return RG_ERR_OOM;
                }
                index->lect_count++;
            }
            for (s = 0; s < form->form.segment_count; s++) {
                size_t id;
                if (form->form.segments[s].grapheme == 0) {
                    continue;
                }
                if (grapheme_table_intern(&index->graphemes,
                                          form->form.segments[s].grapheme, &id) != RG_OK) {
                    corpus_index_clear(index);
                    return RG_ERR_OOM;
                }
            }
        }
    }
    /* Ascending lect id, so the report does not depend on which lect a corpus
     * happens to mention first. */
    for (i = 1; i < index->lect_count; i++) {
        lect_index key = index->lects[i];
        size_t j = i;
        while (j > 0 && strcmp(index->lects[j - 1].lect, key.lect) > 0) {
            index->lects[j] = index->lects[j - 1];
            j--;
        }
        index->lects[j] = key;
    }

    bytes = (index->graphemes.count + 7) / 8;
    for (i = 0; i < index->lect_count; i++) {
        index->lects[i].uses = (unsigned char *)calloc(bytes == 0 ? 1 : bytes, 1);
        index->lects[i].form_of = (const rg_form **)calloc(
            cognate_count == 0 ? 1 : cognate_count, sizeof(*index->lects[i].form_of));
        index->lects[i].sites = (site_list *)calloc(
            index->graphemes.count == 0 ? 1 : index->graphemes.count,
            sizeof(*index->lects[i].sites));
        if (index->lects[i].uses == 0 || index->lects[i].form_of == 0 ||
            index->lects[i].sites == 0) {
            corpus_index_clear(index);
            return RG_ERR_OOM;
        }
    }

    /* Pass two: fill them. */
    for (c = 0; c < cognate_count; c++) {
        for (f = 0; f < cognates[c].form_count; f++) {
            const rg_cognate_form *form = &cognates[c].forms[f];
            lect_index *lect = 0;
            for (i = 0; i < index->lect_count; i++) {
                if (strcmp(index->lects[i].lect, form->lect_id) == 0) {
                    lect = &index->lects[i];
                    break;
                }
            }
            if (lect == 0) {
                continue;
            }
            /* A doublet gives one lect two forms in a set; the first is the one
             * the sites index points at, and both contribute their graphemes. */
            if (lect->form_of[c] == 0) {
                lect->form_of[c] = &form->form;
            }
            for (s = 0; s < form->form.segment_count; s++) {
                size_t id;
                if (form->form.segments[s].grapheme == 0) {
                    continue;
                }
                if (!grapheme_table_find(&index->graphemes,
                                         form->form.segments[s].grapheme, &id)) {
                    continue;
                }
                lect->uses[id / 8] |= (unsigned char)(1u << (id % 8));
                if (site_list_add(&lect->sites[id], c) != RG_OK) {
                    corpus_index_clear(index);
                    return RG_ERR_OOM;
                }
            }
        }
    }
    return RG_OK;
}

/* How many cognate sets have `mine` writing the grapheme with `theirs` also
 * present, and how many of those have `theirs` writing `pieces` as a run of
 * adjacent segments.
 *
 * The second number is what makes the finding a finding. An inventory
 * asymmetry on its own is what two different languages look like; the same
 * cognate sets showing the pieces in the same order is what one language
 * written two ways looks like. */
static void corroborate(
    const lect_index *mine,
    const lect_index *theirs,
    size_t grapheme_id,
    const char *const *pieces,
    size_t piece_count,
    size_t *out_forms,
    size_t *out_corroborated
) {
    const site_list *sites = &mine->sites[grapheme_id];
    size_t i;
    *out_forms = 0;
    *out_corroborated = 0;
    for (i = 0; i < sites->count; i++) {
        const rg_form *there = theirs->form_of[sites->cognates[i]];
        size_t s;
        if (there == 0) {
            continue;
        }
        (*out_forms)++;
        if (there->segment_count < piece_count) {
            continue;
        }
        for (s = 0; s + piece_count <= there->segment_count; s++) {
            size_t k;
            int match = 1;
            for (k = 0; k < piece_count; k++) {
                const char *g = there->segments[s + k].grapheme;
                if (g == 0 || strcmp(g, pieces[k]) != 0) {
                    match = 0;
                    break;
                }
            }
            if (match) {
                (*out_corroborated)++;
                break;
            }
        }
    }
}

static rg_status accumulate(
    drift_accumulator *acc,
    const char *lect,
    const char *other_lect,
    const char *grapheme,
    const char *written_as,
    rg_drift_kind kind,
    size_t forms,
    size_t corroborated
) {
    rg_transcription_drift_row *slot;
    if (acc->count == acc->cap) {
        size_t next_cap = acc->cap == 0 ? 8 : acc->cap * 2;
        rg_transcription_drift_row *next =
            (rg_transcription_drift_row *)realloc(acc->rows, next_cap * sizeof(*next));
        if (next == 0) {
            return RG_ERR_OOM;
        }
        acc->rows = next;
        acc->cap = next_cap;
    }
    slot = &acc->rows[acc->count];
    memset(slot, 0, sizeof(*slot));
    slot->lect = rg_strdup_internal(lect);
    slot->other_lect = rg_strdup_internal(other_lect);
    slot->grapheme = rg_strdup_internal(grapheme);
    slot->written_as = rg_strdup_internal(written_as);
    if (slot->lect == 0 || slot->other_lect == 0 || slot->grapheme == 0 || slot->written_as == 0) {
        rg_free_owned_internal(slot->lect);
        rg_free_owned_internal(slot->other_lect);
        rg_free_owned_internal(slot->grapheme);
        rg_free_owned_internal(slot->written_as);
        memset(slot, 0, sizeof(*slot));
        return RG_ERR_OOM;
    }
    slot->kind = kind;
    slot->forms = forms;
    slot->corroborated = corroborated;
    acc->count++;
    return RG_OK;
}

/* Joins the pieces with spaces, the way a segments cell writes them. */
static rg_status join_pieces(const char *const *pieces, size_t count, char **out) {
    size_t len = 1;
    size_t i;
    char *text;
    char *p;
    for (i = 0; i < count; i++) {
        len += strlen(pieces[i]) + 1;
    }
    text = (char *)malloc(len);
    if (text == 0) {
        return RG_ERR_OOM;
    }
    p = text;
    for (i = 0; i < count; i++) {
        size_t n = strlen(pieces[i]);
        if (i > 0) {
            *p++ = ' ';
        }
        memcpy(p, pieces[i], n);
        p += n;
    }
    *p = '\0';
    *out = text;
    return RG_OK;
}

static int drift_row_cmp(const void *a, const void *b) {
    const rg_transcription_drift_row *ra = (const rg_transcription_drift_row *)a;
    const rg_transcription_drift_row *rb = (const rg_transcription_drift_row *)b;
    int c;
    if (ra->corroborated != rb->corroborated) {
        return ra->corroborated > rb->corroborated ? -1 : 1;
    }
    c = strcmp(ra->grapheme, rb->grapheme);
    if (c != 0) {
        return c;
    }
    c = strcmp(ra->lect, rb->lect);
    if (c != 0) {
        return c;
    }
    return strcmp(ra->other_lect, rb->other_lect);
}

static rg_status report_if_written_apart(
    drift_accumulator *acc,
    const corpus_index *index,
    const lect_index *mine,
    const lect_index *theirs,
    size_t grapheme_id,
    const char *const *pieces,
    size_t piece_count,
    rg_drift_kind kind
) {
    size_t forms = 0;
    size_t corroborated = 0;
    char *written_as = 0;
    size_t k;
    rg_status status;

    for (k = 0; k < piece_count; k++) {
        size_t id;
        if (!grapheme_table_find(&index->graphemes, pieces[k], &id) ||
            !lect_uses(theirs, id)) {
            return RG_OK;
        }
    }
    corroborate(mine, theirs, grapheme_id, pieces, piece_count,
                &forms, &corroborated);
    /* Nothing to say when the other lect never writes the pieces where this one
     * writes the whole. That is two languages, not two conventions. */
    if (corroborated == 0) {
        return RG_OK;
    }
    status = join_pieces(pieces, piece_count, &written_as);
    if (status != RG_OK) {
        return status;
    }
    status = accumulate(acc, mine->lect, theirs->lect,
                        index->graphemes.items[grapheme_id], written_as,
                        kind, forms, corroborated);
    free(written_as);
    return status;
}

/* What the other lect would have to write instead of this grapheme, if it were
 * writing the same sound apart. Up to three answers, one per kind. */
typedef struct decomposition {
    const char *pieces[4];
    size_t count;
    rg_drift_kind kind;
    char *owned;               /* the base string, when one had to be cut */
} decomposition;

static size_t decompositions_of(const char *grapheme, decomposition *out, size_t capacity) {
    size_t count = 0;
    size_t length = strlen(grapheme);
    size_t m;
    size_t mark_len = strlen(LENGTH_MARK);

    /* 1. Segmentation: a grapheme merkmal itself splits, and the pieces are
     *    what the other source writes. Affricates are the case that matters --
     *    tʃ against t ʃ, dʒ against d ʒ. */
    if (count < capacity) {
        mk_string_list *parts = 0;
        if (mk_segment_ipa(grapheme, &parts) == MK_OK) {
            size_t n = mk_string_list_size(parts);
            if (n > 1 && n <= 4) {
                size_t k;
                size_t total = 1;
                char *joined;
                for (k = 0; k < n; k++) {
                    total += strlen(mk_string_list_get(parts, k)) + 1;
                }
                joined = (char *)malloc(total);
                if (joined != 0) {
                    char *p = joined;
                    for (k = 0; k < n; k++) {
                        const char *piece = mk_string_list_get(parts, k);
                        size_t plen = strlen(piece);
                        memcpy(p, piece, plen);
                        out[count].pieces[k] = p;
                        p += plen;
                        *p++ = '\0';
                    }
                    out[count].count = n;
                    out[count].kind = RG_DRIFT_SEGMENTATION;
                    out[count].owned = joined;
                    count++;
                }
            }
            mk_string_list_free(parts);
        }
    }

    /* 2. A modifier letter the other source spells with an ordinary one: tʰ
     *    against t h, kʷ against k w. merkmal keeps these whole -- they are one
     *    segment -- so segmentation above cannot find them. */
    for (m = 0; m < MODIFIER_PAIR_COUNT && count < capacity; m++) {
        size_t mod_len = strlen(modifier_pairs[m].modifier);
        char *base;
        if (length <= mod_len ||
            strcmp(grapheme + length - mod_len, modifier_pairs[m].modifier) != 0) {
            continue;
        }
        base = (char *)malloc(length - mod_len + 1);
        if (base == 0) {
            continue;
        }
        memcpy(base, grapheme, length - mod_len);
        base[length - mod_len] = '\0';
        out[count].pieces[0] = base;
        out[count].pieces[1] = modifier_pairs[m].spelled;
        out[count].count = 2;
        out[count].kind = RG_DRIFT_MODIFIER;
        out[count].owned = base;
        count++;
    }

    /* 3. Length written by doubling: aː against a a. A language with real
     *    geminate vowels writes that too, which is why the corroboration count
     *    is reported rather than a verdict. */
    if (count < capacity && length > mark_len &&
        strcmp(grapheme + length - mark_len, LENGTH_MARK) == 0) {
        char *base = (char *)malloc(length - mark_len + 1);
        if (base != 0) {
            memcpy(base, grapheme, length - mark_len);
            base[length - mark_len] = '\0';
            out[count].pieces[0] = base;
            out[count].pieces[1] = base;
            out[count].count = 2;
            out[count].kind = RG_DRIFT_LENGTH;
            out[count].owned = base;
            count++;
        }
    }
    return count;
}

rg_status rg_find_transcription_drift(
    const rg_context *ctx,
    const rg_cognate_set *cognates,
    size_t cognate_count,
    rg_transcription_drift_row **out,
    size_t *out_count
) {
    corpus_index index;
    drift_accumulator acc;
    decomposition *table = 0;
    size_t *table_offsets = 0;
    size_t a;
    size_t b;
    size_t g;
    size_t d;
    size_t total = 0;
    rg_status status;

    if (out == 0 || out_count == 0) {
        return RG_ERR_INVALID_ARGUMENT;
    }
    *out = 0;
    *out_count = 0;
    memset(&acc, 0, sizeof(acc));
    (void)ctx;

    status = build_corpus_index(cognates, cognate_count, &index);
    if (status != RG_OK) {
        return status;
    }

    /* Every grapheme's decompositions, computed once for the corpus rather than
     * once per lect pair. This is the step that used to call merkmal's
     * segmenter a quarter of a million times on a corpus with 538 lects. */
    table_offsets = (size_t *)calloc(index.graphemes.count + 1, sizeof(*table_offsets));
    table = (decomposition *)calloc(index.graphemes.count * 6 + 1, sizeof(*table));
    if (table_offsets == 0 || table == 0) {
        free(table_offsets);
        free(table);
        corpus_index_clear(&index);
        return RG_ERR_OOM;
    }
    for (g = 0; g < index.graphemes.count; g++) {
        table_offsets[g] = total;
        total += decompositions_of(index.graphemes.items[g], &table[total], 6);
    }
    table_offsets[index.graphemes.count] = total;

    for (a = 0; a < index.lect_count && status == RG_OK; a++) {
        for (b = 0; b < index.lect_count && status == RG_OK; b++) {
            if (a == b) {
                continue;
            }
            for (g = 0; g < index.graphemes.count && status == RG_OK; g++) {
                /* Only graphemes this lect uses and the other never does. One
                 * the other does write is a sound both sources agree about,
                 * whatever else differs. */
                if (!lect_uses(&index.lects[a], g) || lect_uses(&index.lects[b], g)) {
                    continue;
                }
                for (d = table_offsets[g]; d < table_offsets[g + 1] && status == RG_OK; d++) {
                    status = report_if_written_apart(
                        &acc, &index, &index.lects[a], &index.lects[b], g,
                        table[d].pieces, table[d].count, table[d].kind);
                }
            }
        }
    }

    /* Strongest evidence first, and the tie broken on the row's own text so
     * two runs over one corpus print the same thing. A reader with four
     * hundred rows reads the top of the list; a reader with four reads all of
     * them, and neither is served by lect-id order. */
    if (status == RG_OK && acc.count > 1) {
        qsort(acc.rows, acc.count, sizeof(*acc.rows), drift_row_cmp);
    }
    for (d = 0; d < total; d++) {
        free(table[d].owned);
    }
    free(table);
    free(table_offsets);
    corpus_index_clear(&index);
    if (status != RG_OK) {
        rg_transcription_drift_rows_free(acc.rows, acc.count);
        return status;
    }
    *out = acc.rows;
    *out_count = acc.count;
    return RG_OK;
}

void rg_transcription_drift_rows_free(rg_transcription_drift_row *rows, size_t count) {
    size_t i;
    if (rows == 0) {
        return;
    }
    for (i = 0; i < count; i++) {
        rg_free_owned_internal(rows[i].lect);
        rg_free_owned_internal(rows[i].other_lect);
        rg_free_owned_internal(rows[i].grapheme);
        rg_free_owned_internal(rows[i].written_as);
    }
    free(rows);
}
