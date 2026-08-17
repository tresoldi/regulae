#include "regulae.h"
#include "table_access.h"

#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef REGULAE_SOURCE_DIR
#define REGULAE_SOURCE_DIR "."
#endif

typedef struct fixture_pair {
    rg_segment *source_segments;
    size_t source_count;
    int *source_breaks;
    size_t source_break_count;
    rg_segment *target_segments;
    size_t target_count;
    int *target_breaks;
    size_t target_break_count;
    double weight;
} fixture_pair;

/* The test allocated these strings and stored them in fields the API declares
 * `const char *`, because that is what they are to a reader of a form. Freeing
 * them means taking the qualifier back off; going through a copy of the pointer
 * value keeps that defined. The library does the same thing in one place, as
 * rg_free_owned_internal. */
static void free_owned(const void *owned) {
    void *value;
    memcpy(&value, &owned, sizeof(value));
    free(value);
}

static char *dup_string(const char *value) {
    size_t len = strlen(value);
    char *out = (char *)malloc(len + 1);
    assert(out != 0);
    memcpy(out, value, len + 1);
    return out;
}

static void segment_clear(rg_segment *segment) {
    free_owned(segment->grapheme);
    free_owned(segment->tone);
    memset(segment, 0, sizeof(*segment));
}

static void fixture_pair_clear(fixture_pair *pair) {
    size_t i;
    for (i = 0; i < pair->source_count; i++) {
        segment_clear(&pair->source_segments[i]);
    }
    for (i = 0; i < pair->target_count; i++) {
        segment_clear(&pair->target_segments[i]);
    }
    free(pair->source_segments);
    free(pair->target_segments);
    free(pair->source_breaks);
    free(pair->target_breaks);
    memset(pair, 0, sizeof(*pair));
}


static void parse_form(const char *raw, rg_segment **segments_out, size_t *count_out, int **breaks_out, size_t *break_count_out) {
    rg_segment *segments = 0;
    size_t count = 0;
    size_t cap = 0;
    int *breaks = 0;
    size_t break_count = 0;
    size_t break_cap = 0;
    size_t end = strlen(raw);
    size_t i;
    for (i = 0; i < end; i++) {
        if (raw[i] == '+') {
            if (break_count == break_cap) {
                size_t next_cap = break_cap == 0 ? 4 : break_cap * 2;
                int *next = (int *)realloc(breaks, next_cap * sizeof(*breaks));
                assert(next != 0);
                breaks = next;
                break_cap = next_cap;
            }
            breaks[break_count++] = (int)count;
            continue;
        }
        if (raw[i] == '-' || raw[i] == '\0') {
            continue;
        }
        if (count == cap) {
            size_t next_cap = cap == 0 ? 8 : cap * 2;
            rg_segment *next = (rg_segment *)realloc(segments, next_cap * sizeof(*segments));
            assert(next != 0);
            segments = next;
            cap = next_cap;
        }
        memset(&segments[count], 0, sizeof(segments[count]));
        {
            char grapheme[2];
            grapheme[0] = raw[i];
            grapheme[1] = '\0';
            segments[count].grapheme = dup_string(grapheme);
        }
        count++;
    }
    *segments_out = segments;
    *count_out = count;
    *breaks_out = breaks;
    *break_count_out = break_count;
}

static rg_form form_view(const char *lect, const rg_segment *segments, size_t count, const int *breaks, size_t break_count) {
    rg_form form;
    form.lect_id = lect;
    form.segments = segments;
    form.segment_count = count;
    form.syllable_breaks = 0;
    form.syllable_break_count = 0;
    form.morpheme_breaks = breaks;
    form.morpheme_break_count = break_count;
    return form;
}

static rg_form_pair pair_view(const fixture_pair *pair, const char *source_lect, const char *target_lect) {
    rg_form_pair out;
    out.source = form_view(source_lect, pair->source_segments, pair->source_count, pair->source_breaks, pair->source_break_count);
    out.target = form_view(target_lect, pair->target_segments, pair->target_count, pair->target_breaks, pair->target_break_count);
    out.weight = pair->weight;
    return out;
}

static fixture_pair *append_pair(fixture_pair **items, size_t *count, size_t *cap) {
    fixture_pair *next;
    if (*count == *cap) {
        size_t next_cap = *cap == 0 ? 32 : *cap * 2;
        next = (fixture_pair *)realloc(*items, next_cap * sizeof(**items));
        assert(next != 0);
        *items = next;
        *cap = next_cap;
    }
    memset(&(*items)[*count], 0, sizeof((*items)[*count]));
    (*items)[*count].weight = 1.0;
    return &(*items)[(*count)++];
}

static void load_three_column_fixture(const char *path, fixture_pair **items_out, size_t *count_out) {
    char line[4096];
    FILE *fh = fopen(path, "r");
    fixture_pair *items = 0;
    size_t count = 0;
    size_t cap = 0;
    assert(fh != 0);
    assert(fgets(line, sizeof(line), fh) != 0);
    while (fgets(line, sizeof(line), fh) != 0) {
        char *cols[4] = {0, 0, 0, 0};
        char *tok;
        size_t col = 0;
        fixture_pair *pair;
        line[strcspn(line, "\r\n")] = '\0';
        tok = strtok(line, "\t");
        while (tok != 0 && col < 4) {
            cols[col++] = tok;
            tok = strtok(0, "\t");
        }
        if (col < 3) {
            continue;
        }
        pair = append_pair(&items, &count, &cap);
        parse_form(cols[1], &pair->source_segments, &pair->source_count, &pair->source_breaks, &pair->source_break_count);
        parse_form(cols[2], &pair->target_segments, &pair->target_count, &pair->target_breaks, &pair->target_break_count);
        if (col > 3 && cols[3] != 0 && cols[3][0] != '\0') {
            pair->weight = atof(cols[3]);
        }
    }
    fclose(fh);
    *items_out = items;
    *count_out = count;
}

static void fixture_pairs_free(fixture_pair *items, size_t count) {
    size_t i;
    for (i = 0; i < count; i++) {
        fixture_pair_clear(&items[i]);
    }
    free(items);
}

static rg_form_pair *make_views(const fixture_pair *items, size_t count, const char *source_lect, const char *target_lect) {
    rg_form_pair *views = (rg_form_pair *)calloc(count, sizeof(*views));
    size_t i;
    assert(views != 0);
    for (i = 0; i < count; i++) {
        views[i] = pair_view(&items[i], source_lect, target_lect);
    }
    return views;
}

/* The fixture is a wide corpus, so it goes in through the wide loader rather
 * than a parser written for this test: tone reaches the model the same way a
 * user's corpus would, which is the thing worth checking. */
static void test_tone_clean_fixture(rg_context *ctx) {
    rg_corpus *corpus = 0;
    rg_form_pair *views;
    rg_pairwise_model *model = 0;
    rg_train_options options;
    size_t count;
    size_t i;
    int found_voiced = 0;
    int found_voiceless = 0;

    assert(rg_corpus_load_wide_tsv(
        ctx, REGULAE_SOURCE_DIR "/experiments/tone_chinese_like_clean/cognates.tsv",
        0, &corpus, 0) == RG_OK);
    count = rg_corpus_cognate_count(corpus);
    assert(count == 80);
    views = (rg_form_pair *)calloc(count, sizeof(*views));
    assert(views != 0);
    for (i = 0; i < count; i++) {
        const rg_cognate_set *set = rg_corpus_cognate_at(corpus, i);
        assert(set->form_count == 2);
        views[i].source = set->forms[0].form;
        views[i].target = set->forms[1].form;
        views[i].weight = 1.0;
    }
    rg_train_options_init_defaults(&options);
    assert(rg_train_pairwise(ctx, views, count, &options, &model) == RG_OK);
    /* The fixture encodes one conditioned split, and both halves of it are
     * findings: voiced onsets take one tone, voiceless onsets the other. The
     * complementary environment is published under source_value "-".
     *
     * Cantonese kept its voicing here, so the split is found twice: as the
     * cross-lect rule (Mandarin's onset predicts Cantonese's tone) and as the
     * lect-internal one (Cantonese's own onset predicts its own tone, which is
     * tonogenesis). Four rows: two halves, each in both readings. */
    assert(rg_pairwise_model_cross_dimensional_row_count(model) == 4);
    {
        int internal_rows = 0;
        int cross_lect_rows = 0;
        for (i = 0; i < rg_pairwise_model_cross_dimensional_row_count(model); i++) {
            const rg_cross_dimensional_row *row = rg_pairwise_model_cross_dimensional_row_at(model, i);
            /* One predicate, on the preceding segment: the onset's voicing.
             * True of both readings, since the environment is the same onset. */
            assert(row->environment.preceding_count == 1);
            assert(strcmp(row->environment.preceding[0].feature, "voiced") == 0);
            assert(row->environment.self_count == 0);
            assert(row->environment.following_count == 0);
            assert(row->count == 40.0 && row->source_count == 40.0);
            assert(row->confidence == 1.0);
            /* The environment is what makes the difference: the value never
             * occurs outside it. */
            assert(row->contrast_count == 0.0);
            assert(row->contrast_confidence == 0.0);
            assert(row->evidence.delta_bic < 0.0);
            if (row->dimension_from_environment) {
                internal_rows++;
            } else {
                cross_lect_rows++;
                if (strcmp(row->environment.preceding[0].value, "+") == 0 &&
                    strcmp(row->value, "\xe2\x81\xb4\xe2\x81\xb4") == 0) {
                    found_voiced = 1;
                }
                if (strcmp(row->environment.preceding[0].value, "-") == 0 &&
                    strcmp(row->value, "\xc2\xb9\xc2\xb9") == 0) {
                    found_voiceless = 1;
                }
            }
        }
        assert(cross_lect_rows == 2);
        assert(internal_rows == 2);
    }
    assert(found_voiced);
    assert(found_voiceless);
    rg_pairwise_model_free(model);
    free(views);
    rg_corpus_free(corpus);
}

static void test_contaminated_weights_fixture(rg_context *ctx) {
    fixture_pair *items = 0;
    rg_form_pair *views;
    rg_pairwise_model *model = 0;
    rg_train_options options;
    size_t count = 0;
    size_t i;
    int found_pf = 0;
    load_three_column_fixture(REGULAE_SOURCE_DIR "/experiments/contaminated_cognates_synthetic/cognates.tsv", &items, &count);
    assert(count == 25);
    views = make_views(items, count, "proto", "derived");
    rg_train_options_init_defaults(&options);
    assert(rg_train_pairwise(ctx, views, count, &options, &model) == RG_OK);
    for (i = 0; i < rg_pairwise_model_segment_count_row_count(model); i++) {
        const rg_segment_count_row *row = rg_pairwise_model_segment_count_row_at(model, i);
        if (strcmp(row->source, "p") == 0 && strcmp(row->target, "f") == 0) {
            found_pf = 1;
            assert(row->count == 20.0);
        }
    }
    assert(found_pf);
    rg_pairwise_model_free(model);
    free(views);
    fixture_pairs_free(items, count);
}

int main(void) {
    rg_context *ctx = 0;
    assert(rg_context_new_builtin(&ctx) == RG_OK);
    test_tone_clean_fixture(ctx);
    test_contaminated_weights_fixture(ctx);
    rg_context_free(ctx);
    return 0;
}
