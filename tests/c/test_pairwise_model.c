#include "regulae.h"
#include "table_access.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

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

static rg_segment seg(const char *g) {
    rg_segment s = {g, 0, 0, 0};
    return s;
}

static rg_form form(const char *lect, const rg_segment *segments, size_t count) {
    rg_form f;
    f.lect_id = lect;
    f.segments = segments;
    f.segment_count = count;
    f.syllable_breaks = 0;
    f.syllable_break_count = 0;
    f.morpheme_breaks = 0;
    f.morpheme_break_count = 0;
    return f;
}

static rg_form form_with_morphemes(const char *lect, const rg_segment *segments, size_t count, const int *breaks, size_t break_count) {
    rg_form f = form(lect, segments, count);
    f.morpheme_breaks = breaks;
    f.morpheme_break_count = break_count;
    return f;
}

static rg_form one_segment_form(const char *lect, const rg_segment *segment) {
    return form(lect, segment, 1);
}

static char *dup_string(const char *value) {
    size_t len = strlen(value);
    char *out = (char *)malloc(len + 1);
    assert(out != 0);
    memcpy(out, value, len + 1);
    return out;
}

static void segments_free(rg_segment *segments, size_t count) {
    size_t i;
    void *value;
    for (i = 0; i < count; i++) {
        memcpy(&value, &segments[i].grapheme, sizeof(value));
        free(value);
    }
    free(segments);
}

static int has_constraint(const rg_feature_constraint *items, size_t count, const char *feature, const char *value) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (strcmp(items[i].feature, feature) == 0 && strcmp(items[i].value, value) == 0) {
            return 1;
        }
    }
    return 0;
}

static int has_distance_constraint(const rg_distance_constraint *items, size_t count, int offset, const char *feature, const char *value) {
    size_t i;
    for (i = 0; i < count; i++) {
        if (items[i].offset == offset &&
            strcmp(items[i].constraint.feature, feature) == 0 &&
            strcmp(items[i].constraint.value, value) == 0) {
            return 1;
        }
    }
    return 0;
}

static void assert_uncertainty_contains(const rg_uncertainty_estimate *uncertainty, double estimate) {
    assert(fabs(uncertainty->estimate - estimate) < 1e-12);
    assert(uncertainty->lower <= uncertainty->estimate + 1e-12);
    assert(uncertainty->upper + 1e-12 >= uncertainty->estimate);
    assert(uncertainty->lower >= 0.0);
    assert(uncertainty->upper <= 1.0);
}

/* A conditioning environment has to condition something. These are the two
 * ways a rule can be committed on evidence that supports nothing, and both
 * were committed before 2026-08-14: a predicate true of the whole corpus,
 * which partitions nothing, and a predicate that partitions the corpus without
 * moving the target distribution. Neither is a sound change. */
static void test_uninformative_environments_commit_nothing(rg_context *ctx) {
    rg_segment consonant;
    rg_segment vowel_high;
    rg_segment vowel_low;
    rg_segment target_high;
    rg_segment target_low;
    rg_segment source[2];
    rg_segment high_target[2];
    rg_segment low_target[2];
    rg_form_pair pairs[40];
    rg_train_options options;
    rg_pairwise_model *model = 0;
    size_t i;

    memset(&consonant, 0, sizeof(consonant));
    memset(&vowel_high, 0, sizeof(vowel_high));
    memset(&vowel_low, 0, sizeof(vowel_low));
    memset(&target_high, 0, sizeof(target_high));
    memset(&target_low, 0, sizeof(target_low));
    consonant.grapheme = "p";
    vowel_high.grapheme = "i";
    vowel_low.grapheme = "a";
    target_high.grapheme = "i";
    target_high.tone = "H";
    target_low.grapheme = "a";
    target_low.tone = "H";
    rg_train_options_init_defaults(&options);

    /* Every source segment sequence is p + vowel, and every target tone is H.
     * "the preceding segment is a consonant" holds everywhere, so it has no
     * complement and cannot be an environment. */
    source[0] = consonant;
    source[1] = vowel_high;
    high_target[0] = consonant;
    high_target[1] = target_high;
    for (i = 0; i < 20; i++) {
        pairs[i].source = form("A", source, 2);
        pairs[i].target = form("B", high_target, 2);
        pairs[i].weight = 1.0;
    }
    assert(rg_train_pairwise(ctx, pairs, 20, &options, &model) == RG_OK);
    assert(rg_pairwise_model_cross_dimensional_row_count(model) == 0);
    rg_pairwise_model_free(model);
    model = 0;

    /* Now a predicate that does partition the corpus -- the vowel is close in
     * half the pairs -- but the target tone is H either way, so the split
     * explains nothing and buys no rule. */
    low_target[0] = consonant;
    low_target[1] = target_low;
    for (i = 0; i < 40; i++) {
        rg_segment *src = (rg_segment *)malloc(2 * sizeof(*src));
        assert(src != 0);
        src[0] = consonant;
        src[1] = (i % 2 == 0) ? vowel_high : vowel_low;
        pairs[i].source = form("A", src, 2);
        pairs[i].target = form("B", (i % 2 == 0) ? high_target : low_target, 2);
        pairs[i].weight = 1.0;
    }
    assert(rg_train_pairwise(ctx, pairs, 40, &options, &model) == RG_OK);
    assert(rg_pairwise_model_cross_dimensional_row_count(model) == 0);
    rg_pairwise_model_free(model);
    for (i = 0; i < 40; i++) {
        free_owned(pairs[i].source.segments);
    }
}

/* The Middle Chinese register split: a source tone splits by the voicing of
 * the onset before it. Source tone 2 becomes tone 4 after a voiced onset and
 * tone 2 after a voiceless one, while source tone 1 is unaffected either way.
 *
 * Neither predicate alone predicts anything. Voicing alone reported this at
 * confidence 0.50 and read as a weak finding rather than half of one, and the
 * source's own tone was not in the predicate vocabulary at all. A rule needs
 * to name both, which is what an rg_context_spec environment is for. */
static void test_joint_cross_dimensional_rule(rg_context *ctx) {
    rg_corpus *corpus = 0;
    rg_form_pair *views;
    rg_pairwise_model *model = 0;
    rg_train_options options;
    size_t count;
    size_t i;
    int found_voiced = 0;
    int found_voiceless = 0;

    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/corpora/joint_tonogenesis.tsv",
                              0, &corpus, 0) == RG_OK);
    count = rg_corpus_cognate_count(corpus);
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
    for (i = 0; i < rg_pairwise_model_cross_dimensional_row_count(model); i++) {
        const rg_cross_dimensional_row *row = rg_pairwise_model_cross_dimensional_row_at(model, i);
        const rg_context_spec *environment = &row->environment;
        /* Both predicates, and neither is enough: the onset's voicing, and the
         * source segment's own tone. */
        if (environment->preceding_count != 1 || environment->self_count != 1) {
            continue;
        }
        if (strcmp(environment->preceding[0].feature, "voiced") != 0 ||
            strcmp(environment->self[0].feature, "tone") != 0 ||
            strcmp(environment->self[0].value, "2") != 0) {
            continue;
        }
        /* Naming both makes the rule exact where naming one left it at half. */
        assert(row->confidence == 1.0);
        if (strcmp(environment->preceding[0].value, "+") == 0 &&
            strcmp(row->value, "4") == 0) {
            found_voiced = 1;
        }
        if (strcmp(environment->preceding[0].value, "-") == 0 &&
            strcmp(row->value, "2") == 0) {
            found_voiceless = 1;
        }
    }
    assert(found_voiced);
    assert(found_voiceless);
    rg_pairwise_model_free(model);
    free(views);
    rg_corpus_free(corpus);
}

/* Tonogenesis is lect-internal: an onset conditions the tone in the same
 * language. The row can say "this lect's own onset predicts its own tone", not
 * only "one lect's material predicts the other's tone", so
 * the daughter's own voicing-conditioned tone is statable. The fixture keeps
 * voicing on the daughter and splits its tone by that voicing; both the
 * lect-internal rule and the cross-lect one must appear, and no gap must reach
 * the segment side. */
static void test_lect_internal_tonogenesis(rg_context *ctx) {
    rg_corpus *corpus = 0;
    rg_form_pair *views;
    rg_pairwise_model *model = 0;
    rg_train_options options;
    size_t count;
    size_t i;
    int found_internal = 0;
    int found_cross_lect = 0;

    assert(rg_corpus_load_tsv(REGULAE_SOURCE_DIR "/testdata/corpora/tonogenesis_internal.tsv",
                              0, &corpus, 0) == RG_OK);
    count = rg_corpus_cognate_count(corpus);
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
    for (i = 0; i < rg_pairwise_model_cross_dimensional_row_count(model); i++) {
        const rg_cross_dimensional_row *row = rg_pairwise_model_cross_dimensional_row_at(model, i);
        /* The voiced-onset rule, at full confidence: onset voiced -> low tone. */
        if (row->environment.preceding_count != 1 ||
            strcmp(row->environment.preceding[0].feature, "voiced") != 0 ||
            strcmp(row->environment.preceding[0].value, "+") != 0 ||
            strcmp(row->dimension, "tone") != 0 ||
            strcmp(row->value, "\xc2\xb9\xc2\xb9") != 0) {
            continue;
        }
        assert(row->confidence == 1.0);
        if (row->dimension_from_environment && row->context_is_target) {
            /* The daughter (target) preserved voicing, so its own onset
             * conditions its own tone -- read from the target on both sides. */
            found_internal = 1;
        }
        if (!row->dimension_from_environment) {
            found_cross_lect = 1;
        }
    }
    assert(found_internal);
    assert(found_cross_lect);
    rg_pairwise_model_free(model);
    free(views);
    rg_corpus_free(corpus);
}

/* A committed environment is not always identifiable. Where onset voicing and
 * the following vowel's frontness are perfectly confounded, the corpus cannot
 * say which conditions the tone; the rule names one but must flag that another
 * feature at another position carves it the same. Where they vary independently,
 * the environment is pinned and the flag is zero. */
static void test_cross_dimensional_identifiability(rg_context *ctx) {
    struct { const char *path; int expect_confounded; } cases[] = {
        {REGULAE_SOURCE_DIR "/testdata/evaluation/m7/corpora/trap_confounded_environment.tsv", 1},
        {REGULAE_SOURCE_DIR "/testdata/corpora/tonogenesis_internal.tsv", 0},
    };
    size_t ci;
    for (ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        rg_corpus *corpus = 0;
        rg_form_pair *views;
        rg_pairwise_model *model = 0;
        rg_train_options options;
        size_t count;
        size_t i;
        int saw_rule = 0;
        assert(rg_corpus_load_tsv(cases[ci].path, 0, &corpus, 0) == RG_OK);
        count = rg_corpus_cognate_count(corpus);
        views = (rg_form_pair *)calloc(count, sizeof(*views));
        assert(views != 0);
        for (i = 0; i < count; i++) {
            const rg_cognate_set *set = rg_corpus_cognate_at(corpus, i);
            views[i].source = set->forms[0].form;
            views[i].target = set->forms[1].form;
            views[i].weight = 1.0;
        }
        rg_train_options_init_defaults(&options);
        assert(rg_train_pairwise(ctx, views, count, &options, &model) == RG_OK);
        for (i = 0; i < rg_pairwise_model_cross_dimensional_row_count(model); i++) {
            const rg_cross_dimensional_row *row = rg_pairwise_model_cross_dimensional_row_at(model, i);
            if (strcmp(row->dimension, "tone") != 0) {
                continue;
            }
            saw_rule = 1;
            if (cases[ci].expect_confounded) {
                assert(row->environment_alternatives > 0);
            } else {
                assert(row->environment_alternatives == 0);
            }
        }
        assert(saw_rule);
        rg_pairwise_model_free(model);
        free(views);
        rg_corpus_free(corpus);
    }
}

/* The target dimension is not only tone. The scorer has handled stress and
 * length as targets since the port; this stage proposed neither until
 * 2026-08-15, so compensatory lengthening and stress shifts were unreachable
 * however regular they were -- and no loader populated rg_segment.length at
 * all, so the length dimension could not be supplied even by hand. */
static void test_cross_dimensional_dimension_target(rg_context *ctx, const char *corpus_path,
                                                    const char *dimension) {
    rg_corpus *corpus = 0;
    rg_form_pair *views;
    rg_pairwise_model *model = 0;
    rg_train_options options;
    size_t count;
    size_t i;
    int found = 0;

    assert(rg_corpus_load_tsv(corpus_path, 0, &corpus, 0) == RG_OK);
    count = rg_corpus_cognate_count(corpus);
    views = (rg_form_pair *)calloc(count, sizeof(*views));
    assert(views != 0);
    for (i = 0; i < count; i++) {
        const rg_cognate_set *set = rg_corpus_cognate_at(corpus, i);
        views[i].source = set->forms[0].form;
        views[i].target = set->forms[1].form;
        views[i].weight = 1.0;
    }
    rg_train_options_init_defaults(&options);
    assert(rg_train_pairwise(ctx, views, count, &options, &model) == RG_OK);
    for (i = 0; i < rg_pairwise_model_cross_dimensional_row_count(model); i++) {
        const rg_cross_dimensional_row *row = rg_pairwise_model_cross_dimensional_row_at(model, i);
        if (strcmp(row->dimension, dimension) == 0 &&
            (row->environment.preceding_count == 1 ||
             row->environment.following_count == 1)) {
            assert(row->confidence == 1.0);
            found = 1;
        }
    }
    assert(found);
    rg_pairwise_model_free(model);
    free(views);
    rg_corpus_free(corpus);
}

/* A segment answering to nothing is a correspondence like any other, only with
 * ∅ on one side. Six words keep a final -n on side A and drop it on B; the null
 * correspondences must state n ~ ∅ at 6 of 6, the commonest change and the one
 * the 1-to-1 table cannot express. */
static void test_gap_correspondence(rg_context *ctx) {
    const char *words[6] = {"apan", "atan", "akan", "aman", "asan", "alan"};
    rg_train_options options;
    rg_form_pair pairs[6];
    rg_segment *kept[6];
    rg_segment *lost[6];
    size_t kept_n[6];
    size_t lost_n[6];
    rg_pairwise_model *model = 0;
    size_t i;
    int found_deletion = 0;

    rg_train_options_init_defaults(&options);
    for (i = 0; i < 6; i++) {
        size_t n = strlen(words[i]);
        size_t j;
        kept_n[i] = n;
        lost_n[i] = n - 1;
        kept[i] = (rg_segment *)calloc(n, sizeof(rg_segment));
        lost[i] = (rg_segment *)calloc(n - 1, sizeof(rg_segment));
        assert(kept[i] != 0 && lost[i] != 0);
        for (j = 0; j < n; j++) {
            char g[2];
            g[0] = words[i][j];
            g[1] = '\0';
            kept[i][j].grapheme = dup_string(g);
            if (j < n - 1) {
                lost[i][j].grapheme = dup_string(g);
            }
        }
        pairs[i].source = form("A", kept[i], kept_n[i]);
        pairs[i].target = form("B", lost[i], lost_n[i]);
        pairs[i].weight = 1.0;
    }
    assert(rg_train_pairwise(ctx, pairs, 6, &options, &model) == RG_OK);
    assert(rg_pairwise_model_null_correspondence_row_count(model) > 0);
    for (i = 0; i < rg_pairwise_model_null_correspondence_row_count(model); i++) {
        const rg_segment_count_row *row = rg_pairwise_model_null_correspondence_row_at(model, i);
        int deletion = strcmp(row->target, RG_GAP_GRAPHEME) == 0;
        double present_total = deletion ? row->source_total : row->target_total;
        assert(row != 0);
        /* Every present grapheme's count is bounded by how often it appears. */
        assert(row->count <= present_total + 1e-9);
        if (deletion && strcmp(row->source, "n") == 0) {
            assert(row->count == 6.0);
            assert(present_total == 6.0);
            assert_uncertainty_contains(&row->uncertainty, 1.0);
            found_deletion = 1;
        }
    }
    assert(found_deletion);
    rg_pairwise_model_free(model);
    for (i = 0; i < 6; i++) {
        segments_free(kept[i], kept_n[i]);
        segments_free(lost[i], lost_n[i]);
    }
}

int main(void) {
    rg_context *ctx = 0;
    rg_pairwise_model *model = 0;
    rg_train_options options;
    rg_segment pater[] = {{"p", 0, 0, 0}, {"a", 0, 0, 0}, {"t", 0, 0, 0}, {"e", 0, 0, 0}, {"r", 0, 0, 0}};
    rg_segment fadar[] = {{"f", 0, 0, 0}, {"a", 0, 0, 0}, {"d", 0, 0, 0}, {"a", 0, 0, 0}, {"r", 0, 0, 0}};
    rg_segment tone_a[] = {{"a", "1", 0, 0}};
    rg_segment tone_b[] = {{"a", "2", 0, 0}};
    rg_segment voiced_src[] = {{"b", 0, 0, 0}};
    rg_segment voiced_high[] = {{"b", "H", 0, 0}};
    rg_segment voiced_low[] = {{"b", "L", 0, 0}};
    rg_segment voiceless_src[] = {{"p", 0, 0, 0}};
    rg_segment voiceless_low[] = {{"p", "L", 0, 0}};
    rg_segment pa[] = {{"p", 0, 0, 0}, {"a", 0, 0, 0}};
    rg_segment fa[] = {{"f", 0, 0, 0}, {"a", 0, 0, 0}};
    rg_segment pi[] = {{"p", 0, 0, 0}, {"i", 0, 0, 0}};
    rg_segment ka[] = {{"k", 0, 0, 0}, {"a", 0, 0, 0}};
    rg_segment kt[] = {{"k", 0, 0, 0}, {"t", 0, 0, 0}};
    rg_segment tt[] = {{"t", 0, 0, 0}, {"t", 0, 0, 0}};
    int internal_break[] = {1};
    rg_form_pair pairs[2];
    rg_form_pair tone_pairs[1];
    rg_form_pair cross_dim_pairs[6];
    rg_form_pair conditioned_pairs[6];
    rg_form_pair long_range_pairs[16];
    rg_form_pair chunk_pairs[20];
    rg_form_pair boundary_chunk_pairs[20];
    size_t i;
    int found_pf = 0;
    int found_aa = 0;
    int found_disp = 0;
    int found_tone = 0;
    int found_cross_dimensional = 0;
    int found_conditioned = 0;
    int found_long_range = 0;
    int found_chunk = 0;
    double prior_pf = 0.0;
    double learned_pf = 0.0;
    double learned_pb = 0.0;
    rg_segment p = {"p", 0, 0, 0};
    rg_segment f = {"f", 0, 0, 0};
    rg_segment b = {"b", 0, 0, 0};
    rg_alignment *learned_alignment = 0;

    (void)seg;
    assert(rg_context_new_builtin(&ctx) == RG_OK);
    rg_train_options_init_defaults(&options);
    pairs[0].source = form("latin", pater, 5);
    pairs[0].target = form("gothic", fadar, 5);
    pairs[0].weight = 1.0;
    pairs[1].source = form("latin", pater, 5);
    pairs[1].target = form("gothic", fadar, 5);
    pairs[1].weight = 0.5;
    assert(rg_train_pairwise(ctx, pairs, 2, &options, &model) == RG_OK);
    assert(model != 0);
    assert(rg_pairwise_model_segment_count_row_count(model) > 0);
    for (i = 0; i < rg_pairwise_model_segment_count_row_count(model); i++) {
        const rg_segment_count_row *row = rg_pairwise_model_segment_count_row_at(model, i);
        assert(row != 0);
        if (strcmp(row->source, "p") == 0 && strcmp(row->target, "f") == 0) {
            found_pf = 1;
            assert(row->count == 1.5);
            assert(row->source_total == 1.5);
            assert_uncertainty_contains(&row->uncertainty, 1.0);
        }
        if (strcmp(row->source, "a") == 0 && strcmp(row->target, "a") == 0) {
            found_aa = 1;
            assert(row->count == 1.5);
            assert(row->source_total == 1.5);
        }
    }
    assert(found_pf);
    assert(found_aa);
    assert(rg_pairwise_model_displacement_row_count(model) > 0);
    for (i = 0; i < rg_pairwise_model_displacement_row_count(model); i++) {
        const rg_displacement_row *row = rg_pairwise_model_displacement_row_at(model, i);
        size_t d;
        assert(row != 0);
        assert(row->count > 0.0);
        assert(row->total >= row->count);
        for (d = 0; d < row->item_count; d++) {
            if (strcmp(row->items[d].from_value, "present") == 0 ||
                strcmp(row->items[d].to_value, "present") == 0) {
                found_disp = 1;
            }
        }
    }
    assert(found_disp);
    assert(rg_score_link(ctx, &p, 1, &f, 1, &prior_pf) == RG_OK);
    assert(rg_score_link_with_model(ctx, model, &options, &p, 1, &f, 1, &learned_pf) == RG_OK);
    assert(rg_score_link_with_model(ctx, model, &options, &p, 1, &b, 1, &learned_pb) == RG_OK);
    assert(learned_pf < learned_pb);
    assert(learned_pf < prior_pf);
    assert(rg_align_forms_with_model(ctx, model, &options, &pairs[0].source, &pairs[0].target, 0, &learned_alignment) == RG_OK);
    assert(learned_alignment != 0);
    assert(rg_alignment_link_count(learned_alignment) == 5);
    assert(strcmp(rg_alignment_link_at(learned_alignment, 0)->context.position, "initial") == 0);
    assert(has_constraint(
        rg_alignment_link_at(learned_alignment, 0)->context.following,
        rg_alignment_link_at(learned_alignment, 0)->context.following_count,
        "vowel",
        "+"
    ));
    assert(strcmp(rg_alignment_link_at(learned_alignment, 2)->context.position, "medial") == 0);
    assert(has_constraint(
        rg_alignment_link_at(learned_alignment, 2)->context.preceding,
        rg_alignment_link_at(learned_alignment, 2)->context.preceding_count,
        "vowel",
        "+"
    ));
    assert(has_distance_constraint(
        rg_alignment_link_at(learned_alignment, 2)->context.preceding_at_distance,
        rg_alignment_link_at(learned_alignment, 2)->context.preceding_at_distance_count,
        2,
        "consonant",
        "+"
    ));
    assert(has_distance_constraint(
        rg_alignment_link_at(learned_alignment, 3)->context.preceding_at_distance,
        rg_alignment_link_at(learned_alignment, 3)->context.preceding_at_distance_count,
        3,
        "consonant",
        "+"
    ));
    assert(has_constraint(
        rg_alignment_link_at(learned_alignment, 1)->context.somewhere_following,
        rg_alignment_link_at(learned_alignment, 1)->context.somewhere_following_count,
        "consonant",
        "+"
    ));
    assert(has_constraint(
        rg_alignment_link_at(learned_alignment, 3)->context.somewhere_preceding,
        rg_alignment_link_at(learned_alignment, 3)->context.somewhere_preceding_count,
        "vowel",
        "+"
    ));
    assert(strcmp(rg_alignment_link_at(learned_alignment, 4)->context.position, "final") == 0);
    rg_alignment_free(learned_alignment);
    learned_alignment = 0;
    assert(rg_pairwise_model_displacement_row_at(model, 1000000) == 0);
    assert(rg_pairwise_model_tonal_count_row_count(model) == 0);
    assert(rg_pairwise_model_tonal_count_row_at(model, 0) == 0);
    assert(rg_pairwise_model_segment_count_row_at(model, 1000000) == 0);
    rg_pairwise_model_free(model);
    model = 0;

    for (i = 0; i < 3; i++) {
        conditioned_pairs[i].source = form("A", pa, 2);
        conditioned_pairs[i].target = form("B", fa, 2);
        conditioned_pairs[i].weight = 1.0;
        conditioned_pairs[i + 3].source = form("A", pi, 2);
        conditioned_pairs[i + 3].target = form("B", pi, 2);
        conditioned_pairs[i + 3].weight = 1.0;
    }
    assert(rg_train_pairwise(ctx, conditioned_pairs, 6, &options, &model) == RG_OK);
    assert(rg_pairwise_model_conditioned_segment_count_row_count(model) > 0);
    for (i = 0; i < rg_pairwise_model_conditioned_segment_count_row_count(model); i++) {
        const rg_conditioned_segment_count_row *row = rg_pairwise_model_conditioned_segment_count_row_at(model, i);
        assert(row != 0);
        if (strcmp(row->source, "p") == 0 &&
            strcmp(row->target, "p") == 0 &&
            has_constraint(row->context.following, row->context.following_count, "close", "+")) {
            found_conditioned = 1;
            assert(row->count == 3.0);
            assert(row->source_total == 6.0);
            assert_uncertainty_contains(&row->uncertainty, 0.5);
        }
    }
    assert(found_conditioned);
    assert(rg_pairwise_model_conditioned_segment_count_row_at(model, 1000000) == 0);
    rg_pairwise_model_free(model);
    model = 0;

    /* Long-range discovery is a separate pass with stricter gates than the
     * immediate one: both sides of a split need at least
     * LongRangeMinSplitObservations mass, the BIC delta must beat -5, and the
     * YES side must be dominated by a single outcome. The medial consonant
     * varies so chunk promotion cannot swallow whole words and erase the
     * one-to-one links the pass reads. */
    {
        static const char *const medials[8] = {"t", "k", "n", "m", "s", "l", "r", "w"};
        static rg_segment lr_source_a[8][3];
        static rg_segment lr_target_a[8][3];
        static rg_segment lr_source_i[8][3];
        static rg_segment lr_target_i[8][3];
        for (i = 0; i < 8; i++) {
            lr_source_a[i][0] = seg("a");
            lr_source_a[i][1] = seg(medials[i]);
            lr_source_a[i][2] = seg("p");
            lr_target_a[i][0] = seg("a");
            lr_target_a[i][1] = seg(medials[i]);
            lr_target_a[i][2] = seg("f");
            lr_source_i[i][0] = seg("i");
            lr_source_i[i][1] = seg(medials[i]);
            lr_source_i[i][2] = seg("p");
            lr_target_i[i][0] = seg("i");
            lr_target_i[i][1] = seg(medials[i]);
            lr_target_i[i][2] = seg("p");
            long_range_pairs[i].source = form("A", lr_source_a[i], 3);
            long_range_pairs[i].target = form("B", lr_target_a[i], 3);
            long_range_pairs[i].weight = 1.0;
            long_range_pairs[i + 8].source = form("A", lr_source_i[i], 3);
            long_range_pairs[i + 8].target = form("B", lr_target_i[i], 3);
            long_range_pairs[i + 8].weight = 1.0;
        }
    }
    assert(rg_train_pairwise(ctx, long_range_pairs, 16, &options, &model) == RG_OK);
    for (i = 0; i < rg_pairwise_model_conditioned_segment_count_row_count(model); i++) {
        const rg_conditioned_segment_count_row *row = rg_pairwise_model_conditioned_segment_count_row_at(model, i);
        if (strcmp(row->source, "p") == 0 &&
            strcmp(row->target, "p") == 0 &&
            has_constraint(row->context.same_syllable, row->context.same_syllable_count, "close", "+")) {
            found_long_range = 1;
            assert(row->count == 8.0);
            assert(row->source_total == 16.0);
        }
    }
    assert(found_long_range);
    rg_pairwise_model_free(model);
    model = 0;

    for (i = 0; i < 10; i++) {
        chunk_pairs[i].source = form("A", ka, 2);
        chunk_pairs[i].target = form("B", ka, 2);
        chunk_pairs[i].weight = 1.0;
        chunk_pairs[i + 10].source = form("A", kt, 2);
        chunk_pairs[i + 10].target = form("B", tt, 2);
        chunk_pairs[i + 10].weight = 1.0;
    }
    assert(rg_train_pairwise(ctx, chunk_pairs, 20, &options, &model) == RG_OK);
    for (i = 0; i < rg_pairwise_model_chunk_row_count(model); i++) {
        const rg_chunk_row *row = rg_pairwise_model_chunk_row_at(model, i);
        double chunk_cost = 0.0;
        assert(row != 0);
        if (row->source_count == 2 &&
            row->target_count == 2 &&
            strcmp(row->source[0].grapheme, "k") == 0 &&
            strcmp(row->source[1].grapheme, "t") == 0 &&
            strcmp(row->target[0].grapheme, "t") == 0 &&
            strcmp(row->target[1].grapheme, "t") == 0) {
            found_chunk = 1;
            assert(row->count == 10.0);
            assert(row->uncertainty.estimate > 0.0);
            assert(rg_score_link_with_model(ctx, model, &options, kt, 2, tt, 2, &chunk_cost) == RG_OK);
            assert(fabs(chunk_cost - row->cost) < 1e-12);
        }
    }
    assert(found_chunk);
    assert(rg_pairwise_model_chunk_row_at(model, 1000000) == 0);
    rg_pairwise_model_free(model);
    model = 0;
    found_chunk = 0;

    for (i = 0; i < 10; i++) {
        boundary_chunk_pairs[i].source = form("A", ka, 2);
        boundary_chunk_pairs[i].target = form("B", ka, 2);
        boundary_chunk_pairs[i].weight = 1.0;
        boundary_chunk_pairs[i + 10].source = form_with_morphemes("A", kt, 2, internal_break, 1);
        boundary_chunk_pairs[i + 10].target = form_with_morphemes("B", tt, 2, internal_break, 1);
        boundary_chunk_pairs[i + 10].weight = 1.0;
    }
    assert(rg_train_pairwise(ctx, boundary_chunk_pairs, 20, &options, &model) == RG_OK);
    for (i = 0; i < rg_pairwise_model_chunk_row_count(model); i++) {
        const rg_chunk_row *row = rg_pairwise_model_chunk_row_at(model, i);
        if (row->source_count == 2 &&
            row->target_count == 2 &&
            strcmp(row->source[0].grapheme, "k") == 0 &&
            strcmp(row->source[1].grapheme, "t") == 0 &&
            strcmp(row->target[0].grapheme, "t") == 0 &&
            strcmp(row->target[1].grapheme, "t") == 0) {
            found_chunk = 1;
        }
    }
    assert(!found_chunk);
    rg_pairwise_model_free(model);
    model = 0;

    tone_pairs[0].source = form("A", tone_a, 1);
    tone_pairs[0].target = form("B", tone_b, 1);
    tone_pairs[0].weight = 2.0;
    assert(rg_train_pairwise(ctx, tone_pairs, 1, &options, &model) == RG_OK);
    assert(rg_pairwise_model_tonal_count_row_count(model) == 1);
    for (i = 0; i < rg_pairwise_model_tonal_count_row_count(model); i++) {
        const rg_tonal_count_row *row = rg_pairwise_model_tonal_count_row_at(model, i);
        assert(row != 0);
        if (strcmp(row->source_tone, "1") == 0 && strcmp(row->target_tone, "2") == 0) {
            found_tone = 1;
            assert(row->count == 2.0);
            assert(row->source_total == 2.0);
            assert_uncertainty_contains(&row->uncertainty, 1.0);
        }
    }
    assert(found_tone);
    rg_pairwise_model_free(model);
    model = 0;

    for (i = 0; i < 3; i++) {
        cross_dim_pairs[i].source = one_segment_form("A", voiced_src);
        cross_dim_pairs[i].target = one_segment_form("B", voiced_high);
        cross_dim_pairs[i].weight = 1.0;
        cross_dim_pairs[i + 3].source = one_segment_form("A", voiceless_src);
        cross_dim_pairs[i + 3].target = one_segment_form("B", voiceless_low);
        cross_dim_pairs[i + 3].weight = 1.0;
    }
    assert(rg_train_pairwise(ctx, cross_dim_pairs, 6, &options, &model) == RG_OK);
    for (i = 0; i < rg_pairwise_model_cross_dimensional_row_count(model); i++) {
        const rg_cross_dimensional_row *row = rg_pairwise_model_cross_dimensional_row_at(model, i);
        assert(row != 0);
        /* The environment is a context now, so the rule is identified by what
         * it names rather than by three parallel strings. */
        if (row->environment.self_count == 1 &&
            strcmp(row->environment.self[0].feature, "voiced") == 0 &&
            strcmp(row->environment.self[0].value, "+") == 0 &&
            strcmp(row->dimension, "tone") == 0 &&
            strcmp(row->value, "H") == 0) {
            found_cross_dimensional = 1;
            assert(row->count == 3.0);
            assert(row->source_count == 3.0);
            assert(row->confidence == 1.0);
            assert_uncertainty_contains(&row->uncertainty, 1.0);
        }
    }
    assert(found_cross_dimensional);
    assert(rg_pairwise_model_cross_dimensional_row_at(model, 1000000) == 0);
    {
        rg_alignment *predicted = 0;
        rg_alignment *contradicted = 0;
        double predicted_cost = 0.0;
        double contradicted_cost = 0.0;
        rg_form source = one_segment_form("A", voiced_src);
        rg_form high = one_segment_form("B", voiced_high);
        rg_form low = one_segment_form("B", voiced_low);
        assert(rg_align_forms_with_model(ctx, model, &options, &source, &high, 0, &predicted) == RG_OK);
        assert(rg_align_forms_with_model(ctx, model, &options, &source, &low, 0, &contradicted) == RG_OK);
        assert(rg_alignment_cost_with_model(ctx, model, &options, predicted, &predicted_cost) == RG_OK);
        assert(rg_alignment_cost_with_model(ctx, model, &options, contradicted, &contradicted_cost) == RG_OK);
        assert(predicted_cost < contradicted_cost);
        rg_alignment_free(predicted);
        rg_alignment_free(contradicted);
    }
    rg_pairwise_model_free(model);
    rg_pairwise_model_free(0);
    assert(rg_score_link_with_model(ctx, 0, &options, &p, 1, &f, 1, &learned_pf) == RG_OK);
    assert(fabs(learned_pf - prior_pf) < 1e-12);
    assert(rg_align_forms_with_model(ctx, 0, &options, &pairs[0].source, &pairs[0].target, 0, &learned_alignment) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_train_pairwise(ctx, 0, 1, &options, &model) == RG_ERR_INVALID_ARGUMENT);
    test_joint_cross_dimensional_rule(ctx);
    test_lect_internal_tonogenesis(ctx);
    test_cross_dimensional_identifiability(ctx);
    test_cross_dimensional_dimension_target(
        ctx, REGULAE_SOURCE_DIR "/testdata/corpora/stress_dimension_target.tsv", "stress");
    test_cross_dimensional_dimension_target(
        ctx, REGULAE_SOURCE_DIR "/testdata/corpora/length_dimension_target.tsv", "length");
    test_uninformative_environments_commit_nothing(ctx);
    test_gap_correspondence(ctx);
    rg_context_free(ctx);
    return 0;
}
