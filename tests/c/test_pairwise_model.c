#include "regulae.h"

#include <assert.h>
#include <math.h>
#include <string.h>

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
    rg_segment atp[] = {{"a", 0, 0, 0}, {"t", 0, 0, 0}, {"p", 0, 0, 0}};
    rg_segment atf[] = {{"a", 0, 0, 0}, {"t", 0, 0, 0}, {"f", 0, 0, 0}};
    rg_segment itp[] = {{"i", 0, 0, 0}, {"t", 0, 0, 0}, {"p", 0, 0, 0}};
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
        if (strcmp(row->source_feature, "voiced") == 0 &&
            strcmp(row->source_value, "+") == 0 &&
            strcmp(row->source_position, "relative_0") == 0 &&
            strcmp(row->target_dimension, "tone") == 0 &&
            strcmp(row->target_value, "H") == 0) {
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
    rg_context_free(ctx);
    return 0;
}
