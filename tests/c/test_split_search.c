#include "split_search.h"

#include <assert.h>
#include <math.h>
#include <string.h>

static const rg_context_spec inside = {
    .position = "initial"
};

static const rg_context_spec outside = {
    .position = "medial"
};

static double find_delta(
    const rg_split_observation *rows,
    size_t count,
    double log_sample_size
) {
    const rg_split_candidate candidate = {"position", "initial", 0};
    const rg_split_gate gate = {0.0, -1.0, 0.0};
    const rg_split_score_config score = {
        RG_SPLIT_SCORER_CORRECTED_BIC, 1.0, log_sample_size, 0.0, 1, 0.0,
        RG_CLASS_OUTCOME_SISTER_TUPLE, 0, 0
    };
    rg_split_search search;
    rg_split_result result;
    int found = 0;

    assert(rg_split_search_init(&search, count) == RG_OK);
    assert(rg_split_find_best(&search, rows, count, &candidate, &gate, 1,
                              &score, &result, &found) == RG_OK);
    assert(found);
    rg_split_search_clear(&search);
    return result.delta_score;
}

static void test_binary_split_adds_one_parameter(void) {
    const rg_split_observation rows[] = {
        {&inside, "a", 1.0, 0},
        {&inside, "a", 1.0, 0},
        {&inside, "a", 1.0, 0},
        {&inside, "a", 1.0, 0},
        {&outside, "b", 1.0, 0},
        {&outside, "b", 1.0, 0},
        {&outside, "b", 1.0, 0},
        {&outside, "b", 1.0, 0}
    };
    double log_n = log(8.0);
    double expected = -2.0 * rg_split_group_cost(rows, 8) + log_n;

    assert(fabs(find_delta(rows, 8, log_n) - expected) < 1e-12);
}

static void test_three_outcomes_add_two_parameters(void) {
    const rg_split_observation rows[] = {
        {&inside, "a", 1.0, 0},
        {&inside, "a", 1.0, 0},
        {&inside, "a", 1.0, 0},
        {&inside, "a", 1.0, 0},
        {&outside, "b", 1.0, 0},
        {&outside, "b", 1.0, 0},
        {&outside, "c", 1.0, 0},
        {&outside, "c", 1.0, 0}
    };
    double log_n = log(8.0);
    double split_cost = 4.0 * log(2.0);
    double expected = -2.0 * (rg_split_group_cost(rows, 8) - split_cost) + 2.0 * log_n;

    assert(fabs(find_delta(rows, 8, log_n) - expected) < 1e-12);
}

static void test_fractional_mass_does_not_change_the_outcome_dimension(void) {
    const rg_split_observation rows[] = {
        {&inside, "a", 0.5, 0},
        {&inside, "a", 0.5, 0},
        {&inside, "a", 0.5, 0},
        {&inside, "a", 0.5, 0},
        {&outside, "b", 0.5, 0},
        {&outside, "b", 0.5, 0},
        {&outside, "c", 0.5, 0},
        {&outside, "c", 0.5, 0}
    };
    double log_n = log(8.0);
    double split_cost = 2.0 * log(2.0);
    double expected = -2.0 * (rg_split_group_cost(rows, 8) - split_cost) + 2.0 * log_n;

    assert(fabs(find_delta(rows, 8, log_n) - expected) < 1e-12);
}

static void test_equivalent_split_encodings_do_not_increase_search_charge(void) {
    const rg_split_observation rows[] = {
        {&inside, "a", 1.0, 0}, {&inside, "a", 1.0, 0},
        {&inside, "a", 1.0, 0}, {&inside, "a", 1.0, 0},
        {&outside, "b", 1.0, 0}, {&outside, "b", 1.0, 0},
        {&outside, "b", 1.0, 0}, {&outside, "b", 1.0, 0}
    };
    const rg_split_candidate one[] = {
        {"position", "initial", 0}
    };
    const rg_split_candidate three[] = {
        {"position", "initial", 0}, {"position", "medial", 0},
        {"position", "initial", 0}
    };
    const rg_split_gate gates[] = {
        {0.0, -1.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, -1.0, 0.0}
    };
    const rg_split_score_config score = {
        RG_SPLIT_SCORER_CORRECTED_BIC, 1.0, log(8.0), 0.0, 0, 1.0,
        RG_CLASS_OUTCOME_SISTER_TUPLE, 0, 0
    };
    rg_split_search search;
    rg_split_result a;
    rg_split_result b;
    int found = 0;

    assert(rg_split_search_init(&search, 8) == RG_OK);
    assert(rg_split_find_best(&search, rows, 8, one, gates, 1,
                              &score, &a, &found) == RG_OK);
    assert(found);
    found = 0;
    assert(rg_split_find_best(&search, rows, 8, three, gates, 3,
                              &score, &b, &found) == RG_OK);
    assert(found);
    assert(fabs(a.delta_score - b.delta_score) < 1e-12);
    assert(fabs(a.search_margin - b.search_margin) < 1e-12);
    rg_split_search_clear(&search);
}

static void test_tie_ranking_prefers_morphological_over_feature(void) {
    static const rg_feature_constraint voiced_plus = {"voiced", "+"};
    static const rg_context_spec morph_and_voiced = {
        .morphological = "initial",
        .following = &voiced_plus,
        .following_count = 1
    };
    static const rg_context_spec neither = {0};
    const rg_split_observation rows[] = {
        {&morph_and_voiced, "a", 1.0, 0},
        {&morph_and_voiced, "a", 1.0, 0},
        {&morph_and_voiced, "a", 1.0, 0},
        {&morph_and_voiced, "a", 1.0, 0},
        {&neither, "b", 1.0, 0},
        {&neither, "b", 1.0, 0},
        {&neither, "b", 1.0, 0},
        {&neither, "b", 1.0, 0}
    };
    const rg_split_candidate candidates[] = {
        {"following", "voiced", "+"},
        {"morphological", "initial", 0}
    };
    const rg_split_gate gates[] = {
        {0.0, -1.0, 0.0},
        {0.0, -1.0, 0.0}
    };
    const rg_split_score_config score = {
        RG_SPLIT_SCORER_CORRECTED_BIC, 1.0, log(8.0), 0.0, 0, 1.0,
        RG_CLASS_OUTCOME_SISTER_TUPLE, 0, 0
    };
    rg_split_search search;
    rg_split_result result;
    int found = 0;

    assert(rg_split_search_init(&search, 8) == RG_OK);
    assert(rg_split_find_best(&search, rows, 8, candidates, gates, 2,
                              &score, &result, &found) == RG_OK);
    assert(found);
    assert(strcmp(result.candidate.slot, "morphological") == 0);
    assert(strcmp(result.candidate.feature, "initial") == 0);
    rg_split_search_clear(&search);
}

int main(void) {
    test_binary_split_adds_one_parameter();
    test_three_outcomes_add_two_parameters();
    test_fractional_mass_does_not_change_the_outcome_dimension();
    test_equivalent_split_encodings_do_not_increase_search_charge();
    test_tie_ranking_prefers_morphological_over_feature();
    return 0;
}
