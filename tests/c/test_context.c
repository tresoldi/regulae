#include "regulae.h"

#include <assert.h>

int main(void) {
    rg_context_spec empty;
    rg_context_spec table;
    rg_context_spec link;
    rg_context_spec mismatch;
    rg_feature_constraint preceding[] = {{"vowel", "present"}};
    rg_feature_constraint following[] = {{"front", "present"}};
    rg_feature_constraint somewhere[] = {{"voiced", "present"}};
    rg_distance_constraint distance[] = {{2, {"back", "present"}}};
    int ok = 0;

    rg_context_spec_init_empty(&empty);
    assert(rg_context_spec_constraint_count(&empty) == 0);

    rg_context_spec_init_empty(&table);
    table.position = "medial";
    table.preceding = preceding;
    table.preceding_count = 1;
    table.following = following;
    table.following_count = 1;
    table.morphological = "stem";
    table.preceding_at_distance = distance;
    table.preceding_at_distance_count = 1;

    rg_context_spec_init_empty(&link);
    link.position = "medial";
    link.preceding = preceding;
    link.preceding_count = 1;
    link.following = following;
    link.following_count = 1;
    link.morphological = "stem";
    link.preceding_at_distance = distance;
    link.preceding_at_distance_count = 1;
    link.somewhere_preceding = somewhere;
    link.somewhere_preceding_count = 1;

    assert(rg_context_spec_constraint_count(&table) == 5);
    assert(rg_context_spec_constraint_count(&link) == 6);
    assert(rg_context_spec_is_subset(&table, &link, &ok) == RG_OK);
    assert(ok == 1);
    assert(rg_context_spec_is_subset(&empty, &link, &ok) == RG_OK);
    assert(ok == 1);
    assert(rg_context_spec_is_subset(&link, &table, &ok) == RG_OK);
    assert(ok == 0);

    rg_context_spec_init_empty(&mismatch);
    mismatch.position = "final";
    mismatch.preceding = preceding;
    mismatch.preceding_count = 1;
    assert(rg_context_spec_is_subset(&mismatch, &link, &ok) == RG_OK);
    assert(ok == 0);

    assert(rg_context_spec_is_subset(0, &link, &ok) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_context_spec_is_subset(&table, 0, &ok) == RG_ERR_INVALID_ARGUMENT);
    assert(rg_context_spec_is_subset(&table, &link, 0) == RG_ERR_INVALID_ARGUMENT);
    rg_context_spec_init_empty(0);
    return 0;
}
