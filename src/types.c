#include "internal.h"

#include <string.h>

int rg_link_is_reordering_internal(
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count,
    size_t *pairing
) {
    size_t taken[RG_MAX_REORDER_SPAN];
    size_t i;
    int moved = 0;

    if (source_count != target_count || source_count < 2 || source_count > RG_MAX_REORDER_SPAN) {
        return 0;
    }
    for (i = 0; i < source_count; i++) {
        taken[i] = 0;
    }
    /* Greedy left-to-right matching on the grapheme. Ambiguity only arises
     * when a grapheme repeats, and then any consistent choice describes the
     * same reordering. */
    for (i = 0; i < source_count; i++) {
        size_t j;
        int found = 0;
        for (j = 0; j < target_count; j++) {
            if (taken[j]) {
                continue;
            }
            if (strcmp(source[i].grapheme == 0 ? "" : source[i].grapheme,
                       target[j].grapheme == 0 ? "" : target[j].grapheme) == 0) {
                taken[j] = 1;
                pairing[i] = j;
                if (j != i) {
                    moved = 1;
                }
                found = 1;
                break;
            }
        }
        if (!found) {
            return 0;
        }
    }
    return moved;
}
