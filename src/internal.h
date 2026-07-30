#ifndef REGULAE_INTERNAL_H
#define REGULAE_INTERNAL_H

#include "regulae.h"

#include <stddef.h>

#define RG_DEFAULT_GAP_COST 0.5
#define RG_DEFAULT_CHUNK_PENALTY 0.25
#define RG_CHUNK_COMPLEXITY_PENALTY 1e-9

char *rg_strdup_internal(const char *value);
void rg_segment_clear_internal(rg_segment *segment);
rg_status rg_segment_copy_internal(const rg_segment *src, rg_segment *out);
void rg_link_clear_internal(rg_link *link);
rg_status rg_link_copy_chunks_internal(
    rg_link *link,
    const rg_segment *source,
    size_t source_count,
    const rg_segment *target,
    size_t target_count
);

#endif
