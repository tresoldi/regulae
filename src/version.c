#include "regulae.h"

#include <stdlib.h>

const char *rg_version_string(void) {
    return RG_VERSION_STRING;
}

int rg_version_major(void) {
    return RG_VERSION_MAJOR;
}

int rg_version_minor(void) {
    return RG_VERSION_MINOR;
}

int rg_version_patch(void) {
    return RG_VERSION_PATCH;
}

uint32_t rg_abi_version(void) {
    return RG_ABI_VERSION;
}

const char *rg_split_scorer_string(rg_split_scorer scorer) {
    switch (scorer) {
    case RG_SPLIT_SCORER_CORRECTED_BIC:
        return "corrected_bic";
    case RG_SPLIT_SCORER_MULTINOMIAL_NML:
        return "multinomial_nml";
    case RG_SPLIT_SCORER_DIRICHLET_MARGINAL:
        return "dirichlet_marginal";
    default:
        return "unknown";
    }
}

const char *rg_observation_unit_string(rg_observation_unit unit) {
    switch (unit) {
    case RG_OBSERVATION_UNIT_AUTO:
        return "auto";
    case RG_OBSERVATION_UNIT_COGNATE_SET:
        return "cognate_set";
    case RG_OBSERVATION_UNIT_ETYMON_GROUP:
        return "etymon_group";
    case RG_OBSERVATION_UNIT_SOURCE_GROUP:
        return "source_group";
    case RG_OBSERVATION_UNIT_ALIGNED_SPAN:
        return "aligned_span";
    case RG_OBSERVATION_UNIT_ALIGNED_POSITION:
        return "aligned_position";
    default:
        return "unknown";
    }
}

const char *rg_status_string(rg_status status) {
    switch (status) {
    case RG_OK:
        return "ok";
    case RG_ERR_INVALID_ARGUMENT:
        return "invalid argument";
    case RG_ERR_IO:
        return "io error";
    case RG_ERR_PARSE:
        return "parse error";
    case RG_ERR_MERKMAL:
        return "merkmal error";
    case RG_ERR_UNKNOWN_GRAPHEME:
        return "unknown grapheme";
    case RG_ERR_SOURCE_MARKER:
        return "source marker, not a sound";
    case RG_ERR_UNSUPPORTED_OPTION:
        return "unsupported option";
    case RG_ERR_CANCELLED:
        return "cancelled";
    case RG_ERR_OOM:
        return "out of memory";
    default:
        return "unknown status";
    }
}

void rg_string_free(char *value) {
    free(value);
}
