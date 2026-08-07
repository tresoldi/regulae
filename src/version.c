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
