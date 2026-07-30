#include "regulae.h"

#include <stdio.h>
#include <string.h>

static int usage(void) {
    printf("Usage: regulae <command>\n");
    printf("\n");
    printf("Commands:\n");
    printf("  version   print version\n");
    printf("  help      print this help\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "help") == 0 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        return usage();
    }
    if (strcmp(argv[1], "version") == 0 || strcmp(argv[1], "--version") == 0) {
        printf("regulae %s\n", rg_version_string());
        return 0;
    }
    fprintf(stderr, "regulae: unknown command: %s\n", argv[1]);
    usage();
    return 2;
}
