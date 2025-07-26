#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/option.h"

/* 
 * Stub implementations of option functions for fit.1.35
 * These are minimal implementations to allow the code to compile
 */

int OptionAdd(OptionData *opt, const char *name, int type, void *ptr, const char *text) {
    // For now, just return success without doing anything
    (void)opt; (void)name; (void)type; (void)ptr; (void)text;
    return 0;
}

int OptionProcess(int argc, char *argv[], OptionData *opt, int optstr) {
    // For now, just return success without doing anything
    (void)argc; (void)argv; (void)opt; (void)optstr;
    return 0;
}

void OptionPrintInfo(FILE *fp, OptionData *opt) {
    // Do nothing
    (void)fp; (void)opt;
}

void OptionFree(OptionData *opt) {
    // Do nothing
    (void)opt;
}
