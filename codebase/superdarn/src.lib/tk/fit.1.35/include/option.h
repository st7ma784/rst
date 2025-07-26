#ifndef OPTION_H
#define OPTION_H

/* 
 * Stub implementation of option.h for fit.1.35
 * This is a minimal implementation to allow the code to compile
 */

typedef struct {
    char *name;     /* Name of the option */
    int type;       /* Type of the option */
    void *ptr;      /* Pointer to the option value */
    int status;     /* Status of the option */
    char *text;     /* Help text */
} OptionData;

/* Function prototypes used by fitscan.c */
int OptionAdd(OptionData *opt, const char *name, int type, void *ptr, const char *text);
int OptionProcess(int argc, char *argv[], OptionData *opt, int optstr);
void OptionPrintInfo(FILE *fp, OptionData *opt);
void OptionFree(OptionData *opt);

/* Option types */
#define OPTION_END 0
#define OPTION_STRING 1
#define OPTION_INT 2
#define OPTION_FLOAT 3
#define OPTION_BOOL 4

#endif /* OPTION_H */
