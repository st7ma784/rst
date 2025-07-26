#ifndef DATAMAP_ARRAY_H
#define DATAMAP_ARRAY_H

#include <stdint.h>

/* Data types for mapping */
#define DATACHAR     1
#define DATASHORT    2
#define DATAINT      3
#define DATAFLOAT    4
#define DATADOUBLE   5
#define DATASTRING   6

/* DataMapScalar structure */
typedef struct DataMapScalar {
    char name[32];     /* Name of the scalar */
    int type;          /* Data type (e.g., DATAFLOAT, DATAINT, etc.) */
    union {
        float *fptr;   /* Pointer to float data */
        int *iptr;     /* Pointer to int data */
        void *vptr;    /* Generic pointer */
        char *cptr;    /* Pointer to char data */
        short *sptr;   /* Pointer to short data */
    } data;            /* Union of data pointers */
} DataMapScalar;

/* DataMapArray structure */
typedef struct DataMapArray {
    char name[32];     /* Name of the array */
    int type;          /* Data type (e.g., DATAFLOAT, DATAINT, etc.) */
    int dim;           /* Dimension of the array (1 or 2) */
    int rng[2];        /* Range for each dimension */
    union {
        float *fptr;   /* Pointer to float data */
        int *iptr;     /* Pointer to int data */
        void *vptr;    /* Generic pointer */
        char *cptr;    /* Pointer to char data */
        short *sptr;   /* Pointer to short data */
    } data;            /* Union of data pointers */
} DataMapArray;

/* DataMap structure */
typedef struct DataMap {
    DataMapScalar **scl;  /* Array of scalar pointers */
    DataMapArray **arr;   /* Array of array pointers */
    int snum;             /* Number of scalars */
    int anum;             /* Number of arrays */
} DataMap;

/* Function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

/* DataMap structure (simplified) */
typedef struct DataMap DataMap;

/* Function stubs */
int DataMapRead(int fid, DataMap **ptr);
void DataMapFree(DataMap *ptr);
int RadarParmDecode(DataMap *ptr, void *prm);

#ifdef __cplusplus
}
#endif

#endif /* DATAMAP_ARRAY_H */
