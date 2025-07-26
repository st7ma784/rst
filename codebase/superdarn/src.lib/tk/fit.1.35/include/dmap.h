#ifndef DMAP_H
#define DMAP_H

#include <stddef.h>
#include "rtypes.h"
#include "datamap_array.h"

/* Use the DataMap definition from datamap_array.h */
typedef DataMap DataMap;
typedef DataMapScalar DataMapScalar;
typedef DataMapArray DataMapArray;

/* Use the data type macros from datamap_array.h */
#ifndef DMAP_CHAR
#define DMAP_CHAR     DATACHAR
#endif

#ifndef DMAP_SHORT
#define DMAP_SHORT    DATASHORT
#endif

#ifndef DMAP_INT
#define DMAP_INT      DATAINT
#endif

#ifndef DMAP_FLOAT
#define DMAP_FLOAT    DATAFLOAT
#endif

#ifndef DMAP_DOUBLE
#define DMAP_DOUBLE   DATADOUBLE
#endif

#ifndef DMAP_STRING
#define DMAP_STRING   DATASTRING
#endif

/* For backward compatibility - use the macros from datamap_array.h */
#ifndef DATACHAR
#define DATACHAR    1
#endif

#ifndef DATASHORT
#define DATASHORT   2
#endif

#ifndef DATAINT
#define DATAINT     3
#endif

#ifndef DATAFLOAT
#define DATAFLOAT   4
#endif

#ifndef DATADOUBLE
#define DATADOUBLE  5
#endif

#ifndef DATASTRING
#define DATASTRING  6
#endif

/* Function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

/* Core DataMap functions */
DataMap *DataMapMake(void);
void DataMapFree(DataMap *ptr);
int DataMapRead(int fid, DataMap **ptr);
int DataMapWrite(int fid, const DataMap *ptr);
size_t DataMapSize(const DataMap *ptr);

/* Data manipulation functions */
int DataMapAddScalar(DataMap *ptr, const char *name, int type, const void *data);
int DataMapAddArray(DataMap *ptr, const char *name, int type, int dim, const int *rng, const void *data);
int DataMapAddString(DataMap *ptr, const char *name, const char *str);

/* Data retrieval functions */
void *DataMapGetScalar(const DataMap *ptr, const char *name, int *type);
void *DataMapGetArray(const DataMap *ptr, const char *name, int *type, int *dim, int *rng);
char *DataMapGetString(const DataMap *ptr, const char *name);

/* Utility functions */
int DataMapCopy(DataMap *dest, const DataMap *src);
void DataMapClear(DataMap *ptr);
int DataMapMerge(DataMap *dest, const DataMap *src);

/* Helper functions for specific data types */
int DataMapStoreScalar(DataMap *ptr, const char *name, int type, void *value);
int DataMapStoreArray(DataMap *ptr, const char *name, int type, int dim, int *rng, void *data);

/* Radar parameter functions */
int RadarParmDecode(DataMap *ptr, void *prm);
int RadarParmEncode(DataMap *ptr, void *prm);

#ifdef __cplusplus
}
#endif

#endif /* DMAP_H */
