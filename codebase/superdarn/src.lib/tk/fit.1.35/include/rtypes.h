#ifndef RTYPES_H
#define RTYPES_H

#include <stdint.h>  /* Standard integer types */
#include <time.h>    /* For time_t */

/* Basic type definitions */
typedef unsigned char uint8;
typedef short int16;
typedef unsigned short uint16;
typedef int int32;
typedef unsigned int uint32;
typedef float float32;
typedef double float64;

/* Time structures */
typedef struct {
    int16 year;    /* Year (e.g., 2023) */
    int16 month;   /* Month (1-12) */
    int16 day;     /* Day of month (1-31) */
    int16 hour;    /* Hour (0-23) */
    int16 minute;  /* Minute (0-59) */
    int16 second;  /* Second (0-59) */
    int32 usec;    /* Microseconds (0-999999) */
} DateTime;

typedef struct {
    time_t sc;  /* Seconds since epoch */
    int32 us;   /* Microseconds */
} Time;

/* Forward declarations */
struct DataMap;  /* Defined in dmap.h */
typedef struct DataMap DataMap;

/* Forward declaration of RadarParm structure (defined in rprm.h) */
typedef struct RadarParm RadarParm;

/* Forward declaration of CFitdata structure (defined in cfitdata.h) */
struct CFitdata;
typedef struct CFitdata CFitdata;

/* Function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

/* Time conversion functions */
double TimeYMDHMSToEpoch(int16 year, int16 month, int16 day, 
                        int16 hour, int16 minute, int16 second, int32 usec);
void TimeEpochToYMDHMS(double epoch, int16 *year, int16 *month, int16 *day,
                      int16 *hour, int16 *minute, int16 *second, int32 *usec);
void TimeCurrentYMDHMS(int16 *year, int16 *month, int16 *day,
                      int16 *hour, int16 *minute, int16 *second, int32 *usec);
double TimeYMDHMSToEpochDT(const DateTime *dt);
void TimeEpochToYMDHMSDT(double epoch, DateTime *dt);

/* DataMap functions */
DataMap *DataMapMake(void);
void DataMapFree(DataMap *ptr);
int DataMapRead(int fid, DataMap **ptr);
int DataMapWrite(int fid, const DataMap *ptr);
size_t DataMapSize(const DataMap *ptr);
int DataMapAddScalar(DataMap *ptr, const char *name, int type, const void *data);
int DataMapAddArray(DataMap *ptr, const char *name, int type, int dim, const int *rng, const void *data);
int DataMapAddString(DataMap *ptr, const char *name, const char *str);
void *DataMapGetScalar(const DataMap *ptr, const char *name, int *type);
void *DataMapGetArray(const DataMap *ptr, const char *name, int *type, int *dim, int *rng);
char *DataMapGetString(const DataMap *ptr, const char *name);
int DataMapCopy(DataMap *dest, const DataMap *src);
void DataMapClear(DataMap *ptr);
int DataMapMerge(DataMap *dest, const DataMap *src);
int DataMapStoreScalar(DataMap *ptr, const char *name, int type, void *value);
int DataMapStoreArray(DataMap *ptr, const char *name, int type, int dim, int *rng, void *data);

/* Radar parameter functions */
int RadarParmDecode(DataMap *ptr, void *prm);
int RadarParmEncode(DataMap *ptr, void *prm);

/* CFit functions */
CFitdata *CFitMake(void);
void CFitFree(CFitdata *ptr);
int CFitSetRng(CFitdata *ptr, int nrang);

#ifdef __cplusplus
}
#endif

#endif /* RTYPES_H */
