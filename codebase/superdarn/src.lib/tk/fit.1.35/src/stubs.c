#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#include "fitblk.h"
#include "rtypes.h"
#include "dmap.h"
#include "rprm.h"
#include "rtime.h"
#include "rconvert.h"
#include "cfitdata.h"
#include "fitblk.h"

/* Data type constants for DataMap */
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

/* Include the DataMap definition */
#include "datamap_array.h"

/* Global variables for tracking allocated maps */
static DataMap *allocated_maps[100];
static int num_allocated_maps = 0;

/* Implementation of DataMapMake */
DataMap *DataMapMake(void) {
    DataMap *map = (DataMap *)calloc(1, sizeof(DataMap));
    if (!map) return NULL;
    
    /* Initialize the DataMap structure */
    map->scl = NULL;
    map->arr = NULL;
    map->snum = 0;
    map->anum = 0;
    
    /* Add to allocated maps list */
    if (num_allocated_maps < 100) {
        allocated_maps[num_allocated_maps++] = map;
    }
    
    return map;
}

void DataMapFree(DataMap *ptr) {
    if (!ptr) return;
    
    /* Free all scalars */
    for (int i = 0; i < ptr->snum; i++) {
        if (ptr->scl[i]) {
            if (ptr->scl[i]->data.vptr) {
                free(ptr->scl[i]->data.vptr);
            }
            free(ptr->scl[i]);
        }
    }
    if (ptr->scl) {
        free(ptr->scl);
    }
    
    /* Free all arrays */
    for (int i = 0; i < ptr->anum; i++) {
        if (ptr->arr[i]) {
            if (ptr->arr[i]->data.vptr) {
                free(ptr->arr[i]->data.vptr);
            }
            free(ptr->arr[i]);
        }
    }
    if (ptr->arr) {
        free(ptr->arr);
    }
    
    /* Remove from allocated maps list */
    for (int i = 0; i < num_allocated_maps; i++) {
        if (allocated_maps[i] == ptr) {
            for (int j = i; j < num_allocated_maps - 1; j++) {
                allocated_maps[j] = allocated_maps[j + 1];
            }
            num_allocated_maps--;
            break;
        }
    }
    
    free(ptr);
}

/* Implementation of DataMapWrite */
int DataMapWrite(int fid, const DataMap *ptr) {
    // Stub implementation - just return success
    (void)fid;
    (void)ptr;
    return 0;
}

size_t DataMapSize(const DataMap *ptr) {
    if (!ptr) return 0;
    
    size_t size = 0;
    
    /* Calculate size of scalars */
    for (int i = 0; i < ptr->snum; i++) {
        if (ptr->scl[i]) {
            size += sizeof(DataMapScalar);
            if (ptr->scl[i]->data.vptr) {
                size_t data_size = 0;
                switch (ptr->scl[i]->type) {
                    case DATACHAR:   data_size = sizeof(char); break;
                    case DATASHORT:  data_size = sizeof(short); break;
                    case DATAINT:    data_size = sizeof(int); break;
                    case DATAFLOAT:  data_size = sizeof(float); break;
                    case DATADOUBLE: data_size = sizeof(double); break;
                    case DATASTRING: data_size = strlen((char *)ptr->scl[i]->data.vptr) + 1; break;
                    default: break;
                }
                size += data_size;
            }
        }
    }
    
    /* Calculate size of arrays */
    for (int i = 0; i < ptr->anum; i++) {
        if (ptr->arr[i]) {
            size += sizeof(DataMapArray);
            if (ptr->arr[i]->data.vptr) {
                size_t data_size = 1;
                for (int j = 0; j < ptr->arr[i]->dim; j++) {
                    data_size *= ptr->arr[i]->rng[j];
                }
                switch (ptr->arr[i]->type) {
                    case DATACHAR:   data_size *= sizeof(char); break;
                    case DATASHORT:  data_size *= sizeof(short); break;
                    case DATAINT:    data_size *= sizeof(int); break;
                    case DATAFLOAT:  data_size *= sizeof(float); break;
                    case DATADOUBLE: data_size *= sizeof(double); break;
                    default: break;
                }
                size += data_size;
            }
        }
    }
    
    return size;
}

/* Implementation of DataMapStoreArray */
int DataMapStoreArray(DataMap *ptr, const char *name, int type,
                     int dim, int *rng, void *data)
{
    if (!ptr || !name || dim <= 0 || !rng || !data) return -1;
    
    /* Calculate data size */
    size_t size = 1;
    for (int i = 0; i < dim; i++) {
        size *= rng[i];
    }
    
    switch (type) {
        case DATACHAR:   size *= sizeof(char); break;
        case DATASHORT:  size *= sizeof(short); break;
        case DATAINT:    size *= sizeof(int); break;
        case DATAFLOAT:  size *= sizeof(float); break;
        case DATADOUBLE: size *= sizeof(double); break;
        default: return -1;
    }
    
    /* Create a new DataMapArray */
    DataMapArray *arr = (DataMapArray *)calloc(1, sizeof(DataMapArray));
    if (!arr) return -1;
    
    /* Initialize the array */
    strncpy(arr->name, name, sizeof(arr->name) - 1);
    arr->name[sizeof(arr->name) - 1] = '\0';
    arr->type = type;
    arr->dim = dim;
    memcpy(arr->rng, rng, dim * sizeof(int));
    
    /* Allocate and copy the data */
    arr->data.vptr = malloc(size);
    if (!arr->data.vptr) {
        free(arr);
        return -1;
    }
    memcpy(arr->data.vptr, data, size);
    
    /* Add to the DataMap */
    if (!ptr->arr) {
        ptr->arr = (DataMapArray **)malloc(sizeof(DataMapArray *));
        if (!ptr->arr) {
            free(arr->data.vptr);
            free(arr);
            return -1;
        }
        ptr->anum = 0;
    } else {
        DataMapArray **new_arr = (DataMapArray **)realloc(ptr->arr, 
            (ptr->anum + 1) * sizeof(DataMapArray *));
        if (!new_arr) {
            free(arr->data.vptr);
            free(arr);
            return -1;
        }
        ptr->arr = new_arr;
    }
    
    ptr->arr[ptr->anum++] = arr;
    return 0;
}

/* Implementation of DataMapStoreScalar */
int DataMapStoreScalar(DataMap *ptr, const char *name, int type, void *value)
{
    if (!ptr || !name || !value) return -1;
    
    size_t size = 0;
    
    switch (type) {
        case DATACHAR:   size = sizeof(char); break;
        case DATASHORT:  size = sizeof(short); break;
        case DATAINT:    size = sizeof(int); break;
        case DATAFLOAT:  size = sizeof(float); break;
        case DATADOUBLE: size = sizeof(double); break;
        case DATASTRING: size = strlen((char *)value) + 1; break;
        default: return -1;
    }
    
    /* Create a new DataMapScalar */
    DataMapScalar *scl = (DataMapScalar *)calloc(1, sizeof(DataMapScalar));
    if (!scl) return -1;
    
    /* Initialize the scalar */
    strncpy(scl->name, name, sizeof(scl->name) - 1);
    scl->name[sizeof(scl->name) - 1] = '\0';
    scl->type = type;
    
    /* Allocate and copy the data */
    scl->data.vptr = malloc(size);
    if (!scl->data.vptr) {
        free(scl);
        return -1;
    }
    memcpy(scl->data.vptr, value, size);
    
    /* Add to the DataMap */
    if (!ptr->scl) {
        ptr->scl = (DataMapScalar **)malloc(sizeof(DataMapScalar *));
        if (!ptr->scl) {
            free(scl->data.vptr);
            free(scl);
            return -1;
        }
        ptr->snum = 0;
    } else {
        DataMapScalar **new_scl = (DataMapScalar **)realloc(ptr->scl, 
            (ptr->snum + 1) * sizeof(DataMapScalar *));
        if (!new_scl) {
            free(scl->data.vptr);
            free(scl);
            return -1;
        }
        ptr->scl = new_scl;
    }
    
    ptr->scl[ptr->snum++] = scl;
    return 0;
}

int DataMapAddScalar(DataMap *ptr, const char *name, int type, const void *value) {
    return DataMapStoreScalar(ptr, name, type, (void *)value);
}

/* Implementation of RadarParmMake */
RadarParm *RadarParmMake(void) {
    // Allocate and initialize a new RadarParm structure
    RadarParm *prm = (RadarParm *)calloc(1, sizeof(RadarParm));
    if (prm == NULL) {
        return NULL;
    }
    
    // Initialize the revision structure
    prm->revision.major = 1;
    prm->revision.minor = 0;
    
    // Initialize time to current time if needed
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    
    prm->time.yr = tm_info->tm_year + 1900;
    prm->time.mo = tm_info->tm_mon + 1;
    prm->time.dy = tm_info->tm_mday;
    prm->time.hr = tm_info->tm_hour;
    prm->time.mt = tm_info->tm_min;
    prm->time.sc = tm_info->tm_sec;
    prm->time.us = 0;  // No microseconds available from time()
    
    // Initialize other fields to default values
    prm->nrang = 0;
    prm->num = 0;
    prm->rng = NULL;
    
    return prm;
}

/* Implementation of RadarParmFree */
void RadarParmFree(RadarParm *prm) {
    if (prm == NULL) {
        return;
    }
    
    // Free the range gate array if it was allocated
    if (prm->rng != NULL) {
        // Get the guard word before the range array
        uint64_t *guard_before = (uint64_t *)((char *)prm->rng - sizeof(uint64_t));
        
        // Get the original allocation pointer stored before the guard word
        void **original_alloc_ptr = (void **)((char *)guard_before - sizeof(void *));
        
        // Calculate the end of the range array to find the guard word after
        size_t alloc_size = prm->nrang * sizeof(int16_t);
        uint64_t *guard_after = (uint64_t *)((char *)prm->rng + alloc_size);
        
        // Check guard words for corruption
        int corruption_detected = 0;
        if (*guard_before != 0xDEADBEEFDEADBEEF) {
            printf("WARNING: Memory corruption detected in RadarParm range array (before guard)!\n");
            printf("Guard before: 0x%016lX, expected 0xDEADBEEFDEADBEEF\n", *guard_before);
            corruption_detected = 1;
        }
        
        if (*guard_after != 0xDEADBEEFDEADBEEF) {
            printf("WARNING: Memory corruption detected in RadarParm range array (after guard)!\n");
            printf("Guard after: 0x%016lX, expected 0xDEADBEEFDEADBEEF\n", *guard_after);
            corruption_detected = 1;
        }
        
        if (corruption_detected) {
            printf("Range array allocation details:\n");
            printf("  Original pointer: %p\n", *original_alloc_ptr);
            printf("  Range array: %p\n", prm->rng);
            printf("  Guard before: %p\n", (void*)guard_before);
            printf("  Guard after: %p\n", (void*)guard_after);
            printf("  Allocation size: %zu bytes\n", alloc_size);
        }
        
        // Free the original allocation
        free(*original_alloc_ptr);
        prm->rng = NULL;
    }
    
    // Free the RadarParm structure itself
    free(prm);
}

/* Implementation of RadarParmEncode */
int RadarParmEncode(DataMap *ptr, void *prm) {
    // Stub implementation - just return success
    return 0;
}

// Implementation of TimeYMDHMSToEpoch
double TimeYMDHMSToEpoch(int16 year, int16 month, int16 day, 
                        int16 hour, int16 minute, int16 second, int32 usec) {
    struct tm tm;
    time_t t;
    
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    tm.tm_hour = hour;
    tm.tm_min = minute;
    tm.tm_sec = second;
    tm.tm_isdst = -1;  /* Let mktime determine daylight saving time */
    
    t = mktime(&tm);
    if (t == (time_t)-1) {
        return -1.0;
    }
    
    return (double)t + (double)usec / 1e6;
}

// Implementation of CFitSetRng to handle dynamic resizing of range gate and data arrays
/* Implementation of CFitMake */
struct CFitdata *CFitMake()
{
    struct CFitdata *cfit = (struct CFitdata *)calloc(1, sizeof(struct CFitdata));
    if (!cfit) return NULL;
    
    /* Initialize version */
    cfit->version.major = 0;
    cfit->version.minor = 0;
    
    /* Initialize other fields */
    cfit->stid = 0;
    cfit->time = 0.0;
    cfit->scan = 0;
    cfit->cp = 0;
    cfit->bmnum = 0;
    cfit->bmazm = 0.0f;
    cfit->channel = 0;
    cfit->intt.sc = 0;
    cfit->intt.us = 0;
    cfit->frang = 0;
    cfit->rsep = 0;
    cfit->rxrise = 0;
    cfit->tfreq = 0;
    cfit->noise = 0.0f;
    cfit->atten = 0;
    cfit->nave = 0;
    cfit->nrang = 0;
    cfit->num = 0;
    cfit->rng = NULL;
    cfit->data = NULL;
    cfit->txpow = 0.0f;
    cfit->lagfr = 0;
    cfit->smsep = 0;
    cfit->ercod = 0;
    cfit->stat = 0;
    cfit->qflg = 0;
    cfit->channel_side = 0;
    cfit->mppul = 0;
    cfit->mplgs = 0;
    cfit->nlag = 0;
    cfit->lag = NULL;
    
    /* Initialize datetime */
    cfit->datetime.year = 0;
    cfit->datetime.month = 0;
    cfit->datetime.day = 0;
    cfit->datetime.hour = 0;
    cfit->datetime.minute = 0;
    cfit->datetime.second = 0;
    cfit->datetime.usec = 0;
    
    return cfit;
}

/* Implementation of CFitFree */
void CFitFree(struct CFitdata *cfit)
{
    if (!cfit) return;
    
    /* Free dynamically allocated arrays */
    if (cfit->rng) free(cfit->rng);
    if (cfit->data) free(cfit->data);
    if (cfit->lag) free(cfit->lag);
    
    /* Free the structure itself */
    free(cfit);
}

int CFitSetRng(struct CFitdata *cfit, int nrang)
{
    if (!cfit) return -1;
    
    /* If the range is the same, do nothing */
    if (nrang == cfit->nrang) return 0;
    
    /* If nrang is 0, free the existing arrays */
    if (nrang == 0) {
        if (cfit->rng) {
            free(cfit->rng);
            cfit->rng = NULL;
        }
        if (cfit->data) {
            free(cfit->data);
            cfit->data = NULL;
        }
        cfit->nrang = 0;
        cfit->num = 0;
        return 0;
    }
    
    /* Allocate new arrays */
    int16_t *new_rng = (int16_t *)realloc(cfit->rng, nrang * sizeof(int16_t));
    struct CFitCell *new_data = (struct CFitCell *)realloc(cfit->data, nrang * sizeof(struct CFitCell));
    
    if (!new_rng || !new_data) {
        if (new_rng) free(new_rng);
        if (new_data) free(new_data);
        return -1;
    }
    
    /* If we're expanding the arrays, initialize the new elements */
    if (nrang > cfit->nrang) {
        /* Initialize new range gates */
        for (int i = cfit->nrang; i < nrang; i++) {
            new_rng[i] = cfit->frang + i * cfit->rsep;
            memset(&new_data[i], 0, sizeof(struct CFitCell));
        }
    }
    
    /* Update the structure */
    cfit->rng = new_rng;
    cfit->data = new_data;
    cfit->nrang = nrang;
    
    /* Update the number of valid data points */
    if (cfit->num > nrang) {
        cfit->num = nrang;
    }
    
    return 0;
}
