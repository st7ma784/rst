#ifndef SCANDATA_H
#define SCANDATA_H

/* 
 * Stub implementation of scandata.h for fit.1.35
 * This is a minimal implementation to allow the code to compile
 */

#include "rtypes.h"
#include "rtime.h"
#include "radarbeam.h"  /* Include radarbeam.h for RadarBeam and RadarRange definitions */

/* Time structure */
struct Time {
    int yr;     /* Year */
    int mo;     /* Month */
    int dy;     /* Day */
    int hr;     /* Hour */
    int mt;     /* Minute */
    int sc;     /* Second */
    int us;     /* Microseconds */
};

/* 
 * RadarRange and RadarBeam are now defined in radarbeam.h
 * to avoid duplicate definitions
 */

/* Radar scan data structure */
typedef struct RadarScan {
    struct {
        int major;
        int minor;
    } version;                  /* Version number */
    
    double st_time;             /* Start time of the scan */
    double ed_time;             /* End time of the scan */
    
    int stid;                   /* Station ID */
    int channel;                /* Channel number */
    
    int scan;                   /* Scan number */
    int npnt;                   /* Number of beam directions */
    int bmnum;                  /* Beam number */
    
    float bmazm;                /* Beam azimuth */
    
    struct {
        int sc;                 /* Seconds */
        int us;                 /* Microseconds */
    } intt;                     /* Integration time */
    
    int frang;                  /* First range gate */
    int rsep;                   /* Range separation */
    int rxrise;                 /* Receiver rise time */
    
    int tfreq;                  /* Transmit frequency */
    float noise;                /* Noise level */
    int atten;                  /* Attenuation level */
    
    int nave;                   /* Number of integrations */
    int nrang;                  /* Number of ranges */
    
    int *rng;                   /* Range gate array */
    void *data;                 /* Pointer to scan data */
    
    /* Extended parameters */
    int cp;                     /* Control program ID */
    int mppul;                  /* Multi-pulse sequence length */
    int mplgs;                  /* Number of lags */
    int *lag;                   /* Lag array */
    
    /* Beam data */
    int num;                    /* Number of beams */
    struct RadarBeam **bm;      /* Array of beam pointers */
    
    /* Timing information */
    struct Time time;           /* Date and time of the scan */
} RadarScan;

/* Function prototypes */
struct RadarScan *RadarScanMake(void);
void RadarScanFree(struct RadarScan *ptr);
int RadarScanSetRng(struct RadarScan *ptr, int nrang);
struct RadarBeam *RadarScanAddBeam(struct RadarScan *ptr, int nrang);
void RadarScanReset(struct RadarScan *ptr);

#endif /* SCANDATA_H */
