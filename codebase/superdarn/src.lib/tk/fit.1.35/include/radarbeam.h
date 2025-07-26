#ifndef RADARBEAM_H
#define RADARBEAM_H

#include "rtypes.h"
#include "rtime.h"

/* Forward declaration of RadarScan to avoid circular dependency */
struct RadarScan;

/* 
 * Stub implementation of radarbeam.h for fit.1.35
 * This is a minimal implementation to allow the code to compile
 */

/* Radar range data structure */
typedef struct {
    int gsct;       /* Ground scatter flag */
    float p_0;      /* Power at zero lag */
    float p_0_e;    /* Error in p_0 */
    float v;        /* Velocity */
    float v_e;      /* Error in velocity */
    float w_l;      /* Spectral width */
    float w_l_e;    /* Error in spectral width */
    float p_l;      /* Power from lagged product */
    float p_l_e;    /* Error in p_l */
    float phi0;     /* Phase at zero lag */
    float elv;      /* Elevation angle */
} RadarRange;

/* Radar beam data structure */
typedef struct RadarBeam {
    int scan;               /* Scan number */
    int bm;                 /* Beam number */
    float bmazm;            /* Beam azimuth */
    int cpid;               /* Control program ID */
    struct {
        int sc;             /* Seconds */
        int us;             /* Microseconds */
    } intt;                 /* Integration time */
    int nave;               /* Number of coherent integrations */
    int frang;              /* First range gate */
    int rsep;               /* Range separation */
    int rxrise;             /* Receiver rise time */
    float freq;             /* Frequency */
    float noise;            /* Noise level */
    int atten;              /* Attenuation level */
    int channel;            /* Channel number */
    int nrang;              /* Number of range gates */
    
    /* Data arrays */
    int *sct;               /* Scatter flag array */
    RadarRange *rng;        /* Range data array */
    
    /* Time of the beam */
    double time;
    
    /* Navigation data */
    float phi0;             /* Phase at zero lag */
    float elv;              /* Elevation angle */
    
} RadarBeam;

/* Function prototypes */
struct RadarBeam *RadarBeamMake(int nrang);
void RadarBeamFree(struct RadarBeam *bm);
struct RadarBeam *RadarScanAddBeam(struct RadarScan *ptr, int nrang);

#endif /* RADARBEAM_H */
