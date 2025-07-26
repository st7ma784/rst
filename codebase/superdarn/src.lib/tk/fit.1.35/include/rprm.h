#ifndef RPRM_H
#define RPRM_H

#include "stdint_types.h"  /* Include standard integer types */

/* Array limits */
#define ORIGIN_TIME_SIZE 26
#define ORIGIN_COMMAND_SIZE 128
#define PULSE_SIZE 64
#define LAG_SIZE 96
#define MAX_RANGE 300
#define COMBF_SIZE 80
#define MAX_BEAM 255

/* Radar parameters structure */
typedef struct RadarParm {
    /* Version information */
    struct {
        char major;    /* Major revision */
        char minor;    /* Minor revision */
    } revision;

    /* Origin information */
    struct {
        char code;     /* origin code, 0 when generated at radar */
        char time[ORIGIN_TIME_SIZE];  /* ASCII time */
        char command[ORIGIN_COMMAND_SIZE]; /* command line or control program */
    } origin;

    int16 cp;          /* control program identifier */
    int16 stid;        /* station identifier */

    /* Time of the beam */
    struct {
        int16 yr;      /* year */
        int16 mo;      /* month */
        int16 dy;      /* day */
        int16 hr;      /* hour */
        int16 mt;      /* minute */
        int16 sc;      /* second */
        int16 us;      /* micro-second */
    } time;

    int16 txpow;       /* transmitted power (kW) */
    int16 nave;        /* number of pulse sequences transmitted */
    int16 atten;       /* attenuation level */
    int16 lagfr;       /* lag to first range (microseconds) */
    int16 smsep;       /* sample separation (microseconds) */
    int16 ercod;       /* error code */

    /* Status information */
    struct {
        int16 agc;     /* AGC status word */
        int16 lopwr;   /* LOPWR status word */
    } stat;

    /* Noise information */
    struct {
        float search;   /* calculated noise from clear frequency search */
        float mean;     /* average noise across frequency band */
    } noise;

    int16 channel;     /* channel number for a stereo radar */
    int16 bmnum;       /* beam number */
    float bmazm;       /* beam azimuth */
    int16 scan;        /* scan flag */
    int16 rxrise;      /* receiver rise time (microseconds) */

    /* Integration time */
    struct {
        int16 sc;      /* number of seconds integration time */
        int16 us;      /* number of microseconds of integration time */
    } intt;

    int16 txpl;        /* transmit pulse length (microseconds) */
    int16 mpinc;       /* multi-pulse increment (microseconds) */
    int16 mppul;       /* number of pulses in sequence */
    int16 mplgs;       /* number of lags in sequences */
    int16 nrang;       /* number of ranges */
    int16 frang;       /* distance to first range (kilometers) */
    int16 rsep;        /* range separation (kilometers) */
    int16 xcf;         /* XCF flag */
    int16 tfreq;       /* transmitted frequency */
    int16 offset;      /* offset between channels for a stereo radar */

    int32 mxpwr;       /* maximum power */
    int32 lvmax;       /* maximum noise level */

    int16 pulse[PULSE_SIZE];  /* pulse table */
    int16 lag[LAG_SIZE][2];   /* lag table */
    char combf[COMBF_SIZE];   /* comment buffer */
    
    /* Additional fields for compatibility with legacy code */
    int16 num;          /* Number of ranges in the data */
    int16 *rng;         /* Range gate array */
} RadarParm;

/* Function prototypes */
RadarParm *RadarParmMake(void);
void RadarParmFree(RadarParm *prm);
int RadarParmSetRanges(RadarParm *prm, int nrang);

#endif /* RPRM_H */
