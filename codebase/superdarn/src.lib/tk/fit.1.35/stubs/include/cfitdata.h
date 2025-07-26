/* cfitdata.h - Stub for testing */
#ifndef _CFITDATA_H
#define _CFITDATA_H

#include <stdint.h>

#define CFIT_MAJOR_REVISION 2
#define CFIT_MINOR_REVISION 1

/* Time structure */
typedef struct {
    int32_t year;
    int32_t month;
    int32_t day;
    int32_t hour;
    int32_t minute;
    int32_t second;
    int32_t usec;
} DateTime;

struct CFitCell {
    int gsct;
    double p_0;
    double p_0_e;
    double v;
    double v_e;
    double w_l;
    double w_l_e;
    double p_l;
    double p_l_e;
};

struct CFitdata {
    struct {
        int major;  /* major revision */
        int minor;  /* minor revision */
    } hdr;
    
    /* Radar identification */
    int16_t stid;      /* Station ID */
    int16_t channel;   /* Channel number */
    
    /* Scan information */
    int16_t scan;      /* Scan number */
    int16_t cp;        /* Control program number */
    int16_t bmnum;     /* Beam number */
    
    /* Time information */
    DateTime time;
    
    /* Radar parameters */
    int16_t frang;     /* First range gate (km) */
    int16_t rsep;      /* Range separation (km) */
    int16_t rxrise;    /* Receiver rise time (us) */
    int16_t tfreq;     /* Transmit frequency (kHz) */
    int16_t atten;     /* Attenuation level */
    
    /* Data parameters */
    int16_t nave;      /* Number of pulse sequences transmitted */
    int16_t nrang;     /* Number of range gates */
    int16_t num;       /* Number of ranges with data */
    
    /* Data arrays */
    int16_t *rng;      /* Range gate array */
    struct CFitCell *data;  /* Fit data array */
};

#endif /* _CFITDATA_H */
