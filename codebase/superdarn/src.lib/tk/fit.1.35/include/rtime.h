#ifndef RTIME_H
#define RTIME_H

#include <time.h>
#include "rtypes.h"  /* Includes DateTime and type definitions */

/* 
 * This header is now a compatibility layer that forwards to the functions
 * defined in rtypes.h to avoid duplicate definitions.
 */

/* Backward compatibility macros */
#define TimeYMDHMSToEpoch_DateTime TimeYMDHMSToEpochDT
#define TimeEpochToYMDHMS_DateTime TimeEpochToYMDHMSDT

/* 
 * All function declarations are now in rtypes.h
 * This header is kept for backward compatibility only
 */

#endif /* RTIME_H */
