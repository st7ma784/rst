#ifndef _ADAPTER_H
#define _ADAPTER_H
#include "griddata.h"
#include "griddata_parallel.h"
/* Copy original GridData into a pre-allocated GridDataOpt.
   Only the fields that GridSort/Average/Locate touch are copied. */
void grid_copy_to_opt(const struct GridData *src, struct GridDataOpt *dst);
/* Verify two grids hold the same vectors in the same order (by st_id+index
   and field tolerance eps). Returns 0 on equivalence, -1 on mismatch. */
int grid_compare(const struct GridData *a, const struct GridDataOpt *b, double eps);
#endif
