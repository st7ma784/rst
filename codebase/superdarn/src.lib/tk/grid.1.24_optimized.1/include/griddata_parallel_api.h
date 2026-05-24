/* griddata_parallel_api.h
   =======================
   Aggregator header for the public libgrdopt API. Fulfils AUDIT.md item A6:
   consumers can `#include "griddata_parallel_api.h"` and get the full
   *Opt-suffix surface without needing `extern` declarations.

   What lives here vs in griddata_parallel.h:
     - griddata_parallel.h          : core types (GridDataOpt, GridGVecOpt,
                                      GridStats, ...) + the original 7 *Opt
                                      wrappers (Make/Free/Sort/Average/
                                      Integrate/Merge/LocateCell).
     - griddata_parallel_api.h (here): the audit-A5 *Opt wrappers
                                      (Copy/Add/Read/Write/Fread/Fwrite/
                                      Seek/Fseek/IndexLoad/IndexFload/
                                      IndexFree/LinReg/GetTime).

   Phase B will surface the GridFilterCriteria / StatisticalFilterParams /
   SpatialFilterParams type definitions (currently file-private in
   filtergrid_parallel.c) and add prototypes for the Filter*Parallel and
   advanced Sort*Parallel functions. They are intentionally NOT declared
   here yet because their parameter types are not yet in a public header.

   This header is additive over griddata_parallel.h -- including only this
   file is sufficient. */

#ifndef _GRIDDATA_PARALLEL_API_H
#define _GRIDDATA_PARALLEL_API_H

#include "griddata_parallel.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----- Audit A5 *Opt wrappers (defined in src/gridopt_wrappers.c) ----- */

/* Pass-throughs to existing implementations. */
double GridGetTimeOpt(struct DataMap *ptr);

int GridSeekOpt(int fid, int yr, int mo, int dy, int hr, int mt, int sc,
                double *atme, struct GridIndexParallel *inx);

int GridFseekOpt(FILE *fp, int yr, int mo, int dy, int hr, int mt, int sc,
                 double *atme, struct GridIndexParallel *inx);

struct GridIndexParallel *GridIndexLoadOpt(int fid);
struct GridIndexParallel *GridIndexFloadOpt(FILE *fp);
void GridIndexFreeOpt(struct GridIndexParallel *inx);

void GridLinRegOpt(int num, struct GridGVecOpt **data,
                   double *vpar, double *vper);

/* Phase B placeholders. Symbol-level coverage for Phase A; bodies will be
   filled in (or replaced with delegation to libgrd via a GridData<->Opt
   bridge) during Phase B. */
void GridCopyOpt(struct GridDataOpt *a, struct GridDataOpt *b);
void GridAddOpt (struct GridDataOpt *a, struct GridDataOpt *b, int recnum);

int GridReadOpt  (int fid,   struct GridDataOpt *gp);
int GridWriteOpt (int fid,   struct GridDataOpt *ptr);
int GridFreadOpt (FILE *fp,  struct GridDataOpt *ptr);
int GridFwriteOpt(FILE *fp,  struct GridDataOpt *ptr);

#ifdef __cplusplus
}
#endif

#endif /* _GRIDDATA_PARALLEL_API_H */
