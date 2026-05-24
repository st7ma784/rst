/* gridopt_wrappers.c
   ==================
   Phase A *Opt-suffix API wrappers, fulfilling AUDIT.md item A5.

   The optimised library has already grown:
     - 6 *Opt symbols implemented (GridMakeOpt, GridFreeOpt, GridSortOpt,
       GridAverageOpt, GridIntegrateOpt, GridMergeOpt, GridLocateCellOpt).
     - 30+ *Parallel symbols (GridSortParallel, GridAddParallel, ...) and
       a dozen snake_case grid_parallel_* symbols (read/write/seek/index).

   This file adds the remaining *Opt entry points so the optimised library
   exposes a symbol for every public function in libgrd.1.a. Where an
   underlying implementation already exists (LinReg, GetTime, Index*, Seek,
   Fseek), the wrapper is a thin pass-through. Where no implementation
   exists yet (Copy, Add, file I/O on GridDataOpt), the wrapper returns -1
   with a TODO -- correctness is a Phase B job per the audit.

   This file deliberately stays out of grid_parallel_utils.c to keep the
   wrapper layer cleanly inspectable. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <zlib.h>
#include "rtypes.h"
#include "griddata.h"           /* libgrd GridData / GridGVec / GridSVec */
#include "gridread.h"
#include "gridwrite.h"
#include "griddata_parallel.h"

/* Forward declarations of the underlying parallel API used below. The full
   prototypes live in griddata_parallel.h; we re-declare only the handful
   this file needs to keep the include surface small. */
double grid_parallel_get_time(struct DataMap *ptr);
int    grid_parallel_seek(int fid, int yr, int mo, int dy, int hr, int mt,
                          int sc, double *atme,
                          struct GridIndexParallel *inx,
                          struct GridPerformanceStats *stats);
int    grid_parallel_fseek(FILE *fp, int yr, int mo, int dy, int hr, int mt,
                           int sc, double *atme,
                           struct GridIndexParallel *inx,
                           struct GridPerformanceStats *stats);
struct GridIndexParallel *grid_parallel_load_index(
    int fid, struct GridPerformanceStats *stats);
struct GridIndexParallel *grid_parallel_fload_index(
    FILE *fp, struct GridPerformanceStats *stats);
void   grid_parallel_index_free(struct GridIndexParallel *inx);
CUDA_CALLABLE void GridLinRegParallel(struct GridGVecOpt **data, uint32_t num,
                                       double *vpar, double *vper);

/* ----- Pass-throughs to existing implementations ---------------------- */

double GridGetTimeOpt(struct DataMap *ptr) {
    return grid_parallel_get_time(ptr);
}

int GridSeekOpt(int fid, int yr, int mo, int dy, int hr, int mt, int sc,
                double *atme, struct GridIndexParallel *inx) {
    return grid_parallel_seek(fid, yr, mo, dy, hr, mt, sc, atme, inx, NULL);
}

int GridFseekOpt(FILE *fp, int yr, int mo, int dy, int hr, int mt, int sc,
                 double *atme, struct GridIndexParallel *inx) {
    return grid_parallel_fseek(fp, yr, mo, dy, hr, mt, sc, atme, inx, NULL);
}

struct GridIndexParallel *GridIndexLoadOpt(int fid) {
    return grid_parallel_load_index(fid, NULL);
}

struct GridIndexParallel *GridIndexFloadOpt(FILE *fp) {
    return grid_parallel_fload_index(fp, NULL);
}

void GridIndexFreeOpt(struct GridIndexParallel *inx) {
    grid_parallel_index_free(inx);
}

void GridLinRegOpt(int num, struct GridGVecOpt **data,
                   double *vpar, double *vper) {
    /* libgrd's GridLinReg has signature (int, GVec**, double*, double*).
       Parallel implementation takes uint32_t but the public *Opt entry
       keeps the libgrd argument order so callers can switch by symbol
       rename only. */
    GridLinRegParallel(data, (uint32_t)num, vpar, vper);
}

/* ----- Phase D: GridData <-> GridDataOpt conversion bridge -----------
   The optimised library never reimplemented the .grd DataMap parser
   and writer -- that's hundreds of lines of stable, audited code in
   libgrd. Phase D wires the *Opt I/O entry points through libgrd by
   converting to a temporary GridData on the way in / out.

   Field mapping is lossy in one direction: GridGVec has a srng field
   that GridGVecOpt dropped during C1's struct slim. Reads zero srng
   on input; writes lose srng on output (it wasn't carried by the
   *Opt path anyway, so callers were already operating without it). */

static void opt_to_grid(const struct GridDataOpt *src, struct GridData *dst) {
    dst->st_time = src->st_time;
    dst->ed_time = src->ed_time;
    dst->stnum   = src->stnum;
    dst->vcnum   = src->vcnum;
    dst->xtd     = src->xtd;

    if (src->stnum > 0 && src->sdata) {
        dst->sdata = (struct GridSVec*)calloc(src->stnum, sizeof(struct GridSVec));
        for (int i = 0; i < src->stnum; i++) {
            dst->sdata[i].st_id          = src->sdata[i].st_id;
            dst->sdata[i].chn            = src->sdata[i].chn;
            dst->sdata[i].npnt           = src->sdata[i].npnt;
            dst->sdata[i].freq0          = src->sdata[i].freq0;
            dst->sdata[i].major_revision = src->sdata[i].major_revision;
            dst->sdata[i].minor_revision = src->sdata[i].minor_revision;
            dst->sdata[i].prog_id        = src->sdata[i].prog_id;
            dst->sdata[i].gsct           = src->sdata[i].gsct;
            dst->sdata[i].noise.mean     = src->sdata[i].noise.mean;
            dst->sdata[i].noise.sd       = src->sdata[i].noise.sd;
            dst->sdata[i].vel.min        = src->sdata[i].vel.min;
            dst->sdata[i].vel.max        = src->sdata[i].vel.max;
            dst->sdata[i].pwr.min        = src->sdata[i].pwr.min;
            dst->sdata[i].pwr.max        = src->sdata[i].pwr.max;
            dst->sdata[i].wdt.min        = src->sdata[i].wdt.min;
            dst->sdata[i].wdt.max        = src->sdata[i].wdt.max;
            dst->sdata[i].verr.min       = src->sdata[i].verr.min;
            dst->sdata[i].verr.max       = src->sdata[i].verr.max;
        }
    } else {
        dst->sdata = NULL;
    }

    if (src->vcnum > 0 && src->data) {
        dst->data = (struct GridGVec*)calloc(src->vcnum, sizeof(struct GridGVec));
        for (int i = 0; i < src->vcnum; i++) {
            dst->data[i].mlat  = src->data[i].mlat;
            dst->data[i].mlon  = src->data[i].mlon;
            dst->data[i].azm   = src->data[i].azm;
            dst->data[i].srng  = 0.0;   /* dropped by C1 */
            dst->data[i].st_id = src->data[i].st_id;
            dst->data[i].chn   = src->data[i].chn;
            dst->data[i].index = src->data[i].index;
            dst->data[i].vel.median = src->data[i].vel.median;
            dst->data[i].vel.sd     = src->data[i].vel.sd;
            dst->data[i].pwr.median = src->data[i].pwr.median;
            dst->data[i].pwr.sd     = src->data[i].pwr.sd;
            dst->data[i].wdt.median = src->data[i].wdt.median;
            dst->data[i].wdt.sd     = src->data[i].wdt.sd;
        }
    } else {
        dst->data = NULL;
    }
}

static void grid_to_opt(const struct GridData *src, struct GridDataOpt *dst) {
    dst->st_time = src->st_time;
    dst->ed_time = src->ed_time;
    dst->stnum   = src->stnum;
    dst->vcnum   = src->vcnum;
    dst->xtd     = src->xtd;

    if (src->stnum > 0 && src->sdata) {
        if (dst->sdata) free(dst->sdata);
        dst->sdata = (struct GridSVecOpt*)calloc(src->stnum, sizeof(struct GridSVecOpt));
        for (int i = 0; i < src->stnum; i++) {
            dst->sdata[i].st_id          = src->sdata[i].st_id;
            dst->sdata[i].chn            = src->sdata[i].chn;
            dst->sdata[i].npnt           = src->sdata[i].npnt;
            dst->sdata[i].freq0          = src->sdata[i].freq0;
            dst->sdata[i].major_revision = src->sdata[i].major_revision;
            dst->sdata[i].minor_revision = src->sdata[i].minor_revision;
            dst->sdata[i].prog_id        = src->sdata[i].prog_id;
            dst->sdata[i].gsct           = src->sdata[i].gsct;
            dst->sdata[i].noise.mean     = src->sdata[i].noise.mean;
            dst->sdata[i].noise.sd       = src->sdata[i].noise.sd;
            dst->sdata[i].vel.min        = src->sdata[i].vel.min;
            dst->sdata[i].vel.max        = src->sdata[i].vel.max;
            dst->sdata[i].pwr.min        = src->sdata[i].pwr.min;
            dst->sdata[i].pwr.max        = src->sdata[i].pwr.max;
            dst->sdata[i].wdt.min        = src->sdata[i].wdt.min;
            dst->sdata[i].wdt.max        = src->sdata[i].wdt.max;
            dst->sdata[i].verr.min       = src->sdata[i].verr.min;
            dst->sdata[i].verr.max       = src->sdata[i].verr.max;
        }
    }

    if (src->vcnum > 0 && src->data) {
        if (dst->data) free(dst->data);
        dst->data = (struct GridGVecOpt*)calloc(src->vcnum, sizeof(struct GridGVecOpt));
        for (int i = 0; i < src->vcnum; i++) {
            dst->data[i].mlat  = src->data[i].mlat;
            dst->data[i].mlon  = src->data[i].mlon;
            dst->data[i].azm   = src->data[i].azm;
            dst->data[i].st_id = src->data[i].st_id;
            dst->data[i].chn   = src->data[i].chn;
            dst->data[i].index = src->data[i].index;
            dst->data[i].vel.median = src->data[i].vel.median;
            dst->data[i].vel.sd     = src->data[i].vel.sd;
            dst->data[i].pwr.median = src->data[i].pwr.median;
            dst->data[i].pwr.sd     = src->data[i].pwr.sd;
            dst->data[i].wdt.median = src->data[i].wdt.median;
            dst->data[i].wdt.sd     = src->data[i].wdt.sd;
        }
    }
}

static void grid_free_payload(struct GridData *g) {
    if (g->sdata) { free(g->sdata); g->sdata = NULL; }
    if (g->data)  { free(g->data);  g->data  = NULL; }
}

void GridCopyOpt(struct GridDataOpt *a, struct GridDataOpt *b) {
    /* Semantics from libgrd: copy contents of b into a, replacing a's. */
    if (!a || !b) return;
    a->st_time = b->st_time;
    a->ed_time = b->ed_time;
    a->stnum   = b->stnum;
    a->vcnum   = b->vcnum;
    a->xtd     = b->xtd;

    if (a->sdata) { free(a->sdata); a->sdata = NULL; }
    if (a->data)  { free(a->data);  a->data  = NULL; }

    if (b->stnum > 0 && b->sdata) {
        a->sdata = (struct GridSVecOpt*)calloc(b->stnum, sizeof(struct GridSVecOpt));
        memcpy(a->sdata, b->sdata, b->stnum * sizeof(struct GridSVecOpt));
    }
    if (b->vcnum > 0 && b->data) {
        a->data = (struct GridGVecOpt*)calloc(b->vcnum, sizeof(struct GridGVecOpt));
        memcpy(a->data, b->data, b->vcnum * sizeof(struct GridGVecOpt));
    }
}

void GridAddOpt(struct GridDataOpt *a, struct GridDataOpt *b, int recnum) {
    /* Delegate to libgrd: convert both sides to GridData, call GridAdd,
       convert result back into a. recnum follows libgrd's accumulation
       semantics (running record count for the average). */
    if (!a || !b) return;
    struct GridData ga = {0}, gb = {0};
    opt_to_grid(a, &ga);
    opt_to_grid(b, &gb);
    GridAdd(&ga, &gb, recnum);
    grid_to_opt(&ga, a);
    grid_free_payload(&ga);
    grid_free_payload(&gb);
}

int GridReadOpt(int fid, struct GridDataOpt *gp) {
    if (!gp || fid < 0) return -1;
    struct GridData tmp = {0};
    int rc = GridRead(fid, &tmp);
    if (rc >= 0) grid_to_opt(&tmp, gp);
    grid_free_payload(&tmp);
    return rc;
}

int GridWriteOpt(int fid, struct GridDataOpt *ptr) {
    if (!ptr || fid < 0) return -1;
    struct GridData tmp = {0};
    opt_to_grid(ptr, &tmp);
    int rc = GridWrite(fid, &tmp);
    grid_free_payload(&tmp);
    return rc;
}

int GridFreadOpt(FILE *fp, struct GridDataOpt *ptr) {
    if (!fp || !ptr) return -1;
    struct GridData tmp = {0};
    int rc = GridFread(fp, &tmp);
    if (rc >= 0) grid_to_opt(&tmp, ptr);
    grid_free_payload(&tmp);
    return rc;
}

int GridFwriteOpt(FILE *fp, struct GridDataOpt *ptr) {
    if (!fp || !ptr) return -1;
    struct GridData tmp = {0};
    opt_to_grid(ptr, &tmp);
    int rc = GridFwrite(fp, &tmp);
    grid_free_payload(&tmp);
    return rc;
}
