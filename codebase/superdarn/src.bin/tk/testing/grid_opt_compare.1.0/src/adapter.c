#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "adapter.h"

void grid_copy_to_opt(const struct GridData *src, struct GridDataOpt *dst) {
    dst->st_time = src->st_time;
    dst->ed_time = src->ed_time;
    dst->stnum   = src->stnum;
    dst->vcnum   = src->vcnum;
    dst->xtd     = src->xtd;
    if (src->stnum > 0) {
        dst->sdata = calloc(src->stnum, sizeof(struct GridSVecOpt));
        for (int i = 0; i < src->stnum; i++) {
            dst->sdata[i].st_id   = src->sdata[i].st_id;
            dst->sdata[i].chn     = src->sdata[i].chn;
            dst->sdata[i].npnt    = src->sdata[i].npnt;
            dst->sdata[i].freq0   = src->sdata[i].freq0;
            dst->sdata[i].prog_id = src->sdata[i].prog_id;
            dst->sdata[i].gsct    = src->sdata[i].gsct;
        }
    }
    if (src->vcnum > 0) {
        dst->data = calloc(src->vcnum, sizeof(struct GridGVecOpt));
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

int grid_compare(const struct GridData *a, const struct GridDataOpt *b, double eps) {
    if (a->vcnum != b->vcnum) return -1;
    for (int i = 0; i < a->vcnum; i++) {
        if (a->data[i].st_id != b->data[i].st_id)               return -1;
        if (a->data[i].index != b->data[i].index)               return -1;
        if (fabs(a->data[i].mlat - b->data[i].mlat) > eps)      return -1;
        if (fabs(a->data[i].vel.median - b->data[i].vel.median) > eps) return -1;
    }
    return 0;
}
