/* Stubs for symbols referenced by libgrdopt that were never defined in its
   sources. Real impl would use posix_memalign / aligned_alloc; these are
   thin shims so the benchmark links. */
#include <stdlib.h>
void *grid_aligned_malloc(size_t size, size_t alignment) {
    void *p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    return p;
}
void grid_aligned_free(void *ptr) { free(ptr); }
