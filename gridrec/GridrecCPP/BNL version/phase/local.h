/* Function prototypes */

int
fexist (char *filename);

unsigned short *
malloc_vector_us (long n);

void
free_vector_us (unsigned short *v);

float **
malloc_matrix_f (long nr, long nc);

void
free_matrix_f (float **m);

void
writenetCDF (char *name, float **m, int x, int y);

float **
readnetCDF (char *name, int *xsize, int *ysize);

void
usage (void);

void
shellsort (unsigned long n, float a[]);
