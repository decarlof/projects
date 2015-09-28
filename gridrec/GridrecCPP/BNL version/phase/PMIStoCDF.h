/* Header for the raw PMIS files */
#include <time.h>
struct PMISHeader
{
    char           ID[4];          /* File ID - 'PMIS' */
    unsigned short nHeadSize;      /* Header Size - 0x00AC = 172 bytes */
    unsigned short nVersion;       /* File Version - 0x0014 = 20 */
    unsigned short nXorg;          /* Serial Origin - CCD column origin */
    unsigned short nYorg;          /* Parallel Origin - CCD row origin */
    unsigned short nXsiz;          /* Serial Size - Image column count */
    unsigned short nYsiz;          /* Parallel Size - Image row count */
    unsigned short nXbin;          /* Serial Binning - Column binning factor */
    unsigned short nYbin;          /* Parallel Binning - Row binning factor */
    char           szName[40];     /* Image Name - Image name (ASCIIZ) */
    char           szComment[100]; /* Image Comment - Image comment (ASCIIZ) */
    time_t         tCreated;       /* Creation Date - Image Creation date */
    time_t         tModified;      /* Modification - Image Modification date */
    unsigned short nGain;          /* Camera Gain - CCD gain value */
    unsigned short nImages;        /* Sequence Count - Image frame count */
};

/* Function prototypes */

unsigned short
** readPMIS (char *PMISfile, struct PMISHeader *Header, int *xsize, int *ysize);

void
writePMIS(char *PMISfile, struct PMISHeader *Header, unsigned short **buffer);

#include "libhead.h"

unsigned short *
malloc_vector_us (long n);

void
free_vector_us (unsigned short *v);

float *
malloc_vector_f (long n);

void
free_vector_f (float *v);

unsigned short **
malloc_matrix_us (long nr, long nc);

void
free_matrix_us (unsigned short **m);

float **
malloc_matrix_f (long nr, long nc);

void
free_matrix_f (float **m);

unsigned short ***
malloc_tensor_us (long nr, long nc, long nd);

void
free_tensor_us (unsigned short ***t);

float ***
malloc_tensor_f (long nr, long nc, long nd);

void
free_tensor_f (float ***t);

void
saveaspgm (char *name, unsigned short **m, int y, int x);

void
saveaspgm2 (char *name, unsigned short **m, int x, int y, int min, int max);

void
saveaspgm2f (char *name, float **m, int x, int y, float min, float max);

void
saveasnetCDF (char *name, float **m, int x, int y);

void
usage (void);

/* Global defines */
#define MEDIAN 0
#define MAXWHITEFIELD 2
#define BYTESWAP2(x) ((x >> 8) | (x << 8))
#define XBORDER 5
#define YBORDER 10
#define VERSION "PMIStoCDF Version 1.0 Rev: 2/23/95"
