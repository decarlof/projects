/* Header for the raw PMIS files */
#include <time.h>

struct PXLHeader
{

        	long    hsiz,                   /* 0     pointer to image data                  */
                        text,                   /* 4     total history data appended            */
                        startx,                 /* 8     starting col                           */
                        starty,                 /* 12    starting row                           */
                        totalx,                 /* 16    total number of cols                   */
                        totaly,                 /* 20    total number of rows                   */
                        bpp,                    /* 24    bits per pixel [1-8]                   */
                        exp_t,                  /* 28    exposure time * 33 mSEC                */
                        exp_n;                  /* 32    exposure number                        */
        	char    spc[4],                 /* 36    x,y length per unit pixel              */
                        units[4];               /* 40    in ascii format                        */
        	char    date[36],               /* 44    Time-date of aquisition                */
                        drk[36],                /* 80    dark-current and/or bkg                */
                        rad[36],                /* 116   radiometric correction                 */
                        geom[36],               /* 152   geometric correction                   */
                        src[36],                /* 188   image src descriptor                   */
                        opt[36],                /* 224   filters, optics                        */
                        pos[36],                /* 260   world x,y,z position                   */
                        expt[36],               /* 296   experimental protocol                  */
                        label[144];             /* 332   image title                            */
        	char    pixel_type;             /* 476   TYPE_BYTE, _SHORT,                     */
        	char    is_rgb;                 /* 477   Boolean for color images               */
        	short   section_type;           /* 478   Time, Z, Other                         */
        	short   mosaic_x,               /* 480   # horizontal tiles                     */
                        mosaic_y;               /* 482   # vertical tiles                       */
        	long    nbands;                 /* 484   # bands for multispectral              */
        	long    nsections;              /* 488   # images of section_type               */
        	long    version_magic;  /* 492   Magic # for version 2                          */
        	float   zspc;                   /* 496   z length/unit voxel                    */
        	float   xspc,                   /* 500   x length/unit pixel                    */
                        yspc;                   /* 504   y length/unit pixel                    */
        	long    magic;                  /* 508   magic number                           */


};

/* Function prototypes */

#include "libhead.h"

unsigned short **
readPXL (char *PXLfile, struct PXLHeader *Header, int *xsize, int *ysize);

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
