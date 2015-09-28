/* File grid_io.c -- adapted from skeleton.c  2:41 PM 11/7/97 */
/* Revised 7/7/98  */
 /*******************************************
 * Read and interpret command line options.
 * Read in sinugrams in netCFD format.
 * Write out images in netCDF format.
 * Memory allocation and other utilities
 *******************************************/

#include "grid.h"

/**** Header file for netCDF */
#include "netcdf.h"

/*** static function prototypes ***/

static void writenetCDF (char *name, float **m, int x, int y);
static float **readnetCDF (char *name, int *xsize, int *ysize, float *cent);
static int fexist (char *filename);

/*** Global variables ***/
int verbose=0;

/**** Local variables ***/

static char CDFfile[256];
static float centoveride;
static int argc;
static char **argv;
//static char *optstr="f:l:p:vD:c:";

static long nfiles=0;
static int optindr;
static char dirstr[256];  /* Directory path for data files */



void get_parm(int argc,char **argv, grid_struct *A){

	int c;
	float C;
	extern int optind, opterr;   /* Used by getopt() */
	extern char* optarg;		/* " "  */

/*** Set default values ***/

	C=6.0; A->sampl=1.2;   /** Reset these based on experience **/
	A->R=A->MaxPixSiz=1.0;
	A->X0=A->Y0=0.0;
	A->ltbl=LTBL_DEF;
	A->fname[0]='\0';	/* Will be set by get_filter() */
	centoveride=-1.;

/*** Extract gridding parameters from command line ***/

	if(argc<2)usage();
/*	getoptreset();     Unneeded and may not be portable--7/7/98*/
	opterr=0;
//	while((c=getopt(argc,argv,optstr))!=EOF)
	{
		switch(c)
		{
		case 'p':
			sscanf(optarg,"%f,%f,%f,%f,%f,%f",
			&C,&A->sampl,
			&A->MaxPixSiz,
			&A->R,&A->X0,&A->Y0);
			break;
		case 'f':
			strcpy(A->fname,optarg);
			break;
		case 'l':
			A->ltbl=strtol(optarg,NULL,10);
			break;
		case 'D':
			strcpy(dirstr,optarg);
			break;
		case 'v':
			verbose++;
			break;
		case 'c':
			centoveride=atof(optarg);
			break;
		case '?':
			usage();
		}
	}


/*** Get filter function ***/

	A->filter=get_filter(A->fname);

/*** Get pswf data based on input value C ***/

	get_pswf(C, &A->pswf);

	return;

}      /**** End get_parm() ***/


int data_setup(int argc0, char *argv0[]) {

    char name[256];
    extern char *optarg;	/* Used by optarg */
    extern int optind, opterr;	/*     "          */
    int c=0;
    float cent = 0;
    int errflg = 0;

/*** Save args in static storage  */
	argc=argc0;
	argv=argv0; 


/*** Space optind over options */
/** Unneeded and may not be portable:
	opterr=0;
    	getoptreset();
	while(getopt(argc, argv, optstr)!=EOF) ;
		---Deleted 7/7/98 ****/

	optindr=optind;   /* Index to 1st filename arg */
	nfiles = argc - optindr;
	if(!nfiles)
	{
        	fprintf (stderr, "No filename was specified.\n");
    	        exit(1);
	}
							 			
/* Read the file names from the command line; make sure
       that they all exist. */

	if (verbose)
		printf("Attempting to read %d files.\n",nfiles);
	for(optind=optindr; optind < argc; optind++)
	{
		strcpy(name,dirstr);
	        if (!fexist(strcat(name,argv[optind])))
		{
			fprintf (stderr,
		"File %s does not exist. Aborting execution.\n",
                     argv[optind]);
			exit(2);
		}
	}

	if(verbose) printf("data_setup(): All %d files exist.\n",nfiles);
	return  nfiles;  							

} 	/* End data_setup()  */


float **get_sgram(int ifile,sg_struct *A){					

	char name[256];
	float** buf;

	strcpy(name,dirstr);
        strcat (name, argv[optindr+ifile]);

	buf=readnetCDF(name, &A->n_det, &A->n_ang, &A->center);

	if(centoveride>0) A->center=centoveride;

	A->geom=1;  /* For now: Assume uniform angular 
			distribution on half circle */
	A->angles=NULL;

	if(verbose) prn_sgparm(A);
	return buf;

}  /*** End get_sgram() **/

void rel_sgram(float **S)
{
	free_matrix(S);

}  /*** End rel_sgram() ***/

void put_image(int ifile, float **image, int size){

	char name[256];
	strcpy(name,dirstr);
	strcat (name, argv[optindr+ifile]);
	writenetCDF (name, image, size, size);

}   /* End put_image() */


static 
float **readnetCDF (char *name, int *xsize, int *ysize, 
					float *cent)
{
    int ncid, xid, angleid;
    long xdim, angdim;
    int sinogram_dimids[2], sinogram_id, centerid;
    long start[2], count[2];
    float **Data = NULL;
    float center;

    strcpy (CDFfile, name);

    /* Open a netCDF file for reading */

    if (verbose)  printf(
	"Opening sinugram data file %s. \n", name);
    ncid = ncopen (CDFfile, NC_NOWRITE);
    /*Get the dimensions of the Hyperslab*/
    angleid = ncdimid (ncid, "angle");
    ncdiminq (ncid, angleid, (char *) 0, &angdim);
    *ysize = angdim;
    xid = ncdimid (ncid, "x");
    ncdiminq (ncid, xid, (char *) 0, &xdim);
    *xsize = xdim;
    centerid = ncvarid(ncid, "center");


    /*Get the variable id*/
    sinogram_dimids[0] = angleid;
    sinogram_dimids[1] = xid;
    sinogram_id = ncvarid (ncid, "sinogram");

    /*Allocate space for 2D array*/
    if (xdim && angdim)

    Data = malloc_matrix_f (angdim,xdim);

   /* Read the data */
    start[0] = 0;
    start[1] = 0;
    count[0] = angdim;
    count[1] = xdim;
    ncvarget (ncid, sinogram_id, start, count,
		 (void *)&Data[start[0]][start[1]]);
    ncvarget1 (ncid, centerid, (long *)0, (void *)&center);
    *cent = center;	 

    /* Close the netCDF file */
        ncclose (ncid);


    return Data;
}

static
void writenetCDF (char *name, float **m, int x, int y)
{
    char stoken[] = {"."};
    char *token;

    int ncid;
    int xid, yid;
    int slice_dimids[2], slice_id;
    long start[2], count[2];

   /* Create a netCDF file for writing */
    token = strtok(name, stoken);
    strcpy (CDFfile, token);
    strcat (CDFfile, ".cdf");
    if(verbose) printf(
		"Writing image to netCDF file %s\n", CDFfile);

    ncid = nccreate (CDFfile,  NC_CLOBBER);

    /* Define the dimensions */
    yid = ncdimdef (ncid, "y", y);
    xid = ncdimdef (ncid, "x", x);
    slice_dimids[0] = yid;
    slice_dimids[1] = xid;

    /* Define the variables */
    slice_id = ncvardef (ncid, "slice", NC_FLOAT, 2, slice_dimids);
            /* Leave defined mode */
    ncendef (ncid);

    /* Write the data */
    start[0] = 0;
    start[1] = 0;
    count[0] = y;
    count[1] = x;
    ncvarput (ncid, slice_id, start, count, (void **)&m[start[0]][start[1]]);


    /* Close the netCDF file */

    ncclose (ncid);
    return;
}


static
int fexist (char *filename) 
{
    /* Check for existence of a file */
    struct stat stbuf;
	int error;
	error=stat (filename, &stbuf);

	if(error) return 0;
	else return 1;
   
}

void usage(void)
{
	char *msg= "usage:\n\
gridrec [opts] [-p C,[sampl,[MaxPixSiz[,[R,[X0,[Y0]]]]]][opts]filenames\n\
\t\twhere C,sampl, etc. are parameters for the gridding algorithm,\n\
\t\tand where opts are any of the following:\n\
\t\t\t-f <name of filter>\n\
\t\t\t-l<length of convolvent lookup table>\n\n\
\t\t\t-v (verbose mode)\n\
\t\t\t-D <Directory path for data files>\n\
\t\t\t-c <value of center (overrides input data file)>\n";
    fprintf(stderr, msg);
    exit (2);
}

float *malloc_vector_f (long n) 
{
//    float *v = NULL;
    float *v;

    v = (float *) malloc((size_t) (n * sizeof(float)));
    if (!v) {
        fprintf (stderr, "malloc error in malloc_vector_f for length %ld.\n", n);
        v = NULL;
        return v;
    }
    return v;
}

complex *malloc_vector_c (long n) 
{
    complex *v;   // = NULL;
    
    v = (complex *) malloc((size_t) (n * sizeof(complex)));
    if (!v) {
        fprintf (stderr, "malloc error in malloc_vector_c for length %ld.\n", n);
        v = NULL;
        return v;
    }
    return v;
}

float **malloc_matrix_f (long nr, long nc)
{
    float **m;   // = NULL;
    long i;

    /* Allocate pointers to rows */

    m = (float **) malloc((size_t) (nr * sizeof(float *)));
    if (!m) {
        fprintf (stderr, "malloc error in malloc_matrix_f for %ld row pointers.\n", nr);
        m = NULL;
        return m;
    }
    /* Allocate rows and set the pointers to them */

    m[0] = (float *) malloc((size_t) (nr * nc * sizeof(float)));
    if (!m[0]) {
        fprintf (stderr, "malloc error in malloc_matrix_f for %ld row with %ld columns.\n", nr, nc);
        m[0] = NULL;
        free (m);
        m = NULL;
        return m;
    }

    for (i = 1; i < nr; i++) m[i] = m[i-1] + nc;

    return m;
}

complex **malloc_matrix_c (long nr, long nc)
{
    complex **m;   // = NULL;
    long i;

    /* Allocate pointers to rows */
    m = (complex **) malloc((size_t) (nr * sizeof(complex *)));
    if (!m) {
        fprintf (stderr, "malloc error in malloc_matrix_c for %ld row pointers.\n", nr);
        m = NULL;
        return m;
    }
    /* Allocate rows and set the pointers to them */
    m[0] = (complex *) malloc((size_t) (nr * nc * sizeof(complex)));
    if (!m[0]) {
        fprintf (stderr,
		 "malloc error in malloc_matrix_c for %ld row with %ld columns.\n",
			nr, nc);
        m[0] = NULL;
        free (m);
        m = NULL;
        return m;
    }
    for (i = 1; i < nr; i++) m[i] = m[i-1] + nc;

    return m;
}

void prn_gparm(grid_struct *G)
{
	printf("Contents of grid_struct:\n\t");
	prn_pswf(G->pswf);
	printf("\n  sampl=%f MaxPixSiz=%fR=%f X0=%f Y0=%f\n  filter=%s ltbl=%d\n",
		G->sampl,G->MaxPixSiz, G->R, G->X0, G->Y0, G->fname, G->ltbl);
	return;
}

void prn_pswf(pswf_struct *P)
{
	int i;
	printf("Contents of pswf: C=%f nt=%d lmbda=%f\n\tCoefficients:\n",
			P->C, P->nt, P->lmbda);
	for(i=0;i<P->nt/2+1;i++)
	{
		printf("\t\t%f\n",P->coefs[i]);
	}
}

void prn_sgparm(sg_struct *S)
{
	printf("Contents of sg_struct:\n");
	printf("\tn_ang=%d n_det=%d geom=%d center=%f\n",
		S->n_ang,S->n_det,S->geom,S->center);
}



