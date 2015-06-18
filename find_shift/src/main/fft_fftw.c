/*** File fft_fftw.c 2/26/2005
	 This file calls routines from the FFTW library.  It
	 emulates the 1-D and n-D routines from Numerical Recipes, so that
	 gridrec requires essentially no modification

	 Written by Mark Rivers

	 Modified:
		Brian Tieman 8/2/2006
		Modified for use in our tomography cluster.
 **/

#include <stdlib.h>
#include <string.h>
#include <fftw3.h>

/* Note that we are using the routines designed for
 * COMPLEX data type.  C does not normally support COMPLEX, but Gridrec uses
 * a C structure to emulate it.
 */

static int 				n_prev;
static fftwf_complex 	*in_1d,
						*out_1d;
static fftwf_plan 		forward_plan_1d,
						backward_plan_1d;
static int 				nx_prev,
						ny_prev;
static fftwf_complex 	*in_2d,
						*out_2d;
static fftwf_plan 		forward_plan_2d,
						backward_plan_2d;

//_______________________________________________________________________________________________________________

void initFFTMemoryStructures (void)
{
	n_prev = 0;
	in_1d = NULL;
	out_1d = NULL;

	nx_prev = 0;
	ny_prev = 0;
	in_2d = NULL;
	out_2d = NULL;
}

//_______________________________________________________________________________________________________________

void destroyFFTMemoryStructures (void)
{
	if (in_1d != NULL)
	  fftwf_free (in_1d);
	fftwf_destroy_plan (forward_plan_1d);
	fftwf_destroy_plan (backward_plan_1d);

	if (in_2d != NULL)
		fftwf_free (in_2d);
	fftwf_destroy_plan (forward_plan_2d);
	fftwf_destroy_plan (backward_plan_2d);

}

//_______________________________________________________________________________________________________________

void four1(float data[], unsigned long nn, int isign)
{
   int n = nn;

   if (n != n_prev)
   {
	  /* Create plans */
	  if (n_prev != 0)
	    fftwf_free(in_1d);

	  in_1d = fftwf_malloc(sizeof(fftwf_complex)*n);
	  out_1d = in_1d;
	  printf("fft_test1f: creating plans, n=%d, n_prev=%d\n", n, n_prev);
	  n_prev = n;
	  forward_plan_1d = fftwf_plan_dft_1d(n, in_1d, out_1d, FFTW_FORWARD, FFTW_MEASURE);
	  backward_plan_1d = fftwf_plan_dft_1d(n, in_1d, out_1d, FFTW_BACKWARD, FFTW_MEASURE);
   }
   /* The Numerical Recipes routines are passed a pointer to one element
	* before the start of the array - add one */
   memcpy(in_1d, data+1, n*sizeof(fftwf_complex));
   if (isign == -1)
	fftwf_execute(forward_plan_1d);
   else
	fftwf_execute(backward_plan_1d);
   memcpy(data+1, out_1d, n*sizeof(fftwf_complex));
}

//_______________________________________________________________________________________________________________

void fourn(float data[], unsigned long nn[], int ndim, int isign)
{
   int nx = nn[2];
   int ny = nn[1];

   /* NOTE: This function only works for ndim=2 */
   if (ndim != 2)
   {
	  printf("fourn only works with ndim=2\n");
	  return;
   }

   if ((nx != nx_prev) || (ny != ny_prev))
   {
	  /* Create plans */
	  if (nx_prev != 0)
		fftwf_free(in_1d);
	  in_2d = fftwf_malloc(sizeof(fftwf_complex)*nx*ny);
	  out_2d = in_2d;
	  printf("fft_test2f: creating plans, nx=%d, ny=%d, nx_prev=%d, ny_prev=%d\n",
			 nx, ny, nx_prev, ny_prev);
	  nx_prev = nx;
	  ny_prev = ny;
	  forward_plan_2d = fftwf_plan_dft_2d(ny, nx, in_2d, out_2d, FFTW_FORWARD, FFTW_MEASURE);
	  backward_plan_2d = fftwf_plan_dft_2d(ny, nx, in_2d, out_2d, FFTW_BACKWARD, FFTW_MEASURE);
   }
   /* The Numerical Recipes routines are passed a pointer to one element
	* before the start of the array - add one */
   memcpy(in_2d, data+1, nx*ny*sizeof(fftwf_complex));
   if (isign == -1)
	fftwf_execute(forward_plan_2d);
   else
	fftwf_execute(backward_plan_2d);
   memcpy(data+1, out_2d, nx*ny*sizeof(fftwf_complex));
}

//_______________________________________________________________________________________________________________

