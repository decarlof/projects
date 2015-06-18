#ifndef recon_algorihmH
#define recon_algorihmH
//---------------------------------------------------------------------------

#include <string.h>
#include <stdlib.h>
#include <math.h>

// #include "logfileclass.h"
 
 extern "C" {
#include "fft.h"
}

//_____________________________________________________________________________________
 //Defines for reconstruction filter algorithm 
 
#define         FILTER_NONE		                0 
#define         FILTER_SHEPP_LOGAN              1 
#define         FILTER_HANN                     2 
#define         FILTER_HAMMING                  3 
#define         FILTER_RAMP                     4 
#define         FILTER_FBP                      5 
 
//---------------------------------------------------------------------------
 
typedef struct {
	float   r,
	        i;
} complex_struct;


//---------------------------------------------------------------------------

class Filter
{
public:
	Filter ();

	void setFilter (int filter);
	int getFilter (void);

	float filterData (float x);

private:
int     filter_type;

	float shlo (float x);	/* Shepp-Logan filter */
	float hann (float x);	/* Hann filter */ 
	float hamm (float x);	/* Hamming filter */ 
	float ramp (float x);	/* Ramp filter */ 
};

//---------------------------------------------------------------------------

class ReconAlgorithm
{
public:
	ReconAlgorithm (void);

	int numberOfSinogramsNeeded (void);
	int numberOfReconstructions (void);

	void setThetaList (float *theta_list, int theta_list_size);
	int getThetaListSize (void);
	float *getThetaList (void);

	void setSinogramDimensions (int x_dim, int y_dim);
	int getSinogramXDimension (void);
	int getSinogramYDimension (void);

	void setFilter (int filter_name);
	int getFilter (void);

	virtual void setSinoAndReconBuffers (int number, float *sinogram_address, float *reconstruction_address);

	virtual void init (void);
	virtual void reconstruct (void);
	virtual void destroy (void);

	// static void acknowledgements (LogFileClass *acknowledge_file);

protected:
int             num_sinograms_needed,
				theta_list_size;
unsigned long   sinogram_x_dim,
				sinogram_y_dim;
float           *theta_list,
				*sinogram1,
				*sinogram2,
				*reconstruction1,
				*reconstruction2;
Filter          filter;

};
//---------------------------------------------------------------------------

extern char         msg[256];

//---------------------------------------------------------------------------
#endif
