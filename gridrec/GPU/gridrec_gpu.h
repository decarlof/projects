#ifndef gridrecH
#define gridrecH
//---------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h> 
#include <string.h> 
#include <math.h> 
#include <stddef.h> 
#include <time.h> 
#include <sys/stat.h> 

#include <vector>
#include <iostream>
#include <fstream>

//---------------------------------------------------------------------------

#include "recon_algorithm.h"

//---------------------------------------------------------------------------

#define GPU_GRIDREC

#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <cutil_inline.h>
#include <cutil_math.h>

#ifdef GPU_GRIDREC

#define BLOCK_2DIM_X 32
#define BLOCK_2DIM_Y 32

typedef struct{
  int ny;       // index for projection (angle)
  int nx;       // index for ray 
  float wa;    // weight from PSWF
} Sample; 

typedef std::vector<Sample> listSample;

__global__ void freq_filter_kernel( cufftComplex * , cufftComplex * , dim3 );
__global__ void bp_kernel( cufftComplex*, Sample*, float*, int*, int, int, int, dim3);

#define GPU1_IMG_SIZE 4096
#define GPU2_IMG_SIZE 512
#define GPU3_IMG_SIZE 256
#define CPU_IMG_SIZE 128

#define GPU1_SAMPLE_LEN  16 // 32
#define GPU2_SAMPLE_LEN  32
#define GPU3_SAMPLE_LEN  64
#define CPU_SAMPLE_LEN  5120

#endif // GPU_GRIDREC

//--------------------------------------------------------------------------

/**** Macros and typedefs ****/
#define max(A,B) ((A)>(B)?(A):(B)) 
 
#define min(A,B) ((A)<(B)?(A):(B)) 
 
//#define free_matrix(A) (free(*(A)),free(A)) 
 
#define abs(A) ((A)>0 ?(A):-(A)) 
 
#define Cmult(A,B,C) {(A).r=(B).r*(C).r-(B).i*(C).i; (A).i=(B).r*(C).i+(B).i*(C).r;} /** A=B*C for complex_struct A, B, C. A must be distinct, and an lvalue */ 
 
#ifdef INTERP 
#define Cnvlvnt(X) (wtbl[(int)X]+(X-(int)X)*dwtbl[(int)X]) /* Linear interpolation version */ 
#else 
#define Cnvlvnt(X) (wtbl[(int)(X+0.5)])                 /* Nearest nbr version - no interpolation */ 
#endif 
 
#define TOLERANCE 0.1	/* For comparing centers of two sinugrams */ 
#define LTBL_DEF 512	/* Default lookup table length */ 
 
//---------------------------------------------------------------------------
#define NO_PSWFS 5
 
typedef struct PSWF_STRUCT { 
  /*Prolate spheroidal wave fcn (PSWF) data */ 
  float   C,          /* Parameter for particular 0th order pswf being used*/ 
          lmbda;      /* Eigenvalue */ 
  int     nt;         /* Degree of Legendre polynomial expansion */ 
  float   coefs[15];  /* Coeffs for Legendre polynomial expansion */ 
} pswf_struct; 


//--------------------------------------------------------------------------- 

class GridRec : public ReconAlgorithm
{
 public:

  GridRec (void);

  void setSinoAndReconBuffers (int number, float *sinogram_address, float *reconstruction_address);

  void init (void);
  void reconstruct (void);
  void destroy (void);

  void setGPUDeviceID( int id );
  int getGPUDeviceID( );

  // static void acknowledgements (LogFileClass *acknowledge_file);

 private:
  int     flag;

  long    pdim, 
          M, 
          M0, 
          M02, 
          ltbl, 
          imgsiz; 

  float   sampl, 
          scale, 
          L, 
          X0, 
          Y0; 

  float   *SINE, 
          *COSE, 
          *wtbl, 
          *dwtbl, 
          *work, 
          *winv; 

  complex_struct *cproj, 
          *filphase, 
          *H; 

  float   **G1, 
          **G2, 
          **S1, 
          **S2; 

  pswf_struct pswf_db[NO_PSWFS]; 
 
  void phase1 (void);
  void phase2 (void);
  void phase3 (void);

  void trig_su (int geom, int n_ang);
  void filphase_su (long pd, float center, complex_struct *A);
  void pswf_su (pswf_struct *pswf, long ltbl, long linv, float* wtbl, float* dwtbl, float* winv);
  float legendre (int n, float *coefs, float x);
  void get_pswf (float C, pswf_struct **P);

#ifdef GPU_GRIDREC
  // GPU
  cufftHandle plan_fft;
  cufftComplex* data_cproj;
  cufftComplex* d_data_cproj;
  cufftComplex* d_data_cproj_res;

  cufftComplex* data_filphase;
  cufftComplex* d_data_filphase;

  // 
  listSample*  list_pswf_recon; 
  listSample*  list_pswf_sino;

  std::vector<Sample>::iterator it;
  void list_pswf_su( listSample* list_pswf_recon, listSample* list_pswf_sino, float L, long ltbl, 
		     long pdim, long M, float scale );

  Sample* pswf_recon_GPU1;
  int *    pswf_recon_index_GPU1;
  Sample** pswf_recon_GPU2;
  Sample** pswf_recon_GPU3;
  Sample** pswf_recon_CPU; 
  float*   pswf_recon_size; 


  //
  cudaChannelFormatDesc float2Desc;
  float2 * float2_cproj_res;
  cudaArray* d_array_cproj_res;
  float* d_wtbl;

  Sample* d_pswf_recon_GPU1;
  int*    d_pswf_recon_index_GPU1; 
  Sample* d_pswf_recon_GPU2;
  Sample* d_pswf_recon_GPU3;
  Sample* d_pswf_recon_CPU;
  float* d_pswf_recon_size;


  //
  cufftHandle plan_ifft2;
  cufftComplex* data_H;
  cufftComplex* d_data_H;

#endif

  int deviceID; 

};

//---------------------------------------------------------------------------
#endif
