//---------------------------------------------------------------------------

#pragma hdrstop

#include "gridrec.h"

#include <sys/time.h>

//---------------------------------------------------------------------------
#pragma package(smart_init)

#ifdef USE_BRUTE_FORCE_GPU
texture<float2, 2, cudaReadModeElementType> tex_cproj_res;
#endif

//---------------------------------------------------------------------------

GridRec::GridRec (void)
{

  pswf_db[0].C = 4.0;
  pswf_db[0].lmbda = 0.99588549;
  pswf_db[0].nt = 16;
  pswf_db[0].coefs[0] = 0.5239891E+01;
  pswf_db[0].coefs[1] = -0.5308499E+01;
  pswf_db[0].coefs[2] = 0.1184591E+01;
  pswf_db[0].coefs[3] = -0.1230763E-00;
  pswf_db[0].coefs[4] = 0.7371623E-02;
  pswf_db[0].coefs[5] = -0.2864074E-03;
  pswf_db[0].coefs[6] = 0.7789983E-05;
  pswf_db[0].coefs[7] = -0.1564700E-06;
  pswf_db[0].coefs[8] = 0.2414647E-08;
  pswf_db[0].coefs[9] = 0.0;
  pswf_db[0].coefs[10] = 0.0;
  pswf_db[0].coefs[11] = 0.0;
  pswf_db[0].coefs[12] = 0.0;
  pswf_db[0].coefs[13] = 0.0;
  pswf_db[0].coefs[14] = 0.0;

  pswf_db[1].C = 4.2;
  pswf_db[1].lmbda = 0.99657887;
  pswf_db[1].nt = 16;
  pswf_db[1].coefs[0] = 0.6062942E+01;
  pswf_db[1].coefs[1] = -0.6450252E+01;
  pswf_db[1].coefs[2] = 0.1551875E+01;
  pswf_db[1].coefs[3] = -0.1755960E-01;
  pswf_db[1].coefs[4] = 0.1150712E-01;
  pswf_db[1].coefs[5] = -0.4903653E-03;
  pswf_db[1].coefs[6] = 0.1464986E-04;
  pswf_db[1].coefs[7] = -0.3235110E-06;
  pswf_db[1].coefs[8] = 0.5492141E-08;
  pswf_db[1].coefs[9] = 0.0;
  pswf_db[1].coefs[10] = 0.0;
  pswf_db[1].coefs[11] = 0.0;
  pswf_db[1].coefs[12] = 0.0;
  pswf_db[1].coefs[13] = 0.0;
  pswf_db[1].coefs[14] = 0.0;

  pswf_db[2].C = 5.0;
  pswf_db[2].lmbda = 0.99935241;
  pswf_db[2].nt = 18;
  pswf_db[2].coefs[0] = 0.1115509E+02;
  pswf_db[2].coefs[1] = -0.1384861E+02;
  pswf_db[2].coefs[2] = 0.4289811E+01;
  pswf_db[2].coefs[3] = -0.6514303E-00;
  pswf_db[2].coefs[4] = 0.5844993E-01;
  pswf_db[2].coefs[5] = -0.3447736E-02;
  pswf_db[2].coefs[6] = 0.1435066E-03;
  pswf_db[2].coefs[7] = -0.4433680E-05;
  pswf_db[2].coefs[8] = 0.1056040E-06;
  pswf_db[2].coefs[9] = -0.1997173E-08;
  pswf_db[2].coefs[10] = 0.0;
  pswf_db[2].coefs[11] = 0.0;
  pswf_db[2].coefs[12] = 0.0;
  pswf_db[2].coefs[13] = 0.0;
  pswf_db[2].coefs[14] = 0.0;

  pswf_db[3].C = 6.0;
  pswf_db[3].lmbda = 0.9990188;
  pswf_db[3].nt = 18;
  pswf_db[3].coefs[0] = 0.2495593E+02;
  pswf_db[3].coefs[1] = -0.3531124E+02;
  pswf_db[3].coefs[2] = 0.1383722E+02;
  pswf_db[3].coefs[3] = -0.2799028E+01;
  pswf_db[3].coefs[4] = 0.3437217E-00;
  pswf_db[3].coefs[5] = -0.2818024E-01;
  pswf_db[3].coefs[6] = 0.1645842E-02;
  pswf_db[3].coefs[7] = -0.7179160E-04;
  pswf_db[3].coefs[8] = 0.2424510E-05;
  pswf_db[3].coefs[9] = -0.6520875E-07;
  pswf_db[3].coefs[10] = 0.0;
  pswf_db[3].coefs[11] = 0.0;
  pswf_db[3].coefs[12] = 0.0;
  pswf_db[3].coefs[13] = 0.0;
  pswf_db[3].coefs[14] = 0.0;

  pswf_db[4].C = 7.0;
  pswf_db[4].lmbda = 0.99998546;
  pswf_db[4].nt = 20;
  pswf_db[4].coefs[0] = 0.5767616E+02;
  pswf_db[4].coefs[1] = -0.8931343E+02;
  pswf_db[4].coefs[2] = 0.4167596E+02;
  pswf_db[4].coefs[3] = -0.1053599E+02;
  pswf_db[4].coefs[4] = 0.1662374E+01;
  pswf_db[4].coefs[5] = -0.1780527E-00;
  pswf_db[4].coefs[6] = 0.1372983E-01;
  pswf_db[4].coefs[7] = -0.7963169E-03;
  pswf_db[4].coefs[8] = 0.3593372E-04;
  pswf_db[4].coefs[9] = -0.1295941E-05;
  pswf_db[4].coefs[10] = 0.3817796E-07;
  pswf_db[4].coefs[11] = 0.0;
  pswf_db[4].coefs[12] = 0.0;
  pswf_db[4].coefs[13] = 0.0;
  pswf_db[4].coefs[14] = 0.0;

  num_sinograms_needed = 2;

  SINE = NULL;
  COSE = NULL;

  cproj = NULL;
  filphase = NULL; 
  wtbl = NULL; 
       
#ifdef INTERP 
  dwtbl = NULL; 
#endif 
	 
  winv = NULL; 
  work = NULL; 
  H = NULL; 
	       
  G1 = NULL;
  G2 = NULL; 
		 
  S1 = NULL; 
  S2 = NULL; 


#ifdef USE_BRUTE_FORCE_GPU
  // GPU

  data_H = NULL;
  d_data_H = NULL;
  deviceID = 0; 
#endif // USE_BRUTE_FORCE_GPU

  center_shift = 0.0f;
		     
} 

//---------------------------------------------------------------------------

// void GridRec::acknowledgements (LogFileClass *acknowledge_file)
// {
//   acknowledge_file->Message ("__________________________________________________________________");
//   acknowledge_file->Message ("GridRec class");
//   acknowledge_file->Message (""); 
//   acknowledge_file->Message ("Class for performing reconstructions based on the \"GridRec\" algorythm."); 
//   acknowledge_file->Message ("Origional source code developed in C by:"); 
//   acknowledge_file->Message ("       Still trying to find out who--there were no comments in the code "); 
//   acknowledge_file->Message ("Developed and Maintained by:"); 
//   acknowledge_file->Message ("       Brian Tieman & Francesco DeCarlo"); 
//   acknowledge_file->Message ("       Argonne National Laboratory"); 
//   acknowledge_file->Message ("       tieman@aps.anl.gov"); 
//   acknowledge_file->Message (""); 
//   acknowledge_file->Message ("8/20/2003  V1.0   BT  First version with acknowledgements"); 
//   acknowledge_file->Message ("8/20/2003  V1.0   BT  Ported C code to a CPP object structure"); 
//   acknowledge_file->Message (""); 
//   acknowledge_file->Message (""); 
//   acknowledge_file->Message ("__________________________________________________________________"); 
// }

//---------------------------------------------------------------------------

void GridRec::setSinoAndReconBuffers (int number, float *sinogram_address, float *reconstruction_address)
{
  int     loop;

  if (G1 == NULL)
    G1 = (float **) malloc((size_t) (theta_list_size * sizeof(float *)));
  if (G2 == NULL)
    G2 = (float **) malloc((size_t) (theta_list_size * sizeof(float *))); 
     
  if (S1 == NULL) 
    S1 = (float **) malloc((size_t) (imgsiz * sizeof(float *))); 
  if (S2 == NULL) 
    S2 = (float **) malloc((size_t) (imgsiz * sizeof(float *))); 

  if (number == 1)
    {
      sinogram1 = sinogram_address;
      reconstruction1 = reconstruction_address;

      for (loop=0;loop<theta_list_size;loop++)
	G1[loop] = &sinogram1[loop*sinogram_x_dim];
	     
      for (loop=0;loop<imgsiz;loop++) 
	S1[loop] = &reconstruction1[loop*sinogram_x_dim]; 
    }

  if (number == 2)
    {
      sinogram2 = sinogram_address;
      reconstruction2 = reconstruction_address;

      for (loop=0;loop<theta_list_size;loop++)
	G2[loop] = &sinogram2[loop*sinogram_x_dim]; 

      for (loop=0;loop<imgsiz;loop++) 
	S2[loop] = &reconstruction2[loop*sinogram_x_dim]; 
    } 
	   
} 

//---------------------------------------------------------------------------

void GridRec::init (void)
{
  float           center,
                  C, 
                  MaxPixSiz, 
                  R, 
                  D0, 
                  D1; 
  long            itmp; 
  pswf_struct     *pswf; 
	 
  center = sinogram_x_dim / 2 + center_shift; 

  sampl = 1.0; 
  MaxPixSiz = 1.0; 
  R = 1.0; 
  X0 = 0.0; 
  Y0 = 0.0; 
  ltbl = 512; 
  get_pswf (6.0, &pswf); 
  C = pswf->C; 
			   
  if (X0!=0.0||Y0!=0.0) 
    flag = 1; 
  else 
    flag = 0; 
			     
  pdim = 1; 
  itmp = sinogram_x_dim-1; 
  while (itmp) 
    { 
      pdim<<=1; 
      itmp>>=1; 
    } 
				   
  D0 = R*sinogram_x_dim; 
  D1 = sampl*D0; 
				       
  M = 1; 
  itmp = (long int) (D1/MaxPixSiz-1); 
  while (itmp) 
    { 
      M<<=1; 
      itmp>>=1; 
    } 
					     
  M02 = (long int) (floor(M/2/sampl-0.5)); 
  M0 = 2*M02+1; 
						 
  sampl = (float)M/M0; 
  D1 = sampl*D0; 
  L = 2*C*sampl/PI; 
  scale = D1/pdim; 
							 
  cproj = (complex_struct *) malloc ((pdim+1) * sizeof(complex_struct)); 
  filphase = (complex_struct *) malloc (((pdim/2)+1) * sizeof(complex_struct)); 
  wtbl = (float *) malloc ((ltbl+1) * sizeof(float)); 

#ifdef INTERP 
  dwtbl = (float *) malloc ((ltbl+1) * sizeof(float)); 
#endif 

  winv = (float *) malloc (M0 * sizeof(float)); 
  work = (float *) malloc (((int) L+1) * sizeof(float)); 
								     
  H = (complex_struct *) malloc ((M+1)*(M+1)*sizeof(complex_struct)); 
								       
  SINE = (float *) malloc (theta_list_size * sizeof (float)); 
  COSE = (float *) malloc (theta_list_size * sizeof (float)); 
									   
  trig_su (0, theta_list_size); 
									     
  filphase_su (pdim, center, filphase); 
									       
  pswf_su (pswf, ltbl, M02, wtbl, dwtbl, winv); 
										 
  imgsiz = M0; 

#ifdef USE_BRUTE_FORCE_GPU
  // GPU
  cudaSetDevice( deviceID );
  // printf("Use CUDA device %d\n", deviceID);   

  //
  data_H = new cufftComplex[ M * M ];
  cudaMalloc( (void**)&d_data_H, sizeof( cufftComplex ) * M * M );

  cufftResult res = cufftPlan2d( &plan_ifft2, M, M, CUFFT_C2C );

  if( res != CUFFT_SUCCESS )   
    printf("cufftPlan2d failed\n "); 

#endif // USE_BRUTE_FORCE_GPU
										   
} 

//---------------------------------------------------------------------------

void GridRec::reconstruct (void)
{
  memset (H, 0, (M+1)*(M+1)*sizeof(complex_struct));

  // #ifdef USE_BRUTE_FORCE_GPU
  // float timePhase1;
  // unsigned int timerPhase1 = 0;   // test
  // CUT_SAFE_CALL( cutCreateTimer( &timerPhase1 ) );
  // CUT_SAFE_CALL( cutStartTimer( timerPhase1 ) );
  // #endif // USE_BRUTE_FORCE_GPU

  phase1 (); 

  // #ifdef USE_BRUTE_FORCE_GPU
  // CUT_SAFE_CALL(cutStopTimer(timerPhase1));   // test
  // timePhase1 =  cutGetTimerValue(timerPhase1);
  // CUT_SAFE_CALL(cutDeleteTimer(timerPhase1));

  // printf("total time for Phase1 is %f ms \n", timePhase1);

  //

  // float timePhase2;
  // unsigned int timerPhase2 = 0;   // test
  // CUT_SAFE_CALL( cutCreateTimer( &timerPhase2 ) );
  // CUT_SAFE_CALL( cutStartTimer( timerPhase2 ) );
  // #endif 

  phase2 (); 

  // #ifdef USE_BRUTE_FORCE_GPU
  // CUT_SAFE_CALL(cutStopTimer(timerPhase2));   // test
  // timePhase2 =  cutGetTimerValue(timerPhase2);
  // CUT_SAFE_CALL(cutDeleteTimer(timerPhase2));

  // printf("total time for Phase2 is %f ms \n", timePhase2);

  // // 
  // float timePhase3;
  // unsigned int timerPhase3 = 0;   // test
  // CUT_SAFE_CALL( cutCreateTimer( &timerPhase3 ) );
  // CUT_SAFE_CALL( cutStartTimer( timerPhase3 ) );
  // #endif

  phase3 (); 

  // #ifdef USE_BRUTE_FORCE_GPU
  // CUT_SAFE_CALL(cutStopTimer(timerPhase3));   // test
  // timePhase3 =  cutGetTimerValue(timerPhase3);
  // CUT_SAFE_CALL(cutDeleteTimer(timerPhase3));

  // printf("total time for Phase3 is %f ms \n", timePhase3);
  // #endif
	   
  return; 
} 

//---------------------------------------------------------------------------

void GridRec::destroy (void)
{

  if (SINE != NULL)
    free (SINE);
  if (COSE != NULL) 
    free (COSE); 
     
  if (cproj != NULL) 
    free (cproj); 
       
  if (filphase != NULL) 
    free (filphase); 
  if (wtbl != NULL) 
    free (wtbl); 

#ifdef INTERP 
  if (dwtbl != NULL) 
    free (dwtbl); 
#endif 

  if (winv != NULL) 
    free (winv); 
  if (work != NULL) 
    free (work); 
		 
  if (H != NULL) 
    free (H); 
		   
  if (G1 != NULL) 
    free (G1); 
  if (G2 != NULL) 
    free (G2); 
  if (S1 != NULL) 
    free (S1); 
  if (S2 != NULL) 
    free (S2); 

  // 

#ifdef USE_BRUTE_FORCE_GPU
  // GPU

  cufftDestroy( plan_ifft2 );
  cudaFree( d_data_H );
  delete [] data_H;

#endif // USE_BRUTE_FORCE_GPU
			   
} 

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//Private Methods
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

void GridRec::phase1 (void)
{
  /***Phase 1 ***************************************
       
  Loop over the n_ang projection angles. For each angle, do 
  the following: 
       
  1. Copy the real projection data from the two slices into the 
  real and imaginary parts of the first n_det elements of the 
  complex array, cproj[].  Set the remaining pdim-n_det elements 
  to zero (zero-padding). 
       
  2. Carry out a (1D) Fourier transform on the complex data. 
  This results in transform data that is arranged in 
  "wrap-around" order, with non-negative spatial frequencies 
  occupying the first half, and negative frequencies the second 
  half, of the array, cproj[]. 
       
  3. Multiply each element of the 1-D transform by a complex, 
  frequency dependent factor, filphase[].  These factors were 
  precomputed as part of recon_init() and combine the 
  tomographic filtering with a phase factor which shifts the 
  origin in configuration space to the projection of the 
  rotation axis as defined by the parameter, "center".  If a 
  region of interest (ROI) centered on a different origin has 
  been specified [(X0,Y0)!=(0,0)], multiplication by an 
  additional phase factor, dependent on angle as well as 
  frequency, is required. 
       
  4. For each data element, find the Cartesian coordinates, 
  <U,V>, of the corresponding point in the 2D frequency plane, 
  in units of the spacing in the MxM rectangular grid placed 
  thereon; then calculate the upper and lower limits in each 
  coordinate direction of the integer coordinates for the 
  grid points contained in an LxL box centered on <U,V>. 
  Using a precomputed table of the (1-D) convolving function, 
  W, calculate the contribution of this data element to the 
  (2-D) convolvent (the 2_D convolvent is the product of 
  1_D convolvents in the X and Y directions) at each of these 
  grid points, and update the complex 2D array H accordingly. 
       
       
  At the end of Phase 1, the array H[][] contains data arranged in 
  "natural", rather than wrap-around order -- that is, the origin in 
  the spatial frequency plane is situated in the middle, rather than 
  at the beginning, of the array, H[][].  This simplifies the code 
  for carrying out the convolution (step 4 above), but necessitates 
  an additional correction -- See Phase 3 below. 
  **********************************************************************/ 

  complex_struct     Cdata1, Cdata2, Ctmp; 
  float       U, V, 
    rtmp, 
    L2 = L/2.0, 
    convolv, 
    tblspcg = 2*ltbl/L; 
  long        pdim2=pdim>>1, 
    M2=M>>1, 
    iul, 
    iuh, 
    iu, 
    ivl, 
    ivh, 
    iv, 
    n; 

  /* Following are to handle offset ROI case */ 
  float       offset=0.0;    // !!!!!!!!!!!!!!!!! =0.0 l'ho aggiunto io. !!!!!!!!!!!!!!!!!!!!!!!!!! 
  complex_struct     phfac; 

  for (n = 0; n < theta_list_size; n++) {    /*** Start loop on angles */ 
    
    int j, k; 
		 
    if (flag) 
      offset = (X0 * COSE[n] + Y0*SINE[n]) * PI; 

    j = 1; 
    while (j < sinogram_x_dim + 1) {
	
      cproj[j].r = G1[n][j-1]; 
      cproj[j].i = G2[n][j-1]; 
      j++; 
    } 
		       
    while (j < pdim)  { /** Zero fill the rest of array **/ 
	
      cproj[j].r = cproj[j].i = 0.0; 
      j++; 
    } 

    four1 ((float *) cproj+1, pdim, 1); 

    for (j = 1; j < pdim2; j++) {  	/* Start loop on transform data */   // 550 ms

      if (!flag) {
	    
	Ctmp.r = filphase[j].r; 
	Ctmp.i = filphase[j].i; 
      } 
      else { 
	phfac.r = cos(j*offset); 
	phfac.i = -sin(j*offset); 
	Cmult (Ctmp, filphase[j], phfac); 
      } 

      Cmult (Cdata1, Ctmp, cproj[j+1]) 
	Ctmp.i = -Ctmp.i; 
      Cmult (Cdata2, Ctmp, cproj[(pdim-j)+1]) 

      U = (rtmp=scale*j) * COSE[n] + M2;    /* X direction */ 
      V = rtmp * SINE[n] + M2;	            /* Y direction */ 
				       
      /* Note freq space origin is at (M2,M2), but we 
         offset the indices U, V, etc. to range from 0 to M-1 */ 
				       
      iul = (long int) (ceil(U-L2)); 
      iuh = (long int) (floor(U+L2)); 
      ivl = (long int) (ceil(V-L2)); 
      ivh = (long int) (floor(V+L2)); 
					       
      if (iul<1) 
	iul=1; 
      if (iuh>=M) 
	iuh=M-1; 
      if (ivl<1) 
	ivl=1; 
      if (ivh>=M) 
	ivh=M-1; 
						       
      /* Note aliasing value (at index=0) is forced to zero */ 
						       
      for (iv=ivl,k=0;iv<=ivh;iv++,k++) 
	work[k] = Cnvlvnt (abs (V-iv) * tblspcg); 

      for (iu=iul;iu<=iuh;iu++) {
	rtmp=Cnvlvnt (abs(U-iu)*tblspcg); 

	for (iv=ivl,k=0;iv<=ivh;iv++,k++) {
	  convolv = rtmp*work[k]; 

	  H[iu*M+iv+1].r += convolv * Cdata1.r; 
	  H[iu*M+iv+1].i += convolv * Cdata1.i; 
	  H[(M-iu)*M+(M-iv)+1].r += convolv * Cdata2.r; 
	  H[(M-iu)*M+(M-iv)+1].i += convolv * Cdata2.i; 

	} 
      } 
    } /*** End loop on transform data */ 
  }   /*** End loop on angles */ 

} 

//---------------------------------------------------------------------------

void GridRec::phase2 (void)
{
  /*** Phase 2 ********************************************
        
  Carry out a 2D inverse FFT on the array H. 
        
  At the conclusion of this phase, the configuration 
  space data is arranged in wrap-around order with the origin 
  (center of reconstructed images) situated at the start of the 
  array.  The first (resp. second) half of the array 
  contains the  lower, Y<0 (resp, upper Y>0) part of the 
  image, and within each row of the  array, the first 
  (resp. second) half contains data for the right [X>0] 
  (resp. left [X<0]) half of the image. 
        
  ********************************************************************/ 
   
#ifdef USE_BRUTE_FORCE_GPU       

  float time_fft = 0;
  unsigned int timerGridRec = 0;  
  CUT_SAFE_CALL( cutCreateTimer( &timerGridRec ) );
  CUT_SAFE_CALL( cutStartTimer( timerGridRec ) );

  for( int ny = 0; ny < M ; ny++ ){
    for( int nx = 0; nx < M ; nx++ ){
      data_H[ ny * M + nx ].x = H[ ny * M + nx + 1 ].r;
      data_H[ ny * M + nx ].y = H[ ny * M + nx + 1 ].i;
    }
  }

  cudaMemcpy( d_data_H, data_H, sizeof( cufftComplex ) * M * M, cudaMemcpyHostToDevice );

  cufftResult res = cufftExecC2C( plan_ifft2, d_data_H, d_data_H, CUFFT_INVERSE );
  if( res != CUFFT_SUCCESS )
    printf("cufftExecC2C failed\n "); 

  cudaMemcpy( data_H, d_data_H, sizeof( cufftComplex ) * M * M, cudaMemcpyDeviceToHost );

  // Note the coordinate transform here. 
  for( int ny = 0; ny < M ; ny++ ){
    for( int nx = 0; nx < M ; nx++ ){
      H[ ny * M + nx + 1 ].r = data_H[ (M - 1 - ny) * M + M - 1 - nx].x;  
      H[ ny * M + nx + 1 ].i = data_H[ (M - 1 - ny) * M + M - 1 - nx].y;
    }
  }

  CUT_SAFE_CALL(cutStopTimer(timerGridRec));   
  time_fft +=  cutGetTimerValue(timerGridRec);
  CUT_SAFE_CALL(cutDeleteTimer(timerGridRec));

  // printf( "Time for fft in Phase 2 is %f (ms)\n ", time_fft );

#endif


#ifndef USE_BRUTE_FORCE_GPU 
  unsigned long   H_size[3]; 
  H_size[1] = H_size[2] = M; 

  fourn ((float*) H+1, H_size, 2, -1); 
#endif

} 
	 
//---------------------------------------------------------------------------

void GridRec::phase3 (void)
{
  /*** Phase 3 ******************************************************
        
  Copy the real and imaginary parts of the complex data from H[][], 
  into the output buffers for the two reconstructed real images, 
  simultaneously carrying out a final multiplicative correction. 
  The correction factors are taken from the array, winv[], previously 
  computed in pswf_su(), and consist logically of three parts, namely: 
        
  1. A positive real factor, corresponding to the reciprocal 
  of the inverse Fourier transform, of the convolving 
  function, W, and 
        
  2. Multiplication by the cell size, (1/D1)^2, in 2D frequency 
  space.  This correctly normalizes the 2D inverse FFT carried 
  out in Phase 2.  (Note that all quantities are ewxpressed in 
  units in which the detector spacing is one.) 
        
  3. A sign change for the "odd-numbered" elements (in a 
  checkerboard pattern) of the array.  This compensates 
  for the fact that the 2-D Fourier transform (Phase 2) 
  started with a frequency array in which the zero frequency 
  point appears in the middle of the array instead of at 
  its start. 
        
  Only the elements in the square M0xM0 subarray of H[][], centered 
  about the origin, are utilized.  The other elements are not 
  part of the actual region being reconstructed and are 
  discarded.  Because of the  wrap-around ordering, the 
  subarray must actually be taken from the four corners" of the 
  2D array, H[][] -- See Phase 2 description, above. 
  The final data correponds physically to the linear X-ray absorption 
  coefficient expressed in units of the inverse detector spacing -- to 
  convert to inverse cm (say), one must divide the data by the detector 
  spacing in cm. 
        
  *********************************************************************/ 
   
  long    iu, iv, j, k, ustart, vstart, ufin, vfin; 
  float   corrn_u, corrn; 
       
  j = 0; 
  ustart = (M-M02); 
  ufin = M; 
  while (j<M0) {
    
    for (iu = ustart; iu < ufin; j++,iu++) {
	
      corrn_u = winv[j]; 
      k=0; 
      vstart = (M-M02); 
      vfin=M; 

      while (k<M0) {
	for (iv=vstart;iv<vfin;k++,iv++) {
		
	  corrn = corrn_u * winv[k]; 
	  S1[j][k] = corrn * H[iu*M+iv+1].r; 
	  S2[j][k] = corrn * H[iu*M+iv+1].i; 
	} 
	if (k<M0) 
	  (vstart = 0, vfin = M02 + 1); 
      } 
    } 
    if (j<M0) 
      (ustart = 0, ufin = M02 + 1); 
  } 
} 
	       
//---------------------------------------------------------------------------
	       
void GridRec::trig_su (int geom, int n_ang)
{ 
  /*********** Set up tables of sines and cosines. ***********/ 
   
  int     j; 
     
  switch (geom) 
    { 
    case 0 : { 
      float   theta, 
	degtorad = PI/180, 
	*angle = theta_list; 
	   
      for (j=0;j<n_ang;j++) 
	{ 
	  theta = degtorad*angle[j]; 
	  SINE[j] = sin(theta); 
	  COSE[j] = cos(theta); 
	} 
      break; 
    } 
	   
    case 1 : 
    case 2 : { 
      float   dtheta = geom*PI/n_ang, 
	dcos, 
	dsin; 
		 
      dcos = cos (dtheta); 
      dsin = sin (dtheta); 
      SINE[0] = 0.0; 
      COSE[0] = 1.0; 
      for(j=1;j<n_ang;j++) 
	{ 
	  SINE[j] = dcos*SINE[j-1]+dsin*COSE[j-1]; 
	  COSE[j] = dcos*COSE[j-1]-dsin*SINE[j-1]; 
	} 
      break; 
    } 
	   
    default : { 
      fprintf (stderr, "Illegal value for angle geometry indicator.\n"); 
      exit(2); 
    } 
    } 
       
} 
       
//---------------------------------------------------------------------------

void GridRec::filphase_su (long pd, float center, complex_struct *A)
{ 
  /******************************************************************/ 
  /* Set up the complex array, filphase[], each element of which    */ 
  /* consists of a real filter factor [obtained from the function,  */ 
  /* (*pf)()], multiplying a complex phase factor (derived from the */ 
  /* parameter, center}.  See Phase 1 comments in do_recon(), above.*/ 
  /******************************************************************/ 
   
  long    j, 
    pd2=pd>>1; 
  float   x, 
    rtmp1=2*PI*center/pd, 
    rtmp2; 

  float   norm=PI/pd/theta_list_size;	/* Normalization factor for back transform  7/7/98  */ 
  	 
  for (j=0;j<pd2;j++) 
    { 
      x = j*rtmp1; 
      rtmp2 = filter.filterData ((float)j/pd) * norm; 
      A[j].r = rtmp2*cos(x); 
      A[j].i = -rtmp2*sin(x); 
    } 
	   
  // Note: filphase[] (A[]) is of size pdim2 (pd2) + 1. But only the first pdim2 elements
  //       are set. The last one filphase[pdim2] is not assigned any value.
  //       8/24/2011        Yongsheng Pan

} 

//---------------------------------------------------------------------------

void GridRec::setCenterShift (float shift)
{

  // apply this before GridRec::init()
  center_shift = shift; 
}


//---------------------------------------------------------------------------

void GridRec::pswf_su (pswf_struct *pswf, long ltbl, long linv, float* wtbl, float* dwtbl, float* winv)
{ 
  /*************************************************************/ 
  /* Set up lookup tables for convolvent (used in Phase 1 of   */ 
  /* do_recon()), and for the final correction factor (used in */ 
  /* Phase 3).                                                  */ 
  /*************************************************************/ 
   
  float   C, 
    *coefs, 
    lmbda, 
    polyz, 
    norm,fac; 
  long    i; 
  int     nt; 
	 
  C=pswf->C; 
  nt=pswf->nt; 
  coefs=pswf->coefs; 
  lmbda=pswf->lmbda; 
  polyz=legendre(nt,coefs,0.); 
		   
  wtbl[0]=1.0; 
  for (i=1;i<=ltbl;i++) 
    { 
      wtbl[i]=legendre(nt,coefs,(float)i/ltbl)/polyz; 
#ifdef INTERP 
      dwtbl[i]=wtbl[i]-wtbl[i-1]; 
#endif 
    } 
		       
  fac=(float)ltbl/(linv+0.5); 
  norm=sqrt (PI/2/C/lmbda)/sampl;	/* 7/7/98 */ 
			   
  /* Note the final result at end of Phase 3 contains the factor, 
     norm^2.  This incorporates the normalization of the 2D 
     inverse FFT in Phase 2 as well as scale factors involved 
     in the inverse Fourier transform of the convolvent. 
     7/7/98 			*/ 
			   
  winv[linv]=norm/Cnvlvnt(0.); 
  for (i=1;i<=linv;i++) 
    { 
      norm=-norm; 
      /* Minus sign for alternate entries 
	 corrects for "natural" data layout 
	 in array H at end of Phase 1.  */ 
				   
      winv[linv+i] = winv[linv-i] = norm/Cnvlvnt(i*fac); 
    } 
} 
			       
//---------------------------------------------------------------------------

float GridRec::legendre (int n, float *coefs, float x)
{
  /*************************************************** 
   *                                                  * 
   *    Compute SUM(coefs(k)*P(2*k,x), for k=0,n/2)   * 
   *                                                  * 
   *    where P(j,x) is the jth Legendre polynomial   * 
   *                                                  * 
   ***************************************************/ 
				 
  float   penult, last, newer, y; 
  int     j, k, even; 
				     
  if (x>1||x<-1){ 
    
    fprintf(stderr, "\nInvalid argument to legendre()"); 
    exit(2); 
  } 
				       
  y=coefs[0]; 
  penult=1.; 
  last=x; 
  even=1; 
  k=1; 
  for (j=2;j<=n;j++) {
    
    newer=(x*(2*j-1)*last-(j-1)*penult)/j; 
    if (even) { 
      y+=newer*coefs[k]; 
      even=0; 
      k++; 
    } 
    else 
      even=1; 
							 
    penult=last; 
    last=newer; 
  } 
						   
  return y; 
						     
}

//---------------------------------------------------------------------------

void GridRec::get_pswf (float C, pswf_struct **P)
{
  int i=0; 
     
  while (i<NO_PSWFS && abs(C-pswf_db[i].C)>0.01) 
    i++; 
       
  if (i>=NO_PSWFS) 
    { 
      fprintf(stderr, "Prolate parameter, C = %f not in data base\n",C); 
      exit(2); 
    } 
	 
  *P = &pswf_db[i]; 
	   
  return; 
} 

void GridRec::setGPUDeviceID(int id ){
  deviceID = id;
}

int GridRec::getGPUDeviceID( ){
  return deviceID;
}


