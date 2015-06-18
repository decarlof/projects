//---------------------------------------------------------------------------

#pragma hdrstop

#include "gridrec_gpu.h"

#include <sys/time.h>

//---------------------------------------------------------------------------
#pragma package(smart_init)

#ifdef GPU_GRIDREC
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

#ifdef GPU_GRIDREC
  // GPU
  data_cproj = NULL;
  d_data_cproj = NULL;
  d_data_cproj_res = NULL;

  data_filphase = NULL;
  d_data_filphase = NULL;

  //   
  list_pswf_recon = NULL;
  list_pswf_sino = NULL;

  pswf_recon_GPU1 = NULL;
  pswf_recon_GPU2 = NULL;
  pswf_recon_GPU3 = NULL;
  pswf_recon_CPU = NULL;
  pswf_recon_size = NULL;

  // 
  d_wtbl = NULL; 

  d_pswf_recon_GPU1 = NULL;
  d_pswf_recon_GPU2 = NULL;
  d_pswf_recon_GPU3 = NULL;
  d_pswf_recon_CPU = NULL;
  d_pswf_recon_size = NULL; 

  float2_cproj_res = NULL;

  //
  data_H = NULL;
  d_data_H = NULL;
  deviceID = 0; 
#endif // GPU_GRIDREC
		     
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
	 
  center = sinogram_x_dim / 2; 
	   
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

  
#ifdef GPU_GRIDREC
  // GPU
  cudaSetDevice( deviceID );

  data_cproj = new cufftComplex[ M * theta_list_size ];
  cudaMalloc( (void**)&d_data_cproj, sizeof( cufftComplex ) * M * theta_list_size );
  cudaMalloc( (void**)&d_data_cproj_res, sizeof( cufftComplex ) * M * theta_list_size );

  cufftPlan1d( &plan_fft, M, CUFFT_C2C, theta_list_size );

  // 
  data_filphase = new cufftComplex[ M ];
  cudaMalloc( (void**)&d_data_filphase, sizeof( cufftComplex ) * M );

  data_filphase[0].x = 0.0f;
  data_filphase[0].y = 0.0f; 

  data_filphase[ M/2 ].x = 0.0f;
  data_filphase[ M/2 ].y = 0.0f; 
  
  for(int i = 1; i < M / 2; i++ ){
    data_filphase[ i ].x = filphase[ i ].r;  
    data_filphase[ i ].y = filphase[ i ].i;
    
    data_filphase[ M - i ].x = filphase[ i ].r;    // index here
    data_filphase[ M - i ].y = -filphase[ i ].i;   // sign change here
  }

  cudaMemcpy( d_data_filphase, data_filphase, sizeof(cufftComplex) * M, cudaMemcpyHostToDevice );

  // 
  list_pswf_recon = new listSample[ M * M ];
  list_pswf_sino = new listSample[ M * theta_list_size ];

  list_pswf_su( list_pswf_recon, list_pswf_sino, L, ltbl, pdim, M, scale );

  // 
  pswf_recon_size = new float[ M * M ];
  if( !pswf_recon_size ){
    printf("Error allocating memory for pswf_recon_size\n");
    exit(1);
  }

  int sz; 
  for (int n = 0; n < M; n++) {   
    for (int j = 0; j < M; j++) { 
      sz = list_pswf_recon[ n * M + j ].size();
      pswf_recon_size[ n * M + j ] = 1.0f * sz; 
    }
  }

  cudaMalloc( (void**)&d_pswf_recon_size, sizeof( float ) * M * M );  
  CUDA_SAFE_CALL( cudaMemcpy( d_pswf_recon_size, pswf_recon_size, sizeof(float)*M*M, cudaMemcpyHostToDevice ) );

  // 
  int len_GPU1 = 0;
  for (int n = 0; n < GPU1_IMG_SIZE; n++) {   
    for (int j = 0; j < GPU1_IMG_SIZE; j++) { 

      sz = list_pswf_recon[ (M/2 - GPU1_IMG_SIZE/2 + n) * M + (M/2 - GPU1_IMG_SIZE/2 + j) ].size();
      Sample s;

      if( sz > GPU1_SAMPLE_LEN ){
	len_GPU1 += GPU1_SAMPLE_LEN;
      
      }
      else if( sz > 0 && sz <= GPU1_SAMPLE_LEN ){
	len_GPU1 += sz;
      }
    }
  }  

  // 
  pswf_recon_GPU1 = new Sample[ len_GPU1 ];       // explicit programming
  pswf_recon_GPU2 = new Sample*[ GPU2_IMG_SIZE * GPU2_IMG_SIZE];
  pswf_recon_GPU3 = new Sample*[ GPU3_IMG_SIZE * GPU3_IMG_SIZE]; 
  pswf_recon_CPU = new Sample*[ CPU_IMG_SIZE * CPU_IMG_SIZE] ; 

  pswf_recon_index_GPU1 = new int[ GPU1_IMG_SIZE * GPU1_IMG_SIZE];

  if( !pswf_recon_GPU1 || !pswf_recon_GPU2 || !pswf_recon_GPU3 || !pswf_recon_CPU || !pswf_recon_index_GPU1){
    printf("Error allocating memory for pswf_recon!" ); 
    exit(1);
  }

  len_GPU1 = 0; 
  for (int n = 0; n < GPU1_IMG_SIZE; n++) {   
    for (int j = 0; j < GPU1_IMG_SIZE; j++) { 
      sz = list_pswf_recon[ (M/2 - GPU1_IMG_SIZE/2 + n) * M + (M/2 - GPU1_IMG_SIZE/2 + j) ].size();
      Sample s;

      pswf_recon_index_GPU1[ n * GPU1_IMG_SIZE + j ] = len_GPU1;

      if( sz > GPU1_SAMPLE_LEN ){

	for(int k = 0; k < GPU1_SAMPLE_LEN; k++ ){
	  s = list_pswf_recon[ (M/2 - GPU1_IMG_SIZE/2 + n) * M + (M/2 - GPU1_IMG_SIZE/2 + j) ][ k ];

	  pswf_recon_GPU1[ len_GPU1 + k ].nx = s.nx;
	  pswf_recon_GPU1[ len_GPU1 + k ].ny = s.ny;
	  pswf_recon_GPU1[ len_GPU1 + k ].wa = s.wa;
	}
	len_GPU1 += GPU1_SAMPLE_LEN;

      }
      else if( sz > 0 && sz <= GPU1_SAMPLE_LEN ){

	for(int k = 0; k < sz; k++ ){
	  s = list_pswf_recon[ (M/2 - GPU1_IMG_SIZE/2 + n) * M + (M/2 - GPU1_IMG_SIZE/2 + j) ][ k ];

	  pswf_recon_GPU1[ len_GPU1 + k ].nx = s.nx;
	  pswf_recon_GPU1[ len_GPU1 + k ].ny = s.ny;
	  pswf_recon_GPU1[ len_GPU1 + k ].wa = s.wa;
	}
	len_GPU1 += sz;
      }
    }
  }  

  // 
  // int len_GPU2 = 0;
  // for (int n = 0; n < GPU2_IMG_SIZE; n++) {   
  //   for (int j = 0; j < GPU2_IMG_SIZE; j++) { 
  //     sz = list_pswf_recon[ (M/2 - GPU2_IMG_SIZE/2 + n) * M + (M/2 - GPU2_IMG_SIZE/2 + j) ].size();
  //     Sample s;

  //     if( sz > GPU1_SAMPLE_LEN + GPU2_SAMPLE_LEN ){
  // 	pswf_recon_GPU2[ n * GPU2_IMG_SIZE + j ] = new Sample[ GPU2_SAMPLE_LEN ];
  // 	len_GPU2 += GPU2_SAMPLE_LEN;

  // 	for(int k = 0; k < GPU2_SAMPLE_LEN; k++ ){
  // 	  s = list_pswf_recon[ (M/2 - GPU2_IMG_SIZE/2 + n) * M + (M/2 - GPU2_IMG_SIZE/2 + j) ][ k + GPU1_SAMPLE_LEN];

  // 	  pswf_recon_GPU2[ n * GPU2_IMG_SIZE + j ][ k ].nx = s.nx;
  // 	  pswf_recon_GPU2[ n * GPU2_IMG_SIZE + j ][ k ].ny = s.ny;
  // 	  pswf_recon_GPU2[ n * GPU2_IMG_SIZE + j ][ k ].wa = s.wa;
  // 	}

  //     }
  //     else if( sz > GPU1_SAMPLE_LEN && sz <= GPU1_SAMPLE_LEN + GPU2_SAMPLE_LEN ){
  // 	pswf_recon_GPU2[ n * GPU2_IMG_SIZE + j ] = new Sample[ sz - GPU1_SAMPLE_LEN];
  // 	len_GPU2 += sz - GPU1_SAMPLE_LEN;

  // 	for(int k = 0; k < sz - GPU1_SAMPLE_LEN; k++ ){
  // 	  s = list_pswf_recon[ (M/2 - GPU2_IMG_SIZE/2 + n) * M + (M/2 - GPU2_IMG_SIZE/2 + j) ][ k + GPU1_SAMPLE_LEN];

  // 	  pswf_recon_GPU2[ n * GPU2_IMG_SIZE + j ][ k ].nx = s.nx;
  // 	  pswf_recon_GPU2[ n * GPU2_IMG_SIZE + j ][ k ].ny = s.ny;
  // 	  pswf_recon_GPU2[ n * GPU2_IMG_SIZE + j ][ k ].wa = s.wa;
  // 	}

  //     }
  //   }
  // }  

  // // 
  // int len_GPU3 = 0;
  // for (int n = 0; n < GPU3_IMG_SIZE; n++) {   
  //   for (int j = 0; j < GPU3_IMG_SIZE; j++) { 
  //     sz = list_pswf_recon[ (M/2 - GPU3_IMG_SIZE/2 + n) * M + (M/2 - GPU3_IMG_SIZE/2 + j) ].size();
  //     Sample s;

  //     if( sz > GPU1_SAMPLE_LEN + GPU2_SAMPLE_LEN + GPU3_SAMPLE_LEN){
  // 	pswf_recon_GPU3[ n * GPU3_IMG_SIZE + j ] = new Sample[ GPU3_SAMPLE_LEN ];
  // 	len_GPU3 += GPU3_SAMPLE_LEN;

  // 	for(int k = 0; k < GPU3_SAMPLE_LEN; k++ ){
  // 	  s = list_pswf_recon[ (M/2 - GPU3_IMG_SIZE/2 + n) * M + (M/2 - GPU3_IMG_SIZE/2 + j) ][ k + GPU1_SAMPLE_LEN + GPU2_SAMPLE_LEN ];

  // 	  pswf_recon_GPU3[ n * GPU3_IMG_SIZE + j ][ k ].nx = s.nx;
  // 	  pswf_recon_GPU3[ n * GPU3_IMG_SIZE + j ][ k ].ny = s.ny;
  // 	  pswf_recon_GPU3[ n * GPU3_IMG_SIZE + j ][ k ].wa = s.wa;
  // 	}

  //     }
  //     else if( sz > GPU1_SAMPLE_LEN + GPU2_SAMPLE_LEN && sz <= GPU1_SAMPLE_LEN + GPU2_SAMPLE_LEN + GPU3_SAMPLE_LEN ){
  // 	pswf_recon_GPU3[ n * GPU3_IMG_SIZE + j ] = new Sample[ sz - GPU1_SAMPLE_LEN - GPU2_SAMPLE_LEN ];
  // 	len_GPU3 += sz - GPU1_SAMPLE_LEN - GPU2_SAMPLE_LEN;

  // 	for(int k = 0; k < sz - GPU1_SAMPLE_LEN - GPU2_SAMPLE_LEN; k++ ){
  // 	  s = list_pswf_recon[ (M/2 - GPU3_IMG_SIZE/2 + n) * M + (M/2 - GPU3_IMG_SIZE/2 + j) ][ k + GPU1_SAMPLE_LEN + GPU2_SAMPLE_LEN ];

  // 	  pswf_recon_GPU3[ n * GPU3_IMG_SIZE + j ][ k ].nx = s.nx;
  // 	  pswf_recon_GPU3[ n * GPU3_IMG_SIZE + j ][ k ].ny = s.ny;
  // 	  pswf_recon_GPU3[ n * GPU3_IMG_SIZE + j ][ k ].wa = s.wa;
  // 	}

  //     }
  //   }
  // }  

  // // 
  // for (int n = 0; n < CPU_IMG_SIZE; n++) {   
  //   for (int j = 0; j < CPU_IMG_SIZE; j++) { 
  //     sz = list_pswf_recon[ (M/2 - CPU_IMG_SIZE/2 + n) * M + (M/2 - CPU_IMG_SIZE/2 + j) ].size();
  //     Sample s;

  //     if( sz > GPU1_SAMPLE_LEN + GPU2_SAMPLE_LEN + GPU3_SAMPLE_LEN){
  // 	pswf_recon_CPU[ n * CPU_IMG_SIZE + j ] = new Sample[ sz - GPU1_SAMPLE_LEN - GPU2_SAMPLE_LEN - GPU3_SAMPLE_LEN ];

  // 	for(int k = 0; k < sz - GPU1_SAMPLE_LEN - GPU2_SAMPLE_LEN - GPU3_SAMPLE_LEN; k++ ){
  // 	  s = list_pswf_recon[ (M/2 - CPU_IMG_SIZE/2 + n) * M + (M/2 - CPU_IMG_SIZE/2 + j) ][ k + GPU1_SAMPLE_LEN + GPU2_SAMPLE_LEN + GPU3_SAMPLE_LEN ];

  // 	  pswf_recon_CPU[ n * CPU_IMG_SIZE + j ][ k ].nx = s.nx;
  // 	  pswf_recon_CPU[ n * CPU_IMG_SIZE + j ][ k ].ny = s.ny;
  // 	  pswf_recon_CPU[ n * CPU_IMG_SIZE + j ][ k ].wa = s.wa;
  // 	}

  //     }

  //   }
  // }  

  printf( "GPU memory is  %d MB \n", len_GPU1 * sizeof( Sample) / 1024 / 1024 ); 

  CUDA_SAFE_CALL( cudaMalloc( (void**)&d_pswf_recon_GPU1, sizeof( Sample ) * len_GPU1 ) );  
  // CUDA_SAFE_CALL( cudaMalloc( (void**)&d_pswf_recon_GPU2, sizeof( Sample ) * GPU2_IMG_SIZE * GPU2_IMG_SIZE * GPU2_SAMPLE_LEN ) );  
  // CUDA_SAFE_CALL( cudaMalloc( (void**)&d_pswf_recon_GPU3, sizeof( Sample ) * GPU3_IMG_SIZE * GPU3_IMG_SIZE * GPU3_SAMPLE_LEN ) );  
  // cudaMalloc( (void**)&d_pswf_recon_CPU, sizeof( Sample ) * CPU_IMG_SIZE * CPU_IMG_SIZE * CPU_SAMPLE_LEN );  

  CUDA_SAFE_CALL( cudaMemcpy( d_pswf_recon_GPU1, pswf_recon_GPU1, sizeof(Sample) * len_GPU1, cudaMemcpyHostToDevice ) );

  // CUDA_SAFE_CALL( cudaMemcpy( d_pswf_recon_GPU2, pswf_recon_GPU2, sizeof(Sample) * GPU2_IMG_SIZE * GPU2_IMG_SIZE * GPU2_SAMPLE_LEN, cudaMemcpyHostToDevice ) );
  // CUDA_SAFE_CALL( cudaMemcpy( d_pswf_recon_GPU1, pswf_recon_GPU1, sizeof(Sample) * GPU3_IMG_SIZE * GPU1_IMG_SIZE * GPU3_SAMPLE_LEN, cudaMemcpyHostToDevice ) );
  // cudaMemcpy( d_pswf_recon_CPU, pswf_recon_CPU, 
  //	      sizeof(Sample) * * CPU_IMG_SIZE * CPU_IMG_SIZE * CPU_SAMPLE_LEN, cudaMemcpyHostToDevice );

  CUDA_SAFE_CALL( cudaMalloc( (void**)&d_pswf_recon_index_GPU1, sizeof( int ) * GPU1_IMG_SIZE * GPU1_IMG_SIZE ) );   CUDA_SAFE_CALL( cudaMemcpy( d_pswf_recon_index_GPU1, pswf_recon_index_GPU1, sizeof(int) * GPU1_IMG_SIZE * GPU1_IMG_SIZE, cudaMemcpyHostToDevice ) );

  //
  float2Desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
  cudaMallocArray( (cudaArray**) &d_array_cproj_res, &float2Desc, M, theta_list_size);

  cudaMalloc( (void**)&d_wtbl, sizeof( float ) * (ltbl + 1) );  
  cudaMemcpy( d_wtbl, wtbl, sizeof(float) * (ltbl + 1), cudaMemcpyHostToDevice );


  float2_cproj_res = new float2[ M * theta_list_size ];

  //
  data_H = new cufftComplex[ M * M ];
  cudaMalloc( (void**)&d_data_H, sizeof( cufftComplex ) * M * M );

  cufftResult res = cufftPlan2d( &plan_ifft2, M, M, CUFFT_C2C );

  if( res != CUFFT_SUCCESS )
    printf("cufftPlan2d failed\n "); 

#endif // GPU_GRIDREC
										   
} 

//---------------------------------------------------------------------------

void GridRec::reconstruct (void)
{
  memset (H, 0, (M+1)*(M+1)*sizeof(complex_struct));

  // #ifdef GPU_GRIDREC
  float timePhase1;
  unsigned int timerPhase1 = 0;   // test
  CUT_SAFE_CALL( cutCreateTimer( &timerPhase1 ) );
  CUT_SAFE_CALL( cutStartTimer( timerPhase1 ) );
  // #endif // GPU_GRIDREC

  phase1 (); 

  // #ifdef GPU_GRIDREC
  CUT_SAFE_CALL(cutStopTimer(timerPhase1));   // test
  timePhase1 =  cutGetTimerValue(timerPhase1);
  CUT_SAFE_CALL(cutDeleteTimer(timerPhase1));

  printf("total time for Phase1 is %f ms \n", timePhase1);

  //

  float timePhase2;
  unsigned int timerPhase2 = 0;   // test
  CUT_SAFE_CALL( cutCreateTimer( &timerPhase2 ) );
  CUT_SAFE_CALL( cutStartTimer( timerPhase2 ) );
  // #endif 

  phase2 (); 

  // #ifdef GPU_GRIDREC
  CUT_SAFE_CALL(cutStopTimer(timerPhase2));   // test
  timePhase2 =  cutGetTimerValue(timerPhase2);
  CUT_SAFE_CALL(cutDeleteTimer(timerPhase2));

  printf("total time for Phase2 is %f ms \n", timePhase2);

  // 
  float timePhase3;
  unsigned int timerPhase3 = 0;   // test
  CUT_SAFE_CALL( cutCreateTimer( &timerPhase3 ) );
  CUT_SAFE_CALL( cutStartTimer( timerPhase3 ) );
  // #endif

  phase3 (); 

  // #ifdef GPU_GRIDREC
  CUT_SAFE_CALL(cutStopTimer(timerPhase3));   // test
  timePhase3 =  cutGetTimerValue(timerPhase3);
  CUT_SAFE_CALL(cutDeleteTimer(timerPhase3));

  printf("total time for Phase3 is %f ms \n", timePhase3);
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
  for( int n = 0; n< M * M; n++ )
    list_pswf_recon[ n ].clear();
  delete [] list_pswf_recon;

  for( int n = 0; n< M * theta_list_size; n++ )
    list_pswf_sino[ n ].clear();
  delete [] list_pswf_sino;

#ifdef GPU_GRIDREC
  // GPU
  cufftDestroy( plan_fft );

  cudaFree( d_data_cproj );
  cudaFree( d_data_cproj_res );
  delete [] data_cproj;

  // 
  cudaFree( d_data_filphase );
  delete [] data_filphase;

  // 
  delete [] pswf_recon_GPU1;   // needs work 
  delete [] pswf_recon_GPU2;
  delete [] pswf_recon_GPU3;
  delete [] pswf_recon_CPU;
  delete [] pswf_recon_size; 

  cudaFree( d_pswf_recon_GPU1 );
  // cudaFree( d_pswf_recon_GPU2 );
  // cudaFree( d_pswf_recon_GPU3 );
  // cudaFree( d_pswf_recon_CPU );
  cudaFree( d_pswf_recon_size ); 

  delete []  float2_cproj_res;

  // 
  cudaFree( d_wtbl );
  cudaFreeArray( d_array_cproj_res );

  //
  cufftDestroy( plan_ifft2 );
  cudaFree( d_data_H );
  delete [] data_H;

#endif // GPU_GRIDREC
			   
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

#ifdef GPU_GRIDREC

  // always take flag=false in the GPU implementation
  // always take NO INTERP

  // perform fft on the projection data
  for (int n = 0; n < theta_list_size; n++) {   
    for( int j = 0; j < M; j++ ){

      data_cproj[n * M + j].x = G1[n][j] ; 
      data_cproj[n * M + j].y = G2[n][j] ; 

      // data_cproj[n * M + j].x = G1[n][M - 1 - j] ; 
      // data_cproj[n * M + j].y = G2[n][M - 1 - j] ; 

    } 
  }		       

  cudaMemcpy( d_data_cproj, data_cproj, sizeof(cufftComplex) * M * theta_list_size,
	      cudaMemcpyHostToDevice );

  cufftExecC2C( plan_fft, d_data_cproj, d_data_cproj_res, CUFFT_FORWARD );

  // cudaMemcpy( data_cproj, d_data_cproj_res, sizeof(cufftComplex) * M * theta_list_size,
  // 	      cudaMemcpyDeviceToHost );  // for debug

  // perform filtering on the Fourier data
  int blockWidth  = BLOCK_2DIM_X;
  int blockHeight = BLOCK_2DIM_Y;   

  int nBlockX_fil = (int)ceil((float) M / (float)blockWidth);   
  int nBlockY_fil = (int)ceil((float) theta_list_size / (float)blockHeight);

  dim3 dimGrid_fil(nBlockX_fil, nBlockY_fil);                // 3D grid is not supported on G80
  dim3 dimBlock(blockWidth, blockHeight);
  dim3 projdim( M, theta_list_size );

  freq_filter_kernel<<< dimGrid_fil, dimBlock >>>( d_data_cproj_res, d_data_filphase, projdim );

  cudaMemcpy( data_cproj, d_data_cproj_res, sizeof(cufftComplex) * M * theta_list_size,
  	      cudaMemcpyDeviceToHost );  

  // int num_elements_8 = 0; 
  // int num_elements_8_16 = 0; 
  // int num_elements_16_32 = 0; 
  // int num_elements_32_64 = 0; 
  // int num_elements_64_128 = 0; 
  // int num_elements_128_256 = 0; 
  // int num_elements_256_512 = 0; 
  // int num_elements_512_1024 = 0; 
  // int num_elements_1024_4096 = 0; 

  // int max_elements_8 = 0;
  // int max_elements_8_16 = 0;
  // int max_elements_16_32 = 0;
  // int max_elements_32_64 = 0;
  // int max_elements_64_128 = 0;
  // int max_elements_128_256 = 0;
  // int max_elements_256_512 = 0;
  // int max_elements_512_1024 = 0;
  // int max_elements_1024_4096 = 0;

  // for( int ny = 0; ny < M; ny++ ){
  //   for( int nx = 0; nx < M; nx++ ){

  //     int sz = list_pswf_recon[ ny * M + nx ].size();
  //     Sample s;

  //     if( fabs( nx - M/2 ) < 4 &&  fabs( ny - M/2 ) < 4 ){

  // 	if( sz > 0 )
  // 	  num_elements_8 += sz;

  // 	if( max_elements_8 < sz )
  // 	  max_elements_8 = sz;
  //     }
  //     else if( fabs( nx - M/2 ) < 8 &&  fabs( ny - M/2 ) < 8 ){

  // 	if( sz > 0 )
  // 	  num_elements_8_16 += sz;

  // 	if( max_elements_8_16 < sz )
  // 	  max_elements_8_16 = sz;

  //     }
  //     else if (  fabs( nx - M/2 ) < 16 && fabs( ny - M/2 ) < 16 ){

  // 	if( sz > 0 )
  // 	  num_elements_16_32 += sz;

  // 	if( max_elements_16_32 < sz )
  // 	  max_elements_16_32 = sz;

  //     }
  //     else if (  fabs( nx - M/2 ) < 32 && fabs( ny - M/2 ) < 32 ){

  // 	if( sz > 0 )
  // 	  num_elements_32_64 += sz;

  // 	if( max_elements_32_64 < sz )
  // 	  max_elements_32_64 = sz;

  //     }
  //     else if (  fabs( nx - M/2 ) < 64 && fabs( ny - M/2 ) < 64 ){

  // 	if( sz > 0 )
  // 	  num_elements_64_128 += sz;

  // 	if( max_elements_64_128 < sz )
  // 	  max_elements_64_128 = sz;

  //     }
  //     else if (  fabs( nx - M/2 ) < 128 && fabs( ny - M/2 ) < 128 ){

  // 	if( sz > 0 )
  // 	  num_elements_128_256 += sz;

  // 	if( max_elements_128_256 < sz )
  // 	  max_elements_128_256 = sz;

  //     }
  //     else if (  fabs( nx - M/2 ) < 256 && fabs( ny - M/2 ) < 256 ){

  // 	if( sz > 0 )
  // 	  num_elements_256_512 += sz;

  // 	if( max_elements_256_512 < sz )
  // 	  max_elements_256_512 = sz;

  //     }
  //     else if (  fabs( nx - M/2 ) < 512  && fabs( ny - M/2 ) < 512 ){

  // 	if( sz > 0 )
  // 	  num_elements_512_1024 += sz;

  // 	if( max_elements_512_1024 < sz )
  // 	  max_elements_512_1024 = sz;

  //     }
  //     else {

  // 	if( sz > 0 )
  // 	  num_elements_1024_4096 += sz;

  // 	if( max_elements_1024_4096 < sz )
  // 	  max_elements_1024_4096 = sz;

  //     }
  //   }
  // }

  // printf("nonzero elements in [0, 8) is %d\n", num_elements_8 ); 
  // printf("max elements in [0, 8) is %d\n", max_elements_8 ); 

  // printf("nonzero elements in [8, 16) is %d\n", num_elements_8_16 ); 
  // printf("max elements in [8, 16) is %d\n", max_elements_8_16 ); 
   
  // printf("nonzero elements in [16, 32) is %d\n", num_elements_16_32 ); 
  // printf("max elements in [16, 32) is %d\n", max_elements_16_32 ); 

  // printf("nonzero elements in [32, 64) is %d\n", num_elements_32_64 ); 
  // printf("max elements in [32, 64) is %d\n", max_elements_32_64 ); 

  // printf("nonzero elements in [64, 128) is %d\n", num_elements_64_128 ); 
  // printf("max elements in [64, 128) is %d\n", max_elements_64_128 ); 

  // printf("nonzero elements in [128, 256) is %d\n", num_elements_128_256 ); 
  // printf("max elements in [128, 256) is %d\n", max_elements_128_256 ); 

  // printf("nonzero elements in [256, 512) is %d\n", num_elements_256_512 ); 
  // printf("max elements in [256, 512) is %d\n", max_elements_256_512 ); 

  // printf("nonzero elements in [512, 1024) is %d\n", num_elements_512_1024 ); 
  // printf("max elements in [512, 1024) is %d\n", max_elements_512_1024 ); 

  // printf("nonzero elements in [1024, 4096] is %d\n", num_elements_1024_4096 ); 
  // printf("max elements in [1024, 4096] is %d\n", max_elements_1024_4096 ); 

  for (int n = 0; n < theta_list_size; n++) {   
    for( int j = 0; j < M; j++ ){
      float2_cproj_res[ n * M + j ] = make_float2( data_cproj[n * M + j].x, data_cproj[n * M + j].y );
    } 
  }		       

  // perform backprojection of the filtered Fourier data

  // prepare texture memory
  tex_cproj_res.normalized = false;                     
  tex_cproj_res.filterMode = cudaFilterModeLinear;      
  tex_cproj_res.addressMode[0] = cudaAddressModeClamp;  
  tex_cproj_res.addressMode[1] = cudaAddressModeClamp;

  cudaMemcpyToArray( d_array_cproj_res, 0, 0, float2_cproj_res, 
  		     sizeof(float2) * M * theta_list_size, cudaMemcpyHostToDevice );

  CUDA_SAFE_CALL(cudaBindTextureToArray(tex_cproj_res, d_array_cproj_res, float2Desc));

  //
  int blockWidth_GPU  = BLOCK_2DIM_X;
  int blockHeight_GPU = BLOCK_2DIM_Y;   
  dim3 dimBlock_GPU(blockWidth_GPU, blockHeight_GPU);

  int nBlockX_GPU1 = (int)ceil((float) GPU1_IMG_SIZE / (float)blockWidth_GPU);   
  int nBlockY_GPU1 = (int)ceil((float) GPU1_IMG_SIZE / (float)blockHeight_GPU);

  dim3 dimGrid_GPU1(nBlockX_GPU1, nBlockY_GPU1);                // 3D grid is not supported on G80

  bp_kernel<<< dimGrid_GPU1, dimBlock_GPU >>>( d_data_H, d_pswf_recon_GPU1, d_pswf_recon_size, 
  					       d_pswf_recon_index_GPU1, GPU1_IMG_SIZE, 
  					       GPU1_SAMPLE_LEN, 0, projdim );
  CUT_CHECK_ERROR("Kernel execution failed");

  // GPU2
  // int nBlockX_GPU2 = (int)ceil((float) GPU2_IMG_SIZE / (float)blockWidth_GPU);   
  // int nBlockY_GPU2 = (int)ceil((float) GPU2_IMG_SIZE / (float)blockHeight_GPU);

  // dim3 dimGrid_GPU2(nBlockX_GPU2, nBlockY_GPU2);                // 3D grid is not supported on G80

  // bp_kernel<<< dimGrid_GPU2, dimBlock_GPU >>>( d_data_H, d_pswf_recon_GPU2, GPU2_IMG_SIZE, 
  // 						GPU2_SAMPLE_LEN, GPU1_SAMPLE_LEN, projdim );
  // CUT_CHECK_ERROR("Kernel execution failed");

  // // GPU3
  // int nBlockX_GPU3 = (int)ceil((float) GPU3_IMG_SIZE / (float)blockWidth_GPU);   
  // int nBlockY_GPU3 = (int)ceil((float) GPU3_IMG_SIZE / (float)blockHeight_GPU);

  // dim3 dimGrid_GPU3(nBlockX_GPU3, nBlockY_GPU3);                // 3D grid is not supported on G80

  // bp_kernel<<< dimGrid_GPU3, dimBlock_GPU >>>( d_data_H, d_pswf_recon_GPU3, GPU3_IMG_SIZE, 
  // 						GPU3_SAMPLE_LEN, GPU1_SAMPLE_LEN + GPU2_SAMPLE_LEN, projdim );
  // CUT_CHECK_ERROR("Kernel execution failed");

  // CPU

  CUDA_SAFE_CALL( cudaUnbindTexture( tex_cproj_res ) );

  cudaMemcpy( data_H, d_data_H, sizeof( cufftComplex ) * M * M, cudaMemcpyDeviceToHost ); // test

  // continue with CPU compensation
  // for( int ny = 0; ny < GPU2_IMG_SIZE; ny++ ){

  //   int ny_ind = M/2 - GPU2_IMG_SIZE/2 + ny;

  //   for( int nx = 0; nx < GPU2_IMG_SIZE; nx++ ){

  //     int nx_ind = M/2 - GPU2_IMG_SIZE/2 + nx;

  //     Sample s; 
  //     for( int index = GPU1_SAMPLE_LEN; index < list_pswf_recon[ ny_ind * M + nx_ind].size(); index++ ){
  // 	s = list_pswf_recon[ ny_ind * M + nx_ind ][ index ];

  // 	data_H[ ny_ind * M + nx_ind ] += data_cproj[ s.ny * M + s.nx] * s.wa; 
  //     }

  //   }
  // }

#endif // GPU_GRIDREC

  ////////////////////////////////////////

#ifdef CPU_OLD

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

    // #ifndef GPU_GRIDREC

    j = 1; 
    while (j < sinogram_x_dim + 1) {
	
      cproj[j].r = G1[n][j-1]; 
      cproj[j].i = G2[n][j-1]; 

      // cproj[j].r = j-1; 
      // cproj[j].i = j-1; 

      j++; 
    } 
		       
    while (j < pdim)  { /** Zero fill the rest of array **/ 
	
      cproj[j].r = cproj[j].i = 0.0; 
      j++; 
    } 

    four1 ((float *) cproj+1, pdim, 1); 

    if( n == 375 )
      printf("\n"); 

    // #endif // GPU_GRIDREC

    for (j = 1; j < pdim2; j++) {  	/* Start loop on transform data */   // 550 ms

      // #ifndef GPU_GRIDREC

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

	// #endif // GPU_GRIDREC

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

	  if( iu == 3072 && iv == 3072 ){
	    printf( "%d, %d, %f, %E, %E\n", n, j, convolv, Cdata1.r, Cdata1.i );
	  }

// #ifdef GPU_GRIDREC

// 	  H[iu*M+iv+1].r += convolv * data_cproj[ n * M + j ].x; 
// 	  H[iu*M+iv+1].i += convolv * data_cproj[ n * M + j ].y; 
// 	  H[(M-iu)*M+(M-iv)+1].r += convolv * data_cproj[ n * M + M - j ].x; 
// 	  H[(M-iu)*M+(M-iv)+1].i += convolv * data_cproj[ n * M + M - j ].y; 

// #else
	  H[iu*M+iv+1].r += convolv * Cdata1.r; 
	  H[iu*M+iv+1].i += convolv * Cdata1.i; 
	  H[(M-iu)*M+(M-iv)+1].r += convolv * Cdata2.r; 
	  H[(M-iu)*M+(M-iv)+1].i += convolv * Cdata2.i; 
// #endif

	} 
      } 
    } /*** End loop on transform data */ 
  }   /*** End loop on angles */ 

#endif // CPU_OLD

  // This version tries to use the locations and weights stored at init()
  // But it does not improve the efficiency when compared to CPU phase1. 
  // The old version is pretty optimized. 
  // See how GPU goes using the stored information. 8/31/2011

#ifdef CPU_NEW

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

    for (j = 1; j < pdim2; j++) {  	/* Start loop on transform data */ 

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

      int sz = list_pswf_sino[ n * M + j ].size();
      Sample s;

      // timeval tim;
      // gettimeofday( &tim, NULL );
      // double t1 = tim.tv_sec + tim.tv_usec/1000000.0;

      for( int i = 0; i < sz; i++ ){
	s = list_pswf_sino[ n * M + j][ i ];

	H[ s.ny * M + s.nx + 1].r += s.wa * Cdata1.r; 
	H[ s.ny * M + s.nx + 1].i += s.wa * Cdata1.i; 

	H[ (M - s.ny) * M + (M - s.nx) + 1].r += s.wa * Cdata2.r; 
	H[ (M - s.ny) * M + (M - s.nx) + 1].i += s.wa * Cdata2.i; 
      }

      // gettimeofday( &tim, NULL );
      // double t2 = tim.tv_sec + tim.tv_usec/1000000.0;
      // printf("NEW: %.6lf seconds elapsedn", t2-t1);

    } /*** End loop on transform data */ 
  }   /*** End loop on angles */ 
	     
#endif // CPU_NEW.

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
   
#ifdef GPU_GRIDREC       

  float time_fft = 0;
  unsigned int timerGridRec = 0;  
  CUT_SAFE_CALL( cutCreateTimer( &timerGridRec ) );
  CUT_SAFE_CALL( cutStartTimer( timerGridRec ) );

  // for( int ny = 0; ny < M ; ny++ ){
  //   for( int nx = 0; nx < M ; nx++ ){
  //     data_H[ ny * M + nx ].x = H[ ny * M + nx + 1 ].r;
  //     data_H[ ny * M + nx ].y = H[ ny * M + nx + 1 ].i;
  //   }
  // }

  // cudaMemcpy( d_data_H, data_H, sizeof( cufftComplex ) * M * M, cudaMemcpyHostToDevice );

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

  printf( "Time for fft in Phase 2 is %f (ms)\n ", time_fft );

#endif


#ifndef GPU_GRIDREC
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


//---------------------------------------------------------------------------
#ifdef GPU_GRIDREC

__global__ void freq_filter_kernel( cufftComplex * d_data_cproj_res, 
				    cufftComplex * d_data_filphase, 
				    dim3 projdim){

  uint idx, idy;

  idx = blockIdx.x * blockDim.x + threadIdx.x;
  idy = blockIdx.y * blockDim.y + threadIdx.y;

  // shared memory
  
  __shared__ float shmem_cproj_x[BLOCK_2DIM_X][BLOCK_2DIM_Y];
  __shared__ float shmem_cproj_y[BLOCK_2DIM_X][BLOCK_2DIM_Y];
  __shared__ float shmem_filphase_x[BLOCK_2DIM_X];
  __shared__ float shmem_filphase_y[BLOCK_2DIM_X];
 
  dim3 cidx;
  cidx.x = threadIdx.x;
  cidx.y = threadIdx.y;

  // load shared memory
  if(idx < projdim.x && idy < projdim.y)
  {
    shmem_filphase_x[ cidx.x ] = d_data_filphase[ idx ].x; 
    shmem_filphase_y[ cidx.x ] = d_data_filphase[ idx ].y; 
  }
  __syncthreads();

  if(idx < projdim.x && idy < projdim.y)
  {

    shmem_cproj_x[ cidx.x ][ cidx.y ] = d_data_cproj_res[ idy * projdim.x + idx ].x;
    shmem_cproj_y[ cidx.x ][ cidx.y ] = d_data_cproj_res[ idy * projdim.x + idx ].y;

    // for compatibility with CPU version; needs more work
    // shmem_cproj_x[ cidx.x ][ cidx.y ] = d_data_cproj_res[ idy * projdim.x + projdim.x - idx ].x;
    // shmem_cproj_y[ cidx.x ][ cidx.y ] = d_data_cproj_res[ idy * projdim.x + projdim.x - idx ].y;
  }

  // shmem_cproj_x[ 0 ][ cidx.y ] = d_data_cproj_res[ idy * projdim.x ].x;
  // shmem_cproj_y[ 0 ][ cidx.y ] = d_data_cproj_res[ idy * projdim.x ].y;

  __syncthreads();

  // 2. apply kernel and store the result

  d_data_cproj_res[ idy * projdim.x + idx ].x =  shmem_cproj_x[ cidx.x ][ cidx.y ] * shmem_filphase_x[ cidx.x ] - shmem_cproj_y[ cidx.x ][ cidx.y ] * shmem_filphase_y[ cidx.x ];

  d_data_cproj_res[ idy * projdim.x + idx ].y =  shmem_cproj_y[ cidx.x ][ cidx.y ] * shmem_filphase_x[ cidx.x ] +  shmem_cproj_x[ cidx.x ][ cidx.y ] * shmem_filphase_y[ cidx.x ];

  // d_data_cproj_res[ idy * projdim.x + idx ].x =  shmem_cproj_x[ cidx.x ][ cidx.y ];
  // d_data_cproj_res[ idy * projdim.x + idx ].y =  shmem_cproj_y[ cidx.x ][ cidx.y ];

  // d_data_cproj_res[ idy * projdim.x + idx ].x =  shmem_filphase_x[ cidx.x ];
  // d_data_cproj_res[ idy * projdim.x + idx ].y =  shmem_filphase_y[ cidx.x ];

}

void find_angle( float x, float y, float* angle ){

  if( x > 0.0f && y > 0.0f ){
    *angle = atanf( y / x );
  }
  if( x == 0.0f && y > 0.0f ){
    *angle = PI / 2.0f;
  }
  if( x < 0.0f ){
    *angle  = PI + atanf( y / x );
  }

  if( x == 0.0f && y < 0.0f ){
    *angle = 3.0f * PI / 2.0f;
  }

  if( x > 0.0f && y < 0.0f ){
    *angle = 2.0f * PI - atanf( y / x );
  }

}

__global__ void bp_kernel( cufftComplex * d_data_H, Sample * d_pswf_recon, float* d_pswf_recon_size, 
			   int* d_pswf_recon_index,
			   int gpu_img_size, int gpu_sample_len, int gpu_sample_offset, dim3 projdim){

  uint idx, idy;

  idx = blockIdx.x * blockDim.x + threadIdx.x;
  idy = blockIdx.y * blockDim.y + threadIdx.y;

  // shared memory
  // Assume GPU_SAMPLE_LEN(32, 64) is multiples of BLOCK_2DIM_X(Y) (32) 
  // __shared__ float shmem_data_H_x[ BLOCK_2DIM_X ][ BLOCK_2DIM_Y ]; 
  // __shared__ float shmem_data_H_y[ BLOCK_2DIM_X ][ BLOCK_2DIM_Y ]; 

  __shared__ float shmem_recon_size[ BLOCK_2DIM_X ][ BLOCK_2DIM_Y ]; 
  __shared__ int   shmem_recon_index[ BLOCK_2DIM_X ][ BLOCK_2DIM_Y ]; 

  __shared__ float  shmem_sample[ BLOCK_2DIM_X ][ BLOCK_2DIM_Y ]; 
 
  dim3 cidx;
  cidx.x = threadIdx.x;
  cidx.y = threadIdx.y;

  // load shared memory
  if( idx < projdim.x && idy < projdim.x ){
    // shmem_data_H_x[ cidx.x ][ cidx.y ] = d_data_H[ (projdim.x/2 -  gpu_img_size / 2 + idy) * projdim.x +  (projdim.x/2 -  gpu_img_size / 2 + idx) ].x; 
    // shmem_data_H_y[ cidx.x ][ cidx.y ] = d_data_H[  (projdim.x/2 -  gpu_img_size / 2 + idy) * projdim.x +  (projdim.x/2 -  gpu_img_size / 2 + idx) ].y; 

    shmem_recon_size[ cidx.x ][ cidx.y ] = d_pswf_recon_size[  (projdim.x/2 -  gpu_img_size / 2 + idy) * projdim.x +  (projdim.x/2 -  gpu_img_size / 2 + idx) ]; 
    shmem_recon_index[ cidx.x ][ cidx.y ] = d_pswf_recon_index[  (projdim.x/2 -  gpu_img_size / 2 + idy) * projdim.x +  (projdim.x/2 -  gpu_img_size / 2 + idx) ]; 

  }

  // __syncthreads();

  // 2. apply kernel and store the result

  float x_res = 0.0f;
  float y_res = 0.0f;

  int i;
  float x, y, wa;
  int M2 = projdim.x / 2;
  float2 proj_value;

  // int sample_len = gpu_sample_len;
  // if( sample_len + gpu_sample_offset < shmem_recon_size[ cidx.x ][ cidx.y ] )
  //   sample_len =  shmem_recon_size[ cidx.x ][ cidx.y ] - gpu_sample_offset; 

  for( i = 0; i < gpu_sample_len; i++ ){
  // for( i = 0; i < 1; i++ ){

    if( idx < projdim.x && idy < projdim.x ){

      if( i < shmem_recon_size[ cidx.x ][ cidx.y ] - gpu_sample_offset ){
	shmem_sample[ cidx.x ][ cidx.y ] = d_pswf_recon[ shmem_recon_index[ cidx.x ][ cidx.y ] + i ].nx; 
      }
      __syncthreads();
    }

    x = shmem_sample[ cidx.x ][ cidx.y ];

    //
    if( idx < projdim.x && idy < projdim.x ){

      if( i < shmem_recon_size[ cidx.x ][ cidx.y ] - gpu_sample_offset ){
	shmem_sample[ cidx.x ][ cidx.y ] = d_pswf_recon[ shmem_recon_index[ cidx.x ][ cidx.y ] + i ].ny; 
      }
      __syncthreads();
    }

    y = shmem_sample[ cidx.x ][ cidx.y ];

    // 
    if( idx < projdim.x && idy < projdim.x ){

      if( i < shmem_recon_size[ cidx.x ][ cidx.y ] - gpu_sample_offset ){
	shmem_sample[ cidx.x ][ cidx.y ] = d_pswf_recon[ shmem_recon_index[ cidx.x ][ cidx.y ] + i ].wa; 
      }
      __syncthreads();
    }
    wa = shmem_sample[ cidx.x ][ cidx.y ];

    if( i < shmem_recon_size[ cidx.x ][ cidx.y ] - gpu_sample_offset 
	&& x >= 0.0f && x < 1.0f * projdim.x && y >= 0.0f && y < 1.0f * projdim.y ){
      proj_value = tex2D( tex_cproj_res, 1.0f * x + 0.5f, 1.0f * y + 0.5f );

      x_res += wa * proj_value.x;
      y_res += wa * proj_value.y;  
    }    

  }

  __syncthreads();

    // d_data_H[ (projdim.x/2 -  gpu_img_size / 2 + idy ) * projdim.x + (projdim.x/2 -  gpu_img_size / 2 + idx) ].x = shmem_data_H_x[ cidx.x ][ cidx.y ] + x_res; 
    // d_data_H[ (projdim.x/2 -  gpu_img_size / 2 + idy ) * projdim.x + (projdim.x/2 -  gpu_img_size / 2 + idx) ].y = shmem_data_H_y[ cidx.x ][ cidx.y ] + y_res;

    d_data_H[ (projdim.x/2 -  gpu_img_size / 2 + idy ) * projdim.x + (projdim.x/2 -  gpu_img_size / 2 + idx) ].x = x_res; 
    d_data_H[ (projdim.x/2 -  gpu_img_size / 2 + idy ) * projdim.x + (projdim.x/2 -  gpu_img_size / 2 + idx) ].y = y_res;


  // d_data_H[ idy * projdim.x + idx ].x = d_pswf_recon[ (idy * projdim.x + idx) * gpu_sample_len ].nx; // x; 
  // d_data_H[ idy * projdim.x + idx ].y = d_pswf_recon[ (idy * projdim.x + idx) * gpu_sample_len ].ny; //  y;

}


// __global__ void bp_kernel( cufftComplex * d_data_H, 
// 			   float * d_wtbl, 
// 			   int ltbl, float L2, dim3 projdim,
// 			   float tblspcg, float scale,
// 			   float start_rot, float inv_rot){

//   // It turns out that the backprojection in the frequency domain is inappropriate 
//   // for GPU, because of the fact that the number of pixels projected to the frequency
//   // domain changes a lot. For example, the center (2048, 2048) would have 2427, and 
//   // the position (2048, 2045) would have 1139. 

//   uint idx, idy;

//   idx = blockIdx.x * blockDim.x + threadIdx.x;
//   idy = blockIdx.y * blockDim.y + threadIdx.y;

//   // shared memory
  
//   __shared__ float shmem_H_x[BLOCK_2DIM_X][BLOCK_2DIM_Y];
//   __shared__ float shmem_H_y[BLOCK_2DIM_X][BLOCK_2DIM_Y];
//   __shared__ float shmem_wtbl[BLOCK_2DIM_X * BLOCK_2DIM_Y]; // wtbl is of size 512+1. This implementation requires BLOCK_2DIM_X * BLOCK_2DIM_Y > 512 + 1. BLOCK_2DIM_X(Y) is 32 as big enough. 
   
//   dim3 cidx;
//   cidx.x = threadIdx.x;
//   cidx.y = threadIdx.y;

//   // load shared memory
//   // if(idx < projdim.x && idy < projdim.y) // It seems that this line makes a difference, maybe
//                                             // because the image is of irregular size (1025 * 1501)
//                                             // Pad to (2048 * 2048) if problem still happens
//   {

//     shmem_H_x[ cidx.x ][ cidx.y ] = d_data_H[ idy * projdim.x + idx ].x;
//     shmem_H_y[ cidx.x ][ cidx.y ] = d_data_H[ idy * projdim.x + idx ].y;

//     if(  cidx.y * BLOCK_2DIM_X + cidx.x < ltbl + 1 )
//       shmem_wtbl[  cidx.y * BLOCK_2DIM_X + cidx.x ] = d_wtbl[ cidx.y * BLOCK_2DIM_X + cidx.x ]; 
//     else
//       shmem_wtbl[  cidx.y * BLOCK_2DIM_X + cidx.x ] = 0.0f;
//   }

//   __syncthreads();

//   // 2. apply kernel and store the result

//   float x_res = 0.0f;
//   float y_res = 0.0f;

//   float x, y, wtbl_value_y, wtbl_value_x, wtbl_value, angle_value;
//   int wtbl_index ;
//   float M2 = projdim.x / 2.0f;  // assume projdim.x is even
//   float  proj_x, proj_y;
//   float2 proj_value;
//   float jj, j_ub, j_lb;

//   // assume start_rot = 0
//   // assume inv_rot * proj_dim.y ~= 180 degree(e.g., inv_rot = 0.12, proj_dim.y=1501)

//   // For the first projection (0 degree)

//   if( 1.0f * idy - L2 - M2 <= 0.0f && 1.0f * idy + L2 - M2 >= 0.0f ){
//     j_lb = 1.0f * idx - L2 - M2;
//     j_ub = 1.0f * idx + L2 - M2;

//     y = M2;
//     wtbl_index = (int)(fabs( y - idy ) * tblspcg + 0.5f);

//     if( wtbl_index > 1.0f * ltbl )
//       wtbl_value_y = 0.0f; 
//     else{
//       wtbl_value_y = shmem_wtbl[ wtbl_index ];
//     }

//     for( jj = j_lb; jj <= j_ub; jj++ ){

//       proj_x = floor( jj );

//       if( proj_x >= 0.0f && proj_x < 1.0f * projdim.x){
	  
//   	x = scale * proj_x + M2;

//   	wtbl_index = (int)(fabs( x - idx ) * tblspcg + 0.5f);
//   	if( wtbl_index > 1.0f * ltbl )
//   	  wtbl_value_x = 0.0f; 
//   	else{
//   	  wtbl_value_x = shmem_wtbl[ wtbl_index ];
//   	}
//   	wtbl_value = wtbl_value_x * wtbl_value_y;

//   	proj_value = tex2D( tex_cproj_res, proj_x + 0.5f, proj_y + 0.5f );
	
//   	x_res += wtbl_value * proj_value.x;
//   	y_res += wtbl_value * proj_value.y;      

//       }
//     }
//   }

//   __syncthreads();

//   int proj_half = projdim.y / 2; 
  
//   // for projections in (0, 90)

//   if( 1.0f * idx + L2 - M2 >= 0.0f && 1.0f * idy + L2 - M2 >= 0.0f ){

//     for( proj_y = 1.0f; proj_y < 1.0f * proj_half; proj_y++ ){

//       angle_value = start_rot + proj_y * inv_rot;  

//       j_ub = (1.0f * idy + L2 - M2) / sinf( angle_value);
//       jj = (1.0f * idx + L2 - M2) / cosf( angle_value);
      
//       if( j_ub > jj )
// 	j_ub = jj;

//       if( j_ub < 0.0f )
// 	j_ub = 0.0f;

//       if( j_ub > 1.0f * projdim.x )
// 	j_ub = 1.0f * projdim.x -1;

//       j_lb = (1.0f * idy - L2 - M2) / sinf( angle_value);
//       jj = (1.0f * idx - L2 - M2) / cosf( angle_value);
      
//       if( j_lb < jj )
//         j_lb = jj;

//       if( j_lb < 0.0f )
// 	j_lb = 0.0f;

//       if( j_lb > 1.0f * projdim.x )
//         j_lb = 1.0f * projdim.x - 1;

//       for( jj = j_lb; jj <= j_ub; jj++ ){

// 	proj_x = floor( jj );
	  
// 	x = scale * proj_x * cosf(angle_value) + M2;
// 	y = scale * proj_x * sinf(angle_value) + M2;
	    
// 	wtbl_index = (int)(fabs( y - idy ) * tblspcg + 0.5f);

// 	if( wtbl_index > 1.0f * ltbl )
// 	  wtbl_value_y = 0.0f; 
// 	else{
// 	  wtbl_value_y = shmem_wtbl[ wtbl_index ];
// 	}

// 	wtbl_index = (int)(fabs( x - idx ) * tblspcg + 0.5f);
// 	if( wtbl_index > 1.0f * ltbl )
// 	  wtbl_value_x = 0.0f; 
// 	else{
// 	  wtbl_value_x = shmem_wtbl[ wtbl_index ];
// 	}

// 	wtbl_value = wtbl_value_x * wtbl_value_y;
	
// 	proj_value = tex2D( tex_cproj_res, proj_x + 0.5f, proj_y + 0.5f );
	
// 	x_res += wtbl_value * proj_value.x;
// 	y_res += wtbl_value * proj_value.y;      

//       }
//     }
//   }
//    __syncthreads();  

//   // 90 degree 

//   if( 1.0f * idx - L2 - M2 <= 0.0f && 1.0f * idx + L2 - M2 >= 0.0f ){
//     j_lb = 1.0f * idy - L2 - M2;
//     j_ub = 1.0f * idy + L2 - M2;

//     x = M2;
//     wtbl_index = (int)(fabs( x - idx ) * tblspcg + 0.5f);
//     if( wtbl_index > 1.0f * ltbl )
//       wtbl_value_x = 0.0f; 
//     else{
//       wtbl_value_x = shmem_wtbl[ wtbl_index ];
//     }

//     for( jj = j_lb; jj <= j_ub; jj++ ){

//       proj_x = floor( jj );
//       if( proj_x >= 0.0f && proj_x < 1.0f * projdim.x){

//   	y = scale * proj_x + M2;

//   	wtbl_index = (int)(fabs( y - idy ) * tblspcg + 0.5f);

//   	if( wtbl_index > 1.0f * ltbl )
//   	  wtbl_value_y = 0.0f; 
//   	else{
//   	  wtbl_value_y = shmem_wtbl[ wtbl_index ];
//   	}

//   	wtbl_value = wtbl_value_x * wtbl_value_y;

//   	proj_value = tex2D( tex_cproj_res, proj_x + 0.5f, proj_y + 0.5f );
	
//   	x_res += wtbl_value * proj_value.x;
//   	y_res += wtbl_value * proj_value.y;      
//       }
//     }
//   }

//   __syncthreads();

  // (90, 180)

  // for( proj_y = proj_half + 1; proj_y < 2 * proj_half; proj_y++ ){
    
  //   angle_value = start_rot + proj_y * inv_rot; 

  //   jj = (1.0f * idx - L2 - M2) / cosf( angle_value);
  //   j_ub = (1.0f * idy + L2 - M2) / sinf( angle_value);
    
  //   if( j_ub > jj )
  //     j_ub = jj;

  //   jj = (1.0f * idx + L2 - M2) / cosf( angle_value);
  //   j_lb = (1.0f * idy - L2 - M2) / sinf( angle_value);
      
  //   if( j_lb < jj )
  //     j_lb = jj;

  //   for( jj = j_lb; jj <= j_ub; jj++ ){

  //     proj_x = floor( jj );

  //     if( proj_x >= 0.0f && proj_x < 1.0f * projdim.x){	  
  // 	x = scale * proj_x * cosf(angle_value) + M2;
  // 	y = scale * proj_x * sinf(angle_value) + M2;

  // 	wtbl_index = (int)(fabs( y - idy ) * tblspcg + 0.5f);

  // 	if( wtbl_index > 1.0f * ltbl )
  // 	  wtbl_value_y = 0.0f; 
  // 	else{
  // 	  wtbl_value_y = shmem_wtbl[ wtbl_index ];
  // 	}

  // 	wtbl_index = (int)(fabs( x - idx ) * tblspcg + 0.5f);
  // 	if( wtbl_index > 1.0f * ltbl )
  // 	  wtbl_value_x = 0.0f; 
  // 	else{
  // 	  wtbl_value_x = shmem_wtbl[ wtbl_index ];
  // 	}
  // 	wtbl_value = wtbl_value_x * wtbl_value_y;

  // 	proj_value = tex2D( tex_cproj_res, proj_x + 0.5f, proj_y + 0.5f );
	
  // 	x_res += wtbl_value * proj_value.x;
  // 	y_res += wtbl_value * proj_value.y;      
  //     }
  //   }
  // }	

  // __syncthreads();

  // // 180

  // if( 1.0f * idy - L2 - M2 <= 0.0f && 1.0f * idy + L2 - M2 >= 0.0f ){
  //   j_lb = 1.0f * idx + L2 - M2;
  //   j_ub = 1.0f * idx - L2 - M2;

  //   y = M2;
  //   wtbl_index = (int)(fabs( y - idy ) * tblspcg + 0.5f);

  //   if( wtbl_index > 1.0f * ltbl )
  //     wtbl_value_y = 0.0f; 
  //   else{
  //     wtbl_value_y = shmem_wtbl[ wtbl_index ];
  //   }
    	   
  //   for( jj = j_lb; jj <= j_ub; jj++ ){

  //     proj_x = floor( jj );
  //     if( proj_x >= 0.0f && proj_x < 1.0f * projdim.x){
 
  // 	x = -scale * proj_x + M2;

  // 	wtbl_index = (int)(fabs( x - idx ) * tblspcg + 0.5f);
  // 	if( wtbl_index > 1.0f * ltbl )
  // 	  wtbl_value_x = 0.0f; 
  // 	else{
  // 	  wtbl_value_x = shmem_wtbl[ wtbl_index ];
  // 	}
  // 	wtbl_value = wtbl_value_x * wtbl_value_y;

  // 	proj_value = tex2D( tex_cproj_res, proj_x + 0.5f, proj_y + 0.5f );
	
  // 	x_res += wtbl_value * proj_value.x;
  // 	y_res += wtbl_value * proj_value.y;      
  //     }
  //   }
  // }

  // __syncthreads();

  // // projections larger (180, 270)

  // for( proj_y = 2 * proj_half + 1; proj_y < projdim.y; proj_y++ ){
    
  //   angle_value = start_rot + proj_y * inv_rot; 

  //   jj = (1.0f * idx - L2 - M2) / cosf( angle_value);
  //   j_ub = (1.0f * idy - L2 - M2) / sinf( angle_value);
    
  //   if( j_ub > jj )
  //     j_ub = jj;

  //   jj = (1.0f * idx + L2 - M2) / cosf( angle_value);
  //   j_lb = (1.0f * idy + L2 - M2) / sinf( angle_value);
      
  //   if( j_lb < jj )
  //     j_lb = jj;

  //   for( jj = j_lb; jj <= j_ub; jj++ ){

  //     proj_x = floor( jj );

  //     if( proj_x >= 0.0f && proj_x < 1.0f * projdim.x){	  

  // 	x = scale * proj_x * cosf(angle_value) + M2;
  // 	y = scale * proj_x * sinf(angle_value) + M2;

  // 	wtbl_index = (int)(fabs( y - idy ) * tblspcg + 0.5f);

  // 	if( wtbl_index > 1.0f * ltbl )
  // 	  wtbl_value_y = 0.0f; 
  // 	else{
  // 	  wtbl_value_y = shmem_wtbl[ wtbl_index ];
  // 	}

  // 	wtbl_index = (int)(fabs( x - idx ) * tblspcg + 0.5f);
  // 	if( wtbl_index > 1.0f * ltbl )
  // 	  wtbl_value_x = 0.0f; 
  // 	else{
  // 	  wtbl_value_x = shmem_wtbl[ wtbl_index ];
  // 	}
  // 	wtbl_value = wtbl_value_x * wtbl_value_y;

  // 	proj_value = tex2D( tex_cproj_res, proj_x + 0.5f, proj_y + 0.5f );
	
  // 	x_res += wtbl_value * proj_value.x;
  // 	y_res += wtbl_value * proj_value.y;      
  //     }
  //   }
  // }	

  // __syncthreads();

  // version 2

  // for( proj_y = 0.0f; proj_y < 1.0f * projdim.y; proj_y++ ){
    
  //   angle_value = start_rot + proj_y * inv_rot;  // assume angle_value in [0, PI]

  //   // 
  //   if( fabsf( angle_value ) < 1e-5f ){
  //     if( 1.0f * idy - L2 - M2 <= 0.0f && 1.0f * idy + L2 - M2 >= 0.0f ){
  //   	j_lb = 1.0f * idx - L2 - M2;
  //   	j_ub = 1.0f * idx + L2 - M2;
  //   	for( jj = j_lb; jj <= j_ub; jj++ ){

  //   	  proj_x = floor( jj );

  //   	  if( proj_x >= 0.0f && proj_x < 1.0f * projdim.x){
	  
  //   	    x = scale * proj_x * cosf(angle_value) + M2;
  //   	    y = scale * proj_x * sinf(angle_value) + M2;

  //   	    wtbl_index = (int)(fabs( y - idy ) * tblspcg + 0.5f);

  //   	    if( wtbl_index > 1.0f * ltbl )
  //   	      wtbl_value_y = 0.0f; 
  //   	    else{
  //   	      wtbl_index_y = wtbl_index / BLOCK_2DIM_X;
  //   	      wtbl_index_x = wtbl_index - wtbl_index_y * BLOCK_2DIM_X;
  //   	      wtbl_value_y = shmem_wtbl[ wtbl_index_x ][ wtbl_index_y ];
  //   	    }

  //   	    wtbl_index = (int)(fabs( x - idx ) * tblspcg + 0.5f);
  //   	    if( wtbl_index > 1.0f * ltbl )
  //   	      wtbl_value_x = 0.0f; 
  //   	    else{
  //   	      wtbl_index_y = wtbl_index / BLOCK_2DIM_X;
  //   	      wtbl_index_x = wtbl_index - wtbl_index_y * BLOCK_2DIM_X;
  //   	      wtbl_value_x = shmem_wtbl[ wtbl_index_x ][ wtbl_index_y ];
  //   	    }
  //   	    wtbl_value = wtbl_value_x * wtbl_value_y;

  //   	    proj_value = tex2D( tex_cproj_res, proj_x + 0.5f, proj_y + 0.5f );
	
  //   	    x_res += wtbl_value * proj_value.x;
  //   	    y_res += wtbl_value * proj_value.y;      

  //   	  }
  //   	}
  //     }
  //   }

    
  //   if( angle_value > 1e-5f && angle_value < PI/2 - 1e-5f ){

  //     jj = (1.0f * idx + L2 - M2) / cosf( angle_value);
  //     j_ub = (1.0f * idy + L2 - M2) / sinf( angle_value);
      
  //     if( jj < j_ub )
  //   	j_ub = jj;

  //     jj = (1.0f * idx - L2 - M2) / cosf( angle_value);
  //     j_lb = (1.0f * idy - L2 - M2) / sinf( angle_value);
      
  //     if( jj > j_lb )
  //   	j_lb = jj;

  //     for( jj = j_lb; jj <= j_ub; jj++ ){

  //   	proj_x = floor( jj );
	  
  //   	if( proj_x >= 0.0f && proj_x < 1.0f * projdim.x){

  //   	  x = scale * proj_x * cosf(angle_value) + M2;
  //   	  y = scale * proj_x * sinf(angle_value) + M2;

  //   	  wtbl_index = (int)(fabs( y - idy ) * tblspcg + 0.5f);

  //   	  if( wtbl_index > 1.0f * ltbl )
  //   	    wtbl_value_y = 0.0f; 
  //   	  else{
  //   	    wtbl_index_y = wtbl_index / BLOCK_2DIM_X;
  //   	    wtbl_index_x = wtbl_index - wtbl_index_y * BLOCK_2DIM_X;
  //   	    wtbl_value_y = shmem_wtbl[ wtbl_index_x ][ wtbl_index_y ];
  //   	  }

  //   	  wtbl_index = (int)(fabs( x - idx ) * tblspcg + 0.5f);
  //   	  if( wtbl_index > 1.0f * ltbl )
  //   	    wtbl_value_x = 0.0f; 
  //   	  else{
  //   	    wtbl_index_y = wtbl_index / BLOCK_2DIM_X;
  //   	    wtbl_index_x = wtbl_index - wtbl_index_y * BLOCK_2DIM_X;
  //   	    wtbl_value_x = shmem_wtbl[ wtbl_index_x ][ wtbl_index_y ];
  //   	  }
  //   	  wtbl_value = wtbl_value_x * wtbl_value_y;

  //   	  proj_value = tex2D( tex_cproj_res, proj_x + 0.5f, proj_y + 0.5f );
	
  //   	  x_res += wtbl_value * proj_value.x;
  //   	  y_res += wtbl_value * proj_value.y;      
  //   	}
  //     }
  //   }

    
  //   if( fabsf( angle_value - PI/2 ) < 1e-5f ){

  //     if( 1.0f * idx - L2 - M2 <= 0.0f && 1.0f * idx + L2 - M2 >= 0.0f ){
  //   	j_lb = 1.0f * idy - L2 - M2;
  //   	j_ub = 1.0f * idy + L2 - M2;

  //   	for( jj = j_lb; jj <= j_ub; jj++ ){

  //   	  proj_x = floor( jj );
  //   	  if( proj_x >= 0.0f && proj_x < 1.0f * projdim.x){
 
  //   	    x = scale * proj_x * cosf(angle_value) + M2;
  //   	    y = scale * proj_x * sinf(angle_value) + M2;

  //   	    wtbl_index = (int)(fabs( y - idy ) * tblspcg + 0.5f);

  //   	    if( wtbl_index > 1.0f * ltbl )
  //   	      wtbl_value_y = 0.0f; 
  //   	    else{
  //   	      wtbl_index_y = wtbl_index / BLOCK_2DIM_X;
  //   	      wtbl_index_x = wtbl_index - wtbl_index_y * BLOCK_2DIM_X;
  //   	      wtbl_value_y = shmem_wtbl[ wtbl_index_x ][ wtbl_index_y ];
  //   	    }

  //   	    wtbl_index = (int)(fabs( x - idx ) * tblspcg + 0.5f);
  //   	    if( wtbl_index > 1.0f * ltbl )
  //   	      wtbl_value_x = 0.0f; 
  //   	    else{
  //   	      wtbl_index_y = wtbl_index / BLOCK_2DIM_X;
  //   	      wtbl_index_x = wtbl_index - wtbl_index_y * BLOCK_2DIM_X;
  //   	      wtbl_value_x = shmem_wtbl[ wtbl_index_x ][ wtbl_index_y ];
  //   	    }
  //   	    wtbl_value = wtbl_value_x * wtbl_value_y;

  //   	    proj_value = tex2D( tex_cproj_res, proj_x + 0.5f, proj_y + 0.5f );
	
  //   	    x_res += wtbl_value * proj_value.x;
  //   	    y_res += wtbl_value * proj_value.y;      
  //   	  }
  //   	}
  //     }
  //   }

  //   // 
  //   if( angle_value > PI/2 + 1e-5f && angle_value < PI - 1e-5f ){

  //     jj = (1.0f * idx - L2 - M2) / cosf( angle_value);
  //     j_ub = (1.0f * idy + L2 - M2) / sinf( angle_value);
      
  //     if( jj < j_ub )
  //   	j_ub = jj;

  //     jj = (1.0f * idx + L2 - M2) / cosf( angle_value);
  //     j_lb = (1.0f * idy - L2 - M2) / sinf( angle_value);
      
  //     if( jj > j_lb )
  //   	j_lb = jj;

  //     for( jj = j_lb; jj <= j_ub; jj++ ){

  //   	proj_x = floor( jj );

  //   	if( proj_x >= 0.0f && proj_x < 1.0f * projdim.x){	  
  //   	  x = scale * proj_x * cosf(angle_value) + M2;
  //   	  y = scale * proj_x * sinf(angle_value) + M2;

  //   	  wtbl_index = (int)(fabs( y - idy ) * tblspcg + 0.5f);

  //   	  if( wtbl_index > 1.0f * ltbl )
  //   	    wtbl_value_y = 0.0f; 
  //   	  else{
  //   	    wtbl_index_y = wtbl_index / BLOCK_2DIM_X;
  //   	    wtbl_index_x = wtbl_index - wtbl_index_y * BLOCK_2DIM_X;
  //   	    wtbl_value_y = shmem_wtbl[ wtbl_index_x ][ wtbl_index_y ];
  //   	  }

  //   	  wtbl_index = (int)(fabs( x - idx ) * tblspcg + 0.5f);
  //   	  if( wtbl_index > 1.0f * ltbl )
  //   	    wtbl_value_x = 0.0f; 
  //   	  else{
  //   	    wtbl_index_y = wtbl_index / BLOCK_2DIM_X;
  //   	    wtbl_index_x = wtbl_index - wtbl_index_y * BLOCK_2DIM_X;
  //   	    wtbl_value_x = shmem_wtbl[ wtbl_index_x ][ wtbl_index_y ];
  //   	  }
  //   	  wtbl_value = wtbl_value_x * wtbl_value_y;

  //   	  proj_value = tex2D( tex_cproj_res, proj_x + 0.5f, proj_y + 0.5f );
	
  //   	  x_res += wtbl_value * proj_value.x;
  //   	  y_res += wtbl_value * proj_value.y;      
  //   	}
  //     }
  //   }

    // // 
    // if( fabsf( angle_value - PI ) < 1e-5f ){

    //   if( 1.0f * idy - L2 - M2 <= 0.0f && 1.0f * idy + L2 - M2 >= 0.0f ){
    // 	j_lb = 1.0f * idx + L2 - M2;
    // 	j_ub = 1.0f * idx - L2 - M2;
    // 	for( jj = j_lb; jj <= j_ub; jj++ ){

    // 	  proj_x = floor( jj );
    // 	  if( proj_x >= 0.0f && proj_x < 1.0f * projdim.x){
 
    // 	    x = scale * proj_x * cosf(angle_value) + M2;
    // 	    y = scale * proj_x * sinf(angle_value) + M2;

    // 	    wtbl_index = (int)(fabs( y - idy ) * tblspcg + 0.5f);

    // 	    if( wtbl_index > 1.0f * ltbl )
    // 	      wtbl_value_y = 0.0f; 
    // 	    else{
    // 	      wtbl_index_y = wtbl_index / BLOCK_2DIM_X;
    // 	      wtbl_index_x = wtbl_index - wtbl_index_y * BLOCK_2DIM_X;
    // 	      wtbl_value_y = shmem_wtbl[ wtbl_index_x ][ wtbl_index_y ];
    // 	    }

    // 	    wtbl_index = (int)(fabs( x - idx ) * tblspcg + 0.5f);
    // 	    if( wtbl_index > 1.0f * ltbl )
    // 	      wtbl_value_x = 0.0f; 
    // 	    else{
    // 	      wtbl_index_y = wtbl_index / BLOCK_2DIM_X;
    // 	      wtbl_index_x = wtbl_index - wtbl_index_y * BLOCK_2DIM_X;
    // 	      wtbl_value_x = shmem_wtbl[ wtbl_index_x ][ wtbl_index_y ];
    // 	    }
    // 	    wtbl_value = wtbl_value_x * wtbl_value_y;

    // 	    proj_value = tex2D( tex_cproj_res, proj_x + 0.5f, proj_y + 0.5f );
	
    // 	    x_res += wtbl_value * proj_value.x;
    // 	    y_res += wtbl_value * proj_value.y;      
    // 	  }
    // 	}
    //   }
    // }
  
  // }

  // d_data_H[ idy * projdim.x + idx ].x =  shmem_H_x[ cidx.x ][ cidx.y ] + x_res; 
  // d_data_H[ idy * projdim.x + idx ].y =  shmem_H_y[ cidx.x ][ cidx.y ] + y_res;

//   d_data_H[ idy * projdim.x + idx ].x =  x_res; 
//   d_data_H[ idy * projdim.x + idx ].y =  y_res;

// }


#endif // GPU_GRIDREC



void GridRec::list_pswf_su( listSample* list_pswf_recon, listSample* list_pswf_sino, float L, long ltbl, 
			    long pdim, long M, float scale ){

  float U, V, rtmp, convolv;
  float L2 = L/2.0;
  float tblspcg = 2*ltbl/L; 
  long  pdim2=pdim>>1, M2=M>>1, iul, iuh, iu, ivl, ivh, iv, n; 

  for (n = 0; n < theta_list_size; n++) {    /*** Start loop on angles */ 
    
    int j, k; 
    for (j = 1; j < pdim2; j++) {  	/* Start loop on transform data */ 

      rtmp=scale*j;

      U = rtmp * COSE[n] + M2;    /* X direction */ 
      V = rtmp * SINE[n] + M2;	  /* Y direction */ 
				       
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

	  Sample s1;
	  s1.ny = n;
	  s1.nx = j;
	  s1.wa = convolv; 

	  list_pswf_recon[ iu * M + iv ].push_back( s1 );

	  Sample s2;
	  s2.ny = n;
	  s2.nx = pdim - j;
	  s2.wa = convolv; 

	  list_pswf_recon[ (M - iu) * M + (M - iv) ].push_back( s2 );  

	  Sample s3;
	  s3.ny = iu;
	  s3.nx = iv;
	  s3.wa = convolv; 

	  list_pswf_sino[ n * M + j ].push_back( s3 );

	} 
      } 
    } 
  }   

  int sz, j; 
  for (n = 0; n < theta_list_size; n++) {    /*** Start loop on angles */ 
    for (j = 1; j < pdim2; j++) {  	/* Start loop on transform data */ 
      sz = list_pswf_sino[ n * M + j ].size();
      list_pswf_sino[ n * M + j ].resize( sz );
    }
  }

  for (n = 0; n < M; n++) {   
    for (j = 0; j < M; j++) { 
      sz = list_pswf_recon[ n * M + j ].size();
      list_pswf_recon[ n * M + j ].resize( sz );
    }
  }


}

