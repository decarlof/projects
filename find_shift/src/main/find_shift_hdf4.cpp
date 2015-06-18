// Image Concatecation using cross correlation

#include <iostream>   
#include <fstream>
#include <cstdlib>
#include <string>

#include "teem/nrrd.h"
#include "fftw3.h"
#include "nexusbox.h"

#include "find_shift_config.h"  // MACROs and Configurations

#ifdef USE_BRUTE_FORCE_GPU

#include <cutil.h>
#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include <cutil_math.h>

#endif // USE_BRUTE_FORCE_GPU

#define OUTPUT_NRRD
#define OUTPUT_LOG

// #define OUTPUT_TXT  // output nrrd as txt for debug using matlab

#define DIST_BOUNDARY 100

#define TRANS_START_X  -250
#define TRANS_END_X    250
#define TRANS_INV_X     1

#define TRANS_START_Y  -2
#define TRANS_END_Y     2
#define TRANS_INV_Y     1

#define SUBPIXEL_SHIFT  10

using std::cout;
using std::endl;
using std::fstream;
using std::ifstream;
using std::string;
using std::ios;

#define HDF4_DATATYPE unsigned short int  // unit16 for hdf4 
void cross_correlation( double *, double *, double, double, double *, int, int,	int, int, double);
void data_interp(double *, double *, double *, double, double,
		 int, int, int, int, int, int); 

#ifdef USE_BRUTE_FORCE_GPU
extern "C"
void cross_correlation_wrapper(float *, cudaArray*, 
			       double *, 
			       float, float, float, float, int, int, 
			       float, float,
			       int, int, int, int, 
			       int, int);
#endif // USE_BRUTE_FORCE_GPU

int main(int argc, char**  argv){ 

  if( argc < 3 || argc > 13 ){   

#ifdef USE_VERBOSE
    cout << "Info: find_shift version " << FIND_SHIFT_VERSION_MAJOR
         << "." << FIND_SHIFT_VERSION_MINOR << endl;
#ifdef USE_FFTW3
    cout << "Info: libfftw3 is used for computation" << endl;
#else
    cout << "Info: brute-force computation applied" << endl;
#endif // USE_FFTW3

#ifdef USE_BRUTE_FORCE_GPU
    cout << "Info:  GPU is used for brute-force computation" << endl;
#endif // USE_BRUTE_FORCE_GPU
    
    cout << "Usage: find_shift  /data/tom2/Sam06/  Sam06_exp.hdf" << endl;
    cout << "     [transStartX=-150.0 transEndX = 150.0 transStartY=-2 transEndY=2 " << endl;
    cout << "     trans_invX = 1 trans_invY = 1 " << endl;
    cout << "     ROI_x1=100 ROI_y1=100 ROI_x2=xdim-1-100 ROI_y2=ydim-1-100]  " << endl;
#endif

    exit( 1 );
  }

  // log file
#ifdef OUTPUT_LOG
  char strDataPathLog[256];
  strcpy(strDataPathLog, argv[1]);
  strcat(strDataPathLog, "LogCrossCorrelation.txt");

  std::ofstream logFile(strDataPathLog, std::ios::out ); 
#endif

  unsigned int nx, ny, nz, nxx, nyy;
  int indexTemplate, indexImg; 
  double dTemplatePixel, dImgPixel, dInterpPixel, dInterpWeight;

  double dMatchRes, dMatchOptim; 
  int nTransIndX, nTransIndY, nTransXOptim, nTransYOptim;
  int nRotateInd, nRotateOptim;
  int nHdfIndOptim; 
  double dTransX, dTransY, x, y;

  char strDataPath[256];
  strcpy(strDataPath, argv[1]);
  strcat(strDataPath, "raw/");

  //Step1: read the experimental file using NexusBoxClass
  NexusBoxClass exp_file;
  
  exp_file.SetReadScheme (ENTIRE_CONTENTS); 
  exp_file.ReadAll( argv[1], argv[2] ); 

  char index[256];
  int rank, dims[2], type; 

  // get the name for the image data set
  char index_data_group[256];

  strcpy (index_data_group, ";experiment;reconstruction;cluster_config;data_group_index"); 
  if (!exp_file.IndexExists(index_data_group)) { 

#ifdef USE_VERBOSE
    cout << "Index " << index_data_group << " does not exist. Use default." << endl; 
#endif

#ifdef OUTPUT_LOG
    logFile << "Index " << index_data_group << " does not exist. Use default." << endl; 
#endif

    exit(1);
  } 

  int dims_index_data_group; 
  exp_file.GetDatumInfo (index_data_group, &rank, &dims_index_data_group, &type); 

  // Note that the entry for index_data_group is one-dimension only (rank = 1) . 
  if( rank != 1 ){

#ifdef USE_VERBOSE
    cout << "The entry " << index_data_group << " should be one-dimensional "  << endl; 
#endif

#ifdef OUTPUT_LOG
    logFile << "The entry " << index_data_group << " should be one-dimensional "  << endl; 
#endif

    exit(1); 
  }

  char index_data[256]; 
  exp_file.GetDatum (index_data_group, index_data); 

  index_data[ dims_index_data_group ] = '\0';   // This step is very important. 

#ifdef USE_VERBOSE
  cout << "The index for the image data file is " << index_data << endl; 
#endif

#ifdef OUTPUT_LOG
  logFile << "The index for the image data file is is " << index_data << endl; 
#endif

  // check the start angle 
  char index_start_angle[256];
  float start_angle; 
  strcpy (index_start_angle, ";experiment;acquisition;parameters;start_angle"); 
  if (!exp_file.IndexExists(index_start_angle)) {

#ifdef USE_VERBOSE
    cout << "Error: index " << index_start_angle << " does not exist. Take default 0" << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "Error: index " << index_start_angle << " does not exist. Take default 0" << endl;
#endif

  } 
  exp_file.GetDatum (index_start_angle, &start_angle); 

  if( start_angle > 0.5 ){

#ifdef USE_VERBOSE
    cout << "Note the start_angle = " << start_angle << " is far from 0. " << endl;
    cout << "Cross correlation results may not be accurate" << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "Note the start_angle = " << start_angle << " is far from 0. " << endl;
    logFile << "Cross correlation results may not be accurate" << endl;
#endif

  }

#ifdef USE_VERBOSE
  cout << "The start angle is " << start_angle << endl; 
#endif

#ifdef OUTPUT_LOG
  logFile << "The start angle is " << start_angle << endl; 
#endif

  // check the end angle 
  char index_end_angle[256];
  float end_angle;
  strcpy (index_end_angle, ";experiment;acquisition;parameters;end_angle"); 
  if (!exp_file.IndexExists(index_end_angle)) { 

#ifdef USE_VERBOSE
    cout << "Error: index " << index_end_angle << " does not exist. Take default 180." << endl; 
#endif

#ifdef OUTPUT_LOG
    logFile << "Error: index " << index_end_angle << " does not exist. Take default 180." << endl; 
#endif

  } 
  exp_file.GetDatum (index_end_angle, &end_angle); 

  if( end_angle > 180.5 || end_angle < 179.5){

#ifdef USE_VERBOSE
    cout << "Note the end_angle = " << end_angle << " is far from 180. " << endl;
    cout << "Cross correlation results may not be accurate" << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "Note the end_angle = " << end_angle << " is far from 180. " << endl;
    logFile << "Cross correlation results may not be accurate" << endl;
#endif

  }

#ifdef USE_VERBOSE
  cout << "The end angle is " << end_angle << endl; 
#endif

#ifdef OUTPUT_LOG
  logFile << "The end angle is " << end_angle << endl; 
#endif

  //Projection File Names 
  char index_proj_name[256];
  strcpy (index_proj_name, ";experiment;acquisition;projections;names"); 
  if (!exp_file.IndexExists(index_proj_name)) {

#ifdef USE_VERBOSE
    cout << "Index " << index_proj_name << " does not exist." << endl;
#endif 

#ifdef OUTPUT_LOG
    logFile << "Index " << index_proj_name << " does not exist." << endl;
#endif 

    exit(1);
  } 
  exp_file.GetDatumInfo (index_proj_name, &rank, dims, &type); 
 
  char* proj_file_list = (char *) new char[ dims[0]*dims[1] ]; 
  if (proj_file_list == NULL) {

#ifdef USE_VERBOSE
    cout << "Could not allocat memory for file_list." << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "Could not allocat memory for file_list." << endl;
#endif

    exit(1);
  } 
  exp_file.GetDatum (index_proj_name, proj_file_list); 

  char file_name_match[256];    // the first projection file - 0 degree
  char file_name_template[256]; // the last projection file - 180 degree
  
  strncpy(file_name_match, proj_file_list, dims[1]);
  file_name_match[dims[1]] = '\0'; 
  strncpy(file_name_template, &proj_file_list[ (dims[0]-1) * dims[1] ], dims[1]);
  file_name_template[dims[1]] = '\0'; 

  delete [] proj_file_list;

#ifdef USE_VERBOSE
  cout << "The template file name is " << file_name_template << endl; 
  cout << "The match file name is " << file_name_match << endl; 
#endif

#ifdef OUTPUT_LOG
  logFile << "The template file name is " << file_name_template << endl; 
  logFile << "The match file name is " << file_name_match << endl; 
#endif

  //White Field File Names 
  char index_white_name[256];
  strcpy (index_white_name, ";experiment;acquisition;white_field;names"); 
  if (!exp_file.IndexExists(index_white_name)) { 

#ifdef USE_VERBOSE
    cout << "Index " << index_white_name << " does not exist." << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "Index " << index_white_name << " does not exist." << endl;
#endif

    exit(1);
  } 
  exp_file.GetDatumInfo (index_white_name, &rank, dims, &type); 

  char* white_file_list = (char *) new char[ dims[0]*dims[1] ]; 
  exp_file.GetDatum (index_white_name, white_file_list); 
 
  char file_name_white_match[256];    // the first white field file - 0 degree
  char file_name_white_template[256];    // the last white field file - 180 degree

  if( dims[0] <= 1 ){

#ifdef USE_VERBOSE
    cout << "The current program needs at least two white field images! Exiting..." << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "The current program needs at least two white field images! Exiting..." << endl;
#endif

    exit(1);
  }

  strncpy(file_name_white_match, white_file_list, dims[1]); 
  file_name_white_match[ dims[1] ] = '\0'; 
  strncpy(file_name_white_template, &white_file_list[ (dims[0]-1) * dims[1] ], dims[1]);
  file_name_white_template[ dims[1] ] = '\0'; 

  delete [] white_file_list; 

#ifdef USE_VERBOSE
  cout << "The white field file name for " << file_name_template << " is " << file_name_white_template << endl; 
  cout << "The white field file name for " << file_name_match << " is " << file_name_white_match << endl; 
#endif

#ifdef OUTPUT_LOG
  logFile << "The white field file name for " << file_name_template << " is " << file_name_white_template << endl; 
  logFile << "The white field file name for " << file_name_match << " is " << file_name_white_match << endl; 
#endif

  //Dark Field File Names 
  char index_dark_name[256];
  strcpy (index_dark_name, ";experiment;acquisition;black_field;names"); 
  if (!exp_file.IndexExists(index_dark_name)) {

#ifdef USE_VERBOSE
    cout << "Index " << index_dark_name << " does not exist." << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "Index " << index_dark_name << " does not exist." << endl;
#endif

    exit(1);
  } 
  exp_file.GetDatumInfo (index_dark_name, &rank, dims, &type); 
  char* dark_file_list = (char *) new char[ dims[0]*dims[1] ]; 
  exp_file.GetDatum (index_dark_name, dark_file_list); 
 
  if( dims[0] != 1 ){

#ifdef USE_VERBOSE
    cout << "The current version supports one dark field image only! Exiting..." << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "The current version supports one dark field image only! Exiting..." << endl;
#endif

    exit(1);
  }

  char file_name_dark[256];
  strncpy(file_name_dark, dark_file_list, dims[1]); 
  file_name_dark[dims[1]] = '\0'; 

#ifdef USE_VERBOSE
  cout << "The dark field file name is " << file_name_dark << endl; 
#endif

#ifdef OUTPUT_LOG
  logFile << "The dark field file name is " << file_name_dark << endl; 
#endif

  int nHeight, nWidth, volsize;

  //Step2: start to read HDF data 
  NexusBoxClass nexus_template_file;
  nexus_template_file.SetReadScheme (ENTIRE_CONTENTS); 
  nexus_template_file.ReadAll( strDataPath, file_name_template );  
  if (!nexus_template_file.IndexExists(index_data)) {

#ifdef USE_VERBOSE
    cout << "Error: index " << index_data << " does not exist." << endl; 
#endif

#ifdef OUTPUT_LOG
    logFile << "Error: index " << index_data << " does not exist." << endl; 
#endif

    exit(1);
  } 
  nexus_template_file.GetDatumInfo (index_data, &rank, dims, &type); 

  nHeight = dims[0];
  nWidth = dims[1];
  volsize = nWidth * nHeight; 

  if( nHeight & (nHeight - 1 ) != 0  || nWidth & (nWidth - 1 ) != 0 ){

    cout << "This version need the projection dimensions to be power of 2" << endl; // power
    exit(1);
  }

  HDF4_DATATYPE* dataTemplate = new HDF4_DATATYPE[ volsize ];  
  if( !dataTemplate ){

#ifdef USE_VERBOSE
    cout << "Error allocating memory for dataTemplate!" << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "Error allocating memory for dataTemplate!" << endl;
#endif

    exit(1);
  }
  nexus_template_file.GetDatum (index_data, dataTemplate); 

  //
  NexusBoxClass nexus_match_file;
  nexus_match_file.SetReadScheme (ENTIRE_CONTENTS); 
  nexus_match_file.ReadAll( strDataPath, file_name_match ); 
  if (!nexus_match_file.IndexExists(index_data)) {

#ifdef USE_VERBOSE
    cout << "Error: index " << index_data << " does not exist." << endl; 
#endif

#ifdef OUTPUT_LOG
    logFile << "Error: index " << index_data << " does not exist." << endl; 
#endif

    exit(1);
  } 
  nexus_match_file.GetDatumInfo (index_data, &rank, dims, &type); 
  
  if( dims[0] != nHeight || dims[1] != nWidth ){

#ifdef USE_VERBOSE
    cout << "The dimension of data set in " << file_name_match << " is not correct!"<< endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "The dimension of data set in " << file_name_match << " is not correct!"<< endl;
#endif

    exit(1); 
  }

  HDF4_DATATYPE* dataMatch = new HDF4_DATATYPE[ volsize ];
  if( !dataMatch ){

#ifdef USE_VERBOSE
    cout << "Error allocating memory for dataMatch!" << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "Error allocating memory for dataMatch!" << endl;
#endif

    exit(1);
  }
  nexus_match_file.GetDatum (index_data, dataMatch); 

  // 
  NexusBoxClass nexus_white1_file;
  
  nexus_white1_file.SetReadScheme (ENTIRE_CONTENTS); 
  nexus_white1_file.ReadAll( strDataPath, file_name_white_template ); 
  if (!nexus_white1_file.IndexExists(index_data)) {

#ifdef USE_VERBOSE
    cout << "Error: index " << index_data << " does not exist." << endl; 
#endif

#ifdef OUTPUT_LOG
    logFile << "Error: index " << index_data << " does not exist." << endl; 
#endif

    exit(1);
  } 
  nexus_white1_file.GetDatumInfo (index_data, &rank, dims, &type); 

  if( dims[0] != nHeight || dims[1] != nWidth ){

#ifdef USE_VERBOSE
    cout << "The dimension of data set in " << file_name_white_template << " is not correct!"<< endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "The dimension of data set in " << file_name_white_template << " is not correct!"<< endl;
#endif

    exit(1); 
  }

  HDF4_DATATYPE* dataWhite1 = new HDF4_DATATYPE[ volsize ];
  if( !dataWhite1 ){

#ifdef USE_VERBOSE
    cout << "Error allocating memory for dataWhite1!" << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "Error allocating memory for dataWhite1!" << endl;
#endif

    exit(1);
  }

  nexus_white1_file.GetDatum (index_data, dataWhite1); 

  // 
  NexusBoxClass nexus_white2_file;
  
  nexus_white2_file.SetReadScheme (ENTIRE_CONTENTS); 
  nexus_white2_file.ReadAll( strDataPath, file_name_white_match ); 
  if (!nexus_white2_file.IndexExists(index_data)) {

#ifdef USE_VERBOSE
    cout << "Error: index " << index_data << " does not exist." << endl; 
#endif

#ifdef OUTPUT_LOG
    logFile << "Error: index " << index_data << " does not exist." << endl; 
#endif

    exit(1);
  } 
  nexus_white2_file.GetDatumInfo (index_data, &rank, dims, &type); 

  if( dims[0] != nHeight || dims[1] != nWidth ){

#ifdef USE_VERBOSE
    cout << "The dimension of data set in " << file_name_white_match << " is not correct!"<< endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "The dimension of data set in " << file_name_white_match << " is not correct!"<< endl;
#endif

    exit(1); 
  }

  HDF4_DATATYPE* dataWhite2 = new HDF4_DATATYPE[ volsize ];
  if( !dataWhite2 ){

#ifdef USE_VERBOSE
    cout << "Error allocating memory for dataWhite2!" << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "Error allocating memory for dataWhite2!" << endl;
#endif

    exit(1);
  }
  nexus_white2_file.GetDatum (index_data, dataWhite2); 

  //
  NexusBoxClass nexus_dark_file;
  
  nexus_dark_file.SetReadScheme (ENTIRE_CONTENTS); 
  nexus_dark_file.ReadAll( strDataPath, file_name_dark ); 
  if (!nexus_dark_file.IndexExists(index_data)) {

#ifdef USE_VERBOSE
    cout << "Error: index " << index_data << " does not exist." << endl; 
#endif

#ifdef OUTPUT_LOG
    logFile << "Error: index " << index_data << " does not exist." << endl; 
#endif

    exit(1);
  } 
  nexus_dark_file.GetDatumInfo (index_data, &rank, dims, &type); 
  if( dims[0] != nHeight || dims[1] != nWidth ){

#ifdef USE_VERBOSE
    cout << "The dimension of data set in " << file_name_dark << " is not correct!"<< endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "The dimension of data set in " << file_name_dark << " is not correct!"<< endl;
#endif

    exit(1); 
  }

  HDF4_DATATYPE* dataDark = new HDF4_DATATYPE[ volsize ];
  if( !dataDark ){

#ifdef USE_VERBOSE
    cout << "Error allocating memory for dataDark!" << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "Error allocating memory for dataDark!" << endl;
#endif

    exit(1);
  }

  nexus_dark_file.GetDatum (index_data, dataDark); 

  // 
  double dTransStartX = TRANS_START_X; 
  double dTransEndX = TRANS_END_X;   

  if( argc >= 4)
    dTransStartX = atof( argv[3] );

  if( argc >= 5)
    dTransEndX = atof( argv[4] );

  if( dTransStartX > dTransEndX ){

#ifdef USE_VERBOSE
    cout << "The input dTransStartX = " << dTransStartX << " is larger than dTransEndX = " << dTransEndX << endl;
    cout << "Use dTransStartX = " << dTransEndX << " and dTransEndX = " << dTransStartX << " instead" << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "The input dTransStartX = " << dTransStartX << " is larger than dTransEndX = " << dTransEndX << endl;
    logFile << "Use dTransStartX = " << dTransEndX << " and dTransEndX = " << dTransStartX << " instead" << endl;
#endif

    double tmp = dTransStartX;
    dTransStartX = dTransEndX;
    dTransEndX = tmp;
  }

  double dTransStartY = TRANS_START_Y; 
  double dTransEndY = TRANS_END_Y; 

  if( argc >= 6)
    dTransStartY = atof( argv[5] );

  if( argc >= 7)
    dTransEndY = atof( argv[6] );

  if( dTransStartY > dTransEndY ){

#ifdef USE_VERBOSE
    cout << "The input dTransStartY = " << dTransStartY << " is larger than dTransEndY = " << dTransEndY << endl;
    cout << "Use dTransStartY = " << dTransEndY << " and dTransEndY = " << dTransStartY << " instead" << endl;
#endif 

#ifdef OUTPUT_LOG
    logFile << "The input dTransStartY = " << dTransStartY << " is larger than dTransEndY = " << dTransEndY << endl;
    logFile << "Use dTransStartY = " << dTransEndY << " and dTransEndY = " << dTransStartY << " instead" << endl;
#endif 

    double tmp = dTransStartY;
    dTransStartY = dTransEndY;
    dTransEndY = tmp;
  }

  double dTransInvX = TRANS_INV_X; 
  double dTransInvY = TRANS_INV_Y; 

  if( argc >= 8)
    dTransInvX = atof( argv[7] );

  if( argc >= 9)
    dTransInvY = atof( argv[8] );
  
#ifdef USE_VERBOSE
  cout << "The search range in x is [ " << dTransStartX << " , " << dTransEndX << " ] with interval " << dTransInvX << endl;
  cout << "The search range in y is [ " << dTransStartY << " , " << dTransEndY << " ] with interval " << dTransInvY << endl;
#endif 

#ifdef OUTPUT_LOG
  logFile << "The search range in x is [ " << dTransStartX << " , " << dTransEndX << " ] with interval " << dTransInvX << endl;
  logFile << "The search range in y is [ " << dTransStartY << " , " << dTransEndY << " ] with interval " << dTransInvY << endl;
#endif 

  int numTransX = (int) ( (dTransEndX - dTransStartX) / dTransInvX ) + 1; 
  int numTransY = (int) ( (dTransEndY - dTransStartY) / dTransInvY ) + 1; 

  double* score_match = new double[ numTransX * numTransY ];
  if( !score_match ){

#ifdef USE_VERBOSE
    cout << "Error allocating memory for score_match!" << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "Error allocating memory for score_match!" << endl;
#endif

    exit( 1 );
  }

  int nROILowerX = DIST_BOUNDARY;
  int nROILowerY = DIST_BOUNDARY;

  if( argc >= 10 )
    nROILowerX = atoi( argv[9] );

  if( argc >= 11 )
    nROILowerY = atoi( argv[10] );

  int nROIUpperX = nWidth - 1 - DIST_BOUNDARY;
  if( argc >= 12 )
    nROIUpperX = atoi( argv[11] );
  else if( nROIUpperX < 0 ){

#ifdef USE_VERBOSE
    cout << "DIST_BOUNDARY is too large! Use nWidth -1 instead!" << endl;
#endif 

#ifdef OUTPUT_LOG
    logFile << "DIST_BOUNDARY is too large! Use nWidth -1 instead!" << endl;
#endif 

    nROIUpperX = nWidth - 1;
  }

  int nROIUpperY = nHeight - 1 - DIST_BOUNDARY;
  if( argc >= 13 )
    nROIUpperY = atoi( argv[12] );
  else if( nROIUpperY < 0 ){

#ifdef USE_VERBOSE
    cout << "DIST_BOUNDARY is too large! Use nHeight-1 instead!" << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "DIST_BOUNDARY is too large! Use nHeight-1 instead!" << endl;
#endif

    nROIUpperY = nHeight - 1;
  }

  if( nROIUpperX < nROILowerX ){
    nx = nROIUpperX;
    nROIUpperX = nROILowerX;
    nROILowerX = nx;
  }

  if( nROIUpperY < nROILowerY ){
    ny = nROIUpperY;
    nROIUpperY = nROILowerY;
    nROILowerY = ny;
  }

  int nROIArea = ( nROIUpperX - nROILowerX + 1 ) * ( nROIUpperY - nROILowerY + 1 );

  double* dataTemplateROI = new double[ nROIArea ];
  double* dataMatch2 = new double[ volsize ];
  double* dataMatchROI = new double[ nROIArea ];
  if( !dataTemplate || !dataMatch || !dataWhite1 || !dataWhite2
      || !dataDark || !dataTemplateROI || !dataMatchROI){

#ifdef USE_VERBOSE
    cout << "Error allocating memory" << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "Error allocating memory" << endl;
#endif

    exit(1); 
  }

#ifdef USE_VERBOSE
  cout << "ROI region: [ " << nROILowerX << " , " << nROILowerY << " ] -> [ "
       << nROIUpperX << " , " << nROIUpperY << " ] " << endl;
#endif

#ifdef OUTPUT_LOG
  logFile << "ROI region: [ " << nROILowerX << " , " << nROILowerY << " ] -> [ "
	  << nROIUpperX << " , " << nROIUpperY << " ] " << endl;
#endif

  double dTemplateAvg = 0.0;
  double dImgROIAvg = 0.0;
  
  double dTemplate, dMatch, dWhite1, dWhite2, dDark, dValue;

#ifdef OUTPUT_NRRD
  float fTemplateMax = -1e20;
  float fTemplateMin = 1e20;
#endif

  // template for ROI
  for( ny = nROILowerY; ny <= nROIUpperY; ny++ ){
    for( nx = nROILowerX; nx <= nROIUpperX; nx++ ){

      indexTemplate = ( ny - nROILowerY ) * (nROIUpperX - nROILowerX + 1) + nx - nROILowerX ;

      indexImg = ny * nWidth + nWidth - nx - 1 ;

      dTemplate = dataTemplate[ indexImg ]; 
      dWhite1 = dataWhite1[ indexImg ]; 
      dWhite2 = dataWhite2[ indexImg ]; 
      dDark = dataDark[ indexImg ]; 

      dValue = dTemplate - dDark;
      if( dValue <= 0 )
	dataTemplateROI[ indexTemplate ] = 0.0;
      else{

	dValue = (1.0 * dWhite1 - dDark) / dValue;

	if( dValue > 0.0 )
	  dataTemplateROI[ indexTemplate ] = log( dValue);
	else
	  dataTemplateROI[ indexTemplate ] = 0.0;

      }

      dTemplateAvg += dataTemplateROI[ indexTemplate ];

#ifdef OUTPUT_NRRD
      if( dataTemplateROI[ indexTemplate ] > fTemplateMax )
	fTemplateMax = (float) dataTemplateROI[ indexTemplate ];
      if( dataTemplateROI[ indexTemplate ] < fTemplateMin )
	fTemplateMin = (float) dataTemplateROI[ indexTemplate ];
#endif

    }
  }
  dTemplateAvg /= nROIArea; 

  double dTemplateRMS = 0.0;
  for( int ny = 0 ; ny < nROIUpperY - nROILowerY + 1; ny++ ){
    for( int nx = 0; nx < nROIUpperX - nROILowerX + 1; nx++ ){  

      int indexTemplate = ny * ( nROIUpperX - nROILowerX + 1 ) + nx;
      double dTemplatePixel = dataTemplateROI[ indexTemplate ];

      dTemplateRMS += ( dataTemplateROI[ indexTemplate ] - dTemplateAvg ) *
	( dataTemplateROI[ indexTemplate ] - dTemplateAvg );
    }
  }

  dTemplateRMS = sqrt( dTemplateRMS );

  // match for whole image, for interpolation 
  for( ny = 0; ny < nHeight ; ny++ ){
    for( nx = 0; nx < nWidth; nx++ ){

      indexImg = ny * nWidth + nx ;

      dMatch = dataMatch[ indexImg ]; 
      dWhite1 = dataWhite1[ indexImg ]; 
      dWhite2 = dataWhite2[ indexImg ]; 
      dDark = dataDark[ indexImg ]; 

      dValue = dMatch - dDark;
      if( dValue <= 0.0 )
	dataMatch2[ indexImg ] = 0.0;
      else{

	dValue = (1.0 * dWhite2 - dDark) / dValue;

	if( dValue > 0.0 )
	  dataMatch2[ indexImg ] = log( dValue);
	else
	  dataMatch2[ indexImg ] = 0.0;
      }
    }
  }

  // get data for dataMatchROI
  for( ny = nROILowerY; ny <= nROIUpperY; ny++ ){
    for( nx = nROILowerX; nx <= nROIUpperX; nx++ ){

      indexTemplate = ( ny - nROILowerY ) * (nROIUpperX - nROILowerX + 1) + nx - nROILowerX ;
      indexImg = ny * nWidth + nx ;

      dataMatchROI[ indexTemplate ] = dataMatch2[ indexImg ];
    }
  }


#ifdef OUTPUT_NRRD
  // prepare name
  // get the base name for the data set
  char  basename[256];
  strcpy (index, ";experiment;setup;sample;base_name"); 
  if (!exp_file.IndexExists(index)) {

#ifdef OUTPUT_LOG
    logFile << "Error: index " << index << " does not exist." << endl; 
#endif

#ifdef USE_VERBOSE
    cout << "Error: index " << index << " does not exist." << endl; 
#endif

  } 
  exp_file.GetDatumInfo (index, &rank, dims, &type); 
  exp_file.GetDatum (index, basename); 

  basename[ dims[0] ] = '\0';   // This step is very important. 

  // string strBasename = string( basename );

  // #ifdef USE_VERBOSE
  //   cout << "base_name for the dataset is " << strBasename << endl;
  // #endif

  // prepare data
  float* dataNrrd = new float[ volsize * 3 ];
  if( !dataNrrd ){

#ifdef USE_VERBOSE
    cout << "Error allocating memory for dataNrrd " << endl; 
#endif 

#ifdef OUTPUT_LOG
    logFile << "Error allocating memory for dataNrrd " << endl; 
#endif 

    exit(1);
  }

  float fTemplate, fMatch, fWhite1, fWhite2, fDark, fValue;
  float fTemplateAvg = (float)dTemplateAvg;
  float fMatchAvg = 0.0f;
  int indexMatch, indexNrrd;

  float fMatchMax = -1e20;
  float fMatchMin = 1e20;

  for( ny = nROILowerY; ny <= nROIUpperY; ny++ ){
    for( nx = nROILowerX; nx <= nROIUpperX; nx++ ){

      indexImg = ny * nWidth + nx ;

      fValue = (float)dataMatch2[ indexImg ];

      fMatchAvg += fValue;

#ifdef OUTPUT_NRRD 
      if( fValue > fMatchMax )
	fMatchMax = fValue;
      if( fValue < fMatchMin )
	fMatchMin = fValue;
#endif

    }
  }
  fMatchAvg /= nROIArea; 

  if( fTemplateAvg == 0.0f || fMatchAvg == 0.0f ){

#ifdef USE_VERBOSE
    cout << "Error: fTemplateAvg == 0.0f || fMatchAvg == 0.0f " << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "Error: fTemplateAvg == 0.0f || fMatchAvg == 0.0f " << endl;
#endif

    exit(1);
  }
   
  for( ny = nROILowerY; ny <= nROIUpperY; ny++ ){
    for( nx = nROILowerX; nx <= nROIUpperX; nx++ ){

      indexTemplate = (ny - nROILowerY) * (nROIUpperX - nROILowerX + 1) + nx - nROILowerX;
      indexNrrd = ny * nWidth + nx ;

      fValue = (float) dataTemplateROI[ indexTemplate ]; 
      dataNrrd[ indexNrrd ] = 255 * ( fValue - fTemplateMin) / (fTemplateMax - fTemplateMin);
      dataNrrd[ 2 * volsize + indexNrrd ] = 0.5 * dataNrrd[ indexNrrd ]; 

      indexMatch = ny * nWidth + nx ;
      indexNrrd = volsize + ny * nWidth + nx ;

      fValue = (float) dataMatch2[ indexMatch ];
      dataNrrd[ indexNrrd ] = 255 * ( fValue - fMatchMin) / (fMatchMax - fMatchMin);
    }
  }

#endif // OUTPUT_NRRD      

  // Step3: go through the matching image sequence and perform cross correlation

  dMatchOptim = -1.0; 
  nTransXOptim = 0;

#ifdef USE_VERBOSE
  cout << "Start matching using cross correlation method" << endl;
#endif

#ifdef OUTPUT_LOG
  logFile << "Start matching using cross correlation method" << endl;
#endif 

  
#ifdef USE_FFTW3   // perform cross-correlation in Fourier domain

  int nROICenterX = (nROIUpperX - nROILowerX + 1)/2;
  int nROICenterY = (nROIUpperY - nROILowerY + 1)/2;
  int nROIWidth = nROIUpperX - nROILowerX + 1;
  int nROIHeight = nROIUpperY - nROILowerY + 1;

  // DFT for dataTemplateROI
  double* dataTemplateROI2 = new double[ nROIArea ];  // the data in this buffer will be ruined in fftw
  for( int i = 0; i < nROIArea; i++ ){
    dataTemplateROI2[ i ] = dataTemplateROI[ i ];
  }

  fftw_complex *dftTemplateROI = (fftw_complex *) fftw_malloc( sizeof(fftw_complex) * nROIHeight * (nROIWidth/2+1) );
  fftw_plan planForwardTemplateROI = fftw_plan_dft_r2c_2d(nROIHeight, nROIWidth,
							  dataTemplateROI2,
							  dftTemplateROI,
							  FFTW_ESTIMATE);

  fftw_execute(planForwardTemplateROI);

  delete[] dataTemplateROI2;

  // DFT for dataMatchROI
  double* dataMatchROI2 = new double[ nROIArea ];  // the data in this buffer will be ruined in fftw
  for( int i = 0; i < nROIArea; i++ ){
    dataMatchROI2[ i ] = dataMatchROI[ i ];
  }

  fftw_complex *dftMatchROI = (fftw_complex *) fftw_malloc( sizeof(fftw_complex) * nROIHeight * (nROIWidth/2+1) );
  fftw_plan planForwardMatchROI = fftw_plan_dft_r2c_2d(nROIHeight, nROIWidth,
						       dataMatchROI2,
						       dftMatchROI,
						       FFTW_ESTIMATE);

  fftw_execute(planForwardMatchROI);

  delete[] dataMatchROI2;

  // compute dftTemplateROI * conj(dftMatchROI)

  fftw_complex *dftProductROI = (fftw_complex *) fftw_malloc( sizeof(fftw_complex) * nROIHeight * (nROIWidth/2+1) );
  double *dataCrossCorrROI = new double[ nROIWidth * nROIHeight ];

  for(int i = 0; i <  nROIHeight * (nROIWidth/2+1); i++ ){

    dftProductROI[i][0] = dftTemplateROI[i][0] * dftMatchROI[i][0]
      + dftTemplateROI[i][1] * dftMatchROI[i][1];

    dftProductROI[i][1] = -dftTemplateROI[i][0] * dftMatchROI[i][1]
      + dftTemplateROI[i][1] * dftMatchROI[i][0];

  }

  fftw_plan planBackwardProductROI = fftw_plan_dft_c2r_2d(nROIHeight, nROIWidth, 
							  dftProductROI, 
							  dataCrossCorrROI,
							  FFTW_ESTIMATE);

  fftw_execute(planBackwardProductROI);

  // find maximum
  dMatchOptim = -1.0;
  nTransXOptim = 0;
  nTransYOptim = 0;

  for( ny = 0; ny < nROIHeight; ny++ ){
    for( nx = 0; nx < nROIWidth; nx++ ){

      indexTemplate = ny * nROIWidth + nx ;

      dMatchRes = fabs(dataCrossCorrROI[indexTemplate]);

      if( dMatchRes > dMatchOptim ){
	dMatchOptim = dMatchRes;
	nTransXOptim = nx;
	nTransYOptim = ny;

#ifdef USE_VERBOSE
	cout << "FFTW: TransX " << nTransXOptim;
	cout << " TransY " << nTransYOptim;
	cout << " Matching Score " << dMatchRes;
	cout << endl;
#endif // USE_VERBOSE

#ifdef OUTPUT_LOG
	logFile << "FFTW: TransX " << nTransXOptim;
	logFile << " TransY " << nTransYOptim;
	logFile << " Matching Score " << dMatchRes;
	logFile << endl;
#endif // USE_VERBOSE

      }
    }
  }

  if( nTransXOptim > nROIWidth / 2)
    nTransXOptim = -( nROIWidth - nTransXOptim );

  if( nTransYOptim > nROIHeight / 2 )
    nTransYOptim = -( nROIHeight - nTransYOptim );

  fftw_free( dftTemplateROI ); 
  fftw_free( dftMatchROI ); 
  fftw_free( dftProductROI );

  delete[] dataCrossCorrROI;

  fftw_destroy_plan(planForwardTemplateROI);
  fftw_destroy_plan(planForwardMatchROI);
  fftw_destroy_plan(planBackwardProductROI);

  // output matching results for pixel-shift matching

#ifdef USE_VERBOSE
  cout << "Done pixel-shift matching!"<< endl;
  cout << "Best pixel-shift matching results " << endl;
  cout << " TransX " <<  nTransXOptim;
  cout << " TransY " <<  nTransYOptim;
  cout << " Matching Score " << dMatchOptim;
  cout << endl;
#endif

#ifdef OUTPUT_LOG
  logFile << "Done pixel-shift matching!"<< endl;
  logFile << "Best pixel-shift matching results " << endl;
  logFile << " TransX " << nTransXOptim
	  << " TransY " << nTransYOptim
          << " Matching Score " << dMatchOptim
          << endl;

#endif


#else // perform cross-correlation in space domain

  // perform brute-force cross-correlation matching for translations and rotations

#ifdef USE_BRUTE_FORCE_GPU

  float * d_DataTemplate;
  cudaArray*  d_DataMatch2;

  cudaMalloc( (void**) &d_DataTemplate, sizeof( float ) * volsize );

  // pad boundary values outside of ROI in dTemplate to be dTemplateAvg
  // this special design is for the computation of cross correlation in GPU
  // dTemplate[] contains the normalized template data with padded values

  float* fDataTemplate = new float[ volsize ];
  float* fDataMatch2 = new float[ volsize ];

  for( ny = 0; ny < nHeight; ny++ ){
    for( nx = 0; nx < nWidth; nx++ ){

      indexImg = ny * nWidth + nx ;
      if( nx >= nROILowerX && nx <= nROIUpperX 
	  && ny >= nROILowerY && ny <= nROIUpperY ){

	indexTemplate = ( ny - nROILowerY ) * (nROIUpperX - nROILowerX + 1) + nx - nROILowerX ;

	fDataTemplate[ indexImg ] = (float)(double)dataTemplateROI[ indexTemplate ];

      }
      else{
	fDataTemplate[ indexImg ] =  (float)(double)dTemplateAvg; 
      }

      fDataMatch2[ indexImg ] = (float)(double)dataMatch2[ indexImg ];

    }
  }

  cudaMemcpy( d_DataTemplate, fDataTemplate, sizeof( float ) * volsize, 
	      cudaMemcpyHostToDevice );

  cudaChannelFormatDesc float1Desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaMallocArray( (cudaArray**) &d_DataMatch2, &float1Desc, nWidth, nHeight);
  cudaMemcpyToArray( d_DataMatch2, 0, 0, fDataMatch2, sizeof( float ) * volsize, 
		     cudaMemcpyHostToDevice );

  cross_correlation_wrapper( d_DataTemplate, d_DataMatch2, 
			     score_match, 
			     dTransStartX, dTransStartY,
			     dTransInvX, dTransInvY,
			     numTransX, numTransY,
			     (float)(double)dTemplateAvg, (float)(double)dTemplateRMS,
			     nROILowerX, nROIUpperX, nROILowerY, nROIUpperY,
			     nWidth, nHeight);

  for( nTransIndY = 0; nTransIndY < numTransY; nTransIndY++ ){     // for each translation in Y
    for( nTransIndX = 0; nTransIndX < numTransX; nTransIndX++ ){   // for each translation

      if( score_match[ nTransIndY * numTransX + nTransIndX ] > dMatchOptim ){
	dMatchOptim = score_match[ nTransIndY * numTransX + nTransIndX ];
	nTransXOptim = nTransIndX;
	nTransYOptim = nTransIndY;
      }

      dTransX = dTransStartX + nTransIndX * dTransInvX;
      dTransY = dTransStartY + nTransIndY * dTransInvY;
      dMatchRes = score_match[ nTransIndY * numTransX + nTransIndX ];

#ifdef OUTPUT_LOG
      logFile << " TransX " << dTransX
	      << " TransY " << dTransY
	      << " Matching Score " << dMatchRes
	      << endl;
#endif

#ifdef USE_VERBOSE
      cout << " TransX " << dTransX;
      cout << " TransY " << dTransY;
      cout << " Matching Score " << dMatchRes;
      cout << endl;
#endif // USE_VERBOSE

    }
  }

#else // CPU brute force

  for( nTransIndY = 0; nTransIndY < numTransY; nTransIndY++ ){         // for each translation in Y
    for( nTransIndX = 0; nTransIndX < numTransX; nTransIndX++ ){         // for each translation in X

      dTransX = dTransStartX + nTransIndX * dTransInvX;
      dTransY = dTransStartY + nTransIndY * dTransInvY;

      data_interp(dataMatch2,             // input  
		  dataMatchROI,           // output
		  &dImgROIAvg,            // output
		  dTransX, dTransY,       // parameters
		  nROILowerX, nROIUpperX,    
		  nROILowerY, nROIUpperY, 
		  nWidth, nHeight);
	
      dMatchRes = 0.0;

      cross_correlation( dataMatchROI, dataTemplateROI, dTemplateAvg, dImgROIAvg, // input
			 &dMatchRes,                                              // output
			 nROILowerX, nROIUpperX,                                  // parameters  
			 nROILowerY, nROIUpperY,
			 dTemplateRMS);

      score_match[ nTransIndY * numTransX + nTransIndX ] = dMatchRes;

      if( dMatchRes > dMatchOptim ){
	dMatchOptim = dMatchRes;
	nTransXOptim = nTransIndX;
	nTransYOptim = nTransIndY;

      }

#ifdef OUTPUT_LOG
      logFile << " TransX " << dTransX
	      << " TransY " << dTransY
	      << " Matching Score " << dMatchRes
	      << endl;
#endif

#ifdef USE_VERBOSE
      cout << " TransX " << dTransX;
      cout << " TransY " << dTransY;
      cout << " Matching Score " << dMatchRes;
      cout << endl;
#endif // USE_VERBOSE

    }
  }

  // output matching results for pixel-shift matching

#ifdef USE_VERBOSE
  cout << "Done pixel-shift matching!"<< endl;
  cout << "Best pixel-shift matching results " << endl;
  cout << " TransX " <<  dTransStartX + nTransXOptim * dTransInvX;
  cout << " TransY " <<  dTransStartY + nTransYOptim * dTransInvY;
  cout << " Matching Score " << dMatchOptim;
  cout << endl;
#endif

#ifdef OUTPUT_LOG
  logFile << "Done pixel-shift matching!"<< endl;
  logFile << "Best pixel-shift matching results " << endl;
  logFile << " TransX " << dTransStartX + nTransXOptim * dTransInvX
          << " TransY " << dTransStartY + nTransYOptim * dTransInvY
          << " Matching Score " << dMatchOptim
          << endl;

#endif  // OUTPUT_LOG

#endif // USE_BRUTE_FORCE_GPU

#endif // USE_FFTW3


#ifdef OUTPUT_LOG
  logFile << endl << "Start subpixel-shift matching " << endl;
#endif

#ifdef USE_VERBOSE
  cout << endl << "Start subpixel-shift matching " << endl;
#endif

  // subpixel cross-correlation matching 
  // range: [dTransStartX + (nTransXOptim - 1) * dTransInvX, dTransStartX + (nTransXOptim + 1) * dTransInvX]
  // shift: dTransInvX / SUBPIXEL_SHIFT;

#ifdef USE_FFTW3
  dTransY = nTransYOptim; 
  dMatchOptim = -1.0; 
#else
  dTransY = dTransStartY + nTransYOptim * dTransInvY;
#endif

  double dTransInvX_subpixel = dTransInvX / SUBPIXEL_SHIFT; 
  numTransX = 2 * SUBPIXEL_SHIFT + 1;
  int nTransXOptim_subpixel = 0;
  int nTransYOptim_subpixel = 0;

#ifdef USE_BRUTE_FORCE_GPU // GPU BRUTE_FORCE

  double* score_match_subpixel = new double[ numTransX ];

  cross_correlation_wrapper( d_DataTemplate, d_DataMatch2, 
			     score_match_subpixel, 
			     dTransStartX+ (nTransXOptim - 1) * dTransInvX, dTransY,
			     dTransInvX_subpixel, dTransInvY,
			     numTransX, 1,
			     (float)(double)dTemplateAvg, (float)(double)dTemplateRMS,
			     nROILowerX, nROIUpperX, nROILowerY, nROIUpperY,
			     nWidth, nHeight);

  for( nTransIndX = 0; nTransIndX < numTransX; nTransIndX++ ){         // for each translation in X

    dTransX = dTransStartX + (nTransXOptim - 1) * dTransInvX + nTransIndX * dTransInvX_subpixel;
    dMatchRes = score_match_subpixel[ nTransIndX ];

    if( dMatchRes > dMatchOptim ){
      dMatchOptim = dMatchRes;
      nTransXOptim_subpixel = nTransIndX;
      nTransYOptim_subpixel = nTransIndY;

    }

#ifdef OUTPUT_LOG
    logFile << " TransX " << dTransX
	    << " TransY " << dTransY
	    << " Matching Score " << dMatchRes
	    << endl;
#endif

#ifdef USE_VERBOSE
    cout << " TransX " << dTransX;
    cout << " TransY " << dTransY;
    cout << " Matching Score " << dMatchRes;
    cout << endl;
#endif
    
  }
  delete [] score_match_subpixel;

#else // CPU FFTW/BRUTE_FORCE

  for( nTransIndX = 0; nTransIndX <  numTransX; nTransIndX++ ){         // for each translation in X

#ifdef USE_FFTW3  
    dTransX = nTransXOptim - dTransInvX + nTransIndX * dTransInvX_subpixel;
#else
    dTransX = dTransStartX + (nTransXOptim - 1) * dTransInvX + nTransIndX * dTransInvX_subpixel;
#endif

    data_interp(dataMatch2,             // input  
		dataMatchROI,           // output
		&dImgROIAvg,            // output
		dTransX, dTransY,       // parameters
		nROILowerX, nROIUpperX,    
		nROILowerY, nROIUpperY, 
		nWidth, nHeight);
	
    dMatchRes = 0.0;

    cross_correlation( dataMatchROI, dataTemplateROI, dTemplateAvg, dImgROIAvg, // input
		       &dMatchRes,                                              // output
		       nROILowerX, nROIUpperX,                                  // parameters  
		       nROILowerY, nROIUpperY,
		       dTemplateRMS);

    // score_match[ nTransIndY * numTransX + nTransIndX ] = dMatchRes;

    if( dMatchRes > dMatchOptim ){
      dMatchOptim = dMatchRes;
      nTransXOptim_subpixel = nTransIndX;
      nTransYOptim_subpixel = nTransIndY;

    }

#ifdef OUTPUT_LOG
    logFile << " TransX " << dTransX
	    << " TransY " << dTransY
	    << " Matching Score " << dMatchRes
	    << endl;
#endif

#ifdef USE_VERBOSE
    cout << " TransX " << dTransX;
    cout << " TransY " << dTransY;
    cout << " Matching Score " << dMatchRes;
    cout << endl;
#endif
    
  }

#endif // USE_BRUTE_FORCE_GPU

  double dTransXOptim_subpixel; 

#ifdef USE_FFTW3
  dTransXOptim_subpixel = nTransXOptim - dTransInvX + nTransXOptim_subpixel * dTransInvX_subpixel;
#else
  dTransXOptim_subpixel = dTransStartX + (nTransXOptim - 1) * dTransInvX + nTransXOptim_subpixel * dTransInvX_subpixel;
#endif

#ifdef OUTPUT_LOG
  logFile << "Best subpixel-shift matching results " << endl
	  << "  TransX " << dTransXOptim_subpixel
	  << "  TransY " << dTransY
	  << " Matching Score " << dMatchOptim
	  << endl;
#endif

#ifdef USE_VERBOSE
  cout << "Best subpixel-shift matching results " << endl 
       << "  TransX " << dTransXOptim_subpixel
       << "  TransY " << dTransY
       << " Matching Score " << dMatchOptim
       << endl;
#endif

#ifdef OUTPUT_NRRD
  // output nrrd file
  data_interp(dataMatch2,             // input  
	      dataMatchROI,           // output
	      &dImgROIAvg,            // output
	      dTransXOptim_subpixel, dTransY,       // parameters
	      nROILowerX, nROIUpperX,    
	      nROILowerY, nROIUpperY, 
	      nWidth, nHeight);

  for( int ny = nROILowerY; ny <= nROIUpperY; ny++ ){
    for( int nx = nROILowerX; nx <= nROIUpperX; nx++ ){  

      int indexNrrd = ny * nWidth + nx ;

      float fInterpPixel = dataMatchROI[ (ny - nROILowerY) * (nROIUpperX - nROILowerX + 1) + nx - nROILowerX];
      dataNrrd[ 2 * volsize + indexNrrd ] += 0.5f * 255 * ( fInterpPixel - fMatchMin) / (fMatchMax - fMatchMin);

    }
  }

  // 

  size_t v_size[3];
  v_size[0] = nWidth;
  v_size[1] = nHeight;
  v_size[2] = 3;  // fliped 180 proj, 0 proj, combine

  char strNrrdPathName[256];
  strcpy(strNrrdPathName, argv[1]);
  strcat(strNrrdPathName, basename);
  strcat(strNrrdPathName, "_comb_180_flip_0.nrrd");

  Nrrd *nval = nrrdNew();
  if(nrrdWrap_nva(nval, dataNrrd, nrrdTypeFloat, 3, v_size) ||  nrrdSave(strNrrdPathName, nval, NULL)){
    
#ifdef USE_VERBOSE
    cout << "Saving nrrd file failed!" << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "Saving nrrd file failed!" << endl;
#endif

    delete [] dataNrrd; 
    exit(1);
  }
  else{

#ifdef USE_VERBOSE 
    cout << "nrrd file saved successfully !" << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "nrrd file saved successfully !" << endl;
#endif

#ifdef OUTPUT_TXT
    char strTxtPathLog1[256];
    strcpy(strTxtPathLog1, argv[1]);
    strcat(strTxtPathLog1, "txt180.txt");

    std::ofstream txtFile1(strTxtPathLog1, std::ios::out ); 
    for( ny = nROILowerY; ny <= nROIUpperY; ny++ ){
      for( nx = nROILowerX; nx <= nROIUpperX; nx++ ){
	txtFile1 << dataTemplateROI[ (ny - nROILowerY) * (nROIUpperX - nROILowerX + 1) + nx - nROILowerX ]; 
	txtFile1 << "   ";
      }
      txtFile1 << endl;
    }
    txtFile1.close();

    char strTxtPathLog2[256];
    strcpy(strTxtPathLog2, argv[1]);
    strcat(strTxtPathLog2, "txt0.txt");

    std::ofstream txtFile2(strTxtPathLog2, std::ios::out ); 
    for( ny = nROILowerY; ny <= nROIUpperY; ny++ ){
      for( nx = nROILowerX; nx <= nROIUpperX; nx++ ){
	txtFile2 << dataMatchROI[ (ny - nROILowerY) * (nROIUpperX - nROILowerX + 1) + nx - nROILowerX]; 
	txtFile2 << "   ";
      }
      txtFile2 << endl;
    }
    txtFile2.close();
#endif

    delete [] dataNrrd; 
  }

#endif // OUTPUT_NRRD

  // 
  cout << dTransXOptim_subpixel/2.0 << endl;

#ifdef OUTPUT_LOG
  logFile.close();
#endif

  // free allocated resources
#ifdef USE_BRUTE_FORCE_GPU

  cudaFree( d_DataTemplate );
  cudaFreeArray( d_DataMatch2 );

  delete [] fDataTemplate;
  delete [] fDataMatch2;
#endif 

  delete [] dataTemplate;
  delete [] dataTemplateROI;
  delete [] dataMatch;
  delete [] dataMatch2;
  delete [] dataMatchROI;
  delete [] dataWhite1;
  delete [] dataWhite2;
  delete [] dataDark;
  delete [] score_match;

}

void data_interp(double * dataMatch2,              // input  
		 double * dataMatchROI,            // output
		 double * dImgROIAvg,              // output
		 double dTransX, double dTransY,   // parameters
		 int nROILowerX, int nROIUpperX,    
                 int nROILowerY, int nROIUpperY, 
                 int nWidth, int nHeight) {

  int nROIArea = ( nROIUpperX - nROILowerX + 1 ) * ( nROIUpperY - nROILowerY + 1 );

  *dImgROIAvg = 0.0; 
  for( int ny = nROILowerY; ny <= nROIUpperY; ny++ ){
    for( int nx = nROILowerX; nx <= nROIUpperX; nx++ ){  

      // find matching position (rotate first, translation second)
      double x = nx - dTransX;
      double y = ny - dTransY;

      int nxx = (int) floor( x );
      int nyy = (int) floor( y );

      double dInterpPixel = 0.0;
      double dInterpWeight = 0.0;
      int indexImg; 

      if( nxx >= 0 && nxx < nWidth && nyy >= 0 && nyy < nHeight ){   // (nxx, nyy)
	indexImg = nyy *  nWidth + nxx ;
	dInterpPixel += dataMatch2[ indexImg ] * (nxx + 1 - x) * (nyy + 1 - y); 
	dInterpWeight += (nxx + 1 - x) * (nyy + 1 - y);
      }
		
      if( nxx + 1 >= 0 && nxx + 1 < nWidth && nyy >= 0 && nyy < nHeight ){   // (nxx + 1, nyy)
	indexImg = nyy *  nWidth + nxx + 1 ;
	dInterpPixel += dataMatch2[ indexImg ] * (x - nxx) * (nyy + 1 - y); 
	dInterpWeight += (x - nxx) * (nyy + 1 - y);
      }

      if( nxx >= 0 && nxx < nWidth && nyy + 1 >= 0 && nyy + 1 < nHeight ){ // (nxx, nyy + 1)
	indexImg = (nyy + 1) *  nWidth + nxx ;
	dInterpPixel += dataMatch2[ indexImg ] * (nxx + 1 - x) * (y - nyy); 
	dInterpWeight += (nxx + 1 - x) * (y - nyy);
      }

      // (nxx + 1, nyy + 1)
      if( nxx + 1 >= 0 && nxx + 1 < nWidth && nyy + 1 >= 0 && nyy + 1 < nHeight ){
	indexImg = (nyy + 1) *  nWidth + (nxx + 1) ;
	dInterpPixel += dataMatch2[ indexImg ] * (x - nxx) * ( y - nyy); 
	dInterpWeight += (x - nxx) * (y - nyy);
      }
		
      if( dInterpWeight > 1e-5){
	dInterpPixel /= dInterpWeight;
      }
      else{
	
	dInterpPixel = 0.0; 
      }
	
      dataMatchROI[ (ny - nROILowerY) * (nROIUpperX - nROILowerX + 1) + nx - nROILowerX] = dInterpPixel; 
      *dImgROIAvg += dInterpPixel;
    }
  }

  *dImgROIAvg /= nROIArea; 

}

void cross_correlation( double * dataMatchROI,             // input  
			double * dataTemplateROI,          // input
			double dTemplateAvg,
			double dImgROIAvg, 
			double * dMatchRes,                // output
			int nROILowerX, int nROIUpperX,    // parameters  
			int nROILowerY, int nROIUpperY,
			double dTemplateRMS) {

  double dImg = 0.0;
  for( int ny = 0 ; ny < nROIUpperY - nROILowerY + 1; ny++ ){
    for( int nx = 0; nx < nROIUpperX - nROILowerX + 1; nx++ ){  

      int indexTemplate = ny * ( nROIUpperX - nROILowerX + 1 ) + nx;
      double dTemplatePixel = dataTemplateROI[ indexTemplate ];

      *dMatchRes += ( dataTemplateROI[ indexTemplate ] - dTemplateAvg ) *
	( dataMatchROI[ indexTemplate ] - dImgROIAvg );

      dImg += ( dataMatchROI[ indexTemplate ] - dImgROIAvg ) * 
	( dataMatchROI[ indexTemplate ] - dImgROIAvg );

    }
  }

  if( fabs( dTemplateRMS ) > 1e-5 && fabs( dImg ) > 1e-5 ){

    *dMatchRes /= dTemplateRMS * sqrt( dImg );

  }
  else{
    *dMatchRes = 0.0; 
  }

}
