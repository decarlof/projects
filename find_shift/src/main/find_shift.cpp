// Rotational axis detection using cross correlation
       
#include "tinyxml.h"

#include <iostream>   
#include <fstream>
#include <cstdlib>
#include <string>

#include "teem/nrrd.h"
#include "fftw3.h"
#include "nexusbox.h"

#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"

// #define NEW   // The codes inside NEW are compiled, but they are not fully tested. 6/13/2012

#ifdef NEW

#include "itkScalarImageToHistogramGenerator.h"
#include "itkHistogramToEntropyImageFilter.h"

typedef itk::Image<float, 2> Float2DImageType;
typedef itk::Statistics::ScalarImageToHistogramGenerator< Float2DImageType > ScalarImageToHistogramGeneratorType;
typedef ScalarImageToHistogramGeneratorType::HistogramType    HistogramType;

typedef itk::HistogramToEntropyImageFilter< HistogramType > HistogramToEntropyImageFilterType;    

#endif // NEW

typedef itk::Image<unsigned char, 2> UChar2DImageType;


#include "find_shift_config.h"  // MACROs and Configurations

#ifdef USE_BRUTE_FORCE_GPU

#include <cutil.h>
#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include <cutil_math.h>

#endif // USE_BRUTE_FORCE_GPU

#include "gridrec.h"               

#define OUTPUT_NRRD
#define OUTPUT_LOG

// #define OUTPUT_TXT  // output nrrd as txt for debug using matlab

// #define DIST_BOUNDARY 100

// #define TRANS_START_X  -250
// #define TRANS_END_X    250
// #define TRANS_INV_X     1

// #define TRANS_START_Y  -2
// #define TRANS_END_Y     2
// #define TRANS_INV_Y     1

// #define SUBPIXEL_SHIFT  10

using std::cout;
using std::endl;
using std::fstream;
using std::ifstream;
using std::string;
using std::ios;

#define HDF5_TIF_DATA_TYPE unsigned short int  // unit16 for hdf5 and int
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

extern "C"
void Hdf5SerialGetDim(const char* filename, const char* datasetname, 
		      int * xm, int* ym, int* zm );

extern "C"
void Hdf5SerialReadY(const char* filename, const char* datasetname, int nYSliceStart, int numYSlice,  
		     HDF5_TIF_DATA_TYPE * data); 

extern "C"
void Hdf5SerialReadZ(const char* filename, const char* datasetname, int nYSliceStart, int numYSlice,  
		     HDF5_TIF_DATA_TYPE * data); 

float entropy( float* data, float min, float max, int num_hist_bin, int xm, int ym); 

string num2str( int num );

enum{INPUT_IMAGE_HDF5, INPUT_IMAGE_TIF};

int main(int argc, char**  argv){ 

  if( argc != 2 ){   

    cout << "Usage: find_shift  params.xml" << endl;
    cout << "Info: find_shift version " << FIND_SHIFT_VERSION_MAJOR
         << "." << FIND_SHIFT_VERSION_MINOR << endl;
#ifdef USE_FFTW3
    cout << "Info: libfftw3 is used for computation" << endl;
#else
    cout << "Info: brute-force computation applied" << endl;
#endif // USE_FFTW3

#ifdef USE_BRUTE_FORCE_GPU
    cout << "Info: GPU is used for brute-force computation" << endl;
#endif // USE_BRUTE_FORCE_GPU

#ifdef USE_ENTROPY_METRIC  
    cout << "Info: the entropy metric is used for subpixel shift detection" << endl;
#endif 

    cout << "Info: hdf5 data exchange format and tif image sequences supported" << endl;

    exit( 1 );
  }

  unsigned int nx, ny, nz, nxx, nyy;
  int indexTemplate, indexImg; 
  double dTemplatePixel, dImgPixel, dInterpPixel, dInterpWeight;

  double dMatchRes, dMatchOptim; 
  int nTransIndX, nTransIndY, nTransXOptim, nTransYOptim;
  int nRotateInd, nRotateOptim;
  int nHdfIndOptim; 
  double dTransX, dTransY, x, y;


  // step 1: read the xml parameter file for detailed parameters
  int n_input_image_type; 
  string str_hdf5_directory_name, str_hdf5_file_name;

  string str_tif_proj_directory_name, str_tif_proj_180_file_name, str_tif_proj_0_file_name;
  int n_tif_offset_y; 

  string str_tif_white_field_directory_name, str_tif_white_field_base_name;
  int n_tif_white_field_index_start, n_tif_white_field_index_end;
  string str_tif_dark_field_directory_name, str_tif_dark_field_base_name;
  int n_tif_dark_field_index_start, n_tif_dark_field_index_end;

  int n_trans_start_x, n_trans_end_x, n_trans_inv_x;
  int n_trans_start_y, n_trans_end_y, n_trans_inv_y;
  int n_roi_start_x, n_roi_end_x, n_roi_start_y, n_roi_end_y;

  int n_subpixel_shift, n_histogram_bin, n_subpixel_slice; 
  float f_histogram_lower, f_histogram_upper, f_inv_rot; 

  TiXmlDocument doc( argv[1] );
  bool loadOkay = doc.LoadFile();

  if ( !loadOkay ){
    cout << "Could not load test file " << argv[1] << " Error= " << doc.ErrorDesc() << " Exiting.  " << endl;
    exit( 1 );
  }

  TiXmlNode* node = 0;
  TiXmlElement* paramsElement = 0;

  node = doc.FirstChild( "Params" );
  paramsElement = node->ToElement();

  // parameters for projection image
  node = paramsElement->FirstChild("INPUT_IMAGE_TYPE");
  if( node == NULL ){
    cout << "INPUT_IMAGE_TYPE does not exist in XML file. Use default HDF5! " << endl;
    n_input_image_type = INPUT_IMAGE_HDF5;
  }
  else{
    node = node->FirstChild();
    n_input_image_type = atoi( node->Value() );

    if( n_input_image_type != INPUT_IMAGE_HDF5 &&  n_input_image_type != INPUT_IMAGE_TIF ){
      cout << "The INPUT_IMAGE_TYPE should be either " << INPUT_IMAGE_HDF5 << " or " << INPUT_IMAGE_TIF << endl;
      exit(1);
    }
  }

  // hdf5
  if( n_input_image_type == INPUT_IMAGE_HDF5 ){

    node = paramsElement->FirstChild("HDF5_DIRECTORY_NAME");
    if( node == NULL ){
      cout << "HDF5_DIRECTORY_NAME does not exist in XML file. Abort! " << endl;
      exit(1);
    }
    else{
      node = node->FirstChild();
      str_hdf5_directory_name = string( node->Value() );
    }

    // 
    node = paramsElement->FirstChild("HDF5_FILE_NAME");
    if( node == NULL ){
      cout << "HDF5_FILE_NAME does not exist in XML file. Abort! " << endl;
      exit(1);
    }
    else{
      node = node->FirstChild();
      str_hdf5_file_name = string( node->Value() );
    }

  }

  // tif
  if( n_input_image_type == INPUT_IMAGE_TIF ){

    // projection
    node = paramsElement->FirstChild("TIF_PROJ_DIRECTORY_NAME");
    if( node == NULL ){
      cout << "TIF_PROJ_DIRECTORY_NAME does not exist in XML file. Abort! " << endl;
      exit(1);
    }
    else{
      node = node->FirstChild();
      str_tif_proj_directory_name = string( node->Value() );
    }

    // 
    node = paramsElement->FirstChild("TIF_PROJ_180_FILE_NAME");
    if( node == NULL ){
      cout << "TIF_PROJ_180_FILE_NAME does not exist in XML file. Abort! " << endl;
      exit(1);
    }
    else{
      node = node->FirstChild();
      str_tif_proj_180_file_name = string( node->Value() );
    }

    // 
    node = paramsElement->FirstChild("TIF_PROJ_0_FILE_NAME");
    if( node == NULL ){
      cout << "TIF_PROJ_0_FILE_NAME does not exist in XML file. Abort! " << endl;
      exit(1);
    }
    else{
      node = node->FirstChild();
      str_tif_proj_0_file_name = string( node->Value() );
    }

    //
    node = paramsElement->FirstChild("TIF_OFFSET_Y");
    if( node == NULL ){
      cout << "TIF_OFFSET_Y does not exist in XML file. Abort! " << endl;
      exit(1);
    }
    else{
      node = node->FirstChild();
      n_tif_offset_y = atoi( node->Value() );
    }


    //  white filed
    node = paramsElement->FirstChild("TIF_WHITE_FIELD_DIRECTORY_NAME");
    if( node == NULL ){
      cout << "TIF_WHITE_FIELD_DIRECTORY_NAME does not exist in XML file. Abort! " << endl;
      exit(1);
    }
    else{
      node = node->FirstChild();
      str_tif_white_field_directory_name = string( node->Value() );
    }

    // 
    node = paramsElement->FirstChild("TIF_WHITE_FIELD_BASE_NAME");
    if( node == NULL ){
      cout << "TIF_WHITE_FIELD_BASE_NAME does not exist in XML file. Abort! " << endl;
      exit(1);
    }
    else{
      node = node->FirstChild();
      str_tif_white_field_base_name = string( node->Value() );
    }

    node = paramsElement->FirstChild("TIF_WHITE_FIELD_INDEX_START");
    if( node == NULL ){
      cout << "TIF_WHITE_FIELD_INDEX_START does not exist in XML file. Abort! " << endl;
      exit(1);
    }
    else{
      node = node->FirstChild();
      n_tif_white_field_index_start = atoi( node->Value() );
    }

    node = paramsElement->FirstChild("TIF_WHITE_FIELD_INDEX_END");
    if( node == NULL ){
      cout << "TIF_WHITE_FIELD_INDEX_END does not exist in XML file. Abort! " << endl;
      exit(1);
    }
    else{
      node = node->FirstChild();
      n_tif_white_field_index_end = atoi( node->Value() );
    }

    // dark field
    node = paramsElement->FirstChild("TIF_DARK_FIELD_DIRECTORY_NAME");
    if( node == NULL ){
      cout << "TIF_DARK_FIELD_DIRECTORY_NAME does not exist in XML file. Abort! " << endl;
      exit(1);
    }
    else{
      node = node->FirstChild();
      str_tif_dark_field_directory_name = string( node->Value() );
    }

    // 
    node = paramsElement->FirstChild("TIF_DARK_FIELD_BASE_NAME");
    if( node == NULL ){
      cout << "TIF_DARK_FIELD_BASE_NAME does not exist in XML file. Abort! " << endl;
      exit(1);
    }
    else{
      node = node->FirstChild();
      str_tif_dark_field_base_name = string( node->Value() );
    }

    node = paramsElement->FirstChild("TIF_DARK_FIELD_INDEX_START");
    if( node == NULL ){
      cout << "TIF_DARK_FIELD_INDEX_START does not exist in XML file. Abort! " << endl;
      exit(1);
    }
    else{
      node = node->FirstChild();
      n_tif_dark_field_index_start = atoi( node->Value() );
    }

    node = paramsElement->FirstChild("TIF_DARK_FIELD_INDEX_END");
    if( node == NULL ){
      cout << "TIF_DARK_FIELD_INDEX_END does not exist in XML file. Abort! " << endl;
      exit(1);
    }
    else{
      node = node->FirstChild();
      n_tif_dark_field_index_end = atoi( node->Value() );
    }

  }

  // search range in x
  node = paramsElement->FirstChild("TRANS_START_X");
  if( node == NULL ){
    cout << "TRANS_START_X does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_trans_start_x = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("TRANS_END_X");
  if( node == NULL ){
    cout << "TRANS_END_X does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_trans_end_x = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("TRANS_INV_X");
  if( node == NULL ){
    cout << "TRANS_INV_X does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_trans_inv_x = atoi( node->Value() );
  }

  // search range in y
  node = paramsElement->FirstChild("TRANS_START_Y");
  if( node == NULL ){
    cout << "TRANS_START_Y does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_trans_start_y = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("TRANS_END_Y");
  if( node == NULL ){
    cout << "TRANS_END_Y does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_trans_end_y = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("TRANS_INV_Y");
  if( node == NULL ){
    cout << "TRANS_INV_Y does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_trans_inv_y = atoi( node->Value() );
  }


  // ROI
  node = paramsElement->FirstChild("ROI_START_X");
  if( node == NULL ){
    cout << "ROI_START_X does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_roi_start_x = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("ROI_END_X");
  if( node == NULL ){
    cout << "ROI_END_X does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_roi_end_x = atoi( node->Value() );
  }

  // search range in y
  node = paramsElement->FirstChild("ROI_START_Y");
  if( node == NULL ){
    cout << "ROI_START_Y does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_roi_start_y = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("ROI_END_Y");
  if( node == NULL ){
    cout << "ROI_END_Y does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_roi_end_y = atoi( node->Value() );
  }

  // subpixel shift
  node = paramsElement->FirstChild("SUBPIXEL_SHIFT");
  if( node == NULL ){
    cout << "SUBPIXEL_SHIFT does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_subpixel_shift = atoi( node->Value() );
  }

  // 
  node = paramsElement->FirstChild("SUBPIXEL_SLICE");
  if( node == NULL ){
    cout << "SUBPIXEL_SLICE does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_subpixel_slice = atoi( node->Value() );
  }

  // 
  node = paramsElement->FirstChild("SUBPIXEL_HISTOGRAM_LOWER");
  if( node == NULL ){
    cout << "SUBPIXEL_HISTOGRAM_LOWER does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    f_histogram_lower = atof( node->Value() );
  }

  // 
  node = paramsElement->FirstChild("SUBPIXEL_HISTOGRAM_UPPER");
  if( node == NULL ){
    cout << "SUBPIXEL_HISTOGRAM_UPPER does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    f_histogram_upper = atof( node->Value() );
  }

  // 
  node = paramsElement->FirstChild("SUBPIXEL_HISTOGRAM_BIN");
  if( node == NULL ){
    cout << "SUBPIXEL_HISTOGRAM_BIN does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_histogram_bin = atoi( node->Value() );
  }

  //
  node = paramsElement->FirstChild("INV_ROT");
  if( node == NULL ){
    cout << "INV_ROT does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    f_inv_rot = atof( node->Value() );
  }

  // check the data type of the data exchange file: hdf5
  string str_posix;

  if( n_input_image_type == INPUT_IMAGE_HDF5 ){
    str_posix = str_hdf5_file_name.substr( str_hdf5_file_name.length() - 4, 4 );
  }

  if( n_input_image_type == INPUT_IMAGE_TIF ){
    str_posix = str_tif_proj_180_file_name.substr( str_tif_proj_180_file_name.length() - 3, 3 );
  }

  if( str_posix.compare( string( "hdf5" ) ) != 0 && str_posix.compare( string( "tif" ) ) != 0 ){
    cout << "This version support hdf5 and tif only!" << endl;
    exit(1);
  }

  // log file
#ifdef OUTPUT_LOG
  char strDataPathLog[256];
  if( n_input_image_type == INPUT_IMAGE_HDF5 ){
    strcpy(strDataPathLog, str_hdf5_directory_name.c_str());
  }
  if( n_input_image_type == INPUT_IMAGE_TIF ){
    strcpy(strDataPathLog, str_tif_proj_directory_name.c_str());
  }
  strcat(strDataPathLog, "LogCrossCorrelation.txt");

  std::ofstream logFile(strDataPathLog, std::ios::out ); 
#endif


  // step 2: retrieve information from the hdf5 file
  int um, num_proj, vm; 
  string str_exchange_file_name;

  if( n_input_image_type == INPUT_IMAGE_HDF5 ){
    str_exchange_file_name = str_hdf5_directory_name;
    str_exchange_file_name.append( str_hdf5_file_name );
    Hdf5SerialGetDim( str_exchange_file_name.c_str(), "/entry/exchange/data", 
		      &um, &num_proj, &vm ); 
  }

  itk::ImageFileReader<UChar2DImageType>::Pointer tif_image_reader = NULL;
  UChar2DImageType::Pointer tif_image = NULL;
  UChar2DImageType::IndexType index;
  UChar2DImageType::SizeType sizeImg;

  if( n_input_image_type == INPUT_IMAGE_TIF ){
    string str_tif_file_name = str_tif_proj_directory_name;
    str_tif_file_name.append( str_tif_proj_180_file_name );

    tif_image_reader = itk::ImageFileReader<UChar2DImageType>::New();
    tif_image_reader->SetFileName( str_tif_file_name.c_str() );
    tif_image_reader->Update();

    tif_image = tif_image_reader->GetOutput();

    sizeImg = tif_image->GetLargestPossibleRegion().GetSize();
    um = sizeImg[ 0 ];
    vm = sizeImg[ 1 ] - n_tif_offset_y;
  }

  int nHeight, nWidth, volsize;

  nHeight = vm;
  nWidth = um;
  volsize = nWidth * nHeight; 

  // if( nHeight & (nHeight - 1 ) != 0  || nWidth & (nWidth - 1 ) != 0 ){

  //   cout << "This version need the projection dimensions to be power of 2" << endl; // power
  //   exit(1);
  // }

  HDF5_TIF_DATA_TYPE* dataTemplate = new HDF5_TIF_DATA_TYPE[ volsize ];  
  HDF5_TIF_DATA_TYPE* dataMatch = new HDF5_TIF_DATA_TYPE[ volsize ];
  HDF5_TIF_DATA_TYPE* dataWhite = new HDF5_TIF_DATA_TYPE[ 2 * volsize ];  // 2 white field images for hdf5; 1 for tif image sequence
  HDF5_TIF_DATA_TYPE* dataDark = new HDF5_TIF_DATA_TYPE[ volsize ];
  if( !dataTemplate || !dataMatch || !dataWhite || !dataDark ){

#ifdef USE_VERBOSE
    cout << "Error allocating memory for hdf5 data!" << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "Error allocating memory for hdf5 data!" << endl;
#endif

    exit(1);
  }

  if( n_input_image_type == INPUT_IMAGE_HDF5 ){
    Hdf5SerialReadZ( str_exchange_file_name.c_str(), "/entry/exchange/dark_data", 0, 1, dataDark );  
    Hdf5SerialReadZ( str_exchange_file_name.c_str(), "/entry/exchange/white_data", 0, 2, dataWhite );  

    Hdf5SerialReadY( str_exchange_file_name.c_str(), "/entry/exchange/data", 0, 1, dataTemplate );  
    Hdf5SerialReadY( str_exchange_file_name.c_str(), "/entry/exchange/data", num_proj-1, 1, dataMatch );  
  }

  if( n_input_image_type == INPUT_IMAGE_TIF ){

    // read the 180-degree projection tif image
    string str_tif_file_name = str_tif_proj_directory_name;
    str_tif_file_name.append( str_tif_proj_180_file_name );

    tif_image_reader = itk::ImageFileReader<UChar2DImageType>::New();
    tif_image_reader->SetFileName( str_tif_file_name.c_str() );
    tif_image_reader->Update();

    tif_image = tif_image_reader->GetOutput();

    for( int ny = 0; ny < vm; ny++ ){
      for( int nx = 0; nx < um; nx++ ){
	index[ 0 ] = nx;
	index[ 1 ] = ny + n_tif_offset_y;
	dataTemplate[ ny * um + nx ] = tif_image->GetPixel( index );
      }
    }

    // read the 0-degree projection tif image
    string str_tif_file_name_0 = str_tif_proj_directory_name;
    str_tif_file_name_0.append( str_tif_proj_0_file_name );

    tif_image_reader = itk::ImageFileReader<UChar2DImageType>::New();
    tif_image_reader->SetFileName( str_tif_file_name_0.c_str() );
    tif_image_reader->Update();

    tif_image = tif_image_reader->GetOutput();

    sizeImg = tif_image->GetLargestPossibleRegion().GetSize();
    if( sizeImg[ 0 ] != um || sizeImg[ 1 ] != vm + n_tif_offset_y ){
      cout << "The image size of " << str_tif_file_name_0 << " is " << sizeImg << ", not equal to ( " <<  um << " , " << vm + n_tif_offset_y << " ) " << endl;
      exit(1);
    }

    for( int ny = 0; ny < vm; ny++ ){
      for( int nx = 0; nx < um; nx++ ){
	index[ 0 ] = nx;
	index[ 1 ] = ny + n_tif_offset_y;
	dataMatch[ ny * um + nx ] = tif_image->GetPixel( index );
      }
    }
  
    string strPosix = string(".tif"); 

    // read the white field image sequence
    for( int z = n_tif_white_field_index_start; z <= n_tif_white_field_index_end; z++ ){

      string strIndex = num2str( z );
      string strCurrTiffName = str_tif_white_field_directory_name;
      strCurrTiffName.append( str_tif_white_field_base_name );

      int pos = strCurrTiffName.length() - strPosix.length() - strIndex.length();
      strCurrTiffName.replace(pos, strIndex.length(), strIndex); 

      // 
      tif_image_reader->SetFileName( strCurrTiffName.c_str() );  
      tif_image_reader->Update();

      tif_image = tif_image_reader->GetOutput();

      sizeImg = tif_image->GetLargestPossibleRegion().GetSize();
      if( sizeImg[ 0 ] != um || sizeImg[ 1 ] != vm + n_tif_offset_y ){
        cout << "The image size of " << str_tif_file_name_0 << " is " << sizeImg << ", not equal to ( " <<  um << " , " << vm + n_tif_offset_y << " ) " << endl;
        exit(1);
      }

      // 
      for( int ny = 0; ny < vm; ny++ ){
	for( int nx = 0; nx < um; nx++ ){  
	
	  index[0] = nx;
	  index[1] = ny + n_tif_offset_y;
	
	  dataWhite[ ny * um + nx ] += tif_image->GetPixel( index );
	}
      }
    
        // cout << strCurrTiffName << " read " << endl;
    }

    for( int ny = 0; ny < vm; ny++ ){
      for( int nx = 0; nx < um; nx++ ){  
	
	dataWhite[ ny * um + nx ] /=  (n_tif_white_field_index_end - n_tif_white_field_index_start + 1 );

      }
    }
  
    // read the dark field image sequence
    for( int z = n_tif_dark_field_index_start; z <= n_tif_dark_field_index_end; z++ ){

      string strIndex = num2str( z );
      string strCurrTiffName = str_tif_dark_field_directory_name;
      strCurrTiffName.append( str_tif_dark_field_base_name );

      int pos = strCurrTiffName.length() - strPosix.length() - strIndex.length();
      strCurrTiffName.replace(pos, strIndex.length(), strIndex); 

      // 
      tif_image_reader->SetFileName( strCurrTiffName.c_str() );  
      tif_image_reader->Update();

      tif_image = tif_image_reader->GetOutput();

      sizeImg = tif_image->GetLargestPossibleRegion().GetSize();
      if( sizeImg[ 0 ] != um || sizeImg[ 1 ] != vm + n_tif_offset_y ){
        cout << "The image size of " << str_tif_file_name_0 << " is " << sizeImg << ", not equal to ( " <<  um << " , " << vm + n_tif_offset_y << " ) " << endl;
        exit(1);
      }

      // 
      for( int ny = 0; ny < vm; ny++ ){
	for( int nx = 0; nx < um; nx++ ){  
	
	  index[0] = nx;
	  index[1] = ny + n_tif_offset_y;
	
	  dataDark[ ny * um + nx ] += tif_image->GetPixel( index);
	}
      }
    
        // cout << strCurrTiffName << " read " << endl;
    }

    for( int ny = 0; ny < vm; ny++ ){
      for( int nx = 0; nx < um; nx++ ){  
	
	dataDark[ ny * um + nx ] /=  (n_tif_white_field_index_end - n_tif_white_field_index_start + 1 );

      }
    }

  }

  // 
  double dTransStartX = n_trans_start_x;
  double dTransEndX = n_trans_end_x;

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

  double dTransStartY = n_trans_start_y;
  double dTransEndY = n_trans_end_y;

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

  double dTransInvX = n_trans_inv_x;
  double dTransInvY = n_trans_inv_y;
  
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

  int nROILowerX = n_roi_start_x;
  int nROILowerY = n_roi_start_y;
  int nROIUpperX = n_roi_end_x;

  if( nROIUpperX >= nWidth ){

#ifdef USE_VERBOSE
    cout << "DIST_BOUNDARY is too large! Use nWidth -1 instead!" << endl;
#endif 

#ifdef OUTPUT_LOG
    logFile << "DIST_BOUNDARY is too large! Use nWidth -1 instead!" << endl;
#endif 

    nROIUpperX = nWidth - 1;
  }

  int nROIUpperY = n_roi_end_y;
  if( nROIUpperY >= nHeight ){

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
  if( !dataMatch2 || !dataTemplateROI || !dataMatchROI){

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

      dWhite1 = dataWhite[ indexImg ]; 

      if( n_input_image_type == INPUT_IMAGE_HDF5 ){
	dWhite2 = dataWhite[ nWidth * nHeight + indexImg ]; 
      }

      if( n_input_image_type == INPUT_IMAGE_TIF ){
	dWhite2 = dataWhite[ indexImg ]; 
      }

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
      dWhite1 = dataWhite[ indexImg ]; 
      if( n_input_image_type == INPUT_IMAGE_HDF5 ){
	dWhite2 = dataWhite[ nWidth * nHeight + indexImg ]; 
      }

      if( n_input_image_type == INPUT_IMAGE_TIF ){
	dWhite2 = dataWhite[ indexImg ]; 
      }

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
  // shift: dTransInvX / n_subpixel_shift;

#ifdef USE_FFTW3
  dTransY = nTransYOptim; 
  dMatchOptim = -1.0; 
#else
  dTransY = dTransStartY + nTransYOptim * dTransInvY;
#endif

  double dTransInvX_subpixel = dTransInvX / n_subpixel_shift; 
  numTransX = 2 * n_subpixel_shift + 1;
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
  if( n_input_image_type == INPUT_IMAGE_HDF5 ){
    strcpy(strNrrdPathName, str_hdf5_directory_name.c_str());
  }
  if( n_input_image_type == INPUT_IMAGE_TIF ){
    strcpy(strNrrdPathName, str_tif_proj_directory_name.c_str());
  }
  strcat(strNrrdPathName, "find_shift");
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

  dTransXOptim_subpixel = dTransStartX + (nTransXOptim - 1) * dTransInvX + nTransXOptim_subpixel * dTransInvX_subpixel;

#ifndef USE_ENTROPY_METRIC  
  cout << dTransXOptim_subpixel/2.0 << endl;
#else
  // refine the rotational axis shift using entropy metric
  float* vect_angle = (float*) new float[ num_proj ];
  for( int i = 0; i < num_proj; i++ )
    vect_angle[ i ] = i * f_inv_rot;

  int num_ray_padding = 2 * um; 
  int offset_pad = um / 2;   

  GridRec* recon_algorithm = new GridRec ();
  recon_algorithm->setSinogramDimensions ( num_ray_padding, num_proj);
  recon_algorithm->setThetaList (vect_angle, num_proj);
  recon_algorithm->setGPUDeviceID( 0 ); 
  recon_algorithm->setFilter ( 3 );  // 3 for HANN. See recon_algorithm.h for definition
  recon_algorithm->init();

  // 
  HDF5_TIF_DATA_TYPE* dataSinogram = new HDF5_TIF_DATA_TYPE[ 2 * um * num_proj ];  
  float* dataSinogramNorm = new float[ 2 * um * num_proj ];  

  // read sinogram data
  Hdf5SerialReadZ( str_exchange_file_name.c_str(), "/entry/exchange/data", n_subpixel_slice, 2, dataSinogram );  

  // normalization
  for( ny = 0; ny < num_proj; ny++ ){
    for( nx = 0; nx < um; nx++ ){

      indexImg = ny * um + nx;

      // 
      dTemplate = dataSinogram[ indexImg ]; 
      dWhite1 = dataWhite[ n_subpixel_slice * um + nx ]; 
      dWhite2 = dataWhite[ um * vm + n_subpixel_slice * um + nx ]; 
      dDark = dataDark[ n_subpixel_slice * um + nx ]; 

      dValue = dTemplate - dDark;
      if( dValue <= 0 )
	dataSinogramNorm[ indexImg ] = 0.0;
      else{

	dValue = (0.5 * (dWhite1 + dWhite2) - dDark) / dValue;

	if( dValue > 0.0 )
	  dataSinogramNorm[ indexImg ] = log( dValue);
	else
	  dataSinogramNorm[ indexImg ] = 0.0;
      }

      //
      dTemplate = dataSinogram[ um * num_proj + indexImg ]; 
      dWhite1 = dataWhite[ (n_subpixel_slice + 1) * um + nx ]; 
      dWhite2 = dataWhite[ um * vm + (n_subpixel_slice + 1) * um + nx ]; 
      dDark = dataDark[ (n_subpixel_slice + 1) * um + nx ]; 

      dValue = dTemplate - dDark;
      if( dValue <= 0 )
	dataSinogramNorm[ um * num_proj + indexImg ] = 0.0;
      else{

	dValue = (0.5 * (dWhite1 + dWhite2) - dDark) / dValue;

	if( dValue > 0.0 )
	  dataSinogramNorm[ um * num_proj + indexImg ] = log( dValue);
	else
	  dataSinogramNorm[ um * num_proj + indexImg ] = 0.0;
      }
    }
  }

  // recon and entropy comparison
  float* dataSinogramNorm_padding = new float[ 2 * num_ray_padding * num_proj ];  
  float* dataRecon_padding = new float[ 2 * num_ray_padding * num_ray_padding ];
  float* dataRecon = new float[ 2 * um * um ];
  float* dataRecon_optim = new float[ 2 * um * um ];

  float entropy_optim = 1e20f, entropy_cur;
  
  for( nTransIndX = 0; nTransIndX <  numTransX; nTransIndX++ ){         // for each translation in X

    dTransX = (dTransXOptim_subpixel - 1.0 ) / 2.0 + nTransIndX * dTransInvX_subpixel;

    // sinogram shift and padding

    for( int i = 0; i < 2 * num_ray_padding * num_proj; i++ )
      dataSinogramNorm_padding[ i ] = 0.0f;
  
    for( int j = 0; j < num_proj; j++ ){
      for( int k = 0; k < um; k++ ){
  
	float kk = k - dTransX; 
	int nkk = (int)floor(kk);

	float fInterpPixel = 0.0f;
	float fInterpWeight = 0.0f;

	float fInterpPixel2= 0.0f;
	float fInterpWeight2 = 0.0f;

	// pad sinogram using boundary values instead of zero
	if( nkk >= 0 && nkk < um ){
	  fInterpPixel += dataSinogramNorm[ j * um + nkk ] * (nkk + 1 - kk);
	  fInterpPixel2 += dataSinogramNorm[ num_proj * um + j * um + nkk ] * (nkk + 1 - kk);
	}
	else if( nkk < 0 ){
	  fInterpPixel += dataSinogramNorm[ j * um ] * (nkk + 1 - kk);
	  fInterpPixel2 += dataSinogramNorm[ num_proj * um + j * um ] * (nkk + 1 - kk);
	}
	else if( nkk >= um ){
	  fInterpPixel += dataSinogramNorm[ j * um + um - 1] * (nkk + 1 - kk);
	  fInterpPixel2 += dataSinogramNorm[ num_proj * um + j * um + um - 1 ] * (nkk + 1 - kk);
	}
	 
	// 
	if( nkk + 1 >= 0 && nkk + 1 < um ){
	  fInterpPixel += dataSinogramNorm[ j * um + nkk + 1] * (kk - nkk);
	  fInterpPixel2 += dataSinogramNorm[ num_proj * um + j * um + nkk + 1] * (kk - nkk);
	}
	else if( nkk + 1 < 0 ){
	  fInterpPixel += dataSinogramNorm[ j * um ] * (kk - nkk);
	  fInterpPixel2 += dataSinogramNorm[ num_proj * um + j * um ] * (kk - nkk);
	}
	else if( nkk + 1 >= um ){
	  fInterpPixel += dataSinogramNorm[ j * um  + um - 1] * (kk - nkk);
	  fInterpPixel2 += dataSinogramNorm[ num_proj * um + j * um + um - 1] * (kk - nkk);
	}

	dataSinogramNorm_padding[ j * num_ray_padding + k + offset_pad ] = fInterpPixel;
	dataSinogramNorm_padding[ (num_proj + j) * num_ray_padding + k + offset_pad ] = fInterpPixel2;
      }
    }

    // pad sinogram using boundary values instead of zero
    for( int j = 0; j < num_proj; j++ ){

      for( int k = 0; k < offset_pad; k++ ){

	dataSinogramNorm_padding[ j * num_ray_padding + k ] = dataSinogramNorm_padding[ j * num_ray_padding + offset_pad ] ;
	dataSinogramNorm_padding[ (num_proj + j) * num_ray_padding + k ] = dataSinogramNorm_padding[ (num_proj + j) * num_ray_padding + offset_pad ];
    
      }

      for( int k = 0; k < offset_pad; k++ ){
    
	dataSinogramNorm_padding[ j * num_ray_padding + offset_pad + um + k  ] = dataSinogramNorm_padding[ j * num_ray_padding + offset_pad + um - 1 ] ;
	dataSinogramNorm_padding[ (num_proj + j) * num_ray_padding + offset_pad + um + k ] = dataSinogramNorm_padding[ (num_proj + j) * num_ray_padding + offset_pad + um - 1 ];
    
      }

    }

    // // 
    // size_t v_size[3];
    // v_size[0] = um;
    // v_size[1] = num_proj;
    // v_size[2] = 2;  // fliped 180 proj, 0 proj, combine

    // char strNrrdPathName[256];
    // strcpy(strNrrdPathName, str_hdf5_directory_name.c_str());
    // strcat(strNrrdPathName, "find_shift_sino.nrrd");

    // Nrrd *nval = nrrdNew();
    // if(nrrdWrap_nva(nval, dataSinogramNorm, nrrdTypeFloat, 3, v_size) ||  nrrdSave(strNrrdPathName, nval, NULL)){
    //   exit(1);
    // }


    // // 
    // size_t v_size2[3];
    // v_size2[0] = num_ray_padding;
    // v_size2[1] = num_proj;
    // v_size2[2] = 2;  // fliped 180 proj, 0 proj, combine

    // char strNrrdPathName2[256];
    // strcpy(strNrrdPathName2, str_hdf5_directory_name.c_str());
    // strcat(strNrrdPathName2, "find_shift_sino_padding.nrrd");

    // Nrrd *nval2 = nrrdNew();
    // if(nrrdWrap_nva(nval2, dataSinogramNorm_padding, nrrdTypeFloat, 3, v_size2) ||  nrrdSave(strNrrdPathName2, nval2, NULL)){
    //   exit(1);
    // }


    // 
    for (int loop=0; loop < recon_algorithm->numberOfSinogramsNeeded(); loop++){   // num_sinograms_needed = 2;
      recon_algorithm->setSinoAndReconBuffers(loop+1, 
					      &dataSinogramNorm_padding[ loop* num_proj* num_ray_padding ],
					      &dataRecon_padding[ loop* num_ray_padding* num_ray_padding] );
    }

    recon_algorithm->reconstruct();

    // retrieve reconstruction results
    for( int j = 0; j < um; j++ ){        // num_height = num_width for GridRec
      for( int k = 0; k < um; k++ ){
  
	dataRecon[ j * um + k] = dataRecon_padding[ (j + offset_pad) * num_ray_padding + k + offset_pad ];

	dataRecon[ ( um + j ) * um + k] = dataRecon_padding[ (num_ray_padding + j + offset_pad) * num_ray_padding + k + offset_pad ];

      }
    }

#ifdef NEW
    Float2DImageType::Pointer imgRecon = Float2DImageType::New();
    Float2DImageType::SizeType size;
    size[0] = um;
    size[1] = um;
    Float2DImageType::RegionType region;
    region.SetSize( size );
    imgRecon->SetRegions( region );
    imgRecon->Allocate();

    Float2DImageType::IndexType index;
    for( int j = 0; j < um; j++ ){    
      for( int k = 0; k < um; k++ ){
  
	index[0] = k;
	index[1] = j;
	imgRecon->SetPixel( index, dataRecon_padding[ (j + offset_pad) * num_ray_padding + k + offset_pad ] );

      }
    }

    ScalarImageToHistogramGeneratorType::Pointer scalarImageToHistogramGenerator = ScalarImageToHistogramGeneratorType::New();
    scalarImageToHistogramGenerator->SetNumberOfBins( 10000 );
    scalarImageToHistogramGenerator->SetInput( imgRecon );

    HistogramToEntropyImageFilterType::Pointer histogramToEntropyImageFilterPointer = HistogramToEntropyImageFilterType::New();
    histogramToEntropyImageFilterPointer->SetInput( scalarImageToHistogramGenerator->GetOutput() );
    cout << histogramToEntropyImageFilterPointer->GetOutput() << endl;

#endif // NEW
  
    // entropy
    entropy_cur = entropy( dataRecon, f_histogram_lower, f_histogram_upper, n_histogram_bin, 2 * um, um);

#ifdef OUTPUT_LOG
  logFile << "subpixel-shift matching results using entropy" << endl
	  << "  TransX " << dTransX
	  << "  TransY " << dTransY
	  << " Matching Score " << entropy_cur
	  << endl;
#endif

    if( entropy_cur < entropy_optim ){

      entropy_optim = entropy_cur;
      nTransXOptim_subpixel = nTransIndX;
      for( int i = 0; i < 2 * um * um; i++ ){
	dataRecon_optim[ i ] = dataRecon[ i ];
      }

    }

  }

  // output the optim reconstruction

  size_t v_size2[3];
  v_size2[0] = um;
  v_size2[1] = um;
  v_size2[2] = 2;  // fliped 180 proj, 0 proj, combine

  char strNrrdPathName2[256];
  if( n_input_image_type == INPUT_IMAGE_HDF5 ){
    strcpy(strNrrdPathName2, str_hdf5_directory_name.c_str());
  }
  if( n_input_image_type == INPUT_IMAGE_TIF ){
    strcpy(strNrrdPathName2, str_tif_proj_directory_name.c_str());
  }
  strcat(strNrrdPathName2, "find_shift_recon_entropy.nrrd");

  Nrrd *nval2 = nrrdNew();
  if(nrrdWrap_nva(nval2, dataRecon_optim, nrrdTypeFloat, 3, v_size2) ||  nrrdSave(strNrrdPathName2, nval2, NULL)){
    
#ifdef USE_VERBOSE
    cout << "Saving nrrd file failed!" << endl;
#endif

#ifdef OUTPUT_LOG
    logFile << "Saving nrrd file failed!" << endl;
#endif

    exit(1);
  }

  delete [] dataSinogram;
  delete [] dataSinogramNorm;
  delete [] vect_angle; 
  delete [] dataSinogramNorm_padding;
  delete [] dataRecon_padding; 
  delete [] dataRecon;

  // 
  cout << (dTransXOptim_subpixel - 1.0) / 2.0 + nTransXOptim_subpixel * dTransInvX_subpixel << endl;

#endif // USE_ENTROPY_METRIC  

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
  delete [] dataWhite;
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

float entropy( float* data, float min, float max, int num_hist_bin, int xm, int ym){

  float entropy_res = 0.0f;

  float inv = (max - min) / num_hist_bin; 

  int* hist = new int[ num_hist_bin + 1 ];
  for( int i = 0; i < num_hist_bin + 1; i++ )
    hist[ i ] = 0;

  int ind; 
  for( int ny = 0; ny < ym; ny++ ){   
    for( int nx = 0; nx < xm; nx++ ) {

      ind = (int)(float)( ( data[ ny * xm + nx ] - min) / inv ); 

      if( ind >= 0 && ind <= num_hist_bin )
	hist[ ind ]++; 

    }
  }

  float pi;

  for (int i = 0; i < num_hist_bin + 1; i++ ){

    if( hist[ i ] > 0 ){

      pi = 1.0f * hist[ i ] / (xm * ym);
      
      entropy_res += -pi * log(pi) / log(2.0);
    }
  }

  delete [] hist;

  return entropy_res;
}


string num2str( int num ){

  char c_iter = '0';
  string str_ind[10];
  for( int i = 0; i < 10; i++){
    str_ind[i] = string(1, c_iter);
    c_iter = c_iter + 1;
  }

  int iter = num; 
  int res;

  string strNum = "";
  while( iter / 10 != 0 ){
    res = iter % 10;
    iter = iter / 10; 
    strNum.insert( 0, str_ind[res] );
  }
  strNum.insert( 0, str_ind[iter] );

  return strNum;
}
