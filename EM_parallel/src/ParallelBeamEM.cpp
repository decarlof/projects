// This is a simulation program for parallel beam CT reconstruction using EM
// Yongsheng Pan
// 1/28/2011
// All rights reserved


// Definition of the imaging geometry

// Source location: (S * cos(theta), S * sin(theta), 0 )
//                   theta \in [0, PI]

// The detector size (um, vm).

// Definition of the (synthetic) imaging object. 
// Object size: (xm, ym, zm), intensity den_obj = 2.0

// These parameters are stored in the parallel_params.txt data file

#include "tinyxml.h"        // Put it in the front to avoid compiling errors 

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <string>

#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"

// Note: these header files should locate AFTER ITK header files
//       and before CUDA header files

#include "nr3.h"
#include "ran.h"
#include "gamma.h"
#include "deviates.h"

// Note: CUDA header files should locate AFTER ITK header files

#include <cutil.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include <cutil_math.h>

#include <multithreading.h>

using std::cout;
using std::endl;
using std::string;

#include "ParallelBeamEM.h"

typedef itk::Image<unsigned char, 2> UChar2DImageType;
typedef itk::Image<float, 2> Float2DImageType;
typedef itk::Image<float, 3> Float3DImageType;

extern "C"
void proj_cal_wrapper( cudaArray*, float* , int, float*, 
		       float* , 
		       int , int , int , 
		       int, int , int , float , float , float , float , float,
		       float , float , float , float , float);

extern "C" 
void proj_cal_wrapper_sharemem( float* , float* , float* , int , int , int , 
				int , int , int , float , float , float , 
				float , float , unsigned int );

extern "C"
void proj_gen_wrapper( cudaArray*, float*, int, int, int,
		       int, int, int, float, float, float, float, float);
extern "C"
void backproj_cal_wrapper( cudaArray* , float* , float* ,
			   float* , 
			   int , int , int ,
			   int , int , int , float , float , float , float , float, 
			   float , float , float , float , float);

extern "C"
void backproj_cal_wrapper_sharemem( float* , float* ,  float*, float* , int, int, int,  
				    int, int, int, float , float, float, float, float,
				    unsigned int);

extern "C" 
void tv3d_cal_wrapper( float *, float* , float* , int , int , int, int , float, float , float );

static CUT_THREADPROC EM3DThread(ThreadGPUPlan* );

extern "C" 
void Hdf5SerialReadZ(const char* , char* , int , int , HDF5_DATATYPE * );

extern "C" 
void Hdf5SerialReadY(const char* , char* , int , int , HDF5_DATATYPE * );

extern "C"
void Hdf5SerialWrite( const char* filename, const char* datasetname, 
		       int xm, int ym, int zm, float* data ); 

extern "C"
void Hdf5SerialWriteZ(const char* ,const  char* , int , int , int , int , float * );

extern "C"
float reduction_wrapper( float*, float*, int, int, int );

extern "C"
float reduction2_wrapper( float*, float*, float*, int, int, int );

extern "C"
int nextPow2( int );

int GetSharedMemSize(int);

void RingCorrectionSinogram (float *data, float ring_coeff, int um, int num_proj,
			     float *mean_vect, float* mean_sino_line_data,
			     float *low_pass_sino_lines_data); 

string num2str( int num );

//////// TV parameters //////////

float param_fidelity_weight_TV = 0.5; // 0.1; 
float param_epsilon_TV = 1e-5; 
int   param_iter_max_TV = 10; // 1000;
float param_thres_TV = 1.0;

int   param_iter_TV = 1;// 10;
float param_timestep_TV = 0.1;

int main(int argc, char** argv){

  // projection generation by simulation using ray tracing
  // for each angle during source/detector rotation

  if( argc != 2 ){
    cout << "Usage: EM  params.xml" << endl;
#ifdef STEVE_DATA
    cout << "Note: STEVE_DATA defined for bindary projection data!" << endl;
#endif

    exit(1); 
  }

  // check GPU and get the needed parameters
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
    cout << "cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched." << endl;
    exit(0);
  }

  if( deviceCount == 0 ){
    cout << "There is no device supporting CUDA." << endl;
    exit(0);
  }
  
  if( deviceCount > MAX_NUM_GPU ){
    deviceCount = MAX_NUM_GPU;
    cout << "CUDA-capable device count: " << deviceCount << endl;
  }

  cout << deviceCount << " GPUs available for reconstruction." << endl;

  // OS thread ID
  CUTThread* threadID = new CUTThread[ deviceCount ];
  if( !threadID ){
    cout << "Error allocating memory for threadID" << endl;
    exit(1); 
  }

  // Solver configuration
  ThreadGPUPlan *plan = new ThreadGPUPlan[ deviceCount ];
  if( !plan ){
    cout << "Error allocating memory for plan" << endl;
    exit(1); 
  }

  // read the parameter file

  int n_proj_image_type; 
  string str_proj_image_directory;
  string str_proj_image_base_name;
  string str_proj_mask_image_name;
  int n_proj_seq_index_start, n_proj_seq_index_end, n_proj_mask_image = 0;

  int n_recon_image_type; 
  string str_recon_image_directory;
  string str_recon_image_base_name;
  float f_recon_threshold_lower_value, f_recon_threshold_upper_value;
  string str_recon_hdf5_dataset_name;

  unsigned int SOD, um, vm, xm, ym, zm;
  unsigned int num_proj;
  float inv_rot, start_rot, end_rot; 
  float xoffset, yoffset, zoffset; 
  float proj_pixel_size, voxel_size, spacing;
  float lambda;
  int stop_type;
  unsigned int iter_max;
  float proj_thres;
  float diff_thres_percent;
  float diff_thres_value;
  int nSliceStart, nSliceEnd;
  int nReconRadiusCrop;

  // real the xml parameter file for detailed parameters
  TiXmlDocument doc( argv[1] );
  bool loadOkay = doc.LoadFile();

  if ( !loadOkay ){
    cout << "Could not load test file " << argv[1] << " Error= " << doc.ErrorDesc() << " Exiting.  " << endl;
    exit( 1 );
  }

  TiXmlNode* node = 0;
  TiXmlElement* paramsElement = 0;
  TiXmlElement* itemElement = 0;

  node = doc.FirstChild( "Params" );
  paramsElement = node->ToElement();

  // parameters for projection image
  node = paramsElement->FirstChild("PROJ_IMAGE_TYPE");
  if( node == NULL ){
    cout << "Note: PROJ_IMAGE_TYPE does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_proj_image_type = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("PROJ_IMAGE_DIRECTORY_NAME");
  if( node == NULL ){
    cout << "Note: PROJ_IMAGE_TYPE does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    str_proj_image_directory = string( node->Value() );
  }

  node = paramsElement->FirstChild("PROJ_IMAGE_BASE_NAME");
  if( node == NULL ){
    cout << "Note: PROJ_IMAGE_BASE_NAME does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    str_proj_image_base_name = string( node->Value() );
  }

  node = paramsElement->FirstChild("PROJ_SEQ_INDEX_START");
  if( node == NULL ){
    cout << "Note: PROJ_SEQ_INDEX_START does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_proj_seq_index_start = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("PROJ_SEQ_INDEX_END");
  if( node == NULL ){
    cout << "Note: PROJ_SEQ_INDEX_END does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_proj_seq_index_end = atoi( node->Value() );
  }


  node = paramsElement->FirstChild("PROJ_MASK_IMAGE_NAME");
  if( node == NULL ){
    n_proj_mask_image = 0;
  }
  else{
    node = node->FirstChild();
    str_proj_mask_image_name = string( node->Value() );
    n_proj_mask_image = 1;
  }

  // 
  node = paramsElement->FirstChild("RECON_IMAGE_TYPE");
  if( node == NULL ){
    cout << "Note: RECON_IMAGE_TYPE does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_recon_image_type = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("RECON_IMAGE_DIRECTORY_NAME");
  if( node == NULL ){
    cout << "Note: RECON_IMAGE_DIRECTORY_NAME does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    str_recon_image_directory = string( node->Value() );
  }

  node = paramsElement->FirstChild("RECON_IMAGE_BASE_NAME");
  if( node == NULL ){
    cout << "Note: RECON_IMAGE_BASE_NAME does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    str_recon_image_base_name = string( node->Value() );
  }

  node = paramsElement->FirstChild("RECON_THRESHOLD_LOWER_VALUE");
  if( node == NULL ){
    cout << "Note: RECON_THRESHOLD_LOWER_VALUE does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    f_recon_threshold_lower_value = atof( node->Value() );
  }

  node = paramsElement->FirstChild("RECON_THRESHOLD_UPPER_VALUE");
  if( node == NULL ){
    cout << "Note: RECON_THRESHOLD_UPPER_VALUE does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    f_recon_threshold_upper_value = atof( node->Value() );
  }

  node = paramsElement->FirstChild("RECON_HDF5_DATASET_NAME");
  if( node == NULL ){
    cout << "Note: RECON_HDF5_DATASET_NAME does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    str_recon_hdf5_dataset_name = string( node->Value() );
  }

  // parameters for imaging geometry
  node = paramsElement->FirstChild("SOD");
  if( node == NULL ){
    cout << "Note: SOD does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    SOD = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("NUM_PROJ");
  if( node == NULL ){
    cout << "Note: NUM_PROJ does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    num_proj = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("DETECTOR_SIZE_X");
  if( node == NULL ){
    cout << "Note: DETECTOR_SIZE_X does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    um = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("DETECTOR_SIZE_Y");
  if( node == NULL ){
    cout << "Note: DETECTOR_SIZE_Y does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    vm = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("OBJECT_SIZE_X");
  if( node == NULL ){
    cout << "Note: OBJECT_SIZE_X does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    xm = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("OBJECT_SIZE_Y");
  if( node == NULL ){
    cout << "Note: OBJECT_SIZE_Y does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    ym = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("OBJECT_SIZE_Z");
  if( node == NULL ){
    cout << "Note: OBJECT_SIZE_Z does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    zm = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("INV_ROT");
  if( node == NULL ){
    cout << "Note: INV_ROT does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    inv_rot = atof( node->Value() );
  }

  node = paramsElement->FirstChild("ANGLE_START");
  if( node == NULL ){
    cout << "Note: ANGLE_START does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    start_rot = atof( node->Value() );
  }

  node = paramsElement->FirstChild("ANGLE_END");
  if( node == NULL ){
    cout << "Note: ANGLE_END does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    end_rot = atof( node->Value() );
  }

  // parameters for iterative recon algorithms
  node = paramsElement->FirstChild("STOP_TYPE");
  if( node == NULL ){
    cout << "Note: STOP_TYPE does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    stop_type = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("ITER_MAX");
  if( node == NULL ){
    cout << "Note: ITER_MAX does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    iter_max = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("PROJ_THRES");
  if( node == NULL ){
    cout << "Note: PROJ_THRES does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    proj_thres = atof( node->Value() );
  }

  node = paramsElement->FirstChild("DIFF_THRES_PERCENT");
  if( node == NULL ){
    cout << "Note: DIFF_THRES_PERCENT does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    diff_thres_percent = atof( node->Value() );
  }

  node = paramsElement->FirstChild("DIFF_THRES_VALUE");
  if( node == NULL ){
    cout << "Note: DIFF_THRES_VALUE does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    diff_thres_value = atof( node->Value() );
  }

  node = paramsElement->FirstChild("PROJ_PIXEL_SIZE");
  if( node == NULL ){
    cout << "Note: PROJ_PIXEL_SIZE does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    proj_pixel_size = atof( node->Value() );
  }

  node = paramsElement->FirstChild("VOXEL_SIZE");
  if( node == NULL ){
    cout << "Note: VOXEL_SIZE does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    voxel_size = atof( node->Value() );
  }

  node = paramsElement->FirstChild("SPACING");
  if( node == NULL ){
    cout << "Note: SPACING does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    spacing = atof( node->Value() );
  }

  node = paramsElement->FirstChild("STEP_SIZE");
  if( node == NULL ){
    cout << "Note: STEP_SIZE does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    lambda = atof( node->Value() );
  }

  // parameters for reconstruction controls
  node = paramsElement->FirstChild("ROTATION_OFFSET_X");
  if( node == NULL ){
    cout << "Note: ROTATION_OFFSET_X does not exist in XML file! Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    xoffset = atof( node->Value() );
  }

  node = paramsElement->FirstChild("ROTATION_OFFSET_Y");
  if( node == NULL ){
    cout << "Note: ROTATION_OFFSET_Y does not exist in XML file! Use 0!" << endl;
    yoffset = 0;
  }
  else{
    node = node->FirstChild();
    yoffset = atof( node->Value() );
  }

  node = paramsElement->FirstChild("ROTATION_OFFSET_Z");
  if( node == NULL ){
    cout << "Note: ROTATION_OFFSET_Z does not exist in XML file! Use 0!" << endl;
    zoffset = 0;
  }
  else{
    node = node->FirstChild();
    zoffset = atof( node->Value() );
  }

  node = paramsElement->FirstChild("SLICE_START");
  if( node == NULL ){
    cout << "Note: SLICE_START does not exist in XML file! Use 0!" << endl;
    nSliceStart = 0;
  }
  else{
    node = node->FirstChild();
    nSliceStart = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("SLICE_END");
  if( node == NULL ){
    cout << "Note: SLICE_END does not exist in XML file! Use maximum!" << endl;
    nSliceEnd = vm;
  }
  else{
    node = node->FirstChild();
    nSliceEnd = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("RECON_RADIUS_CROP");
  if( node == NULL ){
    cout << "Note: RECON_RADIUS_CROP does not exist in XML file! Use 0!" << endl;
    nReconRadiusCrop = 0;
  }
  else{
    node = node->FirstChild();
    nReconRadiusCrop = atoi( node->Value() );
  }

  if( nSliceEnd > zm ){
    nSliceEnd = zm;
  }

  if( ( n_proj_image_type == PROJ_IMAGE_TYPE_BIN || n_proj_image_type == PROJ_IMAGE_TYPE_BMP 
	|| n_proj_image_type == PROJ_IMAGE_TYPE_TIFF ) 
      && num_proj != n_proj_seq_index_end - n_proj_seq_index_start + 1 ){
    cout << "Error: num_proj != n_proj_seq_index_end - n_proj_seq_index_start + 1!" << endl;
    exit(1); 
  }

  // check the compatibility of parameters for parallel beam
  if( xm != um ){
    cout << "Object width in projection data does not match in parameter file!" << endl;
    exit(1);
  }

  if( zm != vm ){
    cout << "Object height in projection data does not match in parameter file!" << endl;
    exit(1);
  }

  // prepare for the mask image
  float* proj_mask = NULL;
  if( n_proj_mask_image == 1 ){
    proj_mask = new float[ um * vm ]; 
    if( !proj_mask ){
      cout << "Error allocating memory for proj_mask" << endl;
      exit(1);
    }

    string strMask = str_proj_image_directory;
    strMask.append( str_proj_mask_image_name );

    itk::ImageFileReader<UChar2DImageType>::Pointer reader;
    reader = itk::ImageFileReader<UChar2DImageType>::New();
    reader->SetFileName( strMask.c_str() );  
    reader->Update();

    UChar2DImageType::Pointer mask = (reader->GetOutput());
    UChar2DImageType::SizeType size = mask->GetLargestPossibleRegion().GetSize();
    if( size[0] != um || size[1] != vm ){
      cout << "The size of the specified mask image " << size << " does not match the specified detector size" << endl;
      exit(1);
    }

    UChar2DImageType::IndexType index;
    for( unsigned int ny = 0; ny < size[1]; ny++ ){
      for( unsigned int nx = 0; nx < size[0]; nx++ ){
	index[0] = nx;
	index[1] = ny;
	proj_mask[ ny * size[0] + nx ] = mask->GetPixel( index );
      }
    }

  }

  // read projection data

  bool bProjFormatNrrd = ( n_proj_image_type == PROJ_IMAGE_TYPE_NRRD );
  bProjFormatNrrd = bProjFormatNrrd || ( n_proj_image_type == PROJ_IMAGE_TYPE_BIN );
  bProjFormatNrrd = bProjFormatNrrd || ( n_proj_image_type == PROJ_IMAGE_TYPE_BMP );
  bProjFormatNrrd = bProjFormatNrrd || ( n_proj_image_type == PROJ_IMAGE_TYPE_TIFF );

  bool bProjFormatHdf5 = ( n_proj_image_type == PROJ_IMAGE_TYPE_HDF5 );

  if( !bProjFormatNrrd && !bProjFormatHdf5 ){
    cout << "Only NRRD, BIN, BMP, TIFF, HDF5 formats are supported for projection data!" << endl;
    exit(1);
  }

  // prepare for ring artifact removal
  float* mean_vect = new float[ num_proj ];
  float* low_pass_sino_lines_data = new float[ um ]; 
  float* mean_sino_line_data = new float[ um ]; 
  float* data_sino = new float[ um * num_proj ]; 

  Float3DImageType::Pointer ProjPointer = NULL; 
  Float3DImageType::IndexType index;

  if( bProjFormatNrrd ){

    if( n_proj_image_type == PROJ_IMAGE_TYPE_NRRD ){

      string strNrrd = str_proj_image_directory;
      strNrrd.append( str_proj_image_base_name );

      itk::ImageFileReader<Float3DImageType>::Pointer RealProjReader;
      RealProjReader = itk::ImageFileReader<Float3DImageType>::New();
      RealProjReader->SetFileName( strNrrd.c_str() );  
      RealProjReader->Update();

      ProjPointer = (RealProjReader->GetOutput());
      Float3DImageType::SizeType size = ProjPointer->GetLargestPossibleRegion().GetSize();

      // check the compatibility of parameters
      if( size[0] != xm || size[0] != um ){
	cout << "Object width in projection data does not match parameter file!" << endl;
	exit(1);
      }

      if( size[1] != zm || size[1] != vm ){
	cout << "Object height in projection data does not match parameter file!" << endl;
	exit(1);
      }

      if( size[2] != num_proj ){
	cout << "Rotations in projection data does not match parameter file!" << endl;
	exit(1);
      }

      um =  size[0];
      vm =  size[1];
      num_proj =  size[2];

      cout << "Projection Data (nrrd) Read from file !" << endl;
    }
    else if( n_proj_image_type == PROJ_IMAGE_TYPE_BIN ){

      Float3DImageType::SizeType size;
      Float3DImageType::RegionType region;

      size[0] = um;
      size[1] = vm;
      size[2] = num_proj;

      region.SetSize( size );

      ProjPointer = Float3DImageType::New(); 
      ProjPointer->SetRegions( region );
      ProjPointer->Allocate(); 

      string strPosix = string(".bin");

      Float3DImageType::PixelType pixelImg;
      PROJ_IMAGE_BIN_TYPE pixel_bin; 

      for( int z = n_proj_seq_index_start; z <= n_proj_seq_index_end; z++ ){

	string strIndex = num2str( z );
	string strCurrBinName = str_proj_image_directory;
	strCurrBinName.append( str_proj_image_base_name );

	int pos = strCurrBinName.length() - strPosix.length() - strIndex.length();
	strCurrBinName.replace(pos, strIndex.length(), strIndex); 

	fstream datafile( strCurrBinName.c_str(), ios::in | ios::binary );

	if( !datafile.is_open() ){
	  cout << "    Skip reading file " << strCurrBinName << endl;
	  continue; 
	}
	else{
	  cout << "    Reading file " << strCurrBinName << endl;  // test

	  index[2] = z - n_proj_seq_index_start;

	  for( int y = 0; y < vm; y++ ){
	    for( int x = 0; x < um; x++ ){  

	      index[0] = x;
	      index[1] = vm - 1 - y; // Note

	      datafile.read( reinterpret_cast< char*> ( &pixel_bin ), sizeof( float ) );
  	      pixelImg = (Float3DImageType::PixelType) pixel_bin; 

#ifdef STEVE_DATA
	      if( pixelImg > 100.0f ){
		pixelImg = 0.0f;
	      }
	      else if( pixelImg < 0.0f ){
		cout << "Pixel value should not be negative at " << index << endl;
		exit(1);
	      }
	      else{
		pixelImg = log( 100 / pixelImg );
	      }
#endif

    	      ProjPointer->SetPixel( index, pixelImg );

	    }
	  }
	}
      }

      cout << "Projection Data (bin) Read from file !" << endl;

    }
    else if( n_proj_image_type == PROJ_IMAGE_TYPE_BMP ||  n_proj_image_type == PROJ_IMAGE_TYPE_TIFF ){

      Float3DImageType::SizeType size;
      Float3DImageType::RegionType region;

      size[0] = um;
      size[1] = vm;
      size[2] = num_proj;

      region.SetSize( size );

      ProjPointer = Float3DImageType::New(); 
      ProjPointer->SetRegions( region );
      ProjPointer->Allocate(); 

      string strPosix = string(".bmp"); // string(".tif");

      itk::ImageFileReader<UChar2DImageType>::Pointer ProjSliceReader;
      ProjSliceReader = itk::ImageFileReader<UChar2DImageType>::New();

      UChar2DImageType::IndexType indexImg;

      for( int z = n_proj_seq_index_start; z <= n_proj_seq_index_end; z++ ){

	string strIndex = num2str( z );
	string strCurrBmpTiffName = str_proj_image_directory;
	strCurrBmpTiffName.append( str_proj_image_base_name );

	int pos = strCurrBmpTiffName.length() - strPosix.length() - strIndex.length();
	strCurrBmpTiffName.replace(pos, strIndex.length(), strIndex); 

	// 
	ProjSliceReader->SetFileName( strCurrBmpTiffName.c_str() );  
	ProjSliceReader->Update();

	UChar2DImageType::Pointer ProjSlicePointer = (ProjSliceReader->GetOutput());
	UChar2DImageType::SizeType size = ProjSlicePointer->GetLargestPossibleRegion().GetSize();

	// check the compatibility of parameters
	if( size[0] != xm || size[0] != um ){
	  cout << "Object width in projection data does not match parameter file!" << endl;
	  exit(1);
	}

	if( size[1] != zm || size[1] != vm ){
	  cout << "Object height in projection data does not match parameter file!" << endl;
	  exit(1);
	}

	// 
	index[2] = z - n_proj_seq_index_start;

	for( int y = 0; y < vm; y++ ){
	  for( int x = 0; x < um; x++ ){  

	    index[0] = x;
	    index[1] = y;

	    indexImg[0] = x;
	    indexImg[1] = y;

	    ProjPointer->SetPixel( index, (Float3DImageType::PixelType) ProjSlicePointer->GetPixel( indexImg) );
	  }
	}

	cout << strCurrBmpTiffName << " read " << endl;
      }

      if( n_proj_image_type == PROJ_IMAGE_TYPE_BMP ) 
	cout << "Projection Data (bmp) Read from file !" << endl;
      if( n_proj_image_type == PROJ_IMAGE_TYPE_TIFF )
	cout << "Projection Data (tif) Read from file !" << endl;
    }
  
    // ring artifact removal
    Float3DImageType::IndexType index;
    for( int ny = nSliceStart; ny <= nSliceEnd; ny++ ){

      for( int nz = 0; nz < num_proj; nz++ ){
	for( int nx = 0; nx < um; nx++ ){
	  index[ 0 ] = nx;
	  index[ 1 ] = ny;
	  index[ 2 ] = nz;
	  data_sino[ nz * um + nx ] = ProjPointer->GetPixel( index );
	}
      }

      RingCorrectionSinogram ( data_sino, RING_COEFF, um, num_proj,
			      mean_vect, mean_sino_line_data, low_pass_sino_lines_data); 

      for( int nz = 0; nz < num_proj; nz++ ){
	for( int nx = 0; nx < um; nx++ ){
	  index[ 0 ] = nx;
	  index[ 1 ] = ny;
	  index[ 2 ] = nz;
	  ProjPointer->SetPixel( index, data_sino[ nz * um + nx ] );
	}
      }
      cout << "    Ring removal for projection data slice " << ny << " performed!" << endl;
    }
    cout << endl;
    cout << "Ring removal for projection data (nrrd) performed!" << endl;

  }

  HDF5_DATATYPE * proj_darkfield = NULL;
  HDF5_DATATYPE * proj_whitefield = NULL;
  HDF5_DATATYPE * proj_slices = NULL;
  float * proj_slices_sino = new float[ um * (NUM_SLICE_INV * deviceCount) * num_proj ];
  float * obj_slices = new float[ xm * ym * (NUM_SLICE_INV * deviceCount) ];
  if( !proj_slices_sino || !obj_slices){
    cout << "Error allocating memory for proj_slices_sino!" << endl;
    exit(1);
  }

  string str_proj_hdf5_name;

  if( bProjFormatHdf5 ){

    str_proj_hdf5_name = str_proj_image_directory;
    str_proj_hdf5_name.append( str_proj_image_base_name );

    proj_darkfield = new HDF5_DATATYPE[ um * vm ]; 
    proj_whitefield = new HDF5_DATATYPE[ 2 * um * vm ]; 
    proj_slices = new HDF5_DATATYPE[ um * (NUM_SLICE_INV * deviceCount) * num_proj ];


    if( !proj_darkfield || !proj_whitefield || !proj_slices ){
      cout << "Error allocating memory for proj_darkfield,  proj_whitefield and proj_slices!" << endl;
      exit(1);
    }

    Hdf5SerialReadZ( str_proj_hdf5_name.c_str(), "/entry/exchange/dark_data", 0, 1, proj_darkfield );  
    Hdf5SerialReadZ( str_proj_hdf5_name.c_str(), "/entry/exchange/white_data", 0, 2, proj_whitefield );  

  }

  // memory allocation for output
  Float3DImageType::Pointer ImageReconPointer = Float3DImageType::New();
  
  Float3DImageType::RegionType regionObj;
  Float3DImageType::SizeType sizeObj;
  sizeObj[ 0 ] = xm;
  sizeObj[ 1 ] = ym;
  // sizeObj[ 2 ] = nSliceEnd - nSliceStart + 1;
  sizeObj[ 2 ] = deviceCount * NUM_SLICE_INV;

  regionObj.SetSize( sizeObj );  

  ImageReconPointer->SetRegions( regionObj );
  ImageReconPointer->Allocate( );

  // CT reconstruction using EM
  unsigned int ndeviceCompute; 
  unsigned int z = nSliceStart;
  
  while (z <= nSliceEnd){

    // prepare data for threaded GPU
    ndeviceCompute = 0; 
    if( nSliceEnd + 1 - z >= deviceCount * NUM_SLICE_INV){
      cout << "Processing slices " << z << " ~ " << z + NUM_SLICE_INV * deviceCount - 1 << endl;

      ndeviceCompute = deviceCount; 

      if( bProjFormatHdf5 ){  // supports old HDF5 format such as raw_stu.hdf5

	Hdf5SerialReadZ( str_proj_hdf5_name.c_str(), "/entry/exchange/data", z, NUM_SLICE_INV * deviceCount, proj_slices );  

      }
    
      for( int i = 0; i < deviceCount; i++ ){
	plan[ i ].deviceID = i;
	plan[ i ].deviceShareMemSizePerBlock = GetSharedMemSize( i );  

	// geometry
	plan[ i ].num_width = xm;
	plan[ i ].num_height = ym;
	plan[ i ].num_depth = NUM_SLICE_INV;

	plan[ i ].num_ray = um;
	plan[ i ].num_elevation = NUM_SLICE_INV;
	plan[ i ].num_proj = num_proj;

	// algorithm params
	plan[ i ].stop_type = stop_type;
	plan[ i ].iter_max = iter_max;
	plan[ i ].proj_thres = proj_thres;
	plan[ i ].diff_thres_percent = diff_thres_percent;
	plan[ i ].diff_thres_value = diff_thres_value;
	plan[ i ].lambda = lambda;
	plan[ i ].spacing = spacing;
	plan[ i ].voxel_size = voxel_size;
	plan[ i ].proj_pixel_size = proj_pixel_size;
	plan[ i ].SOD = SOD;
	plan[ i ].inv_rot = inv_rot;
	plan[ i ].xoffset = xoffset;
	plan[ i ].yoffset = yoffset;
	plan[ i ].zoffset = zoffset;
	plan[ i ].start_rot = start_rot;
	plan[ i ].end_rot = end_rot;

	// memory for input/output data
	plan[ i ].h_input_data = new float[ plan[i].num_proj * plan[i].num_elevation * plan[i].num_ray ];
	plan[ i ].h_output_data = new float[ plan[i].num_width * plan[i].num_height * plan[i].num_depth ];

	if( !plan[ i ].h_input_data ||  !plan[ i ].h_output_data ){
	  cout << "Error allocating memory for plan [ " << i << " ]" << endl;
	  exit( 1 );
	}

	if( n_proj_mask_image == 1 ){
	  plan[ i ].h_proj_mask_data = new float[ plan[i].num_elevation * plan[i].num_ray ];
	  if( !plan[ i ].h_proj_mask_data ){
	    cout << "Error allocating memory for plan [ " << i << " ]" << endl;
	    exit(1);
	  }

	  for( unsigned int ny = 0; ny <  plan[i].num_elevation; ny++ ){
	    for( unsigned int nx = 0; nx <  plan[i].num_ray; nx++ ){
	      int ny_elevation = z + i * NUM_SLICE_INV + ny; // z: object index; corresponds to elevation in projection index

	      plan[ i ].h_proj_mask_data[ ny * plan[i].num_ray + nx ] = proj_mask[ ny_elevation * plan[i].num_ray + nx ];
	    }
	  }


	  plan[ i ].n_proj_mask_image = 1; 
	}
	else{
	  plan[ i ].h_proj_mask_data = NULL;
	  plan[ i ].n_proj_mask_image = 0; 
	}

	// input(projection) data from file
	if( bProjFormatNrrd){
	  for( int nz = 0; nz < plan[ i ].num_proj; nz++ ){
	    for( int ny = 0; ny < plan[ i ].num_elevation; ny++ ){
	      for( int nx = 0; nx < plan[ i ].num_ray; nx++ ){

		index[ 1 ] = z + i * NUM_SLICE_INV + ny; // z: object index; corresponds to elevation in projection index
		index[ 2 ] = nz;

		// horizontal shift compensation
		float xx = nx - xoffset; 
		int nxx = (int)floor(xx);

		if( nxx >= 0 && nxx < um ){
		  index[ 0 ] = nxx;
		}
		else if( nxx < 0 ){
		  index[ 0 ] = 0; 
		}
		else if( nxx >= um ){
		  index[ 0 ] = um - 1; 
		}
		plan[ i ].h_input_data[ (nz * plan[i].num_elevation + ny) * plan[i].num_ray + nx ] =  ProjPointer->GetPixel( index ) * (nxx + 1 - xx);

		if( nxx + 1 >= 0 && nxx + 1 < um ){
		  index[ 0 ] = nxx + 1;
		}
		else if( nxx + 1 < 0 ){
		  index[ 0 ] = 0; 
		}
		else if( nxx + 1 >= um ){
		  index[ 0 ] = um - 1; 
		}
		plan[ i ].h_input_data[ (nz * plan[i].num_elevation + ny) * plan[i].num_ray + nx ] += ProjPointer->GetPixel( index ) * (xx - nxx);

	      }
	    }
	  }
	}

	if( bProjFormatHdf5 ){

	  float proj, dark, white1, white2;
	  for( int nz = 0; nz < plan[ i ].num_proj; nz++ ){
	    for( int ny = 0; ny < plan[ i ].num_elevation; ny++ ){
	      for( int nx = 0; nx < plan[ i ].num_ray; nx++ ){

		float xx = nx - xoffset; 
		int nxx = (int)floor(xx);
		int nxx_shift;

		if( nxx >= 0 && nxx < um ){
		  nxx_shift = nxx;
		}
		else if( nxx < 0 ){
		  nxx_shift = 0; 
		}
		else if( nxx >= um ){
		  nxx_shift = um - 1; 
		}

		proj = proj_slices[ ( (i * NUM_SLICE_INV + ny ) * num_proj + nz ) * um + nxx_shift ];   
		dark = proj_darkfield[ (z + ny) * um + nxx_shift ];   
		white1 = proj_whitefield[ (z + ny) * um + nxx_shift ];   
		white2 = proj_whitefield[ um * vm + (z + ny) * um + nxx_shift ];   

		proj_slices_sino[ ( (i * NUM_SLICE_INV + ny ) * num_proj + nz ) * um + nx] =  ( 1.0f - 1.0f * (proj - dark) / (0.5f * white1 + 0.5f * white2 - dark)) * (nxx + 1 - xx); // note the index change here for ring removal
		// 
		if( nxx + 1 >= 0 && nxx + 1 < um ){
		  nxx_shift = nxx + 1;
		}
		else if( nxx + 1 < 0 ){
		  nxx_shift = 0; 
		}
		else if( nxx + 1 >= um ){
		  nxx_shift = um - 1; 
		}

		proj = proj_slices[ ( (i * NUM_SLICE_INV + ny ) * num_proj + nz ) * um + nxx_shift ];   
		dark = proj_darkfield[ (z + ny) * um + nxx_shift ];   
		white1 = proj_whitefield[ (z + ny) * um + nxx_shift ];   
		white2 = proj_whitefield[ um * vm + (z + ny) * um + nxx_shift ];   

		proj_slices_sino[ ( (i * NUM_SLICE_INV + ny ) * num_proj + nz ) * um + nx] +=  ( 1.0f - 1.0f * (proj - dark) / (0.5f * white1 + 0.5f * white2 - dark)) * (xx - nxx); // note the index change here for ring removal

	      }
	    }
	  }
	  //
	  for( int p = 0; p < NUM_SLICE_INV * deviceCount; p++ ){
	    RingCorrectionSinogram (&proj_slices_sino[ p * num_proj * um ], RING_COEFF, um, num_proj,
				    mean_vect, mean_sino_line_data, low_pass_sino_lines_data); 
	  }

	  // 
	  for( int nz = 0; nz < plan[ i ].num_proj; nz++ ){
	    for( int ny = 0; ny < plan[ i ].num_elevation; ny++ ){
	      for( int nx = 0; nx < plan[ i ].num_ray; nx++ ){

		plan[ i ].h_input_data[ (nz * plan[i].num_elevation + ny) * plan[i].num_ray + nx ] =  proj_slices_sino[ ( (i * NUM_SLICE_INV + ny ) * num_proj + nz ) * um + nx];

	      }
	    }
	  }
	  // 
	}

      }
  
    }
    else{

      cout << "Processing slices " << z << " ~ " << nSliceEnd << endl;

      if( bProjFormatHdf5 ){
	Hdf5SerialReadY( str_proj_hdf5_name.c_str(), "/entry/exchange/data", z, nSliceEnd - z, proj_slices );  
      }

      for( int i = 0; i < deviceCount; i++ ){
	plan[ i ].deviceID = i;
	plan[ i ].deviceShareMemSizePerBlock = GetSharedMemSize( i );  

	// geometry
	plan[ i ].num_width = xm;
	plan[ i ].num_height = ym;

	plan[ i ].num_ray = um;
	plan[ i ].num_proj = num_proj;

	bool bProjRead = false;
	ndeviceCompute++; 
	if( nSliceEnd - z - i * NUM_SLICE_INV > NUM_SLICE_INV ){
	  plan[ i ].num_depth = NUM_SLICE_INV;
	  plan[ i ].num_elevation =  NUM_SLICE_INV;
	}
	else{
	  plan[ i ].num_depth = nSliceEnd - z - i * NUM_SLICE_INV;
	  plan[ i ].num_elevation =  nSliceEnd - z - i * NUM_SLICE_INV;
	  bProjRead = true;
 	}

	// algorithm params
	plan[ i ].stop_type = stop_type;
	plan[ i ].iter_max = iter_max;
	plan[ i ].proj_thres = proj_thres;
	plan[ i ].diff_thres_percent = diff_thres_percent;
	plan[ i ].diff_thres_value = diff_thres_value;
	plan[ i ].lambda = lambda;
	plan[ i ].spacing = spacing;
	plan[ i ].voxel_size = voxel_size;
	plan[ i ].proj_pixel_size = proj_pixel_size;
	plan[ i ].SOD = SOD;
	plan[ i ].inv_rot = inv_rot;
	plan[ i ].xoffset = xoffset;
	plan[ i ].yoffset = yoffset;
	plan[ i ].zoffset = zoffset;
	plan[ i ].start_rot = start_rot;
	plan[ i ].end_rot = end_rot;

	// memory for input/output data
	plan[ i ].h_input_data = new float[ plan[i].num_proj * plan[i].num_elevation * plan[i].num_ray ];
	plan[ i ].h_output_data = new float[ plan[i].num_width * plan[i].num_height * plan[i].num_depth ];
	if( !plan[ i ].h_input_data ||  !plan[ i ].h_output_data ){
	  cout << "Error allocating memory for plan [ " << i << " ]" << endl;
	  exit( 1 );
	}

	if( n_proj_mask_image == 1 ){
	  plan[ i ].h_proj_mask_data = new float[ plan[i].num_elevation * plan[i].num_ray ];
	  if( !plan[ i ].h_proj_mask_data ){
	    cout << "Error allocating memory for plan [ " << i << " ]" << endl;
	    exit(1);
	  }

	  for( unsigned int ny = 0; ny <  plan[i].num_elevation; ny++ ){
	    for( unsigned int nx = 0; nx <  plan[i].num_ray; nx++ ){
	      int ny_elevation = z + i * NUM_SLICE_INV + ny; // z: object index; corresponds to elevation in projection index

	      plan[ i ].h_proj_mask_data[ ny * plan[i].num_ray + nx ] = proj_mask[ ny_elevation * plan[i].num_ray + nx ];
	    }
	  }


	  plan[ i ].n_proj_mask_image = 1; 
	}
	else{
	  plan[ i ].h_proj_mask_data = NULL;
	  plan[ i ].n_proj_mask_image = 0; 
	}

	if( bProjFormatNrrd ){
	  // input(projection) data from file
	  for( int nz = 0; nz < plan[ i ].num_proj; nz++ ){
	    for( int ny = 0; ny < plan[ i ].num_elevation; ny++ ){
	      for( int nx = 0; nx < plan[ i ].num_ray; nx++ ){

		index[ 1 ] = z + i * NUM_SLICE_INV + ny;
		index[ 2 ] = nz;

		// horizontal shift compensation
		float xx = nx - xoffset; 
		int nxx = (int)floor(xx);

		if( nxx >= 0 && nxx < um ){
		  index[ 0 ] = nxx;
		}
		else if( nxx < 0 ){
		  index[ 0 ] = 0; 
		}
		else if( nxx >= um ){
		  index[ 0 ] = um - 1; 
		}
		plan[ i ].h_input_data[ (nz * plan[i].num_elevation + ny) * plan[i].num_ray + nx ] =  ProjPointer->GetPixel( index ) * (nxx + 1 - xx);

		if( nxx + 1 >= 0 && nxx + 1 < um ){
		  index[ 0 ] = nxx + 1;
		}
		else if( nxx + 1 < 0 ){
		  index[ 0 ] = 0; 
		}
		else if( nxx + 1 >= um ){
		  index[ 0 ] = um - 1; 
		}
		plan[ i ].h_input_data[ (nz * plan[i].num_elevation + ny) * plan[i].num_ray + nx ] += ProjPointer->GetPixel( index ) * (xx - nxx);

	      }
	    }
	  }
	}

	if( bProjFormatHdf5 ){

	  // float proj, dark, white1, white2;
	  // for( int nz = 0; nz < plan[ i ].num_proj; nz++ ){
	  //   for( int ny = 0; ny < plan[ i ].num_elevation; ny++ ){
	  //     for( int nx = 0; nx < plan[ i ].num_ray; nx++ ){

	  // 	proj = proj_slices[ ( nz * (NUM_SLICE_INV * deviceCount) + (i * NUM_SLICE_INV + ny ) ) * um + nx ];   
	  // 	dark = proj_darkfield[ (z + i * NUM_SLICE_INV + ny) * um + nx ];   
	  // 	white1 = proj_whitefield[ (z + i * NUM_SLICE_INV + ny) * um + nx ];   
	  // 	white2 = proj_whitefield[ (vm + z + i * NUM_SLICE_INV + ny) * um + nx ];   
	  // 	plan[ i ].h_input_data[ (nz * plan[i].num_elevation + ny) * plan[i].num_ray + nx ] =  1.0f - 1.0f * (proj - dark) / (0.5f * white1 + 0.5f * white2 - dark);

	  //     }
	  //   }
	  // }

	  float proj, dark, white1, white2;
	  for( int nz = 0; nz < plan[ i ].num_proj; nz++ ){
	    for( int ny = 0; ny < plan[ i ].num_elevation; ny++ ){
	      for( int nx = 0; nx < plan[ i ].num_ray; nx++ ){

		float xx = nx - xoffset; 
		int nxx = (int)floor(xx);
		int nxx_shift;

		if( nxx >= 0 && nxx < um ){
		  nxx_shift = nxx;
		}
		else if( nxx < 0 ){
		  nxx_shift = 0; 
		}
		else if( nxx >= um ){
		  nxx_shift = um - 1; 
		}

		proj = proj_slices[ ( (i * NUM_SLICE_INV + ny ) * num_proj + nz ) * um + nxx_shift ];   
		dark = proj_darkfield[ (z + ny) * um + nxx_shift ];   
		white1 = proj_whitefield[ (z + ny) * um + nxx_shift ];   
		white2 = proj_whitefield[ um * vm + (z + ny) * um + nxx_shift ];   

		proj_slices_sino[ ( (i * NUM_SLICE_INV + ny ) * num_proj + nz ) * um + nx] =  ( 1.0f - 1.0f * (proj - dark) / (0.5f * white1 + 0.5f * white2 - dark)) * (nxx + 1 - xx); // note the index change here for ring removal
		// 
		if( nxx + 1 >= 0 && nxx + 1 < um ){
		  nxx_shift = nxx + 1;
		}
		else if( nxx + 1 < 0 ){
		  nxx_shift = 0; 
		}
		else if( nxx + 1 >= um ){
		  nxx_shift = um - 1; 
		}

		proj = proj_slices[ ( (i * NUM_SLICE_INV + ny ) * num_proj + nz ) * um + nxx_shift ];   
		dark = proj_darkfield[ (z + ny) * um + nxx_shift ];   
		white1 = proj_whitefield[ (z + ny) * um + nxx_shift ];   
		white2 = proj_whitefield[ um * vm + (z + ny) * um + nxx_shift ];   

		proj_slices_sino[ ( (i * NUM_SLICE_INV + ny ) * num_proj + nz ) * um + nx] +=  ( 1.0f - 1.0f * (proj - dark) / (0.5f * white1 + 0.5f * white2 - dark)) * (xx - nxx); // note the index change here for ring removal

	      }
	    }
	  }

	  //
	  for( int p = 0; p < NUM_SLICE_INV * deviceCount; p++ ){
	    RingCorrectionSinogram (&proj_slices_sino[ p * num_proj * um ], RING_COEFF, um, num_proj,
				    mean_vect, mean_sino_line_data, low_pass_sino_lines_data); 
	  }

	  // 
	  for( int nz = 0; nz < plan[ i ].num_proj; nz++ ){
	    for( int ny = 0; ny < plan[ i ].num_elevation; ny++ ){
	      for( int nx = 0; nx < plan[ i ].num_ray; nx++ ){

		plan[ i ].h_input_data[ (nz * plan[i].num_elevation + ny) * plan[i].num_ray + nx ] =  proj_slices_sino[ ( (i * NUM_SLICE_INV + ny ) * num_proj + nz ) * um + nx];

	      }
	    }
	  }
	  // 
	}

	if( bProjRead ){
	  break;
	}
      }

    }

    // run threaded GPU calculation
    for( int i = 0; i < ndeviceCompute; i++ ){
      threadID[ i ] = cutStartThread( (CUT_THREADROUTINE) EM3DThread, (void*) (plan + i) );
    }
    cutWaitForThreads( threadID, ndeviceCompute );

    // retrieve the GPU calculation results sequentially
    for( int i = 0; i < ndeviceCompute; i++ ){

      for( int nz = 0; nz < plan[ i ].num_depth; nz++ ){
	for( int ny = 0; ny < plan[ i ].num_height; ny++ ){
	  for( int nx = 0; nx < plan[ i ].num_width; nx++ ){

	    index[ 0 ] = nx;
	    index[ 1 ] = ny; 
	    // index[ 2 ] = z + i * NUM_SLICE_INV + nz - nSliceStart;
	    index[ 2 ] = i * NUM_SLICE_INV + nz;
	    ImageReconPointer->SetPixel( index, plan[ i ].h_output_data[ (nz * plan[i].num_height + ny) * plan[i].num_width + nx ]);
	  }
	}
      }

    }

    // free the allocated memory
    for( int i = 0; i < ndeviceCompute; i++ ){
      if( plan[ i ].h_input_data && plan[ i ].num_depth > 0) {
	delete [] plan[ i ].h_input_data;
	delete [] plan[ i ].h_output_data;

	if( plan[ i ].n_proj_mask_image == 1 ){
	  delete [] plan[ i ].h_proj_mask_data;
	  plan[ i ].h_proj_mask_data = NULL;
	}

	plan[ i ].h_input_data = NULL;
	plan[ i ].h_output_data = NULL;

      }
    }

    // 
    if( n_recon_image_type == RECON_IMAGE_TYPE_NRRD ){
  
      itk::ImageFileWriter<Float3DImageType>::Pointer ImageReconWriter;
      ImageReconWriter = itk::ImageFileWriter<Float3DImageType>::New();
      ImageReconWriter->SetInput( ImageReconPointer );

      string strPosix = ".nrrd"; 

      string strNrrdOutput = str_recon_image_directory;
      strNrrdOutput.append( str_recon_image_base_name );

      char buf[256];
      sprintf(buf, "_s%d_s%d", z ,  z + deviceCount * NUM_SLICE_INV - 1); 

      string str_end(buf);

      int pos = strNrrdOutput.length() - strPosix.length();
      strNrrdOutput.insert( pos,  str_end );

      cout << "    Writing file " << strNrrdOutput << endl;  // test

      ImageReconWriter->SetFileName( strNrrdOutput );
      ImageReconWriter->Update();  

      //   
      if( nReconRadiusCrop > 0 ){

	for( int nz = 0; nz < NUM_SLICE_INV; nz++ ){
	  for( int ny = 0; ny < ym; ny++ ){
	    for( int nx = 0; nx < xm; nx++ ){

	      index[ 0 ] = nx;
	      index[ 1 ] = ny;
	      index[ 2 ] = nz;

	      if( (nx - xm/2)*(nx-xm/2) + (ny - ym/2) * (ny - ym/2) >= nReconRadiusCrop * nReconRadiusCrop ){
		ImageReconPointer->SetPixel( index, 0.0f );
	      }

	    }
	  }
	}

	string strNrrdCropOutput = strNrrdOutput;
	pos = strNrrdCropOutput.length() - strPosix.length();
	strNrrdCropOutput.insert( pos,  "_crop" );

	ImageReconWriter->SetInput( ImageReconPointer );
	ImageReconWriter->SetFileName( strNrrdCropOutput );
	ImageReconWriter->Update();  

      }

    }
    else if( n_recon_image_type == RECON_IMAGE_TYPE_BIN ){

      string strPosix = string(".bin");

      Float3DImageType::PixelType pixelImg;
      PROJ_IMAGE_BIN_TYPE pixel_bin; 

      for( int nz = z ; nz <= z + deviceCount * NUM_SLICE_INV - 1;  nz++ ){

	string strIndex = num2str( nz );
	string strCurrBinName = str_recon_image_directory;
	strCurrBinName.append( str_recon_image_base_name );

	int pos = strCurrBinName.length() - strPosix.length() - strIndex.length();
	strCurrBinName.replace(pos, strIndex.length(), strIndex); 

	string strCurrBinCropName = strCurrBinName;
	pos = strCurrBinCropName.length() - strPosix.length();
	strCurrBinCropName.insert( pos,  "_crop" );      

	fstream datafile( strCurrBinName.c_str(), ios::out | ios::binary );
	fstream dataCropfile;

	if( nReconRadiusCrop > 0 ){

	  dataCropfile.open( strCurrBinCropName.c_str(), ios::out | ios::binary );

	  if( !dataCropfile.is_open() ){
	    cout << "    Error writing file " << strCurrBinCropName << endl;
	    continue; 
	  }

	}

	if( !datafile.is_open() ){
	  cout << "    Error writing file " << strCurrBinName << endl;
	  continue; 
	}
	else{
	  index[2] = nz - z; 

	  cout << "    Writing file " << strCurrBinName << endl;  // test
	  for( int y = 0; y < ym; y++ ){
	    for( int x = 0; x < xm; x++ ){  

	      index[0] = x;
	      index[1] = y;

	      pixel_bin = ImageReconPointer->GetPixel( index );
	      datafile.write( reinterpret_cast< char*> ( &pixel_bin ), sizeof( float ) );

	      if( nReconRadiusCrop > 0 ){

		if( (x - xm/2)*(x - xm/2) + (y - ym/2) * (y - ym/2) <= nReconRadiusCrop * nReconRadiusCrop ){
		  dataCropfile.write( reinterpret_cast< char*> ( &pixel_bin ), sizeof( float ) );
		}
		else{
		  pixel_bin = 0.0f; 
		  dataCropfile.write( reinterpret_cast< char*> ( &pixel_bin ), sizeof( float ) );
		}

	      }
	    }
	  }

	  datafile.close();
	  dataCropfile.close();
	}
      }
    }
    else if( n_recon_image_type == RECON_IMAGE_TYPE_BMP ||  n_recon_image_type == RECON_IMAGE_TYPE_TIFF){
      string strPosix;

      if( n_recon_image_type == RECON_IMAGE_TYPE_BMP )
	strPosix = string(".bmp");
    
      if( n_recon_image_type == RECON_IMAGE_TYPE_TIFF )
	strPosix = string(".tif");
    
      UChar2DImageType::Pointer ReconSlicePointer = UChar2DImageType::New();

      UChar2DImageType::SizeType size;
      UChar2DImageType::RegionType region;

      size[0] = xm;
      size[1] = ym;

      region.SetSize( size );

      ReconSlicePointer->SetRegions( region );
      ReconSlicePointer->Allocate(); 

      UChar2DImageType::Pointer ReconSliceCropPointer = NULL;
      if( nReconRadiusCrop > 0 ){
	ReconSliceCropPointer = UChar2DImageType::New();
	ReconSliceCropPointer->SetRegions( region );
	ReconSliceCropPointer->Allocate(); 
      }

      Float3DImageType::PixelType pixelImg;

      // 
      float thres_upper = -1e10, thres_lower = 1e10; 
      for( int nz = z ;  nz <=  z +  deviceCount * NUM_SLICE_INV - 1; nz++ ){

	index[2] = nz - z; 

	for( int y = 0; y < ym; y++ ){
	  for( int x = 0; x < xm; x++ ){  

	    index[0] = x;
	    index[1] = y;

	    pixelImg = ImageReconPointer->GetPixel( index );
	    if( thres_upper < pixelImg ){
	      thres_upper = pixelImg;
	    }
	    if( thres_lower > pixelImg ){
	      thres_lower = pixelImg;
	    }
	  }
	}
      }
    
      // 
      if( f_recon_threshold_upper_value > f_recon_threshold_lower_value ){

	if( thres_upper > f_recon_threshold_upper_value )
	  thres_upper = f_recon_threshold_upper_value;

	if( thres_lower < f_recon_threshold_lower_value )
	  thres_lower = f_recon_threshold_lower_value;

      }

      UChar2DImageType::PixelType pixelUShort;
      UChar2DImageType::IndexType indexUChar;

      //
      for( int nz = z ; nz <= z + deviceCount * NUM_SLICE_INV - 1; nz++ ){

	string strIndex = num2str( nz );
	string strCurrBmpTifName = str_recon_image_directory;
	strCurrBmpTifName.append( str_recon_image_base_name) ;

	int pos = strCurrBmpTifName.length() - strPosix.length() - strIndex.length();
	strCurrBmpTifName.replace(pos, strIndex.length(), strIndex); 

	string strCurrBmpTifCropName = strCurrBmpTifName;
	pos = strCurrBmpTifCropName.length() - strPosix.length();
	strCurrBmpTifCropName.insert( pos,  "_crop" );      

	// 
	index[2] = nz - z; 

	cout << "    Writing file " << strCurrBmpTifName << endl;  // test

	for( int y = 0; y < ym; y++ ){
	  for( int x = 0; x < xm; x++ ){  

	    index[0] = x;
	    index[1] = y;

	    indexUChar[0] = x;
	    indexUChar[1] = y;
	  
	    pixelImg = ImageReconPointer->GetPixel( index );
	    if( pixelImg < thres_lower )
	      pixelUShort = 0;
	    else if( pixelImg > thres_upper )
	      pixelUShort = 255;
	    else{
	      pixelUShort = (UChar2DImageType::PixelType ) 255 * (pixelImg - thres_lower) / (thres_upper - thres_lower);
	    }

	    ReconSlicePointer->SetPixel( indexUChar, pixelUShort );

	    if( nReconRadiusCrop > 0 ){

	      if( (x - xm/2)*(x - xm/2) + (y - ym/2) * (y - ym/2) <= nReconRadiusCrop * nReconRadiusCrop ){
		ReconSliceCropPointer->SetPixel( indexUChar, pixelUShort );
	      }
	      else{
		ReconSliceCropPointer->SetPixel( indexUChar, 0 );	  
	      }

	    }
	  }
	}

	itk::ImageFileWriter<UChar2DImageType>::Pointer ImageReconSliceWriter;
	ImageReconSliceWriter = itk::ImageFileWriter<UChar2DImageType>::New();
	ImageReconSliceWriter->SetInput( ReconSlicePointer );
	ImageReconSliceWriter->SetFileName( strCurrBmpTifName.c_str() );
	ImageReconSliceWriter->Update();  
    
	if( nReconRadiusCrop > 0 ){
	  itk::ImageFileWriter<UChar2DImageType>::Pointer ImageReconSliceCropWriter;
	  ImageReconSliceCropWriter = itk::ImageFileWriter<UChar2DImageType>::New();
	  ImageReconSliceCropWriter->SetInput( ReconSliceCropPointer );
	  ImageReconSliceCropWriter->SetFileName( strCurrBmpTifCropName.c_str() );
	  ImageReconSliceCropWriter->Update();  
	}

      }
    }

    if( n_recon_image_type == RECON_IMAGE_TYPE_HDF5 ){
      // retrieve the GPU calculation results sequentially
      for( int i = 0; i < ndeviceCompute; i++ ){

	for( int nz = 0; nz < plan[ i ].num_depth; nz++ ){
	  for( int ny = 0; ny < plan[ i ].num_height; ny++ ){
	    for( int nx = 0; nx < plan[ i ].num_width; nx++ ){

	      int index_obj = ((i * plan[ i ].num_depth + nz) * plan[i].num_height + ny) * plan[i].num_width + nx;
	      int index_plan = (nz * plan[i].num_height + ny) * plan[i].num_width + nx;
	      obj_slices[ index_obj ] = plan[i].h_output_data[ index_plan ];

	    }
	  }
	}

      }


      Hdf5SerialWriteZ(str_proj_hdf5_name.c_str(), str_recon_hdf5_dataset_name.c_str(),  z, NUM_SLICE_INV * ndeviceCompute, xm, ym, obj_slices);

#ifdef VERBOSE
      cout << " Reconstructed slices " << z + 1  << "  ";
      cout << z +  NUM_SLICE_INV * ndeviceCompute;
      cout << " written to hdf5 file" << endl;
#endif // VERBOSE

    }


    // update z;
    // if( zm - z >= deviceCount * NUM_SLICE_INV){
    //   z += deviceCount * NUM_SLICE_INV;
    // }
    // else{
    //   z = zm;
    // }

    if( nSliceEnd - z + 1 >= deviceCount * NUM_SLICE_INV){
      z += deviceCount * NUM_SLICE_INV;
    }
    else{
      z = nSliceEnd + 1;
    }

  }

  // Output the reconstructed Image File

  // if( n_recon_image_type == RECON_IMAGE_TYPE_NRRD ){
  
  //   itk::ImageFileWriter<Float3DImageType>::Pointer ImageReconWriter;
  //   ImageReconWriter = itk::ImageFileWriter<Float3DImageType>::New();
  //   ImageReconWriter->SetInput( ImageReconPointer );

  //   string strNrrdOutput = str_recon_image_directory;
  //   strNrrdOutput.append( str_recon_image_base_name );

  //   ImageReconWriter->SetFileName( strNrrdOutput );
  //   ImageReconWriter->Update();  
  
  //   // 
  //   for( int nz = 0; nz < nSliceEnd - nSliceStart + 1; nz++ ){
  //     for( int ny = 0; ny < ym; ny++ ){
  // 	for( int nx = 0; nx < xm; nx++ ){

  // 	  index[ 0 ] = nx;
  // 	  index[ 1 ] = ny;
  // 	  index[ 2 ] = nz;

  // 	  if( (nx - xm/2)*(nx-xm/2) + (ny - ym/2) * (ny - ym/2) >= nReconRadiusCrop * nReconRadiusCrop ){
  // 	    ImageReconPointer->SetPixel( index, 0.0f );
  // 	  }

  // 	}
  //     }
  //   }

  //   string strPosix = ".nrrd"; 
  //   string strNrrdCropOutput = str_recon_image_directory;
  //   strNrrdCropOutput.append( str_recon_image_base_name );
  //   int pos = strNrrdCropOutput.length() - strPosix.length();
  //   strNrrdCropOutput.insert( pos,  "_crop" );

  //   ImageReconWriter->SetInput( ImageReconPointer );
  //   ImageReconWriter->SetFileName( strNrrdCropOutput );
  //   ImageReconWriter->Update();  
  // }
  // else if( n_recon_image_type == RECON_IMAGE_TYPE_BIN ){

  //   string strPosix = string(".bin");

  //   Float3DImageType::PixelType pixelImg;
  //   PROJ_IMAGE_BIN_TYPE pixel_bin; 

  //   for( int z = nSliceStart; z <= nSliceEnd; z++ ){

  //     string strIndex = num2str( z );
  //     string strCurrBinName = str_recon_image_directory;
  //     strCurrBinName.append( str_recon_image_base_name );

  //     int pos = strCurrBinName.length() - strPosix.length() - strIndex.length();
  //     strCurrBinName.replace(pos, strIndex.length(), strIndex); 

  //     string strCurrBinCropName = strCurrBinName;
  //     pos = strCurrBinCropName.length() - strPosix.length();
  //     strCurrBinCropName.insert( pos,  "_crop" );      

  //     fstream datafile( strCurrBinName.c_str(), ios::out | ios::binary );
  //     fstream dataCropfile( strCurrBinCropName.c_str(), ios::out | ios::binary );

  //     if( !datafile.is_open() || !dataCropfile.is_open() ){
  // 	cout << "    Error writing file " << strCurrBinName << " and " << strCurrBinCropName << endl;
  // 	continue; 
  //     }
  //     else{
  // 	index[2] = z - nSliceStart; 

  // 	cout << "    Writing file " << strCurrBinName << endl;  // test
  // 	for( int y = 0; y < ym; y++ ){
  // 	  for( int x = 0; x < xm; x++ ){  

  // 	    index[0] = x;
  // 	    index[1] = y;

  // 	    pixel_bin = ImageReconPointer->GetPixel( index );
  // 	    datafile.write( reinterpret_cast< char*> ( &pixel_bin ), sizeof( float ) );

  // 	    if( (x - xm/2)*(x - xm/2) + (y - ym/2) * (y - ym/2) <= nReconRadiusCrop * nReconRadiusCrop ){
  // 	      dataCropfile.write( reinterpret_cast< char*> ( &pixel_bin ), sizeof( float ) );
  // 	    }
  // 	    else{
  // 	      pixel_bin = 0.0f; 
  // 	      dataCropfile.write( reinterpret_cast< char*> ( &pixel_bin ), sizeof( float ) );
  // 	    }

  // 	  }
  // 	}

  // 	datafile.close();
  // 	dataCropfile.close();
  //     }
  //   }
  // }
  // else if( n_recon_image_type == RECON_IMAGE_TYPE_BMP ||  n_recon_image_type == RECON_IMAGE_TYPE_TIFF){
  //   string strPosix;

  //   if( n_recon_image_type == RECON_IMAGE_TYPE_BMP )
  //     strPosix = string(".bmp");
    
  //   if( n_recon_image_type == RECON_IMAGE_TYPE_TIFF )
  //     strPosix = string(".tif");
    
  //   UChar2DImageType::Pointer ReconSlicePointer = UChar2DImageType::New();
  //   UChar2DImageType::Pointer ReconSliceCropPointer = UChar2DImageType::New();

  //   UChar2DImageType::SizeType size;
  //   UChar2DImageType::RegionType region;

  //   size[0] = xm;
  //   size[1] = ym;

  //   region.SetSize( size );

  //   ReconSlicePointer->SetRegions( region );
  //   ReconSlicePointer->Allocate(); 

  //   ReconSliceCropPointer->SetRegions( region );
  //   ReconSliceCropPointer->Allocate(); 

  //   Float3DImageType::PixelType pixelImg;

  //   // 
  //   float thres_upper = -1e10, thres_lower = 1e10; 
  //   for( int z = nSliceStart; z <= nSliceEnd; z++ ){

  //     index[2] = z - nSliceStart; 

  //     for( int y = 0; y < ym; y++ ){
  // 	for( int x = 0; x < xm; x++ ){  

  // 	  index[0] = x;
  // 	  index[1] = y;

  // 	  pixelImg = ImageReconPointer->GetPixel( index );
  // 	  if( thres_upper < pixelImg ){
  // 	    thres_upper = pixelImg;
  // 	  }
  // 	  if( thres_lower > pixelImg ){
  // 	    thres_lower = pixelImg;
  // 	  }
  // 	}
  //     }
  //   }
    
  //   // 
  //   if( thres_upper > f_recon_threshold_upper_value )
  //     thres_upper = f_recon_threshold_upper_value;

  //   if( thres_lower < f_recon_threshold_lower_value )
  //     thres_lower = f_recon_threshold_lower_value;

  //   UChar2DImageType::PixelType pixelUShort;
  //   UChar2DImageType::IndexType indexUChar;

  //   //
  //   for( int z = nSliceStart; z <= nSliceEnd; z++ ){

  //     string strIndex = num2str( z );
  //     string strCurrBmpTifName = str_recon_image_directory;
  //     strCurrBmpTifName.append( str_recon_image_base_name) ;

  //     int pos = strCurrBmpTifName.length() - strPosix.length() - strIndex.length();
  //     strCurrBmpTifName.replace(pos, strIndex.length(), strIndex); 

  //     string strCurrBmpTifCropName = strCurrBmpTifName;
  //     pos = strCurrBmpTifCropName.length() - strPosix.length();
  //     strCurrBmpTifCropName.insert( pos,  "_crop" );      

  //     // 
  //     index[2] = z - nSliceStart; 

  //     cout << "    Writing file " << strCurrBmpTifName << endl;  // test

  //     for( int y = 0; y < ym; y++ ){
  // 	for( int x = 0; x < xm; x++ ){  

  // 	  index[0] = x;
  // 	  index[1] = y;

  // 	  indexUChar[0] = x;
  // 	  indexUChar[1] = y;
	  
  // 	  pixelImg = ImageReconPointer->GetPixel( index );
  // 	  if( pixelImg < thres_lower )
  // 	    pixelUShort = 0;
  // 	  else if( pixelImg > thres_upper )
  // 	    pixelUShort = 255;
  // 	  else{
  // 	    pixelUShort = (UChar2DImageType::PixelType ) 255 * (pixelImg - thres_lower) / (thres_upper - thres_lower);
  // 	  }

  // 	  ReconSlicePointer->SetPixel( indexUChar, pixelUShort );

  // 	  if( (x - xm/2)*(x - xm/2) + (y - ym/2) * (y - ym/2) <= nReconRadiusCrop * nReconRadiusCrop ){
  // 	    ReconSliceCropPointer->SetPixel( indexUChar, pixelUShort );
  // 	  }
  // 	  else{
  // 	    ReconSliceCropPointer->SetPixel( indexUChar, 0 );	  
  // 	  }

  // 	}
  //     }

  //     itk::ImageFileWriter<UChar2DImageType>::Pointer ImageReconSliceWriter;
  //     ImageReconSliceWriter = itk::ImageFileWriter<UChar2DImageType>::New();
  //     ImageReconSliceWriter->SetInput( ReconSlicePointer );
  //     ImageReconSliceWriter->SetFileName( strCurrBmpTifName.c_str() );
  //     ImageReconSliceWriter->Update();  

  //     itk::ImageFileWriter<UChar2DImageType>::Pointer ImageReconSliceCropWriter;
  //     ImageReconSliceCropWriter = itk::ImageFileWriter<UChar2DImageType>::New();
  //     ImageReconSliceCropWriter->SetInput( ReconSliceCropPointer );
  //     ImageReconSliceCropWriter->SetFileName( strCurrBmpTifCropName.c_str() );
  //     ImageReconSliceCropWriter->Update();  

  //   }
  // }
  // else if( n_recon_image_type == RECON_IMAGE_TYPE_HDF5 ){

  //   string strHdf5Output = str_recon_image_directory;
  //   strHdf5Output.append( str_recon_image_base_name );

  //   Hdf5SerialWrite( strHdf5Output.c_str(), str_recon_hdf5_dataset_name.c_str(), 
  // 		      xm, ym, nSliceEnd - nSliceStart + 1, 
  // 		      (float*) ImageReconPointer->GetBufferPointer( ) ); 

  //   for( int nz = 0; nz < nSliceEnd - nSliceStart + 1; nz++ ){
  //     for( int ny = 0; ny < ym; ny++ ){
  //   	for( int nx = 0; nx < xm; nx++ ){

  //   	  index[ 0 ] = nx;
  //   	  index[ 1 ] = ny;
  //   	  index[ 2 ] = nz;

  //   	  if( (nx - xm/2)*(nx-xm/2) + (ny - ym/2) * (ny - ym/2) >= nReconRadiusCrop * nReconRadiusCrop ){
  //   	    ImageReconPointer->SetPixel( index, 0.0f );
  //   	  }

  //   	}
  //     }
  //   }

  //   string strPosix = ".hdf5"; 
  //   string strHdf5CropOutput = str_recon_image_directory;
  //   strHdf5CropOutput.append( str_recon_image_base_name );
  //   int pos = strHdf5CropOutput.length() - strPosix.length();
  //   strHdf5CropOutput.insert( pos,  "_crop" );

  //   Hdf5SerialWrite( strHdf5CropOutput.c_str(), str_recon_hdf5_dataset_name.c_str(), 
  // 		      xm, ym, nSliceEnd - nSliceStart + 1, 
  // 		      (float*) ImageReconPointer->GetBufferPointer( ) ); 
  // }

  // free the allocated memory
  if( n_proj_mask_image == 1 ){
    delete [] proj_mask;
  }

  if( bProjFormatHdf5 ){
    delete [] proj_darkfield; 
    delete [] proj_whitefield; 
    delete [] proj_slices;
  }

  delete [] proj_slices_sino;
  delete [] obj_slices;

  delete [] threadID;
  delete [] plan;

  delete [] mean_vect;
  delete [] low_pass_sino_lines_data;
  delete [] mean_sino_line_data; 
  delete [] data_sino;

  return 0; 
}

// threaded multiple GPU EM for CT recon (using texture memory)

static CUT_THREADPROC EM3DThread(ThreadGPUPlan* plan){

  unsigned int   deviceID        = plan->deviceID;
  unsigned int   num_width       = plan->num_width;
  unsigned int   num_height      = plan->num_height;  
  unsigned int   num_depth       = plan->num_depth;
  unsigned int   num_ray         = plan->num_ray;
  unsigned int   num_elevation   = plan->num_elevation;
  unsigned int   num_proj        = plan->num_proj;
  int            stop_type       = plan->stop_type;
  unsigned int   iter_max        = plan->iter_max;
  float          proj_thres      = plan->proj_thres;
  float          diff_thres_percent = plan->diff_thres_percent;
  float          diff_thres_value   = plan->diff_thres_value;
  float lambda          = plan->lambda;
  float spacing         = plan->spacing;
  float voxel_size      = plan->voxel_size;
  float proj_pixel_size = plan->proj_pixel_size;
  float SOD             = plan->SOD;
  float inv_rot         = plan->inv_rot;
  float xoffset         = plan->xoffset;
  float yoffset         = plan->yoffset;
  float zoffset         = plan->zoffset;
  float start_rot       = plan->start_rot;
  float end_rot         = plan->end_rot;

  int n_proj_mask_image = plan->n_proj_mask_image;

  cudaSetDevice( deviceID );
  cout << "device " << deviceID << " used " << endl; 

#ifdef OUTPUT_GPU_TIME
  unsigned int timerEM3D = 0;
  if( deviceID == 0 )
  {
    CUT_SAFE_CALL(cutCreateTimer(&timerEM3D));
    CUT_SAFE_CALL(cutStartTimer(timerEM3D));
  }
#endif 

  int volumeImage = num_depth * num_width * num_height;
  int volumeProj = num_proj * num_elevation * num_ray;

  unsigned int x, y, z, posInVolume;
  float intensity_init = INTENSITY_INIT;

  // EM initialization
  for( z = 0; z < num_depth; z++ ){
    for( y = 0; y < num_height; y++ ){
      for( x = 0; x < num_width; x++ ){

	posInVolume = ( z * num_height + y ) * num_width + x;
	plan->h_output_data[ posInVolume] = intensity_init; 
      }
    }
  }

  // 3D EM Preparation

  int iter = 0; 

  // Allocate memory in the device for GPU implementation

  double time_gpu = 0.0;
  unsigned int timerDataPrepGPU = 0;

#ifdef OUTPUT_GPU_TIME

  if( deviceID == 0 )
  {
    CUT_SAFE_CALL(cutCreateTimer(&timerDataPrepGPU));
    CUT_SAFE_CALL(cutStartTimer(timerDataPrepGPU));
  }

#endif

  float* d_image_prev;
  float* d_image_curr;
  float* d_proj_cur; 
  float* d_proj_mask_data = NULL;

  const cudaExtent volume_size_proj = make_cudaExtent(num_ray, num_elevation, num_proj);  
  const cudaExtent volume_size_voxel = make_cudaExtent(num_width,  num_height, num_depth);  

//   cutilSafeCall( cudaMalloc( (void**)&d_image_prev, sizeof(float) * num_ray * num_elevation * num_proj ) );
//   cutilSafeCall( cudaMalloc( (void**)&d_image_curr, sizeof(float) * num_ray * num_elevation * num_proj ) );

  // Note that cudaMallocPitch() takes more time than cudaMalloc(). It is kept because it is recommended. 
  size_t pitchImage = 0;
  cudaMallocPitch((void**) &d_image_prev, &pitchImage, num_width * sizeof(float), num_height * num_depth );            CUT_CHECK_ERROR("Memory creation failed");

  cudaMallocPitch((void**) &d_image_curr, &pitchImage, num_width * sizeof(float), num_height * num_depth );            CUT_CHECK_ERROR("Memory creation failed");

  cudaMemcpy( d_image_prev, plan->h_output_data, sizeof(float) * volumeImage, cudaMemcpyHostToDevice);
  CUT_CHECK_ERROR("Memory copy for image_prev failed");
  cudaMemcpy( d_image_curr, d_image_prev, sizeof(float) * volumeImage, cudaMemcpyDeviceToDevice);
  CUT_CHECK_ERROR("Memory copy for image_curr failed");


  if( n_proj_mask_image == 1 ){
    cudaMallocPitch((void**) &d_proj_mask_data, &pitchImage, num_ray * sizeof(float), num_elevation ); 
    CUT_CHECK_ERROR("Memory creation failed");

    cudaMemcpy( d_proj_mask_data, plan->h_proj_mask_data, sizeof(float) * num_ray * num_elevation, 
		cudaMemcpyHostToDevice);
    CUT_CHECK_ERROR("Memory copy for proj_mask_data failed");

  }

  // cutilSafeCall( cudaMalloc( (void**)&d_proj_cur, sizeof(float) * num_ray * num_elevation * num_proj ) );

  size_t pitchProj = 0;
  cudaMallocPitch((void**) &d_proj_cur, &pitchProj, num_ray * sizeof(float), num_elevation * num_proj);               CUT_CHECK_ERROR("Memory creation failed");

  cudaArray *d_array_voxel = NULL;
  cudaChannelFormatDesc float1Desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); // float
  CUDA_SAFE_CALL( cudaMalloc3DArray( &d_array_voxel, &float1Desc, volume_size_voxel ) );

  float* d_proj;

  cudaMalloc((void**) &d_proj, sizeof(float) * volumeProj);                         
  CUT_CHECK_ERROR("Memory creation failed");

  cudaMemcpy( d_proj, (float*)plan->h_input_data, sizeof(float) * volumeProj, cudaMemcpyHostToDevice);
  CUT_CHECK_ERROR("Memory copy for Proj failed");

  cudaArray *d_array_proj = NULL;
  CUDA_SAFE_CALL( cudaMalloc3DArray( &d_array_proj, &float1Desc, volume_size_proj ) );
  // cout << "Memcopy allocation for d_array_proj: " << cudaGetErrorString( cudaGetLastError() ) << endl;

  // allocate GPU memory for reduction
  int maxThreads = THREADS_MAX; 

  float * d_proj_sub = NULL;
  int nVolProj = num_ray * num_elevation * num_proj;

  int nThreads = (nVolProj < maxThreads * 2) ? nextPow2((nVolProj + 1)/ 2) : maxThreads;
  int nBlocks = (nVolProj + (nThreads * 2 - 1) ) / (nThreads * 2);

  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_proj_sub, nBlocks* sizeof(float) ) );           
  CUT_CHECK_ERROR("Memory creation failed");
  
  float * d_image_sub = NULL;
  int nVolImg = num_width * num_height * num_depth;

  nThreads = (nVolImg < maxThreads * 2) ? nextPow2((nVolImg + 1)/ 2) : maxThreads;
  nBlocks = (nVolImg + (nThreads * 2 - 1) ) / (nThreads * 2);

  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_image_sub, nBlocks* sizeof(float) ) ); 
  CUT_CHECK_ERROR("Memory creation failed");

  float * d_image_iszero = NULL;

#ifndef WEIGHT_CAL
  float* image_org = new float[ volumeImage ];

  for( x = 0; x < num_width; x++ ){
    for( y = 0; y < num_height; y++ ){
      for( z = 0; z < num_depth; z++ ){

	posInVolume = ( z * num_height + y ) * num_width + x;

#ifdef STEVE_DATA
	// the object locates inside a cycle with center (num_width/2, num_height/2) 
	// and radius min(num_width, num_height) / 2

	int radius = num_width / 2;
	if( num_height < num_width )
	  radius = num_height/2;

	if( (x- num_width/2) * (x-num_width/2) + (y-num_height/2) * (y-num_height/2) <= radius * radius ) 
	  image_org[ posInVolume ] = 1.0f;
	else
	  image_org[ posInVolume ] = 0.0f;
#else
	image_org[ posInVolume ] = 1.0f;
#endif

#ifdef PETER_DATA
	// the object locates inside a cycle with center (512, 512) and radius 512
	if( (x-1024) * (x-1024) + (y-1024) * (y-1024) <= 630 * 630 )
	  image_org[ posInVolume ] = 1.0f;
	else
	  image_org[ posInVolume ] = 0.0f;
#else
	image_org[ posInVolume ] = 1.0f;
#endif

      }
    }
  }

  cudaMallocPitch((void**) &d_image_iszero, &pitchImage, num_width * sizeof(float), num_height * num_depth);             
  CUT_CHECK_ERROR("Memory creation failed");

  cudaMemcpy( d_image_iszero, image_org, sizeof(float) * volumeImage, cudaMemcpyHostToDevice);
  CUT_CHECK_ERROR("Memory copy for d_image_iszero failed");

#endif  // WEIGHT_CAL

  // Perform TV3D on the result
#ifdef USE_TV

  float fidelity_weight_TV = param_fidelity_weight_TV; 
  float epsilon_TV = param_epsilon_TV; 
  int   iter_max_TV = param_iter_max_TV; 
  float thres_TV = param_thres_TV;

  int   iter_TV = param_iter_TV;
  float timestep_TV = param_timestep_TV;

#ifdef OUTPUT_GPU_TIME

  // create timers
  unsigned int timerTVCalGPU = 0;
  if( deviceID == 0 )
   {
   CUT_SAFE_CALL(cutCreateTimer(&timerTVCalGPU));
  
   // start timer
   CUT_SAFE_CALL(cutStartTimer(timerTVCalGPU));
 }

#endif  

  tv3d_cal_wrapper( d_image_prev, d_image_iszero, 
		    d_image_curr,  
		    num_width, num_height, num_depth, 
		    iter_TV, fidelity_weight_TV, epsilon_TV, timestep_TV );

#ifdef OUTPUT_GPU_TIME
  if( deviceID == 0 )
  {
   // stop timer
   cudaThreadSynchronize(); 
   CUT_SAFE_CALL(cutStopTimer(timerTVCalGPU));
   time_gpu += cutGetTimerValue(timerTVCalGPU);
   CUT_SAFE_CALL(cutDeleteTimer(timerTVCalGPU));
 }
#endif

#endif // USE_TV

  // create timer
#ifdef OUTPUT_GPU_TIME

  unsigned int timerTotal = 0;

  if( deviceID == 0 )
  {
    CUT_SAFE_CALL(cutCreateTimer(&timerTotal));
    CUT_SAFE_CALL(cutStartTimer(timerTotal));
  }

  double time_ProjPrep = 0.0;
  double time_ProjCal = 0.0;
  double time_BackprojPrep = 0.0;
  double time_BackprojCal = 0.0;
  double time_ImgTrans = 0.0;

#endif

  // 3D EM
  bool b_iter_max = true;
  bool b_proj_thres = true;
  bool b_diff_thres = true;

  float proj_l1;
  float diff_percent; 

  while( b_iter_max * b_proj_thres * b_diff_thres ){
    // for( int iter = 0; iter < iter_max; iter++ ){
   
#ifdef ADAPTIVE_STEP
    if( iter > 10 )
      lambda = 1.99f;
    else
      lambda = 1.0f;
#endif // ADAPTIVE_STEP
      
    // step 1: calculate the current projection
    // CUDA implementation of the Projection calculation

#ifdef OUTPUT_GPU_TIME
    // create timers
    unsigned int timerProjPrep = 0;

    if( deviceID == 0 )
    {
      CUT_SAFE_CALL(cutCreateTimer(&timerProjPrep));
      CUT_SAFE_CALL(cutStartTimer(timerProjPrep));
    }

#endif

    // copy volume data to 3D array
    cudaMemcpy3DParms copyParamsVoxel = {0};
    copyParamsVoxel.srcPtr   = make_cudaPitchedPtr((void*)d_image_prev,
    						   volume_size_voxel.width*sizeof(float), 
    						   volume_size_voxel.width, volume_size_voxel.height);
    copyParamsVoxel.dstArray = d_array_voxel;
    copyParamsVoxel.extent   = volume_size_voxel;
    copyParamsVoxel.kind     = cudaMemcpyDeviceToDevice;
   
    CUDA_SAFE_CALL( cudaMemcpy3D(&copyParamsVoxel) );
    // cout << "Memcopy from d_image_prev to d_array_voxel: " << cudaGetErrorString( cudaGetLastError() ) << endl;
    cudaMemset( d_proj_cur, 0, sizeof(float) * volumeProj );  // Initialization turns out to be important

#ifdef OUTPUT_GPU_TIME

    // stop timer
    if( deviceID == 0 )
    {

      CUT_SAFE_CALL(cutStopTimer(timerProjPrep));
      double time_tmp =  cutGetTimerValue(timerProjPrep);
      time_ProjPrep += time_tmp;
      time_gpu += time_tmp;

      cout << endl;
      cout << "    Iter " << iter << ": GPU processing time for forward projection preparation: " 
	   << time_tmp << " (ms) " << endl; 

      CUT_SAFE_CALL(cutDeleteTimer(timerProjPrep));
    }

    // create timers
    unsigned int timerProjCal = 0;

    if( deviceID == 0 )
    {
      CUT_SAFE_CALL(cutCreateTimer(&timerProjCal));
      CUT_SAFE_CALL(cutStartTimer(timerProjCal));
    }

#endif

    proj_cal_wrapper( d_array_voxel, d_proj,  n_proj_mask_image, d_proj_mask_data,           
		      d_proj_cur, 
		      num_depth, num_height, num_width, 
		      num_proj, num_elevation, num_ray, 
		      spacing, voxel_size, proj_pixel_size, SOD, inv_rot,
		      xoffset, yoffset, zoffset, start_rot, end_rot) ;  

#ifdef OUTPUT_GPU_TIME

    if( deviceID == 0 ) {
      // stop timer
      cudaThreadSynchronize(); 
      CUT_SAFE_CALL(cutStopTimer(timerProjCal));
      double time_tmp =  cutGetTimerValue(timerProjCal);
      time_ProjCal += time_tmp;
      time_gpu += time_tmp;

      cout << endl;
      cout << "    Iter " << iter << ": GPU processing time for projection calculation: " 
	   << time_tmp << " (ms) " << endl; 

      CUT_SAFE_CALL(cutDeleteTimer(timerProjCal));
    }

#endif

    // // out d_proj_cur for debug
    // float* proj_cur = new float[ volumeProj ];
    // cudaMemcpy( proj_cur, d_proj_cur, sizeof(float) * volumeProj, cudaMemcpyDeviceToHost );

    // Float3DImageType::Pointer outProj = Float3DImageType::New();
    // Float3DImageType::RegionType regionProj;
    // Float3DImageType::SizeType sizeProj;

    // sizeProj[0] = num_ray;
    // sizeProj[1] = num_elevation;
    // sizeProj[2] = num_proj;
    
    // regionProj.SetSize( sizeProj );
    // outProj->SetRegions( regionProj );
    // outProj->Allocate( );

    // for( x = 0; x < num_ray; x++ ){
    //   for( y = 0; y < num_elevation; y++ ){
    // 	for( z = 0; z < num_proj; z++ ){

    // 	  Float3DImageType::IndexType index; 
    // 	  index[ 0 ] = x;
    // 	  index[ 1 ] = y;
    // 	  index[ 2 ] = z;

    // 	  posInVolume = ( z * num_elevation + y ) * num_ray + x;

    // 	  outProj->SetPixel( index, 1000 * proj_cur[posInVolume] );
    // 	}
    //   }
    // }

    // itk::ImageFileWriter<Float3DImageType>::Pointer ImageWriter;
    // ImageWriter = itk::ImageFileWriter<Float3DImageType>::New();
    // ImageWriter->SetInput( outProj );
    // ImageWriter->SetFileName( "proj.nrrd" );
    // ImageWriter->Update(); 

    // delete [] proj_cur;

    // step 1.5: calculate the l2 difference between d_proj_cur and d_proj 
    if( stop_type == 2 && iter > 0 && iter % STOP_INV == 0 ){

      // note that the reduction results from CPU and GPU are not exactly the same (should be), 
      // but they are close enough for stopping iterations (e.g., 1080.34 CPU vs 1079.33 GPU)
      if( 65535 * THREADS_MAX * 2 >= volumeProj ){  // GPU
	proj_l1 = reduction_wrapper( d_proj_cur, d_proj_sub, 
				     num_ray, num_elevation, num_proj );

	b_proj_thres = (proj_l1 > proj_thres);
      }
      else{

	float* proj_cur = new float[ volumeProj ];
	cudaMemcpy( proj_cur, d_proj_cur, sizeof(float) * volumeProj, cudaMemcpyDeviceToHost ) ;
	CUT_CHECK_ERROR("Memory copy for proj_cur failed");

	float proj_l1_cpu = 0.0;
	for( int i = 0; i < volumeProj; i++ ){
	  proj_l1_cpu += fabs( proj_cur[ i] );

	  // if( isnan( proj_l1_cpu ) )  // for patrick data
	  //   cout << endl;
	}
	delete [] proj_cur;

	b_proj_thres = (proj_l1_cpu > proj_thres);
      }
    }
   
    // step 2:  New Implementation of EM 3D

    // GPU implementation of backprojection
#ifdef OUTPUT_GPU_TIME

    unsigned int timerBackprojPrep;
    if( deviceID == 0 ) {
      CUT_SAFE_CALL(cutCreateTimer(&timerBackprojPrep));
      CUT_SAFE_CALL(cutStartTimer(timerBackprojPrep));
    }

#endif

    // Copy data from GPU global memory to GPU array directly. 

    cudaMemcpy3DParms copyParamsProj = {0};
    copyParamsProj.srcPtr   = make_cudaPitchedPtr((void*)d_proj_cur, volume_size_proj.width*sizeof(float), 
     						  volume_size_proj.width, volume_size_proj.height);
    copyParamsProj.dstArray = d_array_proj;
    copyParamsProj.extent   = volume_size_proj;
    copyParamsProj.kind     = cudaMemcpyDeviceToDevice;
    CUDA_SAFE_CALL( cudaMemcpy3D(&copyParamsProj) );

    // the above codes may be accomplished using cudaMemcpyToArray, or cudaMemcpy3DToArray if any

    cudaMemset( d_image_curr, 0, sizeof(float) * volumeImage );  // Initialization turns out to be important

#ifdef OUTPUT_GPU_TIME

    if( deviceID == 0 )
    {
      CUT_SAFE_CALL(cutStopTimer(timerBackprojPrep));
      double time_tmp =  cutGetTimerValue(timerBackprojPrep); 
      time_BackprojPrep += time_tmp;
      time_gpu += time_tmp;

      cout << endl;
      cout << "    Iter " << iter << ": GPU processing time for backward projection preparation: "
	   << time_tmp << "  (ms) " << endl;

      CUT_SAFE_CALL(cutDeleteTimer(timerBackprojPrep));
    }

    unsigned int timerBackprojCal;

    if( deviceID == 0 )
    {
      CUT_SAFE_CALL(cutCreateTimer(&timerBackprojCal));
      CUT_SAFE_CALL(cutStartTimer(timerBackprojCal));
    }

#endif

    backproj_cal_wrapper( d_array_proj, d_image_prev, d_image_iszero,
			  d_image_curr, 
			  num_depth, num_height, num_width,
			  num_proj, num_elevation, num_ray, 
			  lambda, voxel_size, proj_pixel_size, SOD, inv_rot,
			  xoffset, yoffset, zoffset, start_rot, end_rot) ;  

#ifdef OUTPUT_GPU_TIME

    if( deviceID == 0 )
    {
      cudaThreadSynchronize(); 
      CUT_SAFE_CALL(cutStopTimer(timerBackprojCal));
      double time_tmp =  cutGetTimerValue(timerBackprojCal); 
      time_BackprojCal += time_tmp;
      time_gpu += time_tmp;

      cout << endl;
      cout << "    Iter " << iter << ": GPU processing time for backward projection: "
	   << time_tmp << "  (ms) " << endl;

      CUT_SAFE_CALL(cutDeleteTimer(timerBackprojCal));
    }

    unsigned int timerImageTrans;

    if( deviceID == 0 )
    {
      CUT_SAFE_CALL(cutCreateTimer(&timerImageTrans));
      CUT_SAFE_CALL(cutStartTimer(timerImageTrans));
    }

#endif

    // step 2.5: calculate the difference between two iterations
    if( (stop_type == 3 || stop_type == 4 ) && iter > 0 && iter % STOP_INV == 0){

      if( stop_type == 3 ){
	if( 65535 * THREADS_MAX * 2 >= num_width * num_height * num_depth ){  // GPU
	  // 65535: max dimension of grid size
	  // THREADS_MAX: max number of threads per block
	  // Supports 16 slices of 2048 * 2048. Not 32. 
	  diff_percent = reduction2_wrapper( d_image_curr, d_image_prev, d_image_sub, 
					     num_width, num_height, num_depth) ;
	  diff_percent /= reduction_wrapper( d_image_prev, d_image_sub, 
					     num_width, num_height, num_depth) ;
	  b_diff_thres = (diff_percent > diff_thres_percent);

	  cout << diff_percent << " vs " << diff_thres_percent << endl;
	}
	else{ // CPU just in case

	  float* img_prev = new float[ volumeImage ];
	  float* img_curr = new float[ volumeImage ];
	  cudaMemcpy( img_prev, d_image_prev, sizeof(float) * volumeImage, cudaMemcpyDeviceToHost ) ;
	  CUT_CHECK_ERROR("Memory copy for proj_cur failed");
	  cudaMemcpy( img_curr, d_image_curr, sizeof(float) * volumeImage, cudaMemcpyDeviceToHost ) ;
	  CUT_CHECK_ERROR("Memory copy for proj_cur failed");

	  float img_l1_cpu = 0.0, img_diff_cpu = 0.0;
	  for( int i = 0; i < volumeImage; i++ ){
	    img_diff_cpu += fabs( img_prev[ i] - img_curr[i] );
	    img_l1_cpu += fabs( img_prev[ i] );
	  }
	  delete [] img_prev;
	  delete [] img_curr;

	  diff_percent = img_diff_cpu / img_l1_cpu;
	  b_diff_thres = (diff_percent > diff_thres_percent);

	  cout << diff_percent << " vs " << diff_thres_percent << endl;
	}
      }
      if( stop_type == 4 ){

	if( 65535 * THREADS_MAX * 2 >= num_width * num_height * num_depth ){  // GPU
	  // 65535: max dimension of grid size
	  // THREADS_MAX: max number of threads per block
	  // Supports 16 slices of 2048 * 2048. Not 32. 

	  diff_percent = reduction2_wrapper( d_image_curr, d_image_prev, d_image_sub, 
					     num_width, num_height, num_depth) ;
	  b_diff_thres = (diff_percent > diff_thres_value);
	  cout << diff_percent << " vs " << diff_thres_value << endl;
	}
	else
	{ // CPU just in case
	  float* img_prev = new float[ volumeImage ];
	  float* img_curr = new float[ volumeImage ];
	  cudaMemcpy( img_prev, d_image_prev, sizeof(float) * volumeImage, cudaMemcpyDeviceToHost ) ;
	  CUT_CHECK_ERROR("Memory copy for proj_cur failed");
	  cudaMemcpy( img_curr, d_image_curr, sizeof(float) * volumeImage, cudaMemcpyDeviceToHost ) ;
	  CUT_CHECK_ERROR("Memory copy for proj_cur failed");

	  float img_diff_cpu = 0.0;
	  for( int i = 0; i < volumeImage; i++ ){
	    img_diff_cpu += fabs( img_prev[ i] - img_curr[i] );
	  }
	  delete [] img_prev;
	  delete [] img_curr;

	  diff_percent = img_diff_cpu;
	  b_diff_thres = (diff_percent > diff_thres_value);

	  cout << diff_percent << " vs " << diff_thres_value << endl;
	}
      }
    }

    // prepare for next iteration
    cudaMemcpy( d_image_prev, d_image_curr, sizeof(float) * volumeImage, cudaMemcpyDeviceToDevice );	
    CUT_CHECK_ERROR("Memory copy failed");

#ifdef OUTPUT_GPU_TIME

    if( deviceID == 0 )
    {
      CUT_SAFE_CALL(cutStopTimer(timerImageTrans));
      double time_tmp =  cutGetTimerValue(timerImageTrans); 
      time_ImgTrans += time_tmp;
      time_gpu += time_tmp;

      cout << endl;
      cout << "    Iter " << iter << ": GPU processing time for image transfer: "
	   << time_tmp << "  (ms) " << endl;

      CUT_SAFE_CALL(cutDeleteTimer(timerImageTrans));
    }
#endif

    // Perform TV3D on the result
#ifdef USE_TV    

#ifdef OUTPUT_GPU_TIME
    unsigned int timerTVCalGPU;
    if( deviceID == 0 )
    {
      CUT_SAFE_CALL(cutCreateTimer(&timerTVCalGPU));
      CUT_SAFE_CALL(cutStartTimer(timerTVCalGPU));
    }

#endif

    tv3d_cal_wrapper( d_image_prev, d_image_iszero, 
		      d_image_curr,  
		      num_width, num_height, num_depth, 
		      iter_TV, fidelity_weight_TV, epsilon_TV, timestep_TV);

#ifdef OUTPUT_GPU_TIME
    if( deviceID == 0 ){
      cudaThreadSynchronize(); 
      CUT_SAFE_CALL(cutStopTimer(timerTVCalGPU));
      time_gpu += cutGetTimerValue(timerTVCalGPU); 
    
      cout << endl << "    Iter " << iter << ": GPU Total Variation Regularization time: " 
	   << cutGetTimerValue(timerTVCalGPU) << " (ms) " << endl << endl; 
      CUT_SAFE_CALL(cutDeleteTimer(timerTVCalGPU));
    }
#endif

#endif //USE_TV
  
    // stop criteria by number of iterations
    if( stop_type == 1 ){
      b_iter_max = (iter < iter_max);
    }
    iter++; 

  }

#ifdef OUTPUT_GPU_TIME
  if( deviceID == 0 ){
    // stop timer
    CUT_SAFE_CALL(cutStopTimer(timerTotal));
    cout << "    EM3D GPU total processing time: " << cutGetTimerValue(timerTotal) << " ms " << endl;;
    CUT_SAFE_CALL(cutDeleteTimer(timerTotal));
  }
#endif

  cudaMemcpy( plan->h_output_data, d_image_curr, sizeof(float) * volumeImage, cudaMemcpyDeviceToHost );

#ifdef OUTPUT_GPU_TIME
  if( deviceID == 0 ){
    cout << "    Total time for projection preparation is " << time_ProjPrep << " ms " <<  endl;
    cout << "    Total time for projection calculation is " << time_ProjCal << " ms " << endl;
    cout << "    Total time for back projection preparation is " << time_BackprojPrep << " ms " << endl;
    cout << "    Total time for back projection calculation is " << time_BackprojCal << " ms " << endl;
    cout << "    Total time for image transfer is " << time_ImgTrans << " ms " << endl;

    cout << "    EM 3D GPU Processing time: " << time_gpu << " (ms) " << endl;
  }
#endif  

  cudaFree((void*)d_image_prev);        CUT_CHECK_ERROR("Memory free failed"); 
  cudaFree((void*)d_image_curr);        CUT_CHECK_ERROR("Memory free failed"); 
  cudaFree((void*)d_proj_cur);          CUT_CHECK_ERROR("Memory free failed");

  cudaFreeArray( d_array_voxel);        CUT_CHECK_ERROR("Memory free failed");
  cudaFreeArray( d_array_proj);         CUT_CHECK_ERROR("Memory free failed");

  cudaFree((void*)d_proj);              CUT_CHECK_ERROR("Memory free failed");

  cudaFree((void*)d_image_sub);        CUT_CHECK_ERROR("Memory free failed"); 
  cudaFree((void*)d_proj_sub);        CUT_CHECK_ERROR("Memory free failed"); 

  if( n_proj_mask_image == 1 ){
    cudaFree( d_proj_mask_data );
  }

#ifndef WEIGHT_CAL
  cudaFree((void*)d_image_iszero);      CUT_CHECK_ERROR("Memory free failed");
  delete[] image_org;
#endif

#ifdef OUTPUT_GPU_TIME
  if( deviceID == 0 ){
    CUT_SAFE_CALL(cutStopTimer(timerEM3D));
    cout << endl <<  "    EM3D_GPU Total time: " 
	 << cutGetTimerValue(timerEM3D) << " (ms) " << endl; 

    CUT_SAFE_CALL(cutDeleteTimer(timerEM3D));
  }
#endif 

}

int GetSharedMemSize(int dev){

  // get the size of shared memory for each GPU block in bytes
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  return deviceProp.sharedMemPerBlock;
}

void RingCorrectionSinogram (float *data, float ring_coeff, int um, int num_proj,
			   float *mean_vect, float* mean_sino_line_data,
			   float *low_pass_sino_lines_data) 
{ 
  // ring artifact removal from Brian Tieman's MPI cluster codes (RingCorrectionSingle)
  int         i, j, m; 
  float       mean_total; 
  float       tmp; 
 
  for (m=0;m<20;m++) {
    
    // normalization of each projection: mean values estimation 
    for (i=0;i<num_proj;i++) 
      mean_vect[i] = 0.0; 

    mean_total = 0.0; 
 
    for (i=0;i<num_proj;i++) {
        
      for (j=0;j<um;j++) {
	mean_vect[i] += data[i*um+j]; 
      }

      mean_vect[i] /= um; 
      mean_total += mean_vect[i]; 
    } 
    mean_total /= num_proj; 
 
    // renormalization of each projection to the global mean 
    for (i=0;i<num_proj;i++) {
      for (j=0;j<um;j++) {
	if (mean_vect[i] != 0.0) {
	  data[i*um+j] = data[i*um+j]*mean_total/mean_vect[i];        // ring filtering: sum of projection and low-pass filter of the result 
 
	}
      }
    }

    for (i=0;i<um;i++) 
      mean_sino_line_data[i] = 0.0; 
 
    for (i=0;i<num_proj;i++) 
      for (j=0;j<um;j++) 
	mean_sino_line_data[j] += data[i*um+j]; 
 
    for (i=0;i<um;i++) 
      mean_sino_line_data[i] /= num_proj; 
 
    for (j=1;j<um-1;j++) {
      low_pass_sino_lines_data[j] = (mean_sino_line_data[j-1]+mean_sino_line_data[j]+mean_sino_line_data[j+1])/3.0; 
    }

    low_pass_sino_lines_data[0] = mean_sino_line_data[0]; 
    low_pass_sino_lines_data[um-1] = mean_sino_line_data[um-1]; 
 
    // ring corrections 
    for (i=0;i<num_proj;i++) {
      for (j=0;j<um;j++) { 
	tmp = mean_sino_line_data[j]-low_pass_sino_lines_data[j]; 
	if ((data[i*um+j] - (tmp * ring_coeff) ) > 0.0) 
	  data[i*um+j] -= (tmp * ring_coeff); 
	else 
	  data[i*um+j] = 0.0; 
      } 
    } 
  }
 
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
