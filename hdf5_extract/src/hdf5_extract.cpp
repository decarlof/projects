// extract reconstructed hdf5 slices

#include <iostream>
#include "hdf5.h"

#include "itkImage.h"
#include "itkImageFileWriter.h"

typedef unsigned short             HDF5_DATATYPE;
typedef itk::Image<float, 3> VolumeType;

extern "C"
void Hdf5GetDims(char*, char*, int*, int *, int * );

extern "C"
void Hdf5SerialReadY(char* , char* , int , int , HDF5_DATATYPE * );

extern "C"
void Hdf5SerialReadZ(char* , char* , int , int , HDF5_DATATYPE * );

extern "C"                                                                               
void Hdf5SerialReadY_f(char* filename, char* datasetname, int nZSliceStart, int numZSlice,            
                       float * data);        

extern "C"                                                                               
void Hdf5SerialReadZ_f(char* filename, char* datasetname, int nZSliceStart, int numZSlice,            
                       float * data);                                                     
     

using std::cout;
using std::endl;

int main(int argc, char** argv){

  if( argc != 7 && argc != 6 ){
    cout << "Usage: hdf5_extract filename  0(Y)/1(Z) 0(uint16)/1(float32) SliceStart numSlice" << endl;
    cout << "Usage: the default dataset is /entry/exchange/dataRecon" << endl;
    cout << "Usage: hdf5_extract filename  0(Y)/1(Z) 0(uint16)/1(float32) SliceStart numSlice datasetname" << endl;
    exit(1); 
  }

  int nSliceType = atoi( argv[2] );
  int nDatasetType = atoi( argv[3] );
  int nSliceStart = atoi( argv[4] );
  int nSliceNum = atoi( argv[5] );

  int xm, ym, zm; 

  if( argc == 6 )
    Hdf5GetDims( argv[1], "/entry/exchange/dataRecon", &xm, &ym, &zm );
  else if ( argc == 7 )
    Hdf5GetDims( argv[1], argv[6], &xm, &ym, &zm );

  if( nSliceType == 0 && nSliceStart >= ym ){
    cout << "The y range for the dataset is [ 0, " << ym - 1 << " ] " << endl;
  }

  if( nSliceType == 0 && nSliceStart < ym && nSliceStart + nSliceNum - 1 >= ym ){
      cout << "The specified range is larger than ym = " << ym - 1 << ". Use [ " << nSliceStart << " , " << ym -1 << " ] instead" << endl;
      nSliceNum = ym - nSliceStart; 
  }

  if( nSliceType == 1 && nSliceStart >= zm ){
    cout << "The z range for the dataset is [ 0, " << zm - 1 << " ] " << endl;
  }
    
  if( nSliceType == 1 && nSliceStart < zm && nSliceStart + nSliceNum - 1 >= zm ){
    cout << "The specified range is larger than zm = " << zm - 1 << ". Use [ " << nSliceStart << " , " << zm -1 << " ] instead" << endl;
    nSliceNum = zm - nSliceStart; 
  }


  HDF5_DATATYPE * data = NULL; // allocated in Hdf5SerialRead.c and freed here, because of unknown size
  float* data_float = NULL;

  if( nSliceType == 0 && nDatasetType == 0 )
    data = new HDF5_DATATYPE[ xm * nSliceNum * zm ];
  else if( nSliceType == 1  && nDatasetType == 0 )
    data = new HDF5_DATATYPE[ xm * ym * nSliceNum ] ;
  else if( nSliceType == 0 && nDatasetType == 1 )
    data_float = new float[ xm * nSliceNum * zm ];
  else if( nSliceType == 1  && nDatasetType == 1 )
    data_float = new float[ xm * ym * nSliceNum ] ;

  if( ( !data && nDatasetType == 0 ) || ( !data_float && nDatasetType == 1 )){
    cout << "Error allocating memory for data and data_float" << endl;
    exit( 1 );
  }

  if( argc == 6 && nSliceType == 0 && nDatasetType == 0 ){
    Hdf5SerialReadY( argv[1], "/entry/exchange/dataRecon", nSliceStart, nSliceNum,  
		     data);
  }
  else if ( argc == 6 && nSliceType == 1 && nDatasetType == 0 ){
    Hdf5SerialReadZ( argv[1], "/entry/exchange/dataRecon", nSliceStart, nSliceNum,  
		     data);
  }
  else if ( argc == 7 && nSliceType == 0 && nDatasetType == 0 ){
    Hdf5SerialReadY( argv[1], argv[6], nSliceStart, nSliceNum,  
		     data);

  }
  else if ( argc == 7 && nSliceType == 1 && nDatasetType == 0 ){
    Hdf5SerialReadZ( argv[1], argv[6], nSliceStart, nSliceNum,  
		     data);
  }

  // 
  if( argc == 6 && nSliceType == 0 && nDatasetType == 1 ){
    Hdf5SerialReadY_f( argv[1], "/entry/exchange/dataRecon", nSliceStart, nSliceNum,  
		     data_float);
  }
  else if ( argc == 6 && nSliceType == 1 && nDatasetType == 1 ){
    Hdf5SerialReadZ_f( argv[1], "/entry/exchange/dataRecon", nSliceStart, nSliceNum,  
		     data_float);
  }
  else if ( argc == 7 && nSliceType == 0 && nDatasetType == 1 ){
    Hdf5SerialReadY_f( argv[1], argv[6], nSliceStart, nSliceNum,  
		     data_float);

  }
  else if ( argc == 7 && nSliceType == 1 && nDatasetType == 1 ){
    Hdf5SerialReadZ_f( argv[1], argv[6], nSliceStart, nSliceNum,  
		     data_float);
  }


  // output extracted slices as a nrrd file for inspection/visualization
  VolumeType::Pointer SlicePointer = VolumeType::New(); 
  VolumeType::SizeType  size;
  VolumeType::RegionType region;

  if( nSliceType == 0 ){
    size[ 0 ] = xm;
    size[ 1 ] = nSliceNum;
    size[ 2 ] = zm;
  }
  else if ( nSliceType == 1 ){
    size[ 0 ] = xm;
    size[ 1 ] = ym;
    size[ 2 ] = nSliceNum;
  }

  region.SetSize( size );
  SlicePointer->SetRegions( region );
  SlicePointer->Allocate();

  VolumeType::IndexType index;
  VolumeType::PixelType pixel;

  for( int nz = 0; nz < size[2]; nz++ ){
    for( int ny = 0; ny < size[1]; ny++ ){
      for( int nx = 0; nx < size[0]; nx++ ){

	index[ 0 ] = nx;
	index[ 1 ] = ny;
	index[ 2 ] = nz;

	if(  nDatasetType == 0 )
	  pixel = data[ (nz * size[1] + ny) * size[0] + nx ];
	else if ( nDatasetType == 1 )
	  pixel = data_float[ (nz * size[1] + ny) * size[0] + nx ];

	SlicePointer->SetPixel( index, pixel );	
      }
    }
  }

  itk::ImageFileWriter<VolumeType>::Pointer ImageReconWriter;
  ImageReconWriter = itk::ImageFileWriter<VolumeType>::New();
  ImageReconWriter->SetInput( SlicePointer );
  ImageReconWriter->SetFileName( "SliceExtract.nrrd" );
  ImageReconWriter->Update();  

  if( nDatasetType == 0)
    delete [] data; 
  if( nDatasetType == 1)
    delete [] data_float; 
}
