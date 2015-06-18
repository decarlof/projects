#include "itkDiffusionTensor3D.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkNrrdImageIO.h"
#include "itkImage.h"
#include "itkVector.h"


// itk image type definition for anisotropic eikonal solvers

typedef  itk::DiffusionTensor3D<double> DiffusionTensorPixelType; 
typedef  itk::Image<DiffusionTensorPixelType, 3> TensorImageType; // diffusion tensor image type
typedef  itk::Image<unsigned int, 3> SeedImageType; // seed image type

typedef itk::Vector<double, 3> VectorPixelType;
typedef itk::Image<VectorPixelType, 3> VectorImageType; // characteristic image type
typedef itk::Image<double, 3> ScalarImageType; // solution image type

typedef unsigned int MaskPixelType;
typedef itk::Image<MaskPixelType,3> MaskImageType;
typedef itk::ImageFileReader<MaskImageType> MaskImageReaderType;

typedef TensorImageType::Pointer TensorImagePointer;
typedef SeedImageType::Pointer SeedImagePointer;
typedef VectorImageType::Pointer VectorImagePointer;
typedef ScalarImageType::Pointer ScalarImagePointer;
typedef MaskImageType::Pointer MaskImagePointer;

//SolutionImagePointer createSolutionVolume(int x, int y, int z);
//CharacteristicImagePointer createCharVolume(int x, int y, int z);

TensorImagePointer readTensorNrrdImage( const char* fileName, int* volWidth, int* volHeight, int* volDepth);
void writeTensorNrrdImage( const char* fileName, TensorImagePointer image);

ScalarImagePointer readScalarNrrdImage( const char* fileName, int* volWidth, int* volHeight, int* volDepth);
ScalarImagePointer readScalarNrrdImage( const char* fileName);
void writeScalarImage(const char *filename, ScalarImagePointer image );

VectorImagePointer readVectorNrrdImage( const char* fileName, int* volWidth, int* volHeight, int* volDepth);
void writeVectorImage(const char *filename, VectorImagePointer image );

MaskImagePointer readMaskNrrdImage( const char* fileName, int* volWidth, int* volHeight, int* volDepth);
void writeMaskImage(char *filename, MaskImagePointer image );

SeedImagePointer readSeedNrrdImage( const char* fileName, int* volWidth, int* volHeight, int* volDepth);
