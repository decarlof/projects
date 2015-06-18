#include "io.h"

//
// create empty ITK volume
//
/*
SolutionImagePointer
createSolutionVolume(int x, int y, int z)
{
	SolutionImagePointer vol;

	// set size & region
	SolutionImageType::IndexType start;
	start[0] = 0; start[1] = 0; start[2] = 0;
	SolutionImageType::SizeType size;
	size[0] = x; size[1] = y; size[2] = z;
	SolutionImageType::RegionType region;
	region.SetSize( size );	region.SetIndex( start );

	// create
	vol = SolutionImageType::New();
	vol->SetRegions( region );
	vol->Allocate();

	return (vol);
}

CharacteristicImagePointer
createCharVolume(int x, int y, int z)
{
	CharacteristicImagePointer vol;

	// set size & region
	CharacteristicImageType::IndexType start;
	start[0] = 0; start[1] = 0; start[2] = 0;
	CharacteristicImageType::SizeType size;
	size[0] = x; size[1] = y; size[2] = z;
	CharacteristicImageType::RegionType region;
	region.SetSize( size );	region.SetIndex( start );

	// create
	vol = CharacteristicImageType::New();
	vol->SetRegions( region );
	vol->Allocate();

	return (vol);
}
*/

/*-------------------------------------------------------------------
 * Read a Nrrd File of Diffusion Tensor Data
 *-------------------------------------------------------------------*/
TensorImagePointer readTensorNrrdImage( 
		const char* fileName,
		int* volWidth,
		int* volHeight, 
		int* volDepth)
{
	itk::ImageFileReader<TensorImageType>::Pointer reader	= itk::ImageFileReader<TensorImageType>::New();
	reader->SetFileName(fileName);
	try
	{
		reader->Update();
	}
	catch (itk::ExceptionObject & e)
	{
		std::cerr << "exception in file reader " << std::endl;
		std::cerr << e.GetDescription() << std::endl;
		std::cerr << e.GetLocation() << std::endl;
		exit(-1);
	}

	TensorImagePointer image = (reader->GetOutput());
	TensorImageType::SizeType imageSize =	
						image->GetLargestPossibleRegion().GetSize();

	(*volWidth)=imageSize[0];
	(*volHeight)=imageSize[1];
	(*volDepth)=imageSize[2];

	//std::cout << "width:" << imageSize[0] << " height:" << imageSize[1] << " Depth:" << imageSize[2]
	//<< std::endl;
	
	return(image);
}


/*-------------------------------------------------------------------
 * Write a Nrrd File of Diffusion Tensor Data
 *-------------------------------------------------------------------*/
void writeTensorNrrdImage( 
		const char* fileName,
		TensorImagePointer image)
{
	itk::ImageFileWriter<TensorImageType>::Pointer writer	= itk::ImageFileWriter<TensorImageType>::New();
	writer->SetFileName(fileName);
	writer->SetInput(image);

	try
	{
		writer->Update();
	}
	catch (itk::ExceptionObject & e)
	{
		std::cerr << "exception in file writer " << std::endl;
		std::cerr << e.GetDescription() << std::endl;
		std::cerr << e.GetLocation() << std::endl;
		exit(-1);
	}

}

/*-------------------------------------------------------------------
 * Read a Scale File 
 *-------------------------------------------------------------------*/
ScalarImagePointer readScalarNrrdImage( 
		const char* fileName,
		int* volWidth,
		int* volHeight, 
		int* volDepth)
{
	itk::ImageFileReader<ScalarImageType>::Pointer reader = 
					itk::ImageFileReader<ScalarImageType>::New();
	reader->SetFileName(fileName);
	try
	{
		reader->Update();
	}
	catch (itk::ExceptionObject & e)
	{
		std::cerr << "exception in file reader " << std::endl;
		std::cerr << e.GetDescription() << std::endl;
		std::cerr << e.GetLocation() << std::endl;
		exit(-1);
	}

	ScalarImagePointer image = (reader->GetOutput());
	ScalarImageType::SizeType imageSize =	
						image->GetLargestPossibleRegion().GetSize();

	(*volWidth)=imageSize[0];
	(*volHeight)=imageSize[1];
	(*volDepth)=imageSize[2];

	//std::cout << "width:" << imageSize[0] << " height:" << imageSize[1] << " Depth:" << imageSize[2]
	//<< std::endl;
	
	return(image);
}

/*-------------------------------------------------------------------
 * Read a Scale File 
 *-------------------------------------------------------------------*/
ScalarImagePointer readScalarNrrdImage( const char* fileName )
{
	itk::ImageFileReader<ScalarImageType>::Pointer reader = 
					itk::ImageFileReader<ScalarImageType>::New();
	reader->SetFileName(fileName);
	try
	{
		reader->Update();
	}
	catch (itk::ExceptionObject & e)
	{
		std::cerr << "exception in file reader " << std::endl;
		std::cerr << e.GetDescription() << std::endl;
		std::cerr << e.GetLocation() << std::endl;
		exit(-1);
	}

	ScalarImagePointer image = (reader->GetOutput());
	ScalarImageType::SizeType imageSize =	
						image->GetLargestPossibleRegion().GetSize();
	
	return(image);
}


/*---------------------------------------------------------------------------
 * Write solution in NRRD format
/*---------------------------------------------------------------------------*/
void writeScalarImage(const char *filename,  ScalarImagePointer image )
{
	std::cout << "Save Volume : " << filename << std::endl;
	itk::ImageFileWriter<ScalarImageType>::Pointer writer;
	writer = itk::ImageFileWriter<ScalarImageType>::New();
	writer->SetInput( image );
	writer->SetFileName( filename );

	try
	{
		writer->Update();
	}
	catch (itk::ExceptionObject & e)
	{
		std::cerr << "exception in file writer " << std::endl;
		std::cerr << e.GetDescription() << std::endl;
		std::cerr << e.GetLocation() << std::endl;
		exit(-1);
	}

}

VectorImagePointer readVectorNrrdImage( 
		const char* fileName,
		int* volWidth,
		int* volHeight, 
		int* volDepth)
{
	itk::ImageFileReader<VectorImageType>::Pointer reader = 
					itk::ImageFileReader<VectorImageType>::New();
	reader->SetFileName(fileName);
	try
	{
		reader->Update();
	}
	catch (itk::ExceptionObject & e)
	{
		std::cerr << "exception in file reader " << std::endl;
		std::cerr << e.GetDescription() << std::endl;
		std::cerr << e.GetLocation() << std::endl;
		exit(-1);
	}

	VectorImagePointer image = (reader->GetOutput());
	VectorImageType::SizeType imageSize =	
						image->GetLargestPossibleRegion().GetSize();

	(*volWidth)=imageSize[0];
	(*volHeight)=imageSize[1];
	(*volDepth)=imageSize[2];

	//std::cout << "width:" << imageSize[0] << " height:" << imageSize[1] << " Depth:" << imageSize[2]
	//<< std::endl;
	
	return(image);
}


void writeVectorImage(char *filename, VectorImagePointer image )
{
	std::cout << "Save characterictic Vector : " << filename << std::endl;
	itk::ImageFileWriter<VectorImageType>::Pointer writer;
	writer = itk::ImageFileWriter<VectorImageType>::New();
	writer->SetInput( image );
	writer->SetFileName( filename );

	try
	{
		writer->Update();
	}
	catch (itk::ExceptionObject & e)
	{
		std::cerr << "exception in file writer " << std::endl;
		std::cerr << e.GetDescription() << std::endl;
		std::cerr << e.GetLocation() << std::endl;

		exit(-1);
	}
}

MaskImagePointer readMaskNrrdImage( 
		const char* fileName,
		int* volWidth,
		int* volHeight, 
		int* volDepth)
{
	itk::ImageFileReader<MaskImageType>::Pointer reader = 
					itk::ImageFileReader<MaskImageType>::New();
	reader->SetFileName(fileName);
	try
	{
		reader->Update();
	}
	catch (itk::ExceptionObject & e)
	{
		std::cerr << "exception in file reader " << std::endl;
		std::cerr << e.GetDescription() << std::endl;
		std::cerr << e.GetLocation() << std::endl;
		exit(-1);
	}

	MaskImagePointer image = (reader->GetOutput());
	MaskImageType::SizeType imageSize =	
						image->GetLargestPossibleRegion().GetSize();

	(*volWidth)=imageSize[0];
	(*volHeight)=imageSize[1];
	(*volDepth)=imageSize[2];

	//std::cout << "width:" << imageSize[0] << " height:" << imageSize[1] << " Depth:" << imageSize[2]
	//<< std::endl;
	
	return(image);
}

void writeMaskImage(char *filename, MaskImagePointer image )
{
	std::cout << "Save Mask : " << filename << std::endl;
	itk::ImageFileWriter<MaskImageType>::Pointer writer;
	writer = itk::ImageFileWriter<MaskImageType>::New();
	writer->SetInput( image );
	writer->SetFileName( filename );

	try
	{
		writer->Update();
	}
	catch (itk::ExceptionObject & e)
	{
		std::cerr << "exception in file writer " << std::endl;
		std::cerr << e.GetDescription() << std::endl;
		std::cerr << e.GetLocation() << std::endl;

		exit(-1);
	}
}

//----------------------------------------------------------------------------------------------------------

SeedImagePointer readSeedNrrdImage( 
		const char* fileName,
		int* volWidth,
		int* volHeight, 
		int* volDepth)
{
	itk::ImageFileReader<SeedImageType>::Pointer reader = 
					itk::ImageFileReader<SeedImageType>::New();
	reader->SetFileName(fileName);
	try
	{
		reader->Update();
	}
	catch (itk::ExceptionObject & e)
	{
		std::cerr << "exception in file reader " << std::endl;
		std::cerr << e.GetDescription() << std::endl;
		std::cerr << e.GetLocation() << std::endl;
		exit(-1);
	}

	SeedImagePointer image = (reader->GetOutput());
	SeedImageType::SizeType imageSize =	
						image->GetLargestPossibleRegion().GetSize();

	(*volWidth)=imageSize[0];
	(*volHeight)=imageSize[1];
	(*volDepth)=imageSize[2];

	//std::cout << "width:" << imageSize[0] << " height:" << imageSize[1] << " Depth:" << imageSize[2]
	//<< std::endl;
	
	return(image);
}

