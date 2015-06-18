

#ifndef __ITK_GENERETE_PROJ_H__
#define __ITK_GENERETE_PROJ_H__

// itkGenerateParallelProjection2DImageFilter.h
#include <list>

typedef struct{
  int x;            // corresponds to the location of image samples
  int y;
  int z;
  double wa;        // corresponds to a_{ij}
  double wt;        // corresponds to t_{ij}
} Sample;

// structure for SART3D GPU implementation
typedef struct{
  int x;            // corresponds to the location of image samples
  int y;
  int z;
  float wa;        // corresponds to a_{ij}
  float wt;        // corresponds to t_{ij}
} WeightSample;

// information from the projection of a ray i
typedef struct {
  double Proj;                // projection value: p_i
  int    M;                   // number of sampling points: M_i 
  double L;                   // physical length of the ray inside the object: L_i = sum_{j=1}^{N}{ a_{ij} }
  std::list<Sample>  listSample;  // list of Sample structure: { j, a_{ij} } of voxels
}Ray;

typedef struct {
  double W;                       // the sum of voxel weights to all rays. W = sum_{i=1}^{N}{ a_{ij} }
  std::list<Sample>  listSample;  // list of Sample structure: { j, a_{ij} } of rays
}Voxel;


template <class InputImageType>
void GenerateProjection2DImageFilter( InputImageType*, Ray**, double, int, int); 

template <class InputImageType>
void GenerateProjection3DImageFilter( InputImageType*, Ray**, Voxel**, double, int, int, int); 


template <class InputImageType>
void UpdateRay( InputImageType *, Ray**, double, double, int, int, double, double, double*, double* );

template <class OutputImageType>
void SART2D(OutputImageType*, OutputImageType*, Ray**, OutputImageType*, double, double, int, int, int);

template <class OutputImageType>
void SART3D(OutputImageType*, OutputImageType*, Ray**, OutputImageType*, double, double, int, int, int, int);

template <class OutputImageType>
void SART3D_GPU(OutputImageType *, float*, OutputImageType *, Ray**, Voxel**, OutputImageType *, double, double, int, int, int, int, float, float, float, int , int );


template <class OutputImageType>
void SART3D_GPU_Weight(OutputImageType *, float*, OutputImageType *, Ray** , Voxel **, OutputImageType * , 
		       double, double , int , int , int , int );
  
template <class OutputImageType>
void FilteredBackProjectionParallelBeam2D(OutputImageType *, double**, int, int);

template <class OutputImageType>
void FilteredBackProjectionFanBeam2D(OutputImageType *, double**, int, int);

template <class OutputImageType>
void FilteredBackProjectionConeBeam3D(OutputImageType *, double**, int, int);


template <class OutputImageType>
void FilteredBackProjectionConeBeam3D(OutputImageType *, double**, int, int, int);

void  FBPConeBeam3D( double**, double**, int, int, int, int, int, int   ); 


#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8
#define BLOCK_DIM_Z 4

#endif   // #ifndef __ITK_GENERETE_PROJ_H__
