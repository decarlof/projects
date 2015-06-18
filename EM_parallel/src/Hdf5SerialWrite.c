#include "stdio.h"
#include "hdf5.h"

// Write reconstructed data (.hdf5)
void Hdf5SerialWrite( const char* filename, const char* datasetname, int xm, int ym, int zm, 
                       float* data ) {


  hid_t file, dataset, datatype, dataspace;           // handles
  herr_t status;
  
  file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  int rank = 3;

  hsize_t dims_dataset[3];
  dims_dataset[ 0 ] = zm;
  dims_dataset[ 1 ] = ym;
  dims_dataset[ 2 ] = xm;

  dataspace = H5Screate_simple( rank, dims_dataset, NULL );

  datatype = H5Tcopy( H5T_NATIVE_FLOAT);

  dataset = H5Dcreate2( file, datasetname, datatype, dataspace, 
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );

  status = H5Dwrite( dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data );

  H5Sclose( dataspace );
  H5Tclose( datatype );
  H5Dclose( dataset );
  H5Fclose( file );
}
