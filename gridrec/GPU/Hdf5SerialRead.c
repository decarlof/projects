#include "stdio.h"
#include "hdf5.h"

#include "tomo_recon.h"

// Read data sequentially from projection data file (.hdf5). Data type: unsighed short (16 bit)
// Be sure to update H5T_STD_U16LE (unsigned short) when HDF5_DATATYPE changes

void Hdf5SerialReadZ(const char* filename, const char* datasetname, int nZSliceStart, int numZSlice,  
		     HDF5_DATATYPE * data){

  // Read *.hdf5 in the z direction (of size um * vm)

  hid_t file, dataset, dataspace;           // handles

  file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  dataset = H5Dopen2(file, datasetname, H5P_DEFAULT);
  dataspace = H5Dget_space( dataset );
  
  int rank = H5Sget_simple_extent_ndims( dataspace );
  if( rank != 3 ){
    printf("Only rank 3 is supported!\n");
    exit( 1 );
  }

  hsize_t dims_dataset[3];
  H5Sget_simple_extent_dims( dataspace, dims_dataset, NULL );

  if( nZSliceStart + numZSlice > dims_dataset[0] ){ // check the dimension
    printf("Reading slice out of bound!\n");
    exit (1); 
  }
    
  // Define hyperslab in the dataset
  hsize_t count[3];
  hsize_t offset[3];

  offset[0] = nZSliceStart;
  offset[1] = 0;
  offset[2] = 0;

  count[0] = numZSlice;
  count[1] = dims_dataset[1];
  count[2] = dims_dataset[2];

  herr_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);

  // define the memory dataspace
  hsize_t dimsm[3];

  dimsm[0] = numZSlice;
  dimsm[1] = dims_dataset[1];
  dimsm[2] = dims_dataset[2];

  hid_t memspace = H5Screate_simple( 3, dimsm, NULL );
  
  hsize_t count_out[3];
  hsize_t offset_out[3];

  offset_out[0] = 0;
  offset_out[1] = 0;
  offset_out[2] = 0;

  count_out[0] = numZSlice;
  count_out[1] = dims_dataset[1];
  count_out[2] = dims_dataset[2];

  status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL,
			      count_out, NULL);

  status = H5Dread(dataset, H5T_STD_U16LE, memspace, dataspace, H5P_DEFAULT, data);

  H5Dclose( dataset );
  H5Sclose( dataspace );
  H5Sclose( memspace );
  H5Fclose( file );
}

void Hdf5SerialReadZ_f(const char* filename, const char* datasetname, int nZSliceStart, int numZSlice,  
		       float * data){

  // Read *.hdf5 in the z direction (of size um * vm)

  hid_t file, dataset, dataspace;           // handles

  file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  dataset = H5Dopen2(file, datasetname, H5P_DEFAULT);
  dataspace = H5Dget_space( dataset );
  
  int rank = H5Sget_simple_extent_ndims( dataspace );
  if( rank != 3 ){
    printf("Only rank 3 is supported!\n");
    exit( 1 );
  }

  hsize_t dims_dataset[3];
  H5Sget_simple_extent_dims( dataspace, dims_dataset, NULL );

  if( nZSliceStart + numZSlice > dims_dataset[0] ){ // check the dimension
    printf("Reading slice out of bound!\n");
    exit (1); 
  }
    
  // Define hyperslab in the dataset
  hsize_t count[3];
  hsize_t offset[3];

  offset[0] = nZSliceStart;
  offset[1] = 0;
  offset[2] = 0;

  count[0] = numZSlice;
  count[1] = dims_dataset[1];
  count[2] = dims_dataset[2];

  herr_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);

  // define the memory dataspace
  hsize_t dimsm[3];

  dimsm[0] = numZSlice;
  dimsm[1] = dims_dataset[1];
  dimsm[2] = dims_dataset[2];

  hid_t memspace = H5Screate_simple( 3, dimsm, NULL );
  
  hsize_t count_out[3];
  hsize_t offset_out[3];

  offset_out[0] = 0;
  offset_out[1] = 0;
  offset_out[2] = 0;

  count_out[0] = numZSlice;
  count_out[1] = dims_dataset[1];
  count_out[2] = dims_dataset[2];

  status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL,
			      count_out, NULL);

  status = H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, data);

  H5Dclose( dataset );
  H5Sclose( dataspace );
  H5Sclose( memspace );
  H5Fclose( file );
}


void Hdf5SerialReadY(char* filename, char* datasetname, int nYSliceStart, int numYSlice,  
		     HDF5_DATATYPE * data){

  // Read *.hdf5 in the Y direction (of size um * num_proj)

  hid_t file, dataset, dataspace;           // handles

  file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  dataset = H5Dopen2(file, datasetname, H5P_DEFAULT);
  dataspace = H5Dget_space( dataset );
  
  int rank = H5Sget_simple_extent_ndims( dataspace );
  if( rank != 3 ){
    printf("Only rank 3 is supported!\n");
    exit( 1 );
  }

  hsize_t dims_dataset[3];
  H5Sget_simple_extent_dims( dataspace, dims_dataset, NULL );

  if( nYSliceStart + numYSlice > dims_dataset[1] ){ // check the dimension
    printf("Reading slice out of bound!\n");
    exit (1); 
  }
    
  // Define hyperslab in the dataset
  hsize_t count[3];
  hsize_t offset[3];

  offset[0] = 0;
  offset[1] = nYSliceStart;
  offset[2] = 0;

  count[0] = dims_dataset[0];
  count[1] = numYSlice;
  count[2] = dims_dataset[2];

  herr_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);

  // define the memory dataspace
  hsize_t dimsm[3];

  dimsm[0] = dims_dataset[0];
  dimsm[1] = numYSlice;
  dimsm[2] = dims_dataset[2];

  hid_t memspace = H5Screate_simple( 3, dimsm, NULL );
  
  hsize_t count_out[3];
  hsize_t offset_out[3];

  offset_out[0] = 0;
  offset_out[1] = 0;
  offset_out[2] = 0;

  count_out[0] = dims_dataset[0];
  count_out[1] = numYSlice;
  count_out[2] = dims_dataset[2];

  status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL,
			      count_out, NULL);

  status = H5Dread(dataset, H5T_STD_U16LE, memspace, dataspace, H5P_DEFAULT, data);

  H5Dclose( dataset );
  H5Sclose( dataspace );
  H5Sclose( memspace );
  H5Fclose( file );
}

