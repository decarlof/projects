#include "stdio.h"
#include "hdf5.h"
#include <mpi.h>
#include "tomo_recon.h"

// Read data in parallel from projection data file (.hdf5). Data type: unsighed short (16 bit)
// Be sure to update H5T_STD_U16LE (unsigned short) when HDF5_DATATYPE changes

// #define TEST_2D

void Hdf5ParallelReadZ(const char* filename, const char* datasetname, int nZSliceStart, int numZSlice, int commRank,
		     HDF5_DATATYPE * data){

    hid_t       file_id, dset_id;         /* file and dataset identifiers */
    hid_t	plist_id;                 /* property list identifier */

/* #ifdef TEST_2D */
/*     HDF5_DATATYPE data_2d[ 1501 ][2048 ]; */
/* #endif */

    // Set up file access property list with parallel I/O access

    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    // open the file
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, plist_id);
    H5Pclose( plist_id );

    // open the dataset
    dset_id = H5Dopen2(file_id, datasetname, H5P_DEFAULT);

    // Create property list for collective dataset write.
    plist_id = H5Pcreate(H5P_DATASET_XFER);

    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    // 
    hid_t dataspace = H5Dget_space( dset_id );  
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

    offset[0] = nZSliceStart + commRank * numZSlice;
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

/* #ifdef TEST_2D */
/*     status = H5Dread(dset_id, H5T_STD_U16LE, memspace, dataspace, plist_id, data_2d); */
/* #else */
    status = H5Dread(dset_id, H5T_STD_U16LE, memspace, dataspace, plist_id, data);
/* #endif */

    H5Dclose( dset_id );
    H5Sclose( dataspace );
    H5Sclose( memspace );
    H5Pclose( plist_id );
    H5Fclose( file_id );

}

void Hdf5ParallelReadZ_f(const char* filename, const char* datasetname, int nZSliceStart, int numZSlice, int commRank,
			 float * data){

    hid_t       file_id, dset_id;         /* file and dataset identifiers */
    hid_t	plist_id;                 /* property list identifier */

/* #ifdef TEST_2D */
/*     HDF5_DATATYPE data_2d[ 1501 ][2048 ]; */
/* #endif */

    // Set up file access property list with parallel I/O access

    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    // open the file
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, plist_id);
    H5Pclose( plist_id );

    // open the dataset
    dset_id = H5Dopen2(file_id, datasetname, H5P_DEFAULT);

    // Create property list for collective dataset write.
    plist_id = H5Pcreate(H5P_DATASET_XFER);

    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    // 
    hid_t dataspace = H5Dget_space( dset_id );  
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

    offset[0] = nZSliceStart + commRank * numZSlice;
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

/* #ifdef TEST_2D */
/*     status = H5Dread(dset_id, H5T_STD_U16LE, memspace, dataspace, plist_id, data_2d); */
/* #else */
    status = H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, dataspace, plist_id, data);
/* #endif */

    H5Dclose( dset_id );
    H5Sclose( dataspace );
    H5Sclose( memspace );
    H5Pclose( plist_id );
    H5Fclose( file_id );

}


void Hdf5ParallelReadY(char* filename, char* datasetname, int nYSliceStart, int numYSlice,  int commRank,
		     HDF5_DATATYPE * data){


    hid_t       file_id, dset_id;         /* file and dataset identifiers */
    hid_t	plist_id;                 /* property list identifier */

    // Set up file access property list with parallel I/O access
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    // H5Pset_fapl_mpiposix(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);   // Use posix driver. Not a collective function. Even slower. 

    // open the file
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, plist_id);
    H5Pclose( plist_id );

    // open the dataset
    dset_id = H5Dopen2(file_id, datasetname, H5P_DEFAULT);

    // Create property list for collective dataset write.
    plist_id = H5Pcreate(H5P_DATASET_XFER);

    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    /* H5Pset_dxpl_mpio_collective_opt(plist_id, H5FD_MPIO_INDIVIDUAL_IO); */

    hid_t dataspace = H5Dget_space( dset_id );
  
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
    offset[1] = nYSliceStart + commRank * numYSlice;
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

    // status = H5Dread(dset_id, H5T_STD_U16LE, memspace, dataspace, H5P_DEFAULT, data);
    status = H5Dread(dset_id, H5T_STD_U16LE, memspace, dataspace, plist_id, data);

    H5Dclose( dset_id );
    H5Sclose( dataspace );
    H5Sclose( memspace );
    H5Pclose( plist_id );
    H5Fclose( file_id );
}
