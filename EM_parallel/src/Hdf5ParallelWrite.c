#include "stdio.h"
#include "hdf5.h"
#include <mpi.h>
// #include "tomo_recon.h"

// create a new dataset for parallel writing 
void Hdf5ParallelReconDataSet(const char* filename, char* datasetname, int nWidth, int nHeight, int nDepth ){

  /* // Set up file access property list with parallel I/O access */
  /* hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS); */
  /* H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL); */

  /* // open the file collectively and release property list identifier. */
  /* hid_t file_id = H5Fopen(filename, H5F_ACC_RDWR, plist_id); */
  /* H5Pclose(plist_id); */

  /* // Check if the dataset exists */
  /* if( H5Lexists( file_id, datasetname, H5P_DEFAULT ) ){ */

  /*   // check the dimensions if it exists */
  /*   hid_t ds_id = H5Dopen( file_id, datasetname);  */
  /*   hid_t dataspace = H5Dget_space( ds_id ); */

  /*   int rank = H5Sget_simple_extent_ndims( dataspace ); */
  /*   if( rank != 3 ){ */
  /*     printf(" data set exists but the rank is not 3!\n"); */
  /*     exit( 1 ); */
  /*   } */

  /*   hsize_t dims_dataset[3]; */
  /*   H5Sget_simple_extent_dims( dataspace, dims_dataset, NULL ); */

  /*   if( dims_dataset[ 0 ] != nDepth ||  dims_dataset[ 1 ] != nHeight ||  dims_dataset[ 2 ] != nWidth ){ */
  /*     printf(" The size of existing dataset does not match!\n"); */
  /*     exit( 1 );  */
  /*   } */

  /*   H5Dclose(ds_id);  */
  /* } */
  /* else{ */
  /*   // Create the dataspace for the dataset if it does not exist yet. */
  /*   hsize_t dim[3]; */
  /*   dim[0] = nDepth; */
  /*   dim[1] = nHeight; */
  /*   dim[2] = nWidth; */
  /*   hid_t filespace = H5Screate_simple(3, dim, NULL);  */

  /*   // Create the dataset with default properties and close filespace. */
  /*   hid_t dset_id = H5Dcreate(file_id, datasetname, H5T_NATIVE_FLOAT, filespace, */
  /* 			      H5P_DEFAULT); */
  
  /*   // Close/release resources. */
  /*   H5Dclose(dset_id); */
  /*   H5Sclose(filespace); */
  /* } */

  /* H5Fclose(file_id); */
}

void Hdf5SerialWriteZ(const char* filename, const char* datasetname, int nZSliceStart, int numZSlice, int xm, int ym, float * data){

    /* hid_t       file_id, dset_id;         /\* file and dataset identifiers *\/ */
    /* hid_t	plist_id;                 /\* property list identifier *\/ */

    /* // Set up file access property list with parallel I/O access */

    /* plist_id = H5Pcreate(H5P_FILE_ACCESS); */

    /* // open the file */
    /* file_id = H5Fopen(filename, H5F_ACC_RDWR, plist_id); */
    /* H5Pclose( plist_id ); */

    /* // open the dataset  */
    /* dset_id = H5Dopen2( file_id, datasetname, H5P_DEFAULT ); */

    /* // Create property list for collective dataset write. */
    /* plist_id = H5Pcreate(H5P_DATASET_XFER); */

    /* hid_t dataspace = H5Dget_space( dset_id ); */
  
    /* // Define hyperslab in the dataset */
    /* hsize_t count[3]; */
    /* hsize_t offset[3]; */

    /* offset[0] = nZSliceStart + commRank * numZSlice; */
    /* offset[1] = 0; */
    /* offset[2] = 0; */

    /* count[0] = numZSlice; */
    /* count[1] = ym; */
    /* count[2] = xm; */

    /* herr_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL); */

    /* // define the memory dataspace */
    /* hsize_t dimsm[3]; */

    /* dimsm[0] = numZSlice; */
    /* dimsm[1] = ym; */
    /* dimsm[2] = xm; */

    /* hid_t memspace = H5Screate_simple( 3, dimsm, NULL ); */
  
    /* hsize_t count_out[3]; */
    /* hsize_t offset_out[3]; */

    /* offset_out[0] = 0; */
    /* offset_out[1] = 0; */
    /* offset_out[2] = 0; */

    /* count_out[0] = numZSlice; */
    /* count_out[1] = ym; */
    /* count_out[2] = xm; */

    /* status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL, */
    /* 				 count_out, NULL); */

    /* /\* status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, data); *\/ */
    /* status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, dataspace, plist_id, data); */

    /* H5Dclose( dset_id ); */
    /* H5Sclose( dataspace ); */
    /* H5Sclose( memspace ); */
    /* H5Pclose( plist_id ); */
    /* H5Fclose( file_id ); */
}

