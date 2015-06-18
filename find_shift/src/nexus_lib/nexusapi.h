#ifndef NexusAPI_H
#define NexusAPI_H

#include "napi.h"

#include <stdlib.h>
#include <string.h>

#include "nexusexceptionclass.h"

#define CSTRING_SIZE 256

#define FALSE   0
#define TRUE    1

#ifdef __DLL__
class __declspec(dllexport) NexusAPI
#else
class NexusAPI
#endif
{
public:
	NexusAPI ();

	void OpenFile (NXaccess access_mode, char *file_path, char *file_name, NXhandle *file_handle);
	void CloseFile (NXhandle *file_handle);

	void MakeGroup (char *group_name, char *group_class, NXhandle file_handle);
	void OpenGroup (char *group_name, char *group_class, NXhandle file_handle);
	void CloseGroup (NXhandle file_handle);
    long int GetNextEntry (char *entry_name, char *entry_class, int *data_type, NXhandle file_handle);

	void MakeData (char *data_name, int data_type, int rank, int*dimensions, NXhandle file_handle, int compression_scheme);
	void OpenData (char *data_name, NXhandle file_handle);
	void GetData (void *data, NXhandle file_handle);
	void GetDataSlab (void *data, NXhandle file_handle, int *start_dims, int *size_dims);
	void PutData (void *data, NXhandle file_handle);
	void GetDataInfo (int *rank, int *xdim, int *ydim, int *data_type, NXhandle file_handle);
	void GetDataInfo (int *rank, int *dimensions, int *data_type, NXhandle file_handle);
	void CloseData (NXhandle file_handle);

	void GetAttrib (char *attrib_name, void *value, int *length, int *data_type, NXhandle file_handle);
	void PutAttrib (char *attrib_name, void *value, int length, int data_type, NXhandle file_handle);
	long int GetNextAttrib (char *attrib_name, int *length, int *data_type, NXhandle file_handle);

	void GetDatum (char *data_name, void *data, NXhandle file_handle);
	void PutDatum (char *data_name, int data_type, int rank, int *dimensions, void *data, NXhandle file_handle, int compression_scheme);

//	~NexusAPI ();

};

#endif
