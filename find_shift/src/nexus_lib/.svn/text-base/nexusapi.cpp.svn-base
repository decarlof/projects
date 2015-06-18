#pragma hdrstop

#include "nexusapi.h"

#include <iostream.h>

/******************************************************************/
/******************************************************************/
/******************************************************************/

NexusAPI::NexusAPI ()
{
}

/******************************************************************/

void NexusAPI::OpenFile (NXaccess access_mode, char *file_path, char *file_name, NXhandle *file_handle)
{
char				*full_file_name;

	full_file_name = (char *) malloc (256);
	full_file_name = strcpy (full_file_name, file_path);
	full_file_name = strcat (full_file_name, file_name);

    if (NXopen (full_file_name, access_mode, file_handle) != 1)
   	    throw (new NexusExceptionClass ((char *) "File Open Error!"));

 	free (full_file_name);
}

/******************************************************************/

void NexusAPI::CloseFile (NXhandle *file_handle)
{
	if (!(NXclose (file_handle)))
    	throw (new NexusExceptionClass ((char *) "Error closing file!"));
}

/******************************************************************/

void NexusAPI::MakeGroup (char *group_name, char *group_class, NXhandle file_handle)
{
	if (!(NXmakegroup (file_handle, group_name, group_class)))
    	throw (new NexusExceptionClass ((char *) "Error making group!"));
}

/******************************************************************/

void NexusAPI::OpenGroup (char *group_name, char *group_class, NXhandle file_handle)
{
	if (!(NXopengroup (file_handle, group_name, group_class)))
    	throw (new NexusExceptionClass ((char *) "Error opening group!"));
}

/******************************************************************/

void NexusAPI::CloseGroup (NXhandle file_handle)
{
	if (!(NXclosegroup (file_handle)))
	  	throw (new NexusExceptionClass ((char *) "Error closing group!"));
}

/******************************************************************/

long int NexusAPI::GetNextEntry (char *entry_name, char *entry_class, int *data_type, NXhandle file_handle)
{
	if ((NXgetnextentry (file_handle, entry_name, entry_class, (int *) data_type)) < 0)
		return (0);
    else
    	return (1);
}

/******************************************************************/

void NexusAPI::MakeData (char *data_name, int data_type, int rank, int *dimensions, NXhandle file_handle, int compression_scheme)
{
int chunk_size[10];

	for (int loop=0;loop<rank;loop++)
		chunk_size[loop] = dimensions[loop];
		
	if (rank > 1)
	{
		if (!(NXcompmakedata (file_handle, data_name, data_type, rank, dimensions, compression_scheme, chunk_size)))
    		throw (new NexusExceptionClass ((char *) "Error making data!"));
	}
	else
	{
		if (!(NXmakedata (file_handle, data_name, data_type, rank, dimensions)))
    		throw (new NexusExceptionClass ((char *) "Error making data!"));
	}
	
}

/******************************************************************/

void NexusAPI::OpenData (char *data_name, NXhandle file_handle)
{
	if (!(NXopendata (file_handle, data_name)))
    	throw (new NexusExceptionClass ((char *) "Error opening data!"));
}

/******************************************************************/

void NexusAPI::GetData (void *data, NXhandle file_handle)
{
	if (!(NXgetdata (file_handle, data)))
    	throw (new NexusExceptionClass ((char *) "Error getting data!"));
}

/******************************************************************/

 void NexusAPI::GetDataSlab (void *data, NXhandle file_handle, int *start_dims, int *size_dims)
{
	if (!(NXgetslab (file_handle, data, start_dims, size_dims)))
    	throw (new NexusExceptionClass ((char *) "Error getting data!"));
}

/******************************************************************/

void NexusAPI::PutData (void *data, NXhandle file_handle)
{
	if (!(NXputdata (file_handle, data)))
    	throw (new NexusExceptionClass ((char *) "Error putting data!"));
}

/******************************************************************/

void NexusAPI::GetDataInfo (int *rank, int *dimensions, int *data_type, NXhandle file_handle)
{
	if (!(NXgetinfo (file_handle, rank, dimensions, data_type)))
    	throw (new NexusExceptionClass ((char *) "Error getting data info!"));
}

/******************************************************************/

void NexusAPI::CloseData (NXhandle file_handle)
{
	if (!(NXclosedata (file_handle)))
    	throw (new NexusExceptionClass ((char *) "Error closing data!"));
}

/******************************************************************/

void NexusAPI::GetAttrib (char *attrib_name, void *value, int *length, int *data_type, NXhandle file_handle)
{
	if (!(NXgetattr (file_handle, attrib_name, value, length, data_type)))
    	throw (new NexusExceptionClass ((char *) "Error getting attribute!"));
}

/******************************************************************/

void NexusAPI::PutAttrib (char *attrib_name, void *value, int length, int data_type, NXhandle file_handle)
{
	if (!(NXputattr (file_handle, attrib_name, value, length, data_type)))
    	throw (new NexusExceptionClass ((char *) "Error putting attribute!"));
}
/******************************************************************/

long int NexusAPI::GetNextAttrib (char *attrib_name, int *length, int *data_type, NXhandle file_handle)
{
	if (NXgetnextattr (file_handle, attrib_name, length, data_type) < 0)
		return (0);
    else
    	return (1);
}
/******************************************************************/

void NexusAPI::GetDatum (char *data_name, void *data, NXhandle file_handle)
{
    OpenData (data_name, file_handle);

    GetData (data, file_handle);

    CloseData (file_handle);
}

/******************************************************************/

void NexusAPI::PutDatum (char *data_name, int data_type, int rank, int *dimensions, void *data, NXhandle file_handle, int compression_scheme)
{
	MakeData (data_name, data_type, rank, dimensions, file_handle, compression_scheme);

	OpenData (data_name, file_handle);

	PutData (data, file_handle);

	CloseData (file_handle);
}

/******************************************************************/
/*
NexusAPI::~NexusAPI ()
{
	return;
}
*/
/******************************************************************/

