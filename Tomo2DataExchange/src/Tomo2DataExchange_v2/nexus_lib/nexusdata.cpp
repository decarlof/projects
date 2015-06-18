//---------------------------------------------------------------------------
#include "nexusgroup.h"

#pragma hdrstop

#pragma package(smart_init)
//---------------------------------------------------------------------------
//Private Methods for NexusData
//---------------------------------------------------------------------------

NexusData::NexusData (void)
{
	name = NULL;
    type = NULL;

	data = NULL;

#ifdef USECAPV
	pv = NULL;
#endif

    valid = FALSE;

	association = 0;

    required = TRUE;
    uptodate = TRUE;

    if (required)
        use = TRUE;
    else
        use = FALSE;
}

//---------------------------------------------------------------------------

void NexusData::PutDataVal (char *new_data, long int new_size)
{
    data_type = NX_CHAR;
    rank = 1;
    size = new_size;
    arr_dims[0] = size;
    arr_dims[1] = 0;

    if (data != NULL)
        free (data);
    data = (char *) malloc (sizeof (char) * size + 1);
    if (data == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusData::PutDataVal!", (char *) "Memory allocation error!"));

    if (data != NULL)
        strcpy ((char *) data, new_data);

    valid = TRUE;
}

//---------------------------------------------------------------------------

void NexusData::PutDataVal (unsigned char *new_data)
{
    data_type = NX_UINT8;
    rank = 1;
    size = 1;
    arr_dims[0] = size;
    arr_dims[1] = 0;

    if (data != NULL)
        memcpy (data, new_data, sizeof (unsigned char));

    valid = TRUE;
}

//---------------------------------------------------------------------------

void NexusData::PutDataVal (char *new_data)
{
    data_type = NX_INT8;
    rank = 1;
    size = 1;
    arr_dims[0] = size;
    arr_dims[1] = 0;

    if (data != NULL)
        memcpy (data, new_data, sizeof (char));

    valid = TRUE;
}

//---------------------------------------------------------------------------

void NexusData::PutDataVal (unsigned short *new_data)
{
    data_type = NX_UINT16;
    rank = 1;
    size = 1;
    arr_dims[0] = size;
    arr_dims[1] = 0;

    if (data != NULL)
        memcpy (data, new_data, sizeof (unsigned short));

    valid = TRUE;
}

//---------------------------------------------------------------------------

void NexusData::PutDataVal (short *new_data)
{
    data_type = NX_INT16;
    rank = 1;
    size = 1;
    arr_dims[0] = size;
    arr_dims[1] = 0;

    if (data != NULL)
        memcpy (data, new_data, sizeof (short));

    valid = TRUE;
}

//---------------------------------------------------------------------------

void NexusData::PutDataVal (unsigned long *new_data)
{
    data_type = NX_UINT32;
    rank = 1;
    size = 1;
    arr_dims[0] = size;
    arr_dims[1] = 0;

    if (data != NULL)
        memcpy (data, new_data, sizeof (unsigned long));

    valid = TRUE;
}

//---------------------------------------------------------------------------

void NexusData::PutDataVal (long *new_data)
{
    data_type = NX_INT32;
    rank = 1;
    size = 1;
    arr_dims[0] = size;
    arr_dims[1] = 0;

    if (data != NULL)
        memcpy (data, new_data, sizeof (long));

    valid = TRUE;
}

//---------------------------------------------------------------------------

void NexusData::PutDataVal (float *new_data)
{
    data_type = NX_FLOAT32;
    rank = 1;
    size = 1;
    arr_dims[0] = size;
    arr_dims[1] = 0;

    if (data != NULL)
        memcpy (data, new_data, sizeof (float));

    valid = TRUE;
}

//---------------------------------------------------------------------------

void NexusData::PutDataVal (double *new_data)
{
    data_type = NX_FLOAT64;
    rank = 1;
    size = 1;
    arr_dims[0] = size;
    arr_dims[1] = 0;

    if (data != NULL)
        memcpy (data, new_data, sizeof (double));

    valid = TRUE;
}

//---------------------------------------------------------------------------

void NexusData::PutDataVal (void *new_data, int new_rank, int *new_dims, int new_type)
{
int     new_size,
        loop,
        malloc_size;

    data_type = new_type;
    switch (data_type)
    {
        case NX_CHAR :;
        case NX_UINT8 :;
        case NX_INT8 : malloc_size = sizeof (char); break;
        case NX_UINT16 :;
        case NX_INT16 : malloc_size = sizeof (short); break;
        case NX_UINT32 :;
        case NX_INT32 : malloc_size = sizeof (long); break;
        case NX_FLOAT32 : malloc_size = sizeof (float); break;
        case NX_FLOAT64 : malloc_size = sizeof (double); break;
    }

    rank = new_rank;

    new_size = 1;
    for (loop=0;loop<new_rank;loop++)
        new_size = new_size * new_dims[loop];

    size = new_size;

    arr_dims[0] = new_dims[0];
    arr_dims[1] = new_dims[1];

    if (data != NULL)
        free (data);
    data = malloc (malloc_size * size);
    if (data == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusData::PutDataVal!", (char *) "Memory allocation error!"));

    if (data != NULL)
        memcpy (data, new_data, size * malloc_size);

    valid = TRUE;
}

//---------------------------------------------------------------------------

void NexusData::GetDataInfo (int *get_rank, int *get_dims, int *get_type)
{
int loop;

    *get_rank = rank;

    for (loop=0;loop<rank;loop++)
        get_dims[loop] = arr_dims[loop];

    *get_type = data_type;
}

//---------------------------------------------------------------------------

void NexusData::GetDataVal (unsigned char *get_data)
{
    if (data != NULL)
        memcpy (get_data, data, sizeof (unsigned char)*size);
}

//---------------------------------------------------------------------------

void NexusData::GetDataVal (char *get_data)
{
    if (data != NULL)
        memcpy (get_data, data, sizeof (char)*size);
}

//---------------------------------------------------------------------------

void NexusData::GetDataVal (unsigned short *get_data)
{
    if (data != NULL)
        memcpy (get_data, data, sizeof (unsigned short)*size);
}

//---------------------------------------------------------------------------

void NexusData::GetDataVal (short *get_data)
{
    if (data != NULL)
        memcpy (get_data, data, sizeof (short)*size);
}

//---------------------------------------------------------------------------

void NexusData::GetDataVal (unsigned long *get_data)
{
    if (data != NULL)
        memcpy (get_data, data, sizeof (unsigned long)*size);
}

//---------------------------------------------------------------------------

void NexusData::GetDataVal (long *get_data)
{
    if (data != NULL)
        memcpy (get_data, data, sizeof (long)*size);
}

//---------------------------------------------------------------------------

void NexusData::GetDataVal (float *get_data)
{
    if (data != NULL)
        memcpy (get_data, data, sizeof (float)*size);
}

//---------------------------------------------------------------------------

void NexusData::GetDataVal (double *get_data)
{
    if (data != NULL)
        memcpy (get_data, data, sizeof (double)*size);
}

//---------------------------------------------------------------------------

void NexusData::GetDataVal (void *get_data)
{
long int   element_size;

    switch (data_type)
    {
        case NX_CHAR :;
        case NX_UINT8 :;
        case NX_INT8 : element_size = sizeof (char); break;
        case NX_UINT16 :;
        case NX_INT16 : element_size = sizeof (short); break;
        case NX_UINT32 :;
        case NX_INT32 : element_size = sizeof (long); break;
        case NX_FLOAT32 : element_size = sizeof (float); break;
        case NX_FLOAT64 : element_size = sizeof (double); break;
    }

    if (data != NULL)
        if (data_type == NX_CHAR)
            memcpy (get_data, data, (size+1)*element_size);
        else
            memcpy (get_data, data, size*element_size);

}

//---------------------------------------------------------------------------

int NexusData::DataValid (void)
{
    return (valid);
}

//---------------------------------------------------------------------------
#ifdef USECAPV
void NexusData::PVConnect (void)
{
    if (pv->CA_Connect (pv_type) != PV_OKAY)
		throw (new NexusExceptionClass ((char *) "Error in method NexusData::PVConnect!", (char *) "PV connection failed!"));
}
#endif
//---------------------------------------------------------------------------
#ifdef USECAPV
void NexusData::PVDisconnect (void)
{
    if (pv->CA_Disconnect () != PV_OKAY)
		throw (new NexusExceptionClass ((char *) "Error in method NexusData::PVDisconnect!", (char *) "PV connection failed!"));
}
#endif
//---------------------------------------------------------------------------
#ifdef USECAPV
void NexusData::PVReConnect (void)
{
    if (pv->CA_ReConnect () != PV_OKAY)
		throw (new NexusExceptionClass ((char *) "Error in method NexusData::PVReConnect!", (char *) "PV connection failed!"));
}
#endif
//---------------------------------------------------------------------------
#ifdef USECAPV
int NexusData::PVIsConnected (void)
{
	return (pv->CA_IsConnected ());
}
#endif
//---------------------------------------------------------------------------
#ifdef USECAPV
int NexusData::PVLastReconnectAttempt (void)
{
    return (reconnect_time);
}
#endif
//---------------------------------------------------------------------------

NexusData::~NexusData (void)
{
    if (name != NULL)
        free (name);

    if (type != NULL)
        free (type);

//If association = 1, we can't free memory--the parent process owns it.
	if (association == 0)
	  	if (data != NULL)
    	   	free (data);

#ifdef USECAPV
	if (pv != NULL)
    	free (pv);
#endif
}
//---------------------------------------------------------------------------

