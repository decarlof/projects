//---------------------------------------------------------------------------
#include "nexusgroup.h"
#pragma hdrstop

#ifdef WIN32
#pragma package(smart_init)
#endif

#ifdef __unix__
#define stricmp strcasecmp
#endif
//---------------------------------------------------------------------------

NexusAttribute::NexusAttribute (void)
{
    next_attribute = NULL;
}

//---------------------------------------------------------------------------

void NexusAttribute::PutSDSInfo (char *attrib_name, int attrib_length, int attrib_type, void *attrib_data)
{
long int      malloc_size;

    name = (char *) malloc (strlen (attrib_name) + 1);
    if (name == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutSDSInfo!", (char *) "Memory allocation error!"));

    strcpy (name, attrib_name);

	if (attrib_length == 0)
    	attrib_length = 1;
    size = attrib_length;
    rank = 1;
    arr_dims[0] = size;
    arr_dims[1] = 0;

    data_type = attrib_type;
    switch (data_type)
    {
        case NX_CHAR : {
                    type = (char *) malloc (strlen ("NX_CHAR") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutSDSInfo!", (char *) "Memory allocation error!"));
                    strcpy (type, "NX_CHAR");
                    malloc_size = sizeof (char) * size + 1;
                    break;
        		}
        case NX_UINT8 : {
                    type = (char *) malloc (strlen ("NX_UINT8") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutSDSInfo!", (char *) "Memory allocation error!"));
                    strcpy (type, "NX_UINT8");
                    malloc_size = sizeof (unsigned char) * size;
                    break;
        		}
        case NX_INT8 : {
                    type = (char *) malloc (strlen ("NX_INT8") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutSDSInfo!", (char *) "Memory allocation error!"));
                    strcpy (type, "NX_INT8");
                    malloc_size = sizeof (char) * size;
                    break;
        		}
        case NX_UINT16 : {
                    type = (char *) malloc (strlen ("NX_UINT16") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutSDSInfo!", (char *) "Memory allocation error!"));
                    strcpy (type, "NX_UINT16");
                    malloc_size = sizeof (unsigned short) * size;
                    break;
        		}
        case NX_INT16 : {
                    type = (char *) malloc (strlen ("NX_INT16") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutSDSInfo!", (char *) "Memory allocation error!"));
                    strcpy (type, "NX_INT16");
                    malloc_size = sizeof (short) * size;
                    break;
        		}
        case NX_UINT32 : {
                    type = (char *) malloc (strlen ("NX_UINT32") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutSDSInfo!", (char *) "Memory allocation error!"));
                    strcpy (type, "NX_UINT32");
                    malloc_size = sizeof (unsigned long) * size;
                    break;
        		}
        case NX_INT32 : {
                    type = (char *) malloc (strlen ("NX_INT32") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutSDSInfo!", (char *) "Memory allocation error!"));
                    strcpy (type, "NX_INT32");
                    malloc_size = sizeof (long) * size;
                    break;
        		}
        case NX_FLOAT32 : {
                    type = (char *) malloc (strlen ("NX_FLOAT32") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutSDSInfo!", (char *) "Memory allocation error!"));
                    strcpy (type, "NX_FLOAT32");
                    malloc_size = sizeof (float) * size;
                    break;
        		}
        case NX_FLOAT64 : {
                    type = (char *) malloc (strlen ("NX_FLOAT64") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutSDSInfo!", (char *) "Memory allocation error!"));
                    strcpy (type, "NX_FLOAT64");
                    malloc_size = sizeof (double) * size;
                    break;
        		}
    }

    if (data != NULL)
        free (data);
    data = malloc (malloc_size);
    if (data == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutSDSInfo!", (char *) "Memory allocation error!"));
    memcpy (data, attrib_data, malloc_size);

    valid = TRUE;
}

//---------------------------------------------------------------------------

void NexusAttribute::PutConstInfo (char *attrib_name, char *attrib_type, char *attrib_value)
{
    name = (char *) malloc (strlen (attrib_name) + 1);
    if (name == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutConstInfo!", (char *) "Memory allocation error!"));
    strcpy (name, attrib_name);

    size = 1;
    rank = 1;
    arr_dims[0] = size;
    arr_dims[1] = 0;

    type= (char *) malloc (strlen (attrib_type) + 1);
    if (type == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutConstInfo!", (char *) "Memory allocation error!"));
    strcpy (type, attrib_type);

    if (!stricmp (attrib_type, "NX_CHAR"))
    {
        size = strlen (attrib_value);
        arr_dims[0] = size;
        arr_dims[1] = 0;

        data = (char *) malloc (sizeof (char) * size + 1);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutConstInfo!", (char *) "Memory allocation error!"));

        strcpy ((char *) data, attrib_value);
        data_type = NX_CHAR;
    }

    if (!stricmp (attrib_type, "NX_UINT8"))
    {
        data = (unsigned char *) malloc (sizeof (unsigned char) * size);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutConstInfo!", (char *) "Memory allocation error!"));

    	unsigned char temp = (unsigned char) atoi (attrib_value);
		memcpy (data, &temp, sizeof (unsigned char));
        data_type = NX_UINT8;
    }

    if (!stricmp (attrib_type, "NX_INT8"))
    {
        data = (char *) malloc (sizeof (char) * size);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutConstInfo!", (char *) "Memory allocation error!"));

    	char temp = (char) atoi (attrib_value);
		memcpy (data, &temp, sizeof (char));
        data_type = NX_INT8;
    }

    if (!stricmp (attrib_type, "NX_UINT16"))
    {
        data = (unsigned short *) malloc (sizeof (unsigned short) * size);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutConstInfo!", (char *) "Memory allocation error!"));

    	unsigned short temp = (unsigned short) atoi (attrib_value);
		memcpy (data, &temp, sizeof (unsigned short));
        data_type = NX_UINT16;
    }

    if (!stricmp (attrib_type, "NX_INT16"))
    {
        data = (short *) malloc (sizeof (short) * size);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutConstInfo!", (char *) "Memory allocation error!"));

		short temp = (short) atoi (attrib_value);
		memcpy (data, &temp, sizeof (short));
        data_type = NX_INT16;
    }

    if (!stricmp (attrib_type, "NX_UINT32"))
    {
        data = (unsigned long *) malloc (sizeof (unsigned long) * size);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutConstInfo!", (char *) "Memory allocation error!"));

		unsigned long temp = (unsigned long) atoi (attrib_value);
		memcpy (data, &temp, sizeof (unsigned long));
        data_type = NX_UINT32;
    }

    if (!stricmp (attrib_type, "NX_INT32"))
    {
        data = (long *) malloc (sizeof (long) * size);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutConstInfo!", (char *) "Memory allocation error!"));

    	long temp = (long) atoi (attrib_value);
		memcpy (data, &temp, sizeof (long));
        data_type = NX_INT32;
    }

    if (!stricmp (attrib_type, "NX_FLOAT32"))
    {
        data = (float *) malloc (sizeof (float) * size);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutConstInfo!", (char *) "Memory allocation error!"));

    	float temp = (float) atof (attrib_value);
		memcpy (data, &temp, sizeof (float));
        data_type = NX_FLOAT32;
    }

    if (!stricmp (attrib_type, "NX_FLOAT64"))
    {
        data = (double *) malloc (sizeof (double) * size);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutConstInfo!", (char *) "Memory allocation error!"));

    	double temp = (double) atof (attrib_value);
		memcpy (data, &temp, sizeof (double));
        data_type = NX_FLOAT64;
    }

    valid = TRUE;

}

//---------------------------------------------------------------------------

void NexusAttribute::PutVarInfo (char *attrib_name, int attrib_length, int attrib_type, void *var_address)
{
    name = (char *) malloc (strlen (attrib_name) + 1);
    if (name == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutVarInfo!", (char *) "Memory allocation error!"));
    strcpy (name, attrib_name);

	if (attrib_length == 0)
    	attrib_length = 1;
    size = attrib_length;
    rank = 1;
    arr_dims[0] = size;
    arr_dims[1] = 0;

	association = 1;

    data_type = attrib_type;
    switch (data_type)
    {
        case NX_CHAR : {
                    type = (char *) malloc (strlen ("NX_CHAR") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutVarInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_CHAR");
                    break;
        		}
        case NX_UINT8 : {
                    type = (char *) malloc (strlen ("NX_UINT8") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutVarInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_UINT8");
                    break;
        		}
        case NX_INT8 : {
                    type = (char *) malloc (strlen ("NX_INT8") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutVarInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_INT8");
                    break;
        		}
        case NX_UINT16 : {
                    type = (char *) malloc (strlen ("NX_UINT16") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutVarInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_UINT16");
                    break;
        		}
        case NX_INT16 : {
                    type = (char *) malloc (strlen ("NX_INT16") + 1);
				    if (type == NULL)
    					throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutVarInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_INT16");
                    break;
        		}
        case NX_UINT32 : {
                    type = (char *) malloc (strlen ("NX_UINT32") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutVarInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_UINT32");
                    break;
        		}
        case NX_INT32 : {
                    type = (char *) malloc (strlen ("NX_INT32") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutVarInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_INT32");
                    break;
        		}
        case NX_FLOAT32 : {
                    type = (char *) malloc (strlen ("NX_FLOAT32") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutVarInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_FLOAT32");
                    break;
        		}
        case NX_FLOAT64 : {
                    type = (char *) malloc (strlen ("NX_FLOAT64") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutVarInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_FLOAT64");
                    break;
        		}
    }

    data = var_address;

    valid = TRUE;
}

//---------------------------------------------------------------------------
#ifdef USECAPV
void NexusAttribute::PutPVInfo (char *attrib_name, char *attrib_type, char *attrib_value)
{
    name = (char *) malloc (strlen (attrib_name) + 1);
    if (name == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutPVInfo!", (char *) "Memory allocation error!"));

    strcpy (name, attrib_name);

    size = 1;
    rank = 1;
    arr_dims[0] = size;
    arr_dims[1] = 0;

    type= (char *) malloc (strlen (attrib_type) + 1);
    if (type == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutPVInfo!", (char *) "Memory allocation error!"));
    strcpy (type, attrib_type);

    pv = new CAPV (attrib_value);
    if (pv == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutPVInfo!", (char *) "Could not create CAPV object!"));

    if (!stricmp (attrib_type, "NX_CHAR"))
    {
        pv_type = DBR_STRING;
        data_type = NX_CHAR;
        size = strlen ("EMPTY");

        arr_dims[0] = size;
        arr_dims[1] = 0;
        data = (char *) malloc (sizeof (char) * size + 1);
	    if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutPVInfo!", (char *) "Memory allocation error!"));
        strcpy ((char *) data, "EMPTY");
        data_type = NX_CHAR;
    }

    if (!stricmp (attrib_type, "NX_INT8"))
    {
        pv_type = DBR_SHORT;
        data_type = NX_INT8;

		data = (char *) malloc (sizeof (char) * size);
		if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutPVInfo!", (char *) "Memory allocation error!"));
		char temp = 0;
		memcpy (data, &temp, sizeof (char));
	}

	if (!stricmp (attrib_type, "NX_INT32"))
	{
		pv_type = DBR_LONG;
		data_type = NX_INT32;

		data = (long int *) malloc (sizeof (long int) * size);
		if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutPVInfo!", (char *) "Memory allocation error!"));
		long int temp = 0;
        memcpy (data, &temp, sizeof (long int));
    }

    if (!stricmp (attrib_type, "NX_FLOAT32"))
    {
        pv_type = DBR_FLOAT;
        data_type = NX_FLOAT32;

		data = (float *) malloc (sizeof (float) * size);
		if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutPVInfo!", (char *) "Memory allocation error!"));
		float temp = 0.0;
		memcpy (data, &temp, sizeof (float));
	}

	if (!stricmp (attrib_type, "NX_FLOAT64"))
	{
		pv_type = DBR_DOUBLE;
		data_type = NX_FLOAT64;

		data = (double *) malloc (sizeof (double) * size);
		if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutPVInfo!", (char *) "Memory allocation error!"));
		double temp = 0;
        memcpy (data, &temp, sizeof (double));
    }

    valid = TRUE;
}
#endif

//---------------------------------------------------------------------------

void NexusAttribute::UpdateVarInfo (int attrib_length, void *var_address)
{
	if (attrib_length == 0)
    	attrib_length = 1;
    size = attrib_length;
    rank = 1;
    arr_dims[0] = size;
    arr_dims[1] = 0;

    data = var_address;
}

//---------------------------------------------------------------------------

void NexusAttribute::AddAttribute (NexusAttribute *attrib)
{
    next_attribute = attrib;
}

//---------------------------------------------------------------------------

NexusAttribute *NexusAttribute::NextAttribute (void)
{
    return (next_attribute);
}

//---------------------------------------------------------------------------
#ifdef USECAPV
void NexusAttribute::PVUpdate (void)
{
struct timeb	now;

	//if channel is not connected, check to see if last reconnect time was
    //greater than 60 seconds ago.  If it was, try and reconnect.  Otherwise,
    //mark now as the last reconnect time.
    if (!pv->CA_IsConnected ())
	{
        ftime (&now);
		if ((now.time - reconnect_time) > 60)
        	if (pv->CA_ReConnect () != PV_OKAY)
            {
            	reconnect_time = now.time;
				throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutUpdate!", (char *) "PV connection failed!"));
		    }
    }

    if (pv->CA_Get ())
    {
        ftime (&now);
        reconnect_time = now.time;
		throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::PutUpdate!", (char *) "PV connection failed!"));
    }

    switch (data_type)
    {
        case NX_CHAR : PutDataVal ((char *) pv->GetPVValue (), strlen ((char *) pv->GetPVValue ())); break;
		case NX_INT32 : PutDataVal ((long int *) pv->GetPVValue ()); break;
		case NX_FLOAT32 : PutDataVal ((float *) pv->GetPVValue ()); break;
        case NX_FLOAT64 : PutDataVal ((double *) pv->GetPVValue ()); break;
    }
}
#endif
//---------------------------------------------------------------------------

void NexusAttribute::WriteAttribute (NXhandle file_handle)
{
long int    length;
void        *attrib_ptr;

	try
	{
		if (!use)
			return;

		if (!required)
		{
			length = strlen (MISSING_DATA_ERROR_STR);
			PutAttrib (name, (char *) MISSING_DATA_ERROR_STR, length, NX_CHAR, file_handle);
			return;
		}

		if (required && (!uptodate))
			throw (new NexusExceptionClass ((char *) "Error in method NexusAttribute::WriteAttribute!", (char *) "Required attribute not up to date!"));

		attrib_ptr = (void *) data;

		PutAttrib (name, attrib_ptr, size, data_type, file_handle);
	}
	catch (...)
	{
		throw;
	}
}

//---------------------------------------------------------------------------

NexusAttribute::~NexusAttribute (void)
{
    if (next_attribute != NULL)
        delete (next_attribute);
}

