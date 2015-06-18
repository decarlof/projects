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
//Private Methods
//---------------------------------------------------------------------------

NexusField::NexusField (void)
{
    next_field = NULL;
    attribute_list = NULL;
}

//---------------------------------------------------------------------------

void NexusField::PutSDSInfo (char *field_name, int field_rank, int *field_dims, int field_type, void *field_data)
{
long int    malloc_size;
int         loop;

    name = (char *) malloc (strlen (field_name) + 1);
    if (name == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutSDSInfo!", (char *) "Memory allocation error!"));
    strcpy (name, field_name);

    rank = field_rank;
    for (loop=0;loop<rank;loop++)
        arr_dims[loop] = field_dims[loop];

    size = 1;
    if (rank > 1)
        for (loop=0;loop<rank;loop++)
            size = size * arr_dims[loop];
    else
    {
        size = arr_dims[0];
        for (loop=1;loop<rank;loop++)
            arr_dims[loop] = 0;
    }

    data_type = field_type;
    switch (data_type)
    {
        case NX_CHAR : {
                    type= (char *) malloc (strlen ("NX_CHAR") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutSDSInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_CHAR");
                    malloc_size = sizeof (char) * size + 1;
                    break;
                }
        case NX_UINT8 : {
                    type= (char *) malloc (strlen ("NX_UINT8") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutSDSInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_UINT8");
                    malloc_size = sizeof (unsigned char) * size;
                    break;
                }
        case NX_INT8 : {
                    type= (char *) malloc (strlen ("NX_INT8") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutSDSInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_INT8");
                    malloc_size = sizeof (char) * size;
                    break;
                }
        case NX_UINT16 : {
                    type= (char *) malloc (strlen ("NX_UINT16") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutSDSInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_UINT16");
                    malloc_size = sizeof (unsigned short) * size;
                    break;
                }
        case NX_INT16 : {
                    type= (char *) malloc (strlen ("NX_INT16") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutSDSInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_INT16");
                    malloc_size = sizeof (short) * size;
                    break;
                }
        case NX_UINT32 : {
                    type= (char *) malloc (strlen ("NX_UINT32") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutSDSInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_UINT32");
                    malloc_size = sizeof (unsigned long) * size;
                    break;
                }
        case NX_INT32 : {
                    type= (char *) malloc (strlen ("NX_INT32") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutSDSInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_INT32");
                    malloc_size = sizeof (long) * size;
                    break;
                }
        case NX_FLOAT32 : {
                    type= (char *) malloc (strlen ("NX_FLOAT32") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutSDSInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_FLOAT32");
                    malloc_size = sizeof (float) * size;
                    break;
                }
        case NX_FLOAT64 : {
                    type= (char *) malloc (strlen ("NX_FLOAT64") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutSDSInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_FLOAT64");
                    malloc_size = sizeof (double) * size;
                    break;
                }
    }

    data = malloc (malloc_size);
    if (data == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutSDSInfo!", (char *) "Memory allocation error!"));

    if (data != NULL)
        memcpy (data, field_data, malloc_size);

    valid = TRUE;
}

//---------------------------------------------------------------------------

void NexusField::PutSDSInfo (char *field_name, int field_rank, int *field_dims, int field_type)
{
int loop;

    name = (char *) malloc (strlen (field_name) + 1);
    if (name == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutSDSInfo!", (char *) "Memory allocation error!"));
    strcpy (name, field_name);

    rank = field_rank;
    for (loop=0;loop<rank;loop++)
        arr_dims[loop] = field_dims[loop];

    size = 1;
    if (rank > 1)
        for (loop=0;loop<rank;loop++)
            size = size * arr_dims[loop];
    else
    {
        size = arr_dims[0];
        for (loop=1;loop<rank;loop++)
            arr_dims[loop] = 0;
    }

    data_type = field_type;

    valid = FALSE;
}

//---------------------------------------------------------------------------

void NexusField::PutConstInfo (char *field_name, char *field_type, char *field_value)
{
int loop;

    name= (char *) malloc (strlen (field_name) + 1);
    if (name == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutConstInfo!", (char *) "Memory allocation error!"));
    strcpy (name, field_name);

    size = 1;
    rank = 1;
    arr_dims[0] = size;
    for (loop=1;loop<rank;loop++)
        arr_dims[loop] = 0;

    type= (char *) malloc (strlen (field_type) + 1);
    if (type == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutConstInfo!", (char *) "Memory allocation error!"));
    strcpy (type, field_type);

    if (!stricmp (field_type, "NX_CHAR"))
    {
        size = strlen (field_value);
        arr_dims[0] = size;
        for (loop=1;loop<rank;loop++)
            arr_dims[loop] = 0;

        data = (char *) malloc (sizeof (char) * size + 1);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutConstInfo!", (char *) "Memory allocation error!"));

        strcpy ((char *) data, field_value);
        data_type = NX_CHAR;
    }

    if (!stricmp (field_type, "NX_UINT8"))
    {
        data = (unsigned char *) malloc (sizeof (unsigned char) * size);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutConstInfo!", (char *) "Memory allocation error!"));

        unsigned char temp = (unsigned char) atoi (field_value);
    	memcpy (data, &temp, sizeof (unsigned char));
        data_type = NX_UINT8;
    }

    if (!stricmp (field_type, "NX_INT8"))
    {
        data = (char *) malloc (sizeof (char) * size);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutConstInfo!", (char *) "Memory allocation error!"));

    	char temp = (char) atoi (field_value);
		memcpy (data, &temp, sizeof (char));
        data_type = NX_INT8;
    }

    if (!stricmp (field_type, "NX_UINT16"))
    {
        data = (unsigned short *) malloc (sizeof (unsigned short) * size);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutConstInfo!", (char *) "Memory allocation error!"));

    	unsigned short temp = (unsigned short) atoi (field_value);
		memcpy (data, &temp, sizeof (unsigned short));
        data_type = NX_UINT16;
    }

    if (!stricmp (field_type, "NX_INT16"))
    {
        data = (short *) malloc (sizeof (short) * size);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutConstInfo!", (char *) "Memory allocation error!"));

		short temp = (short) atoi (field_value);
		memcpy (data, &temp, sizeof (short));
        data_type = NX_INT16;
    }

    if (!stricmp (field_type, "NX_UINT32"))
    {
        data = (unsigned long *) malloc (sizeof (unsigned long) * size);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutConstInfo!", (char *) "Memory allocation error!"));

		unsigned long temp = (unsigned long) atoi (field_value);
		memcpy (data, &temp, sizeof (unsigned long));
        data_type = NX_UINT32;
    }

    if (!stricmp (field_type, "NX_INT32"))
    {
        data = (long *) malloc (sizeof (long) * size);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutConstInfo!", (char *) "Memory allocation error!"));

		long temp = (long) atoi (field_value);
		memcpy (data, &temp, sizeof (long));
        data_type = NX_INT32;
    }

    if (!stricmp (field_type, "NX_FLOAT32"))
    {
        data = (float *) malloc (sizeof (float) * size);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutConstInfo!", (char *) "Memory allocation error!"));

    	float temp = (float) atof (field_value);
		memcpy (data, &temp, sizeof (float));
        data_type = NX_FLOAT32;
    }

    if (!stricmp (field_type, "NX_FLOAT64"))
    {
        data = (double *) malloc (sizeof (double) * size);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutConstInfo!", (char *) "Memory allocation error!"));

    	double temp = (double) atof (field_value);
		memcpy (data, &temp, sizeof (double));
        data_type = NX_FLOAT64;
    }
}

//---------------------------------------------------------------------------

void NexusField::PutVarInfo (char *field_name, int field_rank, int *field_dims, int field_type, void *var_address)
{
int loop;

    name = (char *) malloc (strlen (field_name) + 1);
    if (name == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutVarInfo!", (char *) "Memory allocation error!"));
    strcpy (name, field_name);

    rank = field_rank;
    for (loop=0;loop<rank;loop++)
        arr_dims[loop] = field_dims[loop];

    size = 1;
    if (rank > 1)
        for (loop=0;loop<rank;loop++)
            size = size * arr_dims[loop];
    else
    {
        size = arr_dims[0];
        for (loop=1;loop<rank;loop++)
            arr_dims[loop] = 0;
    }

	association = 1;

    data_type = field_type;
    switch (data_type)
    {
        case NX_CHAR : {
                    type= (char *) malloc (strlen ("NX_CHAR") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutVarInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_CHAR");
                    break;
                }
        case NX_UINT8 : {
                    type= (char *) malloc (strlen ("NX_UINT8") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutVarInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_UINT8");
                    break;
                }
        case NX_INT8 : {
                    type= (char *) malloc (strlen ("NX_INT8") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutVarInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_INT8");
                    break;
                }
        case NX_UINT16 : {
                    type= (char *) malloc (strlen ("NX_UINT16") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutVarInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_UINT16");
                    break;
                }
        case NX_INT16 : {
                    type= (char *) malloc (strlen ("NX_INT16") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutVarInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_INT16");
                    break;
                }
        case NX_UINT32 : {
                    type= (char *) malloc (strlen ("NX_UINT32") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutVarInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_UINT32");
                    break;
                }
        case NX_INT32 : {
                    type= (char *) malloc (strlen ("NX_INT32") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutVarInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_INT32");
                    break;
                }
        case NX_FLOAT32 : {
                    type= (char *) malloc (strlen ("NX_FLOAT32") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutVarInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_FLOAT32");
                    break;
                }
        case NX_FLOAT64 : {
                    type= (char *) malloc (strlen ("NX_FLOAT64") + 1);
				    if (type == NULL)
						throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutVarInfo!", (char *) "Memory allocation error!"));

                    strcpy (type, "NX_FLOAT64");
                    break;
                }
    }

    data = var_address;
}

//---------------------------------------------------------------------------
#ifdef USECAPV
void NexusField::PutPVInfo (char *field_name, char *field_type, char *field_value)
{
int loop;
NexusAttribute *pv_name_attribute;

    name = (char *) malloc (strlen (field_name) + 1);
    if (name == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutPVInfo!", (char *) "Memory allocation error!"));

    strcpy (name, field_name);

    size = 1;
    rank = 1;
    arr_dims[0] = size;
    for (loop=1;loop<rank;loop++)
        arr_dims[loop] = 0;

    type= (char *) malloc (strlen (field_type) + 1);
    if (type == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutPVInfo!", (char *) "Memory allocation error!"));
    strcpy (type, field_type);

    pv = new CAPV (field_value);
    if (pv == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutPVInfo!", (char *) "Could not create object CAPV!"));

    if (!stricmp (field_type, "NX_CHAR"))
    {
        pv_type = DBR_STRING;
        data_type = NX_CHAR;

        size = strlen ("EMPTY");
        arr_dims[0] = size;
        for (loop=1;loop<rank;loop++)
            arr_dims[loop] = 0;

        data = (char *) malloc (sizeof (char) * size + 1);
        if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutPVInfo!", (char *) "Memory allocation error!"));

        strcpy ((char *) data, "EMPTY");
        data_type = NX_CHAR;
    }

    if (!stricmp (field_type, "NX_INT8"))
    {
        pv_type = DBR_SHORT;
        data_type = NX_INT8;

		data = (char *) malloc (sizeof (char) * size);
		if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutPVInfo!", (char *) "Memory allocation error!"));

		char temp = 0;
		memcpy (data, &temp, sizeof (char));
	}

	if (!stricmp (field_type, "NX_INT32"))
	{
		pv_type = DBR_LONG;
		data_type = NX_INT32;

		data = (long int *) malloc (sizeof (long int) * size);
		if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutPVInfo!", (char *) "Memory allocation error!"));

		long int temp = 0;
		memcpy (data, &temp, sizeof (long int));
	}

	if (!stricmp (field_type, "NX_FLOAT32"))
	{
		pv_type = DBR_FLOAT;
		data_type = NX_FLOAT32;

		data = (float *) malloc (sizeof (float) * size);
		if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutPVInfo!", (char *) "Memory allocation error!"));

		float temp = 0.0;
		memcpy (data, &temp, sizeof (float));
	}

	if (!stricmp (field_type, "NX_FLOAT64"))
	{
		pv_type = DBR_DOUBLE;
		data_type = NX_FLOAT64;

		data = (double *) malloc (sizeof (double) * size);
		if (data == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutPVInfo!", (char *) "Memory allocation error!"));

		double temp = 0;
		memcpy (data, &temp, sizeof (double));
	}

	pv_name_attribute = new NexusAttribute();
	if (pv_name_attribute == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusField::PutPVInfo!", (char *) "Memory allocation error!"));
	pv_name_attribute->PutConstInfo ((char *) "PV_name", (char *) "NX_CHAR", field_value);
	AddToAttributeList (pv_name_attribute);
}
#endif
//---------------------------------------------------------------------------

void NexusField::UpdateVarInfo (int field_rank, int *field_dims, int field_type, void *var_address)
{
int loop;

	rank = field_rank;
	for (loop=0;loop<rank;loop++)
		arr_dims[loop] = field_dims[loop];

    size = 1;
    if (rank > 1)
        for (loop=0;loop<rank;loop++)
            size = size * arr_dims[loop];
    else
    {
        size = arr_dims[0];
        for (loop=1;loop<rank;loop++)
            arr_dims[loop] = 0;
    }

    data_type = field_type;

    data = var_address;
}

//---------------------------------------------------------------------------

void NexusField::AddField (NexusField *field)
{
    next_field = field;
}

//---------------------------------------------------------------------------

NexusField *NexusField::NextField (void)
{
    return (next_field);
}

//---------------------------------------------------------------------------

NexusAttribute *NexusField::AttributeList (void)
{
    return (attribute_list);
}

//---------------------------------------------------------------------------

void NexusField::AddToAttributeList (NexusAttribute *new_attrib)
{
NexusAttribute      *current_attrib;

    if (attribute_list == NULL)
        attribute_list = new_attrib;
    else
    {
        current_attrib = attribute_list;
        while (current_attrib->NextAttribute () != NULL)
            current_attrib = current_attrib->NextAttribute ();

		current_attrib->AddAttribute (new_attrib);
    }
}

//---------------------------------------------------------------------------
#ifdef USECAPV
void NexusField::PVUpdate (void)
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
				throw (new NexusExceptionClass ((char *) "Error in method NexusField::PVUpdate!", (char *) "PV connection failed!"));
		    }
    }

	if (pv->CA_Get ())
    {
        ftime (&now);
        reconnect_time = now.time;
		throw (new NexusExceptionClass ((char *) "Error in method NexusField::PVUpdate!", (char *) "PV connection failed!"));
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

void NexusField::WriteField (NXhandle file_handle, int compression_scheme)
{
int             length;
NexusAttribute  *current_attribute;

	try
	{
		if (!use)
			return;

		if (!required)
		{
			length = strlen (MISSING_DATA_ERROR_STR);
			MakeData (name, NX_CHAR, 1, &length, file_handle, compression_scheme);

			OpenData (name, file_handle);

			PutData ((char *) MISSING_DATA_ERROR_STR, file_handle);

			CloseData (file_handle);

			return;
		}

		if (required && (!uptodate))
			throw (new NexusExceptionClass ((char *) "Error in method NexusField::WriteField!", (char *) "Required field not up to date!"));

		MakeData (name, data_type, rank, arr_dims, file_handle, compression_scheme);

		OpenData (name, file_handle);

		PutData (data, file_handle);

		current_attribute = attribute_list;
		while (current_attribute != NULL)
		{
			current_attribute->WriteAttribute (file_handle);
			current_attribute = current_attribute->NextAttribute ();
		}

		CloseData (file_handle);
	}
	catch (...)
	{
		throw;
	}
}

//---------------------------------------------------------------------------

NexusField::~NexusField (void)
{
    if (attribute_list != NULL)
        delete (attribute_list);

    if (next_field != NULL)
        delete (next_field);

}

