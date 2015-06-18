//---------------------------------------------------------------------------
#ifdef __unix__
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define stricmp strcasecmp //unix uses strcasecmp instead of stricmp
#endif

#include <fcntl.h>
#include <sys/stat.h>

#include "nexusbox.h"

#ifdef WIN32
#include <io.h>
#include "windows.h"
#endif

//---------------------------------------------------------------------------
#ifdef WIN32
#pragma package(smart_init)
#endif

//---------------------------------------------------------------------------
//Public Methods for NexusVarAssoc
//---------------------------------------------------------------------------

NexusVarAssoc::NexusVarAssoc (void)
{
    name = NULL;
    address = NULL;
    next_in_list = NULL;

};

//---------------------------------------------------------------------------

NexusVarAssoc::NexusVarAssoc (char *new_name, int new_rank, int *new_dims, int new_data_type, void *new_address)
{
int loop;

    name = NULL;
    address = NULL;
    next_in_list = NULL;

    name = (char *) malloc ((strlen (new_name) * sizeof (char)) + 1);
    if (name == NULL)
    	return;
    strcpy (name, new_name);

    rank = new_rank;
    for (loop=0;loop<rank;loop++)
        dims[loop] = new_dims[loop];

    data_type = new_data_type;

    address = new_address;
};

//---------------------------------------------------------------------------

void NexusVarAssoc::UpdateVarInfo (int *new_dims, void *new_address)
{
int loop;

    for (loop=0;loop<rank;loop++)
        dims[loop] = new_dims[loop];

    address = new_address;
}

//---------------------------------------------------------------------------

void NexusVarAssoc::UpdateVarInfo (int *new_dims, int type, void *new_address)
{
int loop;

    for (loop=0;loop<rank;loop++)
        dims[loop] = new_dims[loop];

    data_type = type;

    address = new_address;
}

//---------------------------------------------------------------------------

NexusVarAssoc::~NexusVarAssoc ()
{
    if (name != NULL)
        free (name);

	if (next_in_list != NULL)
	{
    	delete ((NexusVarAssoc *) next_in_list);
		next_in_list = NULL;
	}
};

//---------------------------------------------------------------------------
//Public Methods for NexusFieldList
//---------------------------------------------------------------------------

NexusFieldList::NexusFieldList (void)
{
	nexus_field = NULL;
    var_info = NULL;
    next_in_list = NULL;
};

//---------------------------------------------------------------------------

void NexusFieldList::UpdateFromAssoc (void)
{
    nexus_field->UpdateVarInfo (var_info->rank, var_info->dims, var_info->data_type, var_info->address);
}

//---------------------------------------------------------------------------

NexusFieldList::~NexusFieldList (void)
{
	if (next_in_list != NULL)
	{
    	delete ((NexusFieldList *) next_in_list);
		next_in_list = NULL;
	}
};

//---------------------------------------------------------------------------
//Public Methods for NexusAttribList
//---------------------------------------------------------------------------

NexusAttribList::NexusAttribList (void)
{
	nexus_attribute = NULL;
    next_in_list = NULL;
};

//---------------------------------------------------------------------------

void NexusAttribList::UpdateFromAssoc (void)
{
    nexus_attribute->UpdateVarInfo (var_info->dims[0], var_info->address);
}

//---------------------------------------------------------------------------

NexusAttribList::~NexusAttribList (void)
{
	if (next_in_list != NULL)
	{
    	delete ((NexusAttribList *) next_in_list);
		next_in_list = NULL;
	}
};

//---------------------------------------------------------------------------
//Public Methods for NexusFileDirectory
//---------------------------------------------------------------------------

NexusFileDirectory::NexusFileDirectory (char *entry)
{
	nexus_group = NULL;
	nexus_field = NULL;
    nexus_attribute = NULL;
    next_in_list = NULL;

    directory_entry = (char *) malloc (sizeof(char)*strlen(entry)+1);
    strcpy (directory_entry, entry);
};

//---------------------------------------------------------------------------

NexusFileDirectory::NexusFileDirectory (char *entry, NexusGroup *group)
{
    nexus_group = group;
    nexus_field = NULL;
    nexus_attribute = NULL;
    next_in_list = NULL;

    directory_entry = (char *) malloc (sizeof(char)*strlen(entry)+1);
    strcpy (directory_entry, entry);
};

//---------------------------------------------------------------------------

NexusFileDirectory::NexusFileDirectory (char *entry, NexusField *field)
{
	nexus_group = NULL;
    nexus_field = field;
    nexus_attribute = NULL;
    next_in_list = NULL;

    directory_entry = (char *) malloc (sizeof(char)*strlen(entry)+1);
    strcpy (directory_entry, entry);
};

//---------------------------------------------------------------------------

NexusFileDirectory::NexusFileDirectory (char *entry, NexusAttribute *attribute)
{
	nexus_group = NULL;
    nexus_field = NULL;
    nexus_attribute = attribute;
    next_in_list = NULL;

    directory_entry = (char *) malloc (sizeof(char)*strlen(entry)+1);
    strcpy (directory_entry, entry);
};

//---------------------------------------------------------------------------

NexusFileDirectory *NexusFileDirectory::FindByIndex (char *search_index)
{
NexusFileDirectory 		*current_entry;

	current_entry = this;
   	while (stricmp (current_entry->directory_entry, search_index))
	{
   	    current_entry = (NexusFileDirectory *) current_entry->NextInList ();
       	if (current_entry == NULL)
            return (NULL);
//			throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::FindByIndex!", (char *) "Index not found!"));
   	}

    return (current_entry);
}

//---------------------------------------------------------------------------

NexusFileDirectory::~NexusFileDirectory (void)
{
    if (directory_entry != NULL)
        free (directory_entry);

	if (next_in_list != NULL)
	{
    	delete ((NexusFileDirectory *) next_in_list);
		next_in_list = NULL;
	}
};

//---------------------------------------------------------------------------
//Public Methods for NexusBox
//---------------------------------------------------------------------------

NexusBoxClass::NexusBoxClass (void)
{
    InitFileSystem ();
}

//---------------------------------------------------------------------------

void NexusBoxClass::acknowledgements (LogFileClass *acknowledge_file)
{
    acknowledge_file->Message ("__________________________________________________________________");
    acknowledge_file->Message ("NexusBox library");
    acknowledge_file->Message ("");
    acknowledge_file->Message ("Top level interface to a group of classes used to read and write HDF files.");
    acknowledge_file->Message ("Developed and maintained by:");
    acknowledge_file->Message ("       Brian Tieman");
    acknowledge_file->Message ("       Argonne National Laboratory");
    acknowledge_file->Message ("       tieman@aps.anl.gov");
    acknowledge_file->Message ("");
    acknowledge_file->Message ("8/20/2003  V4.0   BT  First version with acknowledgements");
    acknowledge_file->Message ("8/20/2003  V4.0   BT  Ported to Kylix");
    acknowledge_file->Message ("1/10/2006  V5.0   BT  Upgraded to make use of Nexus 3.0.0.  This allows for");
    acknowledge_file->Message ("		use of HDF5 to read/write files.  Modifications were made to handle");
    acknowledge_file->Message ("		data compression with the HDF5 data format.  HDF5 only supports LZW");
    acknowledge_file->Message ("		compression--thus, for HDF5, if RLE or HUF is requested, I made the");
    acknowledge_file->Message ("		to force LZW compression.  Also, the ability to write XML files is not");
    acknowledge_file->Message ("		tested very well--and isn't even built into the linux stuff.  The XML");
    acknowledge_file->Message ("		format is incredibly slow so it's unlikely anyone will want it anyway.");
    acknowledge_file->Message ("");
    acknowledge_file->Message ("");
    acknowledge_file->Message ("");
    acknowledge_file->Message ("");
    acknowledge_file->Message ("");
    acknowledge_file->Message ("");
    acknowledge_file->Message ("");
    acknowledge_file->Message ("");
    acknowledge_file->Message ("");
    acknowledge_file->Message ("__________________________________________________________________");
}

//---------------------------------------------------------------------------

void NexusBoxClass::InitFileSystem()
{
char        *group_type,
            *group_name;
int 		loop,
			dims[10];

	//Default to hdf5
	file_mode = HDF5_MODE;

	//Default to no compression
	compression_scheme = NX_COMP_NONE;

    //Default to read entire contents
    read_scheme = ENTIRE_CONTENTS;

    group_type = (char *) malloc (strlen ("TreeTop") + 1);
    if (group_type == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::NexusBoxClass!", (char *) "Memory allocation error!"));
    strcpy (group_type, "TreeTop");
    group_name = (char *) malloc (strlen ("Tree1") + 1);
    if (group_name == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::NexusBoxClass!", (char *) "Memory allocation error!"));
    strcpy (group_name, "Tree1");

    tree_top = new NexusGroup (group_type, group_name);
    if (tree_top == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::NexusBoxClass!", (char *) "Memory allocation error!"));

    free (group_type);
    free (group_name);

    top_of_varlist_address = malloc (sizeof (char) * (strlen ("Top of List")+1));
    if (top_of_varlist_address == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::NexusBoxClass!", (char *) "Memory allocation error!"));
    strcpy ((char *) top_of_varlist_address, "Top of List");
    dims[0] = strlen ("Test var attribute.");
    dims[1] = 0;
    var_associations = new NexusVarAssoc ((char *) "List_Top", 1, dims, NX_INT32, top_of_varlist_address);
    if (var_associations == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::NexusBoxClass!", (char *) "Memory allocation error!"));

    pv_fields = NULL;
    var_fields = NULL;

    pv_attribs = NULL;
    var_attribs = NULL;

    directory = new NexusFileDirectory ((char *) ";");

    data_path_name = NULL;
    data_file_name = NULL;

    for (loop=0;loop<10000;loop++)
    	tag_list[loop] = NULL;
}

//---------------------------------------------------------------------------

long int NexusBoxClass::InitTemplate (char *file_path, char *file_name)
{
char            *unparsed_template;
int 			template_file;
char			cfilename[256];
int             tag_index;
struct stat     statbuf;

	try
    {
		DeleteDataStructures ();

		strcpy (cfilename, file_path);
	    strcat (cfilename, file_name);

        template_file = open (cfilename, O_RDONLY);
	    if (template_file < 0)
			throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::InitTemplate!", (char *) "File open error!"));

	    error_code = stat (cfilename, &statbuf);
    	if (error_code < 0)
			throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::InitTemplate!", (char *) "File open error!"));

    	unparsed_template = (char *) malloc (sizeof(char)*(statbuf.st_size + 1));
	    if (unparsed_template == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::InitTemplate!", (char *) "Memory allocation error!"));

	    error_code = read (template_file, unparsed_template, statbuf.st_size);
    	if (error_code < 0)
			throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::InitTemplate!", (char *) "File read error!"));
        unparsed_template[statbuf.st_size] = '\0';

    	close (template_file);

	    GenerateTagList (unparsed_template);

	    tag_index = 0;
    	index_entry[0] = '\0';
	    directory = new NexusFileDirectory ((char *) ";");
    	if (directory == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::InitTemplate!", (char *) "Could not create object NexusFileDirectory!"));
    	tree_top->AddSubgroupList (ParseTags (tree_top, &tag_index));

	    DestroyTagList ();

    	free (unparsed_template);

#ifdef USECAPV
	    ConnectPVs ();
#endif

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif

        return (1);
    }

}

//---------------------------------------------------------------------------

long int NexusBoxClass::InsertTemplate (char * index, char *file_path, char *file_name)
{
char                *unparsed_template;
int 			    template_file;
char			    cfilename[256];
int                 tag_index;
struct stat         statbuf;
NexusFileDirectory  *insertion_entry;

	try
    {
		strcpy (cfilename, file_path);
	    strcat (cfilename, file_name);

    	template_file = open (cfilename, O_RDONLY);
	    if (template_file < 0)
			throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::InitTemplate!", (char *) "File open error!"));

	    error_code = stat (cfilename, &statbuf);
    	if (error_code < 0)
			throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::InitTemplate!", (char *) "File open error!"));

    	unparsed_template = (char *) malloc (sizeof(char)*(statbuf.st_size + 1));
	    if (unparsed_template == NULL)
			throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::InitTemplate!", (char *) "Memory allocation error!"));

	    error_code = read (template_file, unparsed_template, statbuf.st_size);
    	if (error_code < 0)
			throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::InitTemplate!", (char *) "File read error!"));
        unparsed_template[statbuf.st_size] = '\0';

    	close (template_file);

	    GenerateTagList (unparsed_template);

	    tag_index = 0;
    	index_entry[0] = '\0';

	    insertion_entry = directory->FindByIndex (index);
        if (insertion_entry->nexus_group != NULL)
        	insertion_entry->nexus_group->AddSubgroupList (ParseTags (insertion_entry->nexus_group, &tag_index));

	    DestroyTagList ();

    	free (unparsed_template);

#ifdef USECAPV
	    ConnectPVs ();
#endif
        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif

        return (1);
    }
}

//---------------------------------------------------------------------------

void NexusBoxClass::ResetFileSystem()
{
    DeleteDataStructures();
    InitFileSystem();
}

//---------------------------------------------------------------------------

void NexusBoxClass::GetIndexSize (long int *elements)
{
NexusFileDirectory  *current_entry;

    current_entry = directory;

    *elements = 0;
    while (current_entry->NextInList () != NULL)
    {
        current_entry = (NexusFileDirectory *) current_entry->NextInList ();

        if ((current_entry->nexus_group != NULL) || (current_entry->nexus_field != NULL) || (current_entry->nexus_attribute != NULL))
            *elements = *elements + 1;
    }
}

//---------------------------------------------------------------------------

char *NexusBoxClass::GetIndex (long int element_number)
{
NexusFileDirectory  *current_entry;
long int			element;

    current_entry = directory;

    element = -1;
    while ((current_entry->NextInList () != NULL) && (element != element_number))
    {
        current_entry =  (NexusFileDirectory *) current_entry->NextInList ();

        if ((current_entry->nexus_group != NULL) || (current_entry->nexus_field != NULL) || (current_entry->nexus_attribute != NULL))
            element = element + 1;
    }

    return (current_entry->directory_entry);

}

//---------------------------------------------------------------------------

char *NexusBoxClass::GetIndex (long int element_number, int *index_type)
{
NexusFileDirectory  *current_entry;
long int			element;

    current_entry = directory;

    element = -1;
    while ((current_entry->NextInList () != NULL) && (element != element_number))
    {
        current_entry =  (NexusFileDirectory *) current_entry->NextInList ();

        if ((current_entry->nexus_group != NULL) || (current_entry->nexus_field != NULL) || (current_entry->nexus_attribute != NULL))
            element = element + 1;
    }

    if (current_entry->nexus_group != NULL)
        *index_type = INDEX_OF_GROUP;
    if (current_entry->nexus_field != NULL)
        *index_type = INDEX_OF_FIELD;
    if (current_entry->nexus_attribute != NULL)
        *index_type = INDEX_OF_ATTRIBUTE;

    return (current_entry->directory_entry);

}

//---------------------------------------------------------------------------

int NexusBoxClass::IndexExists (char *index)
{
    try
    {
        //if this command doesn't throw an exception--the index must exist//
        if (directory == NULL)
            return (0);

        if (directory->FindByIndex (index) != NULL)
            return(1);
        else
            return(0);
    }
    catch (NexusExceptionClass &exception)
    {
        return (0);
    }
}

//---------------------------------------------------------------------------

void NexusBoxClass::PutDatum (char *index, char *new_val)
{
NexusFileDirectory  *current_entry;

	try
    {
	    current_entry = directory->FindByIndex (index);

	    if (current_entry->nexus_field != NULL)
		    current_entry->nexus_field->PutDataVal (new_val, strlen (new_val));
	    else
		    if (current_entry->nexus_attribute != NULL)
			    current_entry->nexus_attribute->PutDataVal (new_val, strlen (new_val));
	        else
				throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::PutDatum!", (char *) "Index not found!"));
    }
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
    }
}

//---------------------------------------------------------------------------

void NexusBoxClass::PutDatum (char *index, unsigned short *new_val)
{
NexusFileDirectory  *current_entry;

	try
    {
	    current_entry = directory->FindByIndex (index);

	    if (current_entry->nexus_field != NULL)
		    current_entry->nexus_field->PutDataVal (new_val);
	   else
		    if (current_entry->nexus_attribute != NULL)
			    current_entry->nexus_attribute->PutDataVal (new_val);
	        else
				throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::PutDatum!", (char *) "Index not found!"));
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
    }
}

//---------------------------------------------------------------------------

void NexusBoxClass::PutDatum (char *index, short *new_val)
{
NexusFileDirectory  *current_entry;

	try
    {
	    current_entry = directory->FindByIndex (index);

	    if (current_entry->nexus_field != NULL)
		    current_entry->nexus_field->PutDataVal (new_val);
	    else
		    if (current_entry->nexus_attribute != NULL)
			    current_entry->nexus_attribute->PutDataVal (new_val);
        	else
				throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::PutDatum!", (char *) "Index not found!"));
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
    }

}

//---------------------------------------------------------------------------

void NexusBoxClass::PutDatum (char *index, unsigned long *new_val)
{
NexusFileDirectory  *current_entry;

	try
    {
	    current_entry = directory->FindByIndex (index);

    	if (current_entry->nexus_field != NULL)
	    	current_entry->nexus_field->PutDataVal (new_val);
	    else
		    if (current_entry->nexus_attribute != NULL)
			    current_entry->nexus_attribute->PutDataVal (new_val);
	        else
   				throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::PutDatum!", (char *) "Index not found!"));
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
    }

}

//---------------------------------------------------------------------------

void NexusBoxClass::PutDatum (char *index, long *new_val)
{
NexusFileDirectory  *current_entry;

	try
    {
	    current_entry = directory->FindByIndex (index);

    	if (current_entry->nexus_field != NULL)
	    	current_entry->nexus_field->PutDataVal (new_val);
	    else
		    if (current_entry->nexus_attribute != NULL)
			    current_entry->nexus_attribute->PutDataVal (new_val);
	        else
				throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::PutDatum!", (char *) "Index not found!"));
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
    }

}

//---------------------------------------------------------------------------

void NexusBoxClass::PutDatum (char *index, float *new_val)
{
NexusFileDirectory  *current_entry;

	try
    {
	    current_entry = directory->FindByIndex (index);

    	if (current_entry->nexus_field != NULL)
	    	current_entry->nexus_field->PutDataVal (new_val);
	    else
		    if (current_entry->nexus_attribute != NULL)
			    current_entry->nexus_attribute->PutDataVal (new_val);
	        else
				throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::PutDatum!", (char *) "Index not found!"));
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
    }

}

//---------------------------------------------------------------------------

void NexusBoxClass::PutDatum (char *index, double *new_val)
{
NexusFileDirectory  *current_entry;

	try
    {
	    current_entry = directory->FindByIndex (index);

    	if (current_entry->nexus_field != NULL)
	    	current_entry->nexus_field->PutDataVal (new_val);
	    else
		    if (current_entry->nexus_attribute != NULL)
			    current_entry->nexus_attribute->PutDataVal (new_val);
	        else
				throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::PutDatum!", (char *) "Index not found!"));
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
    }

}

//---------------------------------------------------------------------------

void NexusBoxClass::PutDatum (char *index, void *new_val, int new_rank, int *new_dims, int new_type)
{
NexusFileDirectory  *current_entry;

	try
    {
	    current_entry = directory->FindByIndex (index);

    	if (current_entry->nexus_field != NULL)
	    	current_entry->nexus_field->PutDataVal (new_val, new_rank, new_dims, new_type);
	    else
		    if (current_entry->nexus_attribute != NULL)
			    current_entry->nexus_attribute->PutDataVal (new_val, new_rank, new_dims, new_type);
	        else
				throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::PutDatum!", (char *) "Index not found!"));
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
    }

}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatumInfo (char *index, int *get_rank, int *get_dims, int *get_type)
{
NexusFileDirectory  *current_entry;

	try
    {
	    current_entry = directory->FindByIndex (index);

	    if (current_entry->nexus_field != NULL)
		    current_entry->nexus_field->GetDataInfo (get_rank, get_dims, get_type);
	    else
		    if (current_entry->nexus_attribute != NULL)
			    current_entry->nexus_attribute->GetDataInfo (get_rank, get_dims, get_type);
    	    else
    		    if (current_entry->nexus_group != NULL)
                    *get_rank = -1;
                else
    				throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::GetDatumInfo!", (char *) "Index not found!"));

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }

}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatum (char *index, unsigned char *get_data)
{
NexusFileDirectory  *current_entry;

	try
    {
        if (read_scheme != ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::GetDatum!", (char *) "File must be in ENTIRE_CONTENTS mode!"));

	    current_entry = directory->FindByIndex (index);

    	if (current_entry->nexus_field != NULL)
   		    current_entry->nexus_field->GetDataVal (get_data);
	    else
		    if (current_entry->nexus_attribute != NULL)
			    current_entry->nexus_attribute->GetDataVal (get_data);
	        else
				throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::GetDatum!", (char *) "Index not found!"));
        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }

}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatumSlab (char *index, unsigned char *get_data, int *start_dims, int *size_dims)
{
	try
    {
        if (read_scheme == ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::GetDatumSlab!", (char *) "File must be in INDEX_ONLY mode!"));

	    GetDataFromIndex (index, get_data, start_dims, size_dims);

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }
}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatum (char *index, char *get_data)
{
NexusFileDirectory  *current_entry;

	try
    {
        if (read_scheme != ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *) (char *) "Error in method NexusBoxClass::GetDatumSlab!", (char *) "File must be in ENTIRE_CONTENTS mode!"));

	    current_entry = directory->FindByIndex (index);

	    if (current_entry->nexus_field != NULL)
   		    current_entry->nexus_field->GetDataVal (get_data);
	    else
		    if (current_entry->nexus_attribute != NULL)
			    current_entry->nexus_attribute->GetDataVal (get_data);
	        else
				throw (new NexusExceptionClass ((char *) "Error in method NexusBoxClass::GetDatum!", (char *) "Index not found!"));
        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }

}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatumSlab (char *index, char *get_data, int *start_dims, int *size_dims)
{
	try
    {
        if (read_scheme == ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatumSlab!", (char *)"File must be in INDEX_ONLY mode!"));

	    GetDataFromIndex (index, get_data, start_dims, size_dims);

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }
}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatum (char *index, unsigned short *get_data)
{
NexusFileDirectory  *current_entry;

	try
    {
        if (read_scheme != ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatum!", (char *)"File must be in ENTIRE_CONTENTS mode!"));

	    current_entry = directory->FindByIndex (index);

	    if (current_entry->nexus_field != NULL)
   		    current_entry->nexus_field->GetDataVal (get_data);
	    else
		    if (current_entry->nexus_attribute != NULL)
			    current_entry->nexus_attribute->GetDataVal (get_data);
	        else
				throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatum!", (char *)"Index not found!"));

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }

}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatumSlab (char *index, unsigned short *get_data, int *start_dims, int *size_dims)
{
	try
    {
        if (read_scheme == ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatumSlab!", (char *)"File must be in INDEX_ONLY mode!"));

	    GetDataFromIndex (index, get_data, start_dims, size_dims);

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }
}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatum (char *index, short *get_data)
{
NexusFileDirectory  *current_entry;

	try
    {
        if (read_scheme != ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatum!", (char *)"File must be in ENTIRE_CONTENTS mode!"));

	    current_entry = directory->FindByIndex (index);

	    if (current_entry->nexus_field != NULL)
   		    current_entry->nexus_field->GetDataVal (get_data);
	    else
		    if (current_entry->nexus_attribute != NULL)
			    current_entry->nexus_attribute->GetDataVal (get_data);
	        else
				throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatum!", (char *)"Index not found!"));

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }

}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatumSlab (char *index, short *get_data, int *start_dims, int *size_dims)
{
	try
    {
        if (read_scheme == ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatumSlab!", (char *)"File must be in INDEX_ONLY mode!"));

	    GetDataFromIndex (index, get_data, start_dims, size_dims);

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }
}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatum (char *index, unsigned long *get_data)
{
NexusFileDirectory  *current_entry;

	try
    {
        if (read_scheme != ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatum!", (char *)"File must be in ENTIRE_CONTENTS mode!"));

	    current_entry = directory->FindByIndex (index);

	    if (current_entry->nexus_field != NULL)
   		    current_entry->nexus_field->GetDataVal (get_data);
	    else
		    if (current_entry->nexus_attribute != NULL)
			    current_entry->nexus_attribute->GetDataVal (get_data);
	        else
				throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatum!", (char *)"Index not found!"));

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }

}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatumSlab (char *index, unsigned long *get_data, int *start_dims, int *size_dims)
{
	try
    {
        if (read_scheme == ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatumSlab!", (char *)"File must be in INDEX_ONLY mode!"));

	    GetDataFromIndex (index, get_data, start_dims, size_dims);

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }
}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatum (char *index, long *get_data)
{
NexusFileDirectory  *current_entry;

	try
    {
        if (read_scheme != ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatum!", (char *)"File must be in ENTIRE_CONTENTS mode!"));

	    current_entry = directory->FindByIndex (index);

    	if (current_entry->nexus_field != NULL)
   		    current_entry->nexus_field->GetDataVal (get_data);
	    else
		    if (current_entry->nexus_attribute != NULL)
			    current_entry->nexus_attribute->GetDataVal (get_data);
	        else
				throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatum!", (char *)"Index not found!"));

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }

}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatumSlab (char *index, long *get_data, int *start_dims, int *size_dims)
{
	try
    {
        if (read_scheme == ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatumSlab!", (char *)"File must be in INDEX_ONLY mode!"));

	    GetDataFromIndex (index, get_data, start_dims, size_dims);

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }
}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatum (char *index, float *get_data)
{
NexusFileDirectory  *current_entry;

	try
    {
        if (read_scheme != ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatum!", (char *)"File must be in ENTIRE_CONTENTS mode!"));

	    current_entry = directory->FindByIndex (index);

    	if (current_entry->nexus_field != NULL)
   		    current_entry->nexus_field->GetDataVal (get_data);
	    else
		    if (current_entry->nexus_attribute != NULL)
			    current_entry->nexus_attribute->GetDataVal (get_data);
	        else
				throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatum!", (char *)"Index not found!"));

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }

}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatumSlab (char *index, float *get_data, int *start_dims, int *size_dims)
{
	try
    {
        if (read_scheme == ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatumSlab!", (char *)"File must be in INDEX_ONLY mode!"));

	    GetDataFromIndex (index, get_data, start_dims, size_dims);

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }
}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatum (char *index, double *get_data)
{
NexusFileDirectory  *current_entry;

	try
    {
        if (read_scheme != ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatum!", (char *)"File must be in ENTIRE_CONTENTS mode!"));

	    current_entry = directory->FindByIndex (index);

    	if (current_entry->nexus_field != NULL)
   		    current_entry->nexus_field->GetDataVal (get_data);
	    else
		    if (current_entry->nexus_attribute != NULL)
			    current_entry->nexus_attribute->GetDataVal (get_data);
	        else
				throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatum!", (char *)"Index not found!"));

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }

}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatumSlab (char *index, double *get_data, int *start_dims, int *size_dims)
{
	try
    {
        if (read_scheme == ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatumSlab!", (char *)"File must be in INDEX_ONLY mode!"));

	    GetDataFromIndex (index, get_data, start_dims, size_dims);

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }
}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatum (char *index, void *get_data)
{
NexusFileDirectory  *current_entry;

	try
    {
        if (read_scheme != ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatumSlab!", (char *)"File must be in ENTIRE_CONTENTS mode!"));

	    current_entry = directory->FindByIndex (index);

    	if (current_entry->nexus_field != NULL)
   		    current_entry->nexus_field->GetDataVal (get_data);
	    else
		    if (current_entry->nexus_attribute != NULL)
			    current_entry->nexus_attribute->GetDataVal (get_data);
	        else
				throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatum!", (char *)"Index not found!"));

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }

}

//---------------------------------------------------------------------------

int NexusBoxClass::GetDatumSlab (char *index, void *get_data, int *start_dims, int *size_dims)
{
	try
    {
        if (read_scheme == ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GetDatumSlab!", (char *)"File must be in INDEX_ONLY mode!"));

	    GetDataFromIndex (index, get_data, start_dims, size_dims);

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }
}

//---------------------------------------------------------------------------

int NexusBoxClass::CreateGroup (char *index, char *group_name, char *group_type)
{
NexusFileDirectory  *current_entry;
NexusGroup          *new_group;
char                temp_index_entry[256];

	try
    {
	    current_entry = directory->FindByIndex (index);

        new_group = new NexusGroup (group_name, group_type);
        if (new_group == NULL)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::CreateGroup!", (char *)"Could not create object NexusGroup!"));

        if (current_entry->nexus_group == NULL)
            if (tree_top->SubGroup () == NULL)
                tree_top->AddSubgroupList (new_group);
            else
                tree_top->AddToGroupList (tree_top->SubGroup (), new_group);
        else
            if (current_entry->nexus_group->SubGroup () == NULL)
                current_entry->nexus_group->AddSubgroupList (new_group);
            else
                current_entry->nexus_group->AddToGroupList (current_entry->nexus_group->SubGroup (), new_group);

        if (strlen(index) != 1)
            strcpy (temp_index_entry, index);
        else
            strcpy (temp_index_entry, "");
        strcat (temp_index_entry, INDEX_SEPERATOR_STR);
        strcat (temp_index_entry, group_name);

	    directory->AddToList (new NexusFileDirectory (temp_index_entry, (NexusGroup *) new_group));

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }

}

//---------------------------------------------------------------------------

int NexusBoxClass::CreateField (char *index, char *field_name, int rank, int *dims, int type, void *data)
{
NexusFileDirectory  *current_entry;
NexusField          *new_field;
char                temp_index_entry[256];

	try
    {
	    current_entry = directory->FindByIndex (index);

        new_field = new NexusField ();
        if (new_field == NULL)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::CreateField!", (char *)"Could not create object NexusField!"));

        new_field->PutSDSInfo (field_name, rank, dims, type, data);
        current_entry->nexus_group->AddToFieldList (new_field);

        strcpy (temp_index_entry, index);
        strcat (temp_index_entry, INDEX_SEPERATOR_STR);
        strcat (temp_index_entry, field_name);

	    directory->AddToList (new NexusFileDirectory (temp_index_entry, (NexusField *) new_field));

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }

}

//---------------------------------------------------------------------------

int NexusBoxClass::CreateAttribute (char *index, char *attribute_name, int length, int type, void *data)
{
NexusFileDirectory  *current_entry;
NexusAttribute      *new_attribute;
char                temp_index_entry[256];

	try
    {
	    current_entry = directory->FindByIndex (index);

        new_attribute = new NexusAttribute ();
        if (new_attribute == NULL)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::CreateAttribute!", (char *)"Could not create object NexusAttribute!"));

        new_attribute->PutSDSInfo (attribute_name, length, type, data);
        current_entry->nexus_field->AddToAttributeList (new_attribute);

        strcpy (temp_index_entry, index);
        strcat (temp_index_entry, INDEX_SEPERATOR_STR);
        strcat (temp_index_entry, attribute_name);

	    directory->AddToList (new NexusFileDirectory (temp_index_entry, (NexusAttribute *) new_attribute));

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }

}

//---------------------------------------------------------------------------
#ifdef USECAPV
int NexusBoxClass::ConnectPVs (void)
{
NexusFieldList  *current_field;
NexusAttribList *current_attribute;

	try
    {
	    current_field = pv_fields;

    	while (current_field != NULL)
	    {
    	    current_field->nexus_field->PVConnect ();
        	current_field = (NexusFieldList *) current_field->NextInList ();
	    }

    	current_attribute = pv_attribs;

	    while (current_attribute != NULL)
    	{
        	current_attribute->nexus_attribute->PVConnect ();
	        current_attribute = (NexusAttribList *) current_attribute->NextInList ();
    	}

        return (0);
	}
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }
}
#endif
//---------------------------------------------------------------------------
#ifdef USECAPV
int NexusBoxClass::UpdatePVs (void)
{
NexusFieldList  *current_field;
NexusAttribList *current_attribute;

	try
    {
	    current_field = pv_fields;

    	while (current_field != NULL)
	    {
    	    current_field->nexus_field->PVUpdate ();
        	current_field = (NexusFieldList *) current_field->NextInList ();
	    }

    	current_attribute = pv_attribs;

	    while (current_attribute != NULL)
    	{
        	current_attribute->nexus_attribute->PVUpdate ();
	        current_attribute = (NexusAttribList *) current_attribute->NextInList ();
    	}

        return (0);
    }
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }
}
#endif
//---------------------------------------------------------------------------
#ifdef USECAPV
int NexusBoxClass::DisconnectPVs (void)
{
NexusFieldList  *current_field;
NexusAttribList *current_attribute;

    try
    {
	    current_field = pv_fields;

    	while (current_field != NULL)
	    {
    	    current_field->nexus_field->PVDisconnect ();
        	current_field = (NexusFieldList *) current_field->NextInList ();
	    }

    	current_attribute = pv_attribs;

	    while (current_attribute != NULL)
    	{
        	current_attribute->nexus_attribute->PVDisconnect ();
	        current_attribute = (NexusAttribList *) current_attribute->NextInList ();
    	}

        return (0);
    }
	catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }
}
#endif
//---------------------------------------------------------------------------

int NexusBoxClass::UpdateVars (void)
{
NexusFieldList  *current_field;
NexusAttribList *current_attribute;

	try
    {
    	current_field = var_fields;

	    while (current_field != NULL)
    	{
        	current_field->UpdateFromAssoc ();
	        current_field = (NexusFieldList *) current_field->NextInList ();
    	}

	    current_attribute = var_attribs;

    	while (current_attribute != NULL)
	    {
    	    current_attribute->UpdateFromAssoc ();
        	current_attribute = (NexusAttribList *) current_attribute->NextInList ();
	    }

        return (0);
    }
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }
}

//---------------------------------------------------------------------------

int NexusBoxClass::RegisterVar (char *name, int rank, int *dims, int data_type, void *address)
{
NexusVarAssoc  *current_var;

	try
    {
	    current_var = var_associations;

    	while (current_var->next_in_list != NULL)
        	current_var = (NexusVarAssoc *) current_var->NextInList ();

	    current_var->next_in_list = new NexusVarAssoc (name, rank, dims, data_type, address);
    	if (current_var->next_in_list == NULL)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::RegisterVar!", (char *)"Could not create object NexusVarAssoc!"));

        return (0);
    }
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }
}

//---------------------------------------------------------------------------

int NexusBoxClass::UpdateVarInfo (char *name, int *dims, void *address)
{
NexusVarAssoc  *current_var;

    try
    {
	    current_var = var_associations;

    	while (stricmp (current_var->name, name))
	    {
    	    current_var = (NexusVarAssoc *) current_var->NextInList ();
        	if (current_var == NULL)
				throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::UpdateVarInfo!", (char *)"Variable not found!"));
		}

    	current_var->UpdateVarInfo (dims, address);

        return (0);
    }
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }
}

//---------------------------------------------------------------------------

int NexusBoxClass::UpdateVarInfo (char *name, int *dims, int type, void *address)
{
NexusVarAssoc  *current_var;

    try
    {
	    current_var = var_associations;

    	while (stricmp (current_var->name, name))
	    {
    	    current_var = (NexusVarAssoc *) current_var->NextInList ();
        	if (current_var == NULL)
				throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::UpdateVarInfo!", (char *)"Variable not found!"));
		}

    	current_var->UpdateVarInfo (dims, type, address);

        return (0);
    }
    catch (NexusExceptionClass &exception)
    {
/*#ifdef WIN32
    	MessageBox (NULL, (LPCTSTR)exception.getErrorString(),(LPCTSTR)"Error!", MB_OK | MB_ICONINFORMATION);
#endif*/
        return (1);
    }
}

//---------------------------------------------------------------------------

int NexusBoxClass::CreateTemplate (char *path_name, char *file_name)
{
    read_scheme = INDEX_ONLY;

    ReadAll (path_name, file_name);

	return (0);
}

//---------------------------------------------------------------------------

int NexusBoxClass::ChangeFile (char *path_name, char *file_name)
{
	if (data_path_name != NULL)
		free (data_path_name);
   	data_path_name = (char *) malloc (sizeof (char) * (strlen (path_name) + 1));
   	strcpy (data_path_name, path_name);

	if (data_file_name != NULL)
       	free (data_file_name);
	data_file_name = (char *) malloc (sizeof (char) * (strlen (file_name) + 1));
	strcpy (data_file_name, file_name);

	return (0);
}

//---------------------------------------------------------------------------

int NexusBoxClass::SetFileMode (int file_mode)
{
    this->file_mode = file_mode;

    return (0);
}

//---------------------------------------------------------------------------

int NexusBoxClass::GetFileMode (void)
{
    return (file_mode);
}

//---------------------------------------------------------------------------

int NexusBoxClass::ReadAll (char *path_name, char *file_name)
{
	try
	{

 		if (data_path_name != NULL)
			free (data_path_name);
		data_path_name = (char *) malloc (sizeof (char) * (strlen (path_name) + 1));
		strcpy (data_path_name, path_name);

		if (data_file_name != NULL)
			free (data_file_name);
		data_file_name = (char *) malloc (sizeof (char) * (strlen (file_name) + 1));
		strcpy (data_file_name, file_name);

		DeleteDataStructures ();

		OpenFile (NXACC_READ, data_path_name, data_file_name, &file_handle);

		index_entry[0] = '\0';
		directory = new NexusFileDirectory ((char *) ";");
		if (directory == NULL)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::ReadAll!", (char *)"Could not create object NexusFileDirectory!"));
		tree_top->AddSubgroupList (ParseHDFFile (tree_top));

		CloseFile (&file_handle);

		return (0);
	}
	catch (...)
	{
		return (1);
	}
}

//---------------------------------------------------------------------------

int NexusBoxClass::WriteAll (char *path_name, char *file_name)
{
NexusGroup  *current_group;

	try
	{
	    if (read_scheme != ENTIRE_CONTENTS)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::WriteAll!", (char *)"Filesystem in INDEX_ONLY mode--can not write file!"));

        switch (file_mode)
        {
            case HDF4_MODE: OpenFile (NXACC_CREATE4, path_name, file_name, &file_handle); break;
            case HDF5_MODE: OpenFile (NXACC_CREATE5, path_name, file_name, &file_handle); break;
            case XML_MODE: OpenFile (NXACC_CREATEXML, path_name, file_name, &file_handle); break;
            default : OpenFile (NXACC_CREATE4, path_name, file_name, &file_handle); break;
        }

	    current_group = tree_top->SubGroup ();

		while (current_group != NULL)
		{
			current_group->WriteGroup (file_handle, compression_scheme);

			current_group = current_group->NextGroup ();
		}

		CloseFile (&file_handle);

		return (0);
	}
	catch (...)
	{
		return (1);
    }
}

//---------------------------------------------------------------------------

void NexusBoxClass::CompressionScheme (int new_scheme)
{
    compression_scheme = new_scheme;
}

//---------------------------------------------------------------------------

int NexusBoxClass::CompressionScheme (void)
{
    return (compression_scheme);
}

//---------------------------------------------------------------------------

void NexusBoxClass::SetReadScheme (int new_scheme)
{
    read_scheme = new_scheme;
}

//---------------------------------------------------------------------------

NexusBoxClass::~NexusBoxClass (void)
{
    //Deleting the top of the tree deletes the entire tree recursively...
    if (tree_top != NULL)
        delete (tree_top);

    if (pv_fields != NULL)
        delete (pv_fields);

    if (var_fields != NULL)
        delete (var_fields);

    if (pv_attribs != NULL)
        delete (pv_attribs);

    if (var_attribs != NULL)
        delete (var_attribs);

    if (top_of_varlist_address != NULL)
        free (top_of_varlist_address);

    if (data_path_name != NULL)
        free (data_path_name);

    if (data_file_name != NULL)
        free (data_file_name);

    if (directory != NULL)
        delete (directory);

    if (var_associations != NULL)
        delete (var_associations);

}

//---------------------------------------------------------------------------
//Private Methods
//---------------------------------------------------------------------------

void NexusBoxClass::GenerateTagList (char *unparsed_template)
{
char            start_delimiters[5],
                end_delimiters[5],
                newline_delimiter[5],
                *tag_start;
int             num_chars,
                tag_num;

    strcpy (start_delimiters, "<[{;");
    strcpy (end_delimiters, ">]}");
    strcpy (newline_delimiter, "\n");

    tag_num = 0;

    tag_start = unparsed_template;
    tag_start = strpbrk (tag_start, start_delimiters);
    while (tag_start != NULL)
    {
        if (tag_start[0] == ';')
            tag_start = strpbrk (tag_start, newline_delimiter);
        else
        {
            num_chars = strcspn (tag_start, end_delimiters);
            tag_list[tag_num] = (char *) malloc (num_chars + 2);
            if (tag_list[tag_num] == NULL)
				throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::GenerateTagList!", (char *)"Memory allocation error!"));

            strncpy (tag_list[tag_num], tag_start, num_chars + 1);
            tag_list[tag_num][num_chars + 1] = '\0';
            tag_start = strpbrk (tag_start, end_delimiters);
            tag_num++;
        }

        if (tag_start != NULL)
            tag_start = strpbrk (tag_start, start_delimiters);
    }
}

//---------------------------------------------------------------------------

void NexusBoxClass::DestroyTagList (void)
{
int     tag_num;

    tag_num = 0;
    while (tag_list[tag_num] != NULL)
    {
        free (tag_list[tag_num]);
        tag_num++;
    }

}

//---------------------------------------------------------------------------

void NexusBoxClass::DeleteDataStructures (void)
{
char        *group_type,
            *group_name;

#ifdef USECAPV
	DisconnectPVs ();
#endif

    if (directory != NULL)
        delete (directory);
    directory = NULL;

    //Deleting the top of the tree deletes the entire tree recursively...
    if (tree_top != NULL)
        delete (tree_top);
    tree_top = NULL;

    if (pv_fields != NULL)
        delete (pv_fields);
    pv_fields = NULL;

    if (var_fields != NULL)
        delete (var_fields);
	var_fields = NULL;

    if (pv_attribs != NULL)
        delete (pv_attribs);
    pv_attribs = NULL;

    if (var_attribs != NULL)
        delete (var_attribs);
	var_attribs = NULL;

//Recreate top of list.
    group_type = (char *) malloc (strlen ("TreeTop") + 1);
    if (group_type == NULL)
		throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::DeleteDataStructures!", (char *)"Memory allocation error!"));

    strcpy (group_type, "TreeTop");
    group_name = (char *) malloc (strlen ("Tree1") + 1);
    if (group_name == NULL)
		throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::DeleteDataStructures!", (char *)"Memory allocation error!"));
    strcpy (group_name, "Tree1");

    tree_top = new NexusGroup (group_type, group_name);
    if (tree_top == NULL)
		throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::DeleteDataStructures!", (char *)"Memory allocation error!"));

    free (group_type);
    free (group_name);
}

//---------------------------------------------------------------------------

void NexusBoxClass::GetGroupInfo (char *group_type, char *group_name, int tag_index)
{
char        group_seperator[5],
            end_delimiter[5],
            *tag_start;
int         num_chars;

    strcpy (group_seperator, ":");
    strcpy (end_delimiter, ">");

    tag_start = &tag_list[tag_index][1];
    num_chars = strcspn (tag_start, group_seperator);
    strncpy (group_type, tag_start, num_chars);
    group_type[num_chars] = '\0';

    tag_start = strpbrk (tag_start, group_seperator);
    tag_start++; //skip past the group seperator
    num_chars = strcspn (tag_start, end_delimiter);
    strncpy (group_name, tag_start, num_chars);
    group_name[num_chars] = '\0';
}

//---------------------------------------------------------------------------

void NexusBoxClass::GetFieldInfo (char *field_name, char *field_location, char *field_value,
                                    char *field_type, int tag_index)
{
char        field_seperator[5],
            end_delimiter[5],
            *tag_start;
int         num_chars;

    strcpy (field_seperator, ",");
    strcpy (end_delimiter, "]");

    tag_start = &tag_list[tag_index][1];
    num_chars = strcspn (tag_start, field_seperator);
    strncpy (field_name, tag_start, num_chars);
    field_name[num_chars] = '\0';

    tag_start = strpbrk (tag_start, field_seperator);
    tag_start++; //skip past the field seperator
    num_chars = strcspn (tag_start, field_seperator);
    strncpy (field_location, tag_start, num_chars);
    field_location[num_chars] = '\0';

    tag_start = strpbrk (tag_start, field_seperator);
    tag_start++; //skip past the field seperator
    tag_start++; //should be "
    num_chars = strcspn (tag_start, field_seperator);
    strncpy (field_value, tag_start, num_chars-1);
    field_value[num_chars-1] = '\0';

    tag_start = strpbrk (tag_start, field_seperator);
    tag_start++; //skip past the field seperator
    num_chars = strcspn (tag_start, end_delimiter);
    strncpy (field_type, tag_start, num_chars);
    field_type[num_chars] = '\0';
}

//---------------------------------------------------------------------------

void NexusBoxClass::GetAttributeInfo (char *attrib_name, char *attrib_location, char *attrib_value, char *attrib_type, int tag_index)
{
char        attrib_seperator[5],
            end_delimiter[5],
            *tag_start;
int         num_chars;

    strcpy (attrib_seperator, ",");
    strcpy (end_delimiter, "}");

    tag_start = &tag_list[tag_index][1];
    num_chars = strcspn (tag_start, attrib_seperator);
    strncpy (attrib_name, tag_start, num_chars);
    attrib_name[num_chars] = '\0';

    tag_start = strpbrk (tag_start, attrib_seperator);
    tag_start++; //skip past the field seperator
    num_chars = strcspn (tag_start, attrib_seperator);
    strncpy (attrib_location, tag_start, num_chars);
    attrib_location[num_chars] = '\0';

    tag_start = strpbrk (tag_start, attrib_seperator);
    tag_start++; //skip past the field seperator
    tag_start++; //should be "
    num_chars = strcspn (tag_start, attrib_seperator);
    strncpy (attrib_value, tag_start, num_chars-1);
    attrib_value[num_chars-1] = '\0';

    tag_start = strpbrk (tag_start, attrib_seperator);
    tag_start++; //skip past the field seperator
    num_chars = strcspn (tag_start, end_delimiter);
    strncpy (attrib_type, tag_start, num_chars);
    attrib_type[num_chars] = '\0';
}

//---------------------------------------------------------------------------

NexusGroup *NexusBoxClass::ParseTags (NexusGroup *parent_group, int *tag_index)
{
NexusGroup      *group_list = NULL,
                *current_group = NULL,
                *new_group = NULL;
NexusField      *new_field = NULL;
NexusAttribute  *new_attrib = NULL;
NexusVarAssoc   *var_association;
char            group_delimiter[5],
                field_delimiter[5],
                attribute_delimiter[5],

                group_type[50],
                group_name[50],

                field_name[50],
                field_location[50],
                field_value[50],
                field_type[50],

                attrib_name[50],
                attrib_location[50],
                attrib_value[50],
                attrib_type[50],

                *temp_chr;

    strcpy (group_delimiter, "<");
    strcpy (field_delimiter, "[");
    strcpy (attribute_delimiter, "{");

    while (stricmp (tag_list[*tag_index], "<EndTemplate>"))
    {
        switch (tag_list[*tag_index][0])
        {
            case '<' : {
                GetGroupInfo (group_type, group_name, *tag_index);

                if (stricmp (group_name, "end"))
                {
                    new_group = new NexusGroup (group_name, group_type);
                    if (new_group == NULL)
						throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::ParseTags!", (char *)"Could not create object NexusGroup!"));
                    strcat (index_entry, INDEX_SEPERATOR_STR);
                    strcat (index_entry, group_name);

                    if (group_list == NULL)
                    {
                        group_list = new_group;
                        current_group = group_list;
                    }
                    else
                    {
                        current_group->AddToGroupList (current_group, new_group);
                        current_group = current_group->NextGroup ();
                    }

				    directory->AddToList (new NexusFileDirectory (index_entry, (NexusGroup *) new_group));

                    *tag_index = *tag_index + 1;
                    current_group->AddSubgroupList (ParseTags (current_group, tag_index));
                }
                else
                {
                    *tag_index = *tag_index + 1;

                    temp_chr = strrchr (index_entry, INDEX_SEPERATOR_CHR);
                    if (temp_chr != NULL)
						temp_chr[0] = '\0';

                    return (group_list);
                }

                break;
            }

            case '[' : {
                GetFieldInfo (field_name, field_location, field_value, field_type, *tag_index);

//If CONST
                if (!stricmp (field_location, "const"))
                {
                    new_field = new NexusField ();
                    if (new_field == NULL)
						throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::ParseTags!", (char *)"Could not create object NexusField!"));
                    new_field->PutConstInfo (field_name, field_type, field_value);
                }

//If VAR
                if (!stricmp (field_location, "var"))
                {
                    var_association = FindVarAssociation (field_value);
                    new_field = new NexusField ();
                    if (new_field == NULL)
						throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::ParseTags!", (char *)"Could not create object NexusField!"));
                    new_field->PutVarInfo (field_name, var_association->rank, var_association->dims, var_association->data_type, var_association->address);
                    if (var_fields == NULL)
                    {
                        var_fields = new NexusFieldList ();
                        if (var_fields == NULL)
							throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::ParseTags!", (char *)"Could not create object NexusFieldList!"));

                        var_fields->nexus_field = new_field;
                        var_fields->var_info = var_association;
                    }
                    else
                        AddToFieldList (var_fields, new_field, var_association);
                }

//If PV
#ifdef USECAPV
                if (!stricmp (field_location, "pv"))
                {
                    new_field = new NexusField ();
                    if (new_field == NULL)
						throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::ParseTags!", (char *)"Could not create object NexusField!"));
                    new_field->PutPVInfo (field_name, field_type, field_value);
                    if (pv_fields == NULL)
                    {
                        pv_fields = new NexusFieldList ();
                        if (pv_fields == NULL)
							throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::ParseTags!", (char *)"Could not create object NexusFieldList!"));

                        pv_fields->nexus_field = new_field;
                    }
                    else
                        AddToFieldList (pv_fields, new_field);
                }
#endif
                parent_group->AddToFieldList (new_field);

                strcat (index_entry, INDEX_SEPERATOR_STR);
                strcat (index_entry, field_name);

			    directory->AddToList (new NexusFileDirectory (index_entry, (NexusField *) new_field));

                *tag_index = *tag_index + 1;

                while (tag_list[*tag_index][0] == '{')
                {
                    GetAttributeInfo (attrib_name, attrib_location, attrib_value, attrib_type, *tag_index);

//If CONST
                    if (!stricmp (attrib_location, "const"))
                    {
                        new_attrib = new NexusAttribute ();
	                    if (new_attrib == NULL)
							throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::ParseTags!", (char *)"Could not create object NexusAttribute!"));
                        new_attrib->PutConstInfo (attrib_name, attrib_type, attrib_value);
                    }
//If VAR
                    if (!stricmp (attrib_location, "var"))
                    {
                        var_association = FindVarAssociation (attrib_value);
                        new_attrib = new NexusAttribute ();
	                    if (new_attrib == NULL)
							throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::ParseTags!", (char *)"Could not create object NexusAttribute!"));
                        new_attrib->PutVarInfo (attrib_name, var_association->dims[0], var_association->data_type, var_association->address);
	                    if (var_attribs == NULL)
    	                {
        	                var_attribs = new NexusAttribList ();
            	            if (var_attribs == NULL)
								throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::ParseTags!", (char *)"Could not create object NexusAttributeList!"));

    	                    var_attribs->nexus_attribute = new_attrib;
        	                var_attribs->var_info = var_association;
            	        }
                	    else
                    	    AddToAttribList (var_attribs, new_attrib, var_association);
                    }
//If PV
#ifdef USECAPV
                    if (!stricmp (attrib_location, "pv"))
                    {
                        new_attrib = new NexusAttribute ();
	                    if (new_attrib == NULL)
							throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::ParseTags!", (char *)"Could not create object NexusAttribute!"));
                        new_attrib->PutPVInfo (attrib_name, attrib_type, attrib_value);
	                    if (pv_attribs == NULL)
    	                {
        	                pv_attribs = new NexusAttribList ();
            	            if (pv_attribs == NULL)
								throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::ParseTags!", (char *)"Could not create object NexusAttributeList!"));

    	                    pv_attribs->nexus_attribute = new_attrib;
        	            }
            	        else
                	        AddToAttribList (pv_attribs, new_attrib);
                    	}
#endif

                    new_field->AddToAttributeList (new_attrib);

	                strcat (index_entry, INDEX_SEPERATOR_STR);
                    strcat (index_entry, attrib_name);

				    directory->AddToList (new NexusFileDirectory (index_entry, (NexusAttribute *) new_attrib));

                    temp_chr = strrchr (index_entry, INDEX_SEPERATOR_CHR);
					temp_chr[0] = '\0';

                    *tag_index = *tag_index + 1;
                }

                temp_chr = strrchr (index_entry, INDEX_SEPERATOR_CHR);
				temp_chr[0] = '\0';

                break;
            }

            default : *tag_index = *tag_index + 1;

        }
    }

    return (group_list);

}

//---------------------------------------------------------------------------

NexusGroup *NexusBoxClass::ParseHDFFile (NexusGroup *parent_group)
{
char            name[256],
                type[256];
char            group_name[256],
                group_type[256],
                attrib_name[256],
                *temp_chr;
int            rank,
                dimensions[10],
                data_type,
                size,
                malloc_size,
                loop,
                length;
NexusGroup      *new_group = NULL,
                *group_list = NULL,
                *current_group = NULL;
NexusField      *new_field;
NexusAttribute  *new_attrib;
void            *temp_data;

    while (GetNextEntry (name, type, &data_type, file_handle))
    {
        
        strcpy (group_name, name);
        strcpy (group_type, type);

        if (stricmp (group_type, "SDS"))
        {
            if (stricmp (group_type, "CDF0.0") && stricmp (group_type, "df"))
            {
            //it's a group
                new_group = new NexusGroup (group_name, group_type);
                if (new_group == NULL)
					throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::ParseHDFFile!", (char *)"Could not create object NexusGroup!"));

                strcat (index_entry, INDEX_SEPERATOR_STR);
                strcat (index_entry, group_name);

                if (group_list == NULL)
                {
                    group_list = new_group;
                    current_group = group_list;
                }
                else
                {
                    current_group->AddToGroupList (current_group, new_group);
                    current_group = current_group->NextGroup ();
                }

              	OpenGroup (group_name, group_type, file_handle);

			    directory->AddToList (new NexusFileDirectory (index_entry, (NexusGroup *) new_group));

                current_group->AddSubgroupList (ParseHDFFile (current_group));

            	CloseGroup (file_handle);

                temp_chr = strrchr (index_entry, INDEX_SEPERATOR_CHR);
                if (temp_chr != NULL)
					temp_chr[0] = '\0';
            }
        }
        else
        {
            OpenData (group_name, file_handle);
            GetDataInfo (&rank, dimensions, &data_type, file_handle);

            new_field = new NexusField ();
            if (new_field == NULL)
				throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::ParseHDFFile!", (char *)"Could not create object NexusField!"));

            if (read_scheme == ENTIRE_CONTENTS)
            {
                size = 1;
                for (loop=0;loop<rank;loop++)
                    size = size * dimensions[loop];

                switch (data_type)
                {
                    case NX_CHAR : malloc_size = ((sizeof (char) * size) + 1); break;
                    case NX_UINT8 :;
                    case NX_INT8 : malloc_size = sizeof (char) * size; break;
                    case NX_UINT16 :;
                    case NX_INT16 : malloc_size = sizeof (short) * size; break;
                    case NX_UINT32 :;
                    case NX_INT32 : malloc_size = sizeof (long) * size; break;
                    case NX_FLOAT32 : malloc_size = sizeof (float) * size; break;
                    case NX_FLOAT64 : malloc_size = sizeof (double) * size; break;
                }

                temp_data = malloc (malloc_size);
                if (temp_data == NULL)
				    throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::ParseHDFFile!", (char *)"Memmory allocation error!"));

                GetData (temp_data, file_handle);

                new_field->PutSDSInfo (group_name, rank, dimensions, data_type, temp_data);

                free (temp_data);
            }
            else
                new_field->PutSDSInfo (group_name, rank, dimensions, data_type);

            parent_group->AddToFieldList (new_field);

            strcat (index_entry, INDEX_SEPERATOR_STR);
            strcat (index_entry, group_name);

		    directory->AddToList (new NexusFileDirectory (index_entry, (NexusField *) new_field));

            while (GetNextAttrib (name, &length, &data_type, file_handle))
            {
                strcpy (attrib_name, name);

                switch (data_type)
                {
                    case NX_CHAR : {
                                malloc_size = sizeof (char);
                                length++;
                                break;
                            }
                    case NX_UINT8 :;
                    case NX_INT8 : malloc_size = sizeof (char); break;
                    case NX_UINT16 :;
                    case NX_INT16 : malloc_size = sizeof (short); break;
                    case NX_UINT32 :;
                    case NX_INT32 : malloc_size = sizeof (long); break;
                    case NX_FLOAT32 : malloc_size = sizeof (float); break;
                    case NX_FLOAT64 : malloc_size = sizeof (double); break;
                }

                temp_data = malloc (malloc_size * length);
                if (temp_data == NULL)
					throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::ParseHDFFile!", (char *)"Memmory allocation error!"));

                GetAttrib (attrib_name, temp_data, &length, &data_type, file_handle);

                new_attrib = new NexusAttribute ();
                if (new_attrib == NULL)
					throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::ParseHDFFile!", (char *)"Could not create object NexusAttribute!"));
                new_attrib->PutSDSInfo (attrib_name, length, data_type, temp_data);

                free (temp_data);

                new_field->AddToAttributeList (new_attrib);

                strcat (index_entry, INDEX_SEPERATOR_STR);
                strcat (index_entry, attrib_name);

			    directory->AddToList (new NexusFileDirectory (index_entry, (NexusAttribute *) new_attrib));

                temp_chr = strrchr (index_entry, INDEX_SEPERATOR_CHR);
				temp_chr[0] = '\0';
            }

            temp_chr = strrchr (index_entry, INDEX_SEPERATOR_CHR);
			temp_chr[0] = '\0';

            CloseData (file_handle);
        }

    }

    return (group_list);

}

//---------------------------------------------------------------------------

void NexusBoxClass::AddToFieldList (NexusFieldList *parent_field, NexusField *new_field)
{
NexusFieldList  *current_field;

    current_field = parent_field;

    while (current_field->next_in_list != NULL)
        current_field = (NexusFieldList *) current_field->NextInList ();

    current_field->next_in_list = new NexusFieldList ();
    if (current_field->next_in_list == NULL)
		throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::AddToFieldList!", (char *)"Could not create object NexusFieldList!"));

    current_field = (NexusFieldList *) current_field->NextInList ();
    current_field->nexus_field = new_field;
}

//---------------------------------------------------------------------------

void NexusBoxClass::AddToFieldList (NexusFieldList *parent_field, NexusField *new_field, NexusVarAssoc *var_info)
{
NexusFieldList  *current_field;

    current_field = parent_field;

    while (current_field->next_in_list != NULL)
        current_field = (NexusFieldList *) current_field->NextInList ();

    current_field->next_in_list = new NexusFieldList ();
    if (current_field->next_in_list == NULL)
		throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::AddToFieldList!", (char *)"Could not create object NexusFieldList!"));

    current_field = (NexusFieldList *) current_field->NextInList ();
    current_field->nexus_field = new_field;
    current_field->var_info = var_info;
}

//---------------------------------------------------------------------------

void NexusBoxClass::AddToAttribList (NexusAttribList *parent_attrib, NexusAttribute *new_attrib)
{
NexusAttribList  *current_attribute;

    current_attribute = parent_attrib;

    while (current_attribute->next_in_list != NULL)
        current_attribute = (NexusAttribList *) current_attribute->NextInList ();

    current_attribute->next_in_list = new NexusAttribList ();
    if (current_attribute->next_in_list == NULL)
		throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::AddToAttribList!", (char *)"Could not create object NexusAttribList!"));

    current_attribute = (NexusAttribList *) current_attribute->NextInList ();
    current_attribute->nexus_attribute = new_attrib;
}

//---------------------------------------------------------------------------


void NexusBoxClass::AddToAttribList (NexusAttribList *parent_attrib, NexusAttribute *new_attrib, NexusVarAssoc *var_info)
{
NexusAttribList  *current_attribute;

    current_attribute = parent_attrib;

    while (current_attribute->next_in_list != NULL)
        current_attribute = (NexusAttribList *) current_attribute->NextInList ();

    current_attribute->next_in_list = new NexusAttribList ();
    if (current_attribute->next_in_list == NULL)
		throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::AddToAttribList!", (char *)"Could not create object NexusAttribList!"));

    current_attribute = (NexusAttribList *) current_attribute->NextInList ();
    current_attribute->nexus_attribute = new_attrib;
    current_attribute->var_info = var_info;
}

//---------------------------------------------------------------------------

NexusVarAssoc *NexusBoxClass::FindVarAssociation (char *var_name)
{
NexusVarAssoc   *current_var;

    current_var = var_associations;

    while (stricmp (current_var->name, var_name))
    {
        current_var = (NexusVarAssoc *) current_var->NextInList ();
        if (current_var == NULL)
			throw (new NexusExceptionClass ((char *)"Error in method NexusBoxClass::FindVarAssociation!", (char *)"Could not find variable association!"));
    }

    return (current_var);

}

//---------------------------------------------------------------------------

void NexusBoxClass::GetDataFromIndex (char *index, void *get_data, int *start_dims, int *length_dims)
{
char	group_name[256],
		*field_name;
int		start_index,
        num_chars;
NexusGroup	*current_group;
NexusField	*current_field;

	OpenFile (NXACC_READ, data_path_name, data_file_name, &file_handle);

	start_index = 1;
    current_group = tree_top;

    field_name = strrchr (index, INDEX_SEPERATOR_CHR);
    field_name++;	//advance beyond the INDEX_SEPERATOR

    while (stricmp (&index[start_index], field_name))
    {
	    current_group = current_group->SubGroup ();

	    num_chars = strcspn (&index[start_index], INDEX_SEPERATOR_STR);
    	strncpy (group_name, &index[start_index], num_chars);
	    group_name[num_chars] = '\0';
    	start_index = start_index + num_chars + 1;

        while (stricmp (current_group->name, group_name))
        {
        	current_group = current_group->NextGroup ();
            if (current_group == NULL)
		    	throw (new NexusExceptionClass ((char *)"Could not find requested index!"));
        }

       	OpenGroup (current_group->name, current_group->type, file_handle);
    }

    current_field = current_group->FieldList ();

    while (stricmp (current_field->name, field_name))
    {
       	current_field = current_field->NextField ();
        if (current_field == NULL)
	    	throw (new NexusExceptionClass ((char *)"Could not find requested index!"));
    }

    OpenData (current_field->name, file_handle);

    GetDataSlab (get_data, file_handle, start_dims, length_dims);

    CloseFile (&file_handle);

}

//---------------------------------------------------------------------------


