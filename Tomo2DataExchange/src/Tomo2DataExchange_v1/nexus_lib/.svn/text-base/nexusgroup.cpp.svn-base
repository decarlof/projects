//---------------------------------------------------------------------------
#include "nexusgroup.h"
#pragma hdrstop

#ifdef win32
#pragma package(smart_init)
#endif
//---------------------------------------------------------------------------
//Public Methods
//---------------------------------------------------------------------------

NexusGroup::NexusGroup (char *group_name, char *group_type)
{
    name = (char *) malloc (strlen (group_name) + 1);
    if (name == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusGroup:NexusGroup!", (char *) "Memmory allocation error!"));
    strcpy (name, group_name);

    type = (char *) malloc (strlen (group_type) + 1);
    if (type == NULL)
		throw (new NexusExceptionClass ((char *) "Error in method NexusGroup:NexusGroup!", (char *) "Memmory allocation error!"));
    strcpy (type, group_type);

    next_group = NULL;
    subgroup_list = NULL;
    field_list = NULL;
    attribute_list = NULL;
}

//---------------------------------------------------------------------------

void NexusGroup::AddGroup (NexusGroup *group)
{
    next_group = group;
}

//---------------------------------------------------------------------------

void NexusGroup::AddToGroupList (NexusGroup *parent_group, NexusGroup *new_group)
{
    while (parent_group->NextGroup () != NULL)
        parent_group = parent_group->NextGroup ();

    parent_group->AddGroup (new_group);
}

//---------------------------------------------------------------------------

void NexusGroup::AddSubgroupList (NexusGroup *new_group)
{
    subgroup_list = new_group;
}

//---------------------------------------------------------------------------

NexusGroup *NexusGroup::NextGroup (void)
{
    return (next_group);
}

//---------------------------------------------------------------------------

NexusGroup *NexusGroup::SubGroup (void)
{
    return (subgroup_list);
}

//---------------------------------------------------------------------------

NexusField *NexusGroup::FieldList (void)
{
    return (field_list);
}

//---------------------------------------------------------------------------

void NexusGroup::AddToFieldList (NexusField *new_field)
{
NexusField      *current_field;

    if (field_list == NULL)
        field_list = new_field;
    else
    {
        current_field = field_list;
        while (current_field->NextField () != NULL)
            current_field = current_field->NextField ();

        current_field->AddField (new_field);
    }
}

//---------------------------------------------------------------------------

void NexusGroup::WriteGroup (NXhandle file_handle, long int compression_scheme)
{
NexusGroup      *current_group;
NexusField      *current_field;

	try
	{
		MakeGroup (name, type, file_handle);
		OpenGroup (name, type, file_handle);

		current_field = field_list;
		while (current_field != NULL)
		{
			current_field->WriteField (file_handle, compression_scheme);

			current_field = current_field->NextField ();
		}

		current_group = subgroup_list;
		while (current_group != NULL)
		{
			current_group->WriteGroup (file_handle, compression_scheme);

			current_group = current_group->NextGroup ();
		}

		CloseGroup (file_handle);
	}
	catch (...)
	{
		throw;
	}

}

//---------------------------------------------------------------------------

NexusGroup::~NexusGroup (void)
{

	if (name != NULL)
        free (name);

    if (type != NULL)
        free (type);

    if (attribute_list != NULL)
        delete (attribute_list);

    if (field_list != NULL)
        delete (field_list);

    if (subgroup_list != NULL)
		delete (subgroup_list);

    if (next_group != NULL)
        delete (next_group);

}

//---------------------------------------------------------------------------

