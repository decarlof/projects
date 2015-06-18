//---------------------------------------------------------------------------
#include <stdio.h>

#ifdef _WIN32_
#include <windows.h>
#endif

#pragma hdrstop

#include "linkedlistclass.h"

#pragma package(smart_init)
//---------------------------------------------------------------------------
LinkedListClass::LinkedListClass ()
{
	previous_in_list = NULL;
    next_in_list = NULL;

    index = 0;
}
//---------------------------------------------------------------------------
int LinkedListClass::Index (void)
{
	return (index);
}
//---------------------------------------------------------------------------
LinkedListClass *LinkedListClass::PreviousInList (void)
{
	return (previous_in_list);
}
//---------------------------------------------------------------------------
LinkedListClass *LinkedListClass::NextInList (void)
{
	return (next_in_list);
}
//---------------------------------------------------------------------------
void LinkedListClass::AddToList (LinkedListClass *new_info)
{
LinkedListClass		*last_item;

	last_item = FindLastItem ();

	last_item->next_in_list = new_info;
    last_item->next_in_list->previous_in_list = this;
    last_item->next_in_list->index = index + 1;
}
//---------------------------------------------------------------------------
LinkedListClass *LinkedListClass::FindLastItem (void)
{
LinkedListClass		*current_entry;

	current_entry = this;
    while (current_entry->NextInList () != NULL)
        current_entry = (LinkedListClass *) current_entry->NextInList ();

    return (current_entry);
}
//---------------------------------------------------------------------------
LinkedListClass::~LinkedListClass ()
{
	if (next_in_list != NULL)
	{
    	delete (next_in_list);
		next_in_list = NULL;
	}
}
//---------------------------------------------------------------------------

