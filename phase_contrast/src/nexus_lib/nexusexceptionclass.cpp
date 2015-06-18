//---------------------------------------------------------------------------
#pragma hdrstop

#include "string.h"
#include "nexusexceptionclass.h"

//---------------------------------------------------------------------------
#pragma package(smart_init)
//---------------------------------------------------------------------------

NexusExceptionClass::NexusExceptionClass (NexusExceptionClass &old_class)
{
	strcpy (error_string, old_class.getErrorString());
	strcpy (error_type, old_class.getErrorType());
}

//---------------------------------------------------------------------------

NexusExceptionClass::NexusExceptionClass (char *error)
{
	strcpy (error_string, error);
}

//---------------------------------------------------------------------------

NexusExceptionClass::NexusExceptionClass (char *error, char *type)
{
	strcpy (error_string, error);
	strcpy (error_type, type);
}

//---------------------------------------------------------------------------

char *NexusExceptionClass::getErrorString ()
{
	return (error_string);
}

//---------------------------------------------------------------------------

char *NexusExceptionClass::getErrorType ()
{
	return (error_type);
}

//---------------------------------------------------------------------------

