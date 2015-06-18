#include "errorlogclass.h"

//__________________________________________________________________________

errorlogclass::errorlogclass()
{
    strcpy (error_file_name, "none");

    error_file = NULL;
}

//__________________________________________________________________________

void errorlogclass::setErrorFileLocation (const char *file_path, const char *file_name)
{
    strcpy (error_file_path, file_path);
    if (error_file_path[strlen (error_file_path)-1] != '/')
        strcat (error_file_path, "/");

    strcpy (error_file_name, file_name);

    sprintf (error_file_location, "%s%s", error_file_path, error_file_name);
}

//__________________________________________________________________________

void errorlogclass::addError (const char *error_str, const char *error_location)
{
char    output[1024];

    if (error_file == NULL)
    {
        error_file = new ofstream ();
        if (!strcmp (error_file_name, "none"))
            error_file->open (error_file_location);
        else
            error_file->open (error_file_location);
    }

    sprintf (output, "%s::%s\n", error_location, error_str);
    error_file->write (output, strlen (output));
    error_file->flush ();

}
//__________________________________________________________________________

void errorlogclass::addAutoResolution (const char *resolution_str)
{
char    output[1024];

    if (error_file == NULL)
    {
        error_file = new ofstream ();
        if (!strcmp (error_file_name, "none"))
            error_file->open (error_file_location);
        else
            error_file->open (error_file_location);
    }

    sprintf (output, "%s\n", resolution_str);
    error_file->write (output, strlen (output));
    error_file->flush ();

}

//__________________________________________________________________________

void errorlogclass::close (void)
{
    if (error_file != NULL)
    {
        error_file->close ();
        delete (error_file);
        error_file = NULL;
    }
}

//__________________________________________________________________________

errorlogclass::~errorlogclass()
{
    if (error_file != NULL)
    {
        error_file->close ();
        delete (error_file);
    }
}

//__________________________________________________________________________
