//---------------------------------------------------------------------------
#ifndef NexusExceptionClassH
#define NexusExceptionClassH
//---------------------------------------------------------------------------

class NexusExceptionClass
{
public:
    NexusExceptionClass (NexusExceptionClass &old_class);

	NexusExceptionClass (char *error_string);
	NexusExceptionClass (char *error_string, char *error_type);

    char *getErrorString (void);
    char *getErrorType (void);

private:
char	error_string[256],
		error_type[256];
};

#endif

