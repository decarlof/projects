//---------------------------------------------------------------------------
#ifndef NexusBoxH
#define NexusBoxH

//#define INTEL86
//#define UNIX386
//#define __unix__

#include "linkedlistclass.h"
#include "logfileclass.h"

#include "nexusapi.h"
#include "nexusgroup.h"
#include "nexusexceptionclass.h"

//---------------------------------------------------------------------------

#define OKAY    0

#define HDF4_MODE           0
#define HDF5_MODE           1
#define XML_MODE            2

#define INDEX_ONLY          0
#define ENTIRE_CONTENTS     1

#define INDEX_OF_GROUP      0
#define INDEX_OF_FIELD      1
#define INDEX_OF_ATTRIBUTE  2

#define INDEX_SEPERATOR_STR ";"
#define INDEX_SEPERATOR_CHR ';'

//---------------------------------------------------------------------------

#ifdef WIN32
//class __declspec(dllexport) NexusVarAssoc : public LinkedListClass
class NexusVarAssoc : public LinkedListClass
#else
class NexusVarAssoc : public LinkedListClass
#endif
{
public:
    char            *name;
    int             rank,
                    dims[10],
                    data_type;
    void            *address;

    NexusVarAssoc (void);
    NexusVarAssoc (char *new_name, int new_rank, int *new_dims, int new_data_type, void *new_address);

    void UpdateVarInfo (int *new_dims, void *new_address);
    void UpdateVarInfo (int *new_dims, int type, void *new_address);

    ~NexusVarAssoc ();
};

//---------------------------------------------------------------------------

class NexusFieldList : public LinkedListClass
{
public:
    NexusField          *nexus_field;
    NexusVarAssoc       *var_info;

    NexusFieldList (void);

    void UpdateFromAssoc (void);

    ~NexusFieldList (void);
};

//---------------------------------------------------------------------------

class NexusAttribList : public LinkedListClass
{
public:
    NexusAttribute      *nexus_attribute;
    NexusVarAssoc       *var_info;

    NexusAttribList (void);

    void UpdateFromAssoc (void);

    ~NexusAttribList (void);
};

//---------------------------------------------------------------------------

class NexusFileDirectory : public LinkedListClass
{
public:
    NexusGroup          *nexus_group;
    NexusField          *nexus_field;
    NexusAttribute      *nexus_attribute;

    char				*directory_entry;

    NexusFileDirectory (char *entry);
    NexusFileDirectory (char *entry, NexusGroup *group);
    NexusFileDirectory (char *entry, NexusField *field);
    NexusFileDirectory (char *entry, NexusAttribute *attribute);

    NexusFileDirectory *FindByIndex (char *index);

    ~NexusFileDirectory (void);
};

//---------------------------------------------------------------------------

#ifdef WIN32
class __declspec(dllexport) NexusBoxClass : private NexusAPI
#else
class NexusBoxClass : private NexusAPI
#endif
{
public:
    NexusBoxClass (void);

    void InitFileSystem ();
    void ResetFileSystem ();

	long int InitTemplate (char *file_path, char *file_name);
    long int InsertTemplate (char * index, char *file_path, char *file_name);

    void GetIndexSize (long int *elements);
    char *GetIndex (long int element_number);
    char *GetIndex (long int element_number, int *index_type);
    int IndexExists(char *index);

    void PutDatum (char *index, char *new_val);
    void PutDatum (char *index, unsigned char *new_val);
    void PutDatum (char *index, unsigned short *new_val);
    void PutDatum (char *index, short *new_val);
    void PutDatum (char *index, unsigned long *new_val);
    void PutDatum (char *index, long *new_val);
    void PutDatum (char *index, float *new_val);
    void PutDatum (char *index, double *new_val);
    void PutDatum (char *index, void *new_val, int new_rank, int *new_dims, int new_type);

	int GetDatumInfo (char *index, int *get_rank, int *get_dims, int *get_type);
	int GetDatum (char *index, unsigned char *get_data);
	int GetDatum (char *index, char *get_data);
	int GetDatum (char *index, unsigned short *get_data);
	int GetDatum (char *index, short *get_data);
	int GetDatum (char *index, unsigned long *get_data);
	int GetDatum (char *index, long *get_data);
	int GetDatum (char *index, float *get_data);
	int GetDatum (char *index, double *get_data);
	int GetDatum (char *index, void *get_data);

	int GetDatumSlab (char *index, unsigned char *get_data, int *start_dims, int *size_dims);
	int GetDatumSlab (char *index, char *get_data, int *start_dims, int *size_dims);
	int GetDatumSlab (char *index, unsigned short *get_data, int *start_dims, int *size_dims);
	int GetDatumSlab (char *index, short *get_data, int *start_dims, int *size_dims);
	int GetDatumSlab (char *index, unsigned long *get_data, int *start_dims, int *size_dims);
	int GetDatumSlab (char *index, long *get_data, int *start_dims, int *size_dims);
	int GetDatumSlab (char *index, float *get_data, int *start_dims, int *size_dims);
	int GetDatumSlab (char *index, double *get_data, int *start_dims, int *size_dims);
	int GetDatumSlab (char *index, void *get_data, int *start_dims, int *size_dims);

    int CreateGroup (char *index, char *group_name, char *group_type);
    int CreateField (char *index, char *field_name, int rank, int *dims, int type, void *data);
    int CreateAttribute (char *index, char *attribute_name, int length, int type, void *data);

#ifdef USECAPV
	int ConnectPVs (void);
    int UpdatePVs (void);
	int DisconnectPVs (void);
#endif

	int RegisterVar (char *name, int rank, int *dims, int data_type, void *address);
	int UpdateVarInfo (char *name, int *dims, void *address);
    int UpdateVarInfo (char *name, int *dims, int type, void *address);
    int UpdateVars (void);

    int CreateTemplate (char *path_name, char *file_name);
    int ChangeFile (char *path_name, char *file_name);

    int SetFileMode (int file_mode);
    int GetFileMode (void);

    int ReadAll (char *path_name, char *file_name);
    int WriteAll (char *path_name, char *file_name);

    void CompressionScheme (int new_scheme);
    int CompressionScheme (void);

    void SetReadScheme (int new_scheme);

    static void acknowledgements (LogFileClass *acknowledge_file);

	~NexusBoxClass (void);

private:
    NXhandle            file_handle;

    NexusGroup          *tree_top;

    NexusFileDirectory	*directory;

    NexusFieldList      *pv_fields,
                        *var_fields;
    NexusAttribList     *pv_attribs,
    					*var_attribs;

    NexusVarAssoc       *var_associations;

    char                *tag_list[10000],
    					index_entry[1024],
                        *data_path_name,
                        *data_file_name;

    int                 compression_scheme,
                        read_scheme,
                        file_mode;

	void				*top_of_varlist_address;

    int     	        error_code;

    void GenerateTagList (char *unparsed_template);
    void DestroyTagList (void);

    void DeleteDataStructures (void);

    void GetGroupInfo (char *group_type, char *group_name, int tag_index);
    void GetFieldInfo (char *field_name, char *field_location, char *field_value, char *field_type, int tag_index);
    void GetAttributeInfo (char *attrib_name, char *attrib_location, char *attrib_value, char *attrib_type, int tag_index);

    NexusGroup *ParseTags (NexusGroup *parent_group, int *tag_index);
    NexusGroup *ParseHDFFile (NexusGroup *parent_group);

    void AddToFieldList (NexusFieldList *parent_field, NexusField *new_field);
    void AddToFieldList (NexusFieldList *parent_field, NexusField *new_field, NexusVarAssoc *var_info);

    void AddToAttribList (NexusAttribList *parent_attrib, NexusAttribute *new_attrib);
    void AddToAttribList (NexusAttribList *parent_attrib, NexusAttribute *new_attrib, NexusVarAssoc *var_info);

    NexusVarAssoc *FindVarAssociation (char *var_name);

	void GetDataFromIndex (char *index, void *get_data, int *start_dims, int *end_dims);

};

//---------------------------------------------------------------------------
#endif

