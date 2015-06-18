//---------------------------------------------------------------------------
#ifndef NexusGroupH
#define NexusGroupH

#include "nexusapi.h"
#include "nexusexceptionclass.h"
#include <sys/timeb.h>

#ifdef USECAPV
#include "capv.h"
#endif

//---------------------------------------------------------------------------
#define PV_ERROR                    2001
//---------------------------------------------------------------------------

#define STRING_UNASSIGNED       "Unassigned"
#define MISSING_DATA_ERROR_STR  "Missing Data"

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

class NexusData : public NexusAPI
{
public:
    char                *name,
                        *type;

    int                 data_type;
    int                 rank;
    int                 arr_dims[10];

    void                *data;

    long int            size;

    int                 valid,
                        required,
                        uptodate,
                        read_only,
                        use,
                        association;

#ifdef USECAPV
    CAPV                *pv;
    long int				pv_type;
    long int				reconnect_time;
#endif

    NexusData ();

    void PutDataVal (char *new_data, long int new_size);
    void PutDataVal (unsigned char *new_data);
    void PutDataVal (char *new_data);
    void PutDataVal (unsigned short *new_data);
    void PutDataVal (short *new_data);
    void PutDataVal (unsigned long int *new_data);
    void PutDataVal (long int *new_data);
    void PutDataVal (float *new_data);
    void PutDataVal (double *new_data);
    void PutDataVal (void *new_data, int new_rank, int *new_dims, int new_type);

    void GetDataInfo (int *get_rank, int *get_dims, int *get_type);
    void GetDataVal (unsigned char *get_data);
    void GetDataVal (char *get_data);
    void GetDataVal (unsigned short *get_data);
    void GetDataVal (short *get_data);
    void GetDataVal (unsigned long int *get_data);
    void GetDataVal (long int *get_data);
    void GetDataVal (float *get_data);
    void GetDataVal (double *get_data);
    void GetDataVal (void *get_data);

    int DataValid (void);

#ifdef USECAPV
    void PVConnect (void);
    void PVDisconnect (void);
	void PVReConnect (void);
    int PVIsConnected (void);
	int PVLastReconnectAttempt (void);
#endif

    virtual ~NexusData (void);

private:

};

//---------------------------------------------------------------------------

class NexusAttribute : public NexusData
{
public:
    NexusAttribute (void);

    void PutSDSInfo (char *attrib_name, int attrib_length, int attrib_type, void *attrib_data);

    void PutConstInfo (char *attrib_name, char *attrib_type, char *attrib_value);
    void PutVarInfo (char *attrib_name, int attrib_length, int attrib_type, void *attrib_data);

#ifdef USECAPV
    void PutPVInfo (char *attrib_name, char *attrib_type, char *attrib_value);
    void PVUpdate (void);
#endif

	void UpdateVarInfo (int attrib_length, void *var_address);

    void AddAttribute (NexusAttribute *attrib);
    NexusAttribute *NextAttribute (void);

	void WriteAttribute (NXhandle file_handle);

    ~NexusAttribute (void);

private:
    NexusAttribute      *next_attribute;
};

//---------------------------------------------------------------------------

class NexusField : public NexusData
{
public:
    NexusField (void);

    void PutSDSInfo (char *field_name, int field_rank, int *field_dims, int field_type, void *field_data);
    void PutSDSInfo (char *field_name, int field_rank, int *field_dims, int field_type);

    void PutConstInfo (char *field_name, char *field_type, char *field_value);
    void PutVarInfo (char *field_name, int field_rank, int *field_dims, int field_type, void *field_data);

#ifdef USECAPV
    void PutPVInfo (char *field_name, char *field_type, char *field_value);
    void PVUpdate (void);
#endif

    void UpdateVarInfo (int field_rank, int *field_dims, int type, void *var_address);

    void AddField (NexusField *field);
    NexusField *NextField (void);

    NexusAttribute *AttributeList (void);
    void AddToAttributeList (NexusAttribute *new_attrib);

    void WriteField (NXhandle file_handle, int compression_scheme);

    ~NexusField (void);

private:
    NexusField          *next_field;

    NexusAttribute      *attribute_list;
};

//---------------------------------------------------------------------------

class NexusGroup : public NexusAPI
{
public:
    char                *name,
                        *type;

    NexusGroup (char *group_name, char *group_type);

    void AddGroup (NexusGroup *group);
    void AddToGroupList (NexusGroup *parent_group, NexusGroup *new_group);
    void AddSubgroupList (NexusGroup *new_group);

    NexusGroup *NextGroup ();
    NexusGroup *SubGroup ();

    NexusField *FieldList (void);
    void AddToFieldList (NexusField *new_field);

    void WriteGroup (NXhandle file_handle, long int compression_scheme);

    ~NexusGroup (void);

private:
    long int               error_code;

    NexusGroup          *next_group,
                        *subgroup_list;

    NexusField          *field_list;

    NexusAttribute      *attribute_list;
};

//---------------------------------------------------------------------------
#endif
