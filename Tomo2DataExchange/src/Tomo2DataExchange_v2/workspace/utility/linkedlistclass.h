//---------------------------------------------------------------------------
#ifndef LinkedListClassH
#define LinkedListClassH
//---------------------------------------------------------------------------

#ifdef WIN32
class __declspec(dllexport) LinkedListClass
#else
class LinkedListClass
#endif
{
public:
LinkedListClass		*previous_in_list,
					*next_in_list;

	LinkedListClass (void);

	int Index ();

	LinkedListClass *PreviousInList (void);
	LinkedListClass *NextInList (void);

	//Adds to end of list started with this object
    void AddToList (LinkedListClass *new_info);

    //Finds last item in list starting with this object
	LinkedListClass *FindLastItem (void);

	~LinkedListClass ();

private:
int			index;

};

//---------------------------------------------------------------------------
#endif

