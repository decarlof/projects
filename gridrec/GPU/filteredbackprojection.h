#ifndef filteredbackprojectionH
#define filteredbackprojectionH
//---------------------------------------------------------------------------

#include <stdio.h>
#include <math.h>

// #include "logfileclass.h"
#include "recon_algorithm.h"

//---------------------------------------------------------------------------

#define GRID_PRECISION 1000000

//---------------------------------------------------------------------------

class FBP : public ReconAlgorithm
{
public:

    FBP (void);

    void init (void);
    void reconstruct (void);
    void destroy (void);

    // static void acknowledgements (LogFileClass *acknowledge_file);

protected:
float   *fft_proj,
        *filtered_proj,
        *sines,
        *cosines,
        *filter_lut;
};

//---------------------------------------------------------------------------

class OptimizedFBP : public FBP
{
public:
        
    void init (void);
    void reconstruct (void);
    void destroy (void);

    // static void acknowledgements (LogFileClass *acknowledge_file);

};

//---------------------------------------------------------------------------

class CircleFBP : public FBP
{
public:

    void init (void);
    void reconstruct (void);
    void destroy (void);

    // static void acknowledgements (LogFileClass *acknowledge_file);

private:
long    *sines,
        *cosines;
};

//---------------------------------------------------------------------------

#endif
