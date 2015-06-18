//---------------------------------------------------------------------------
#pragma hdrstop

#include "recon_algorithm.h"

//---------------------------------------------------------------------------
#pragma package(smart_init)

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//Filter
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

Filter::Filter()
{
  filter_type = FILTER_NONE;
}

//---------------------------------------------------------------------------

void Filter::setFilter (int filter)
{
  filter_type = filter;
}

//---------------------------------------------------------------------------

int Filter::getFilter (void)
{
  return (filter_type);
}

//---------------------------------------------------------------------------

float Filter::filterData (float x)
{
  switch (filter_type) {
  case FILTER_NONE : break;
  case FILTER_SHEPP_LOGAN : x = shlo (x); break;
  case FILTER_HANN : x = hann (x); break;
  case FILTER_HAMMING : x = hamm (x); break;
  case FILTER_RAMP : x = ramp (x); break;
  default : break;
  }

  return (x);
}

//---------------------------------------------------------------------------

float Filter::shlo (float x)
{	/* Shepp-Logan filter */
  return fabs(sin(PI*x)/PI);
}

//---------------------------------------------------------------------------
float Filter::hann (float x)
{	/* Hann filter */
  return fabs(x)*0.5*(1.0+cos(2*PI*x));
}

//---------------------------------------------------------------------------
float Filter::hamm (float x)
{	/* Hamming filter */
  return fabs(x)*(0.54+0.46*cos(2*PI*x));
}

//---------------------------------------------------------------------------
float Filter::ramp (float x)
{	/* Ramp filter */
  return fabs(x);
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//Reconalgorithm
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

ReconAlgorithm::ReconAlgorithm (void)
{
  theta_list = NULL;

  sinogram1 = NULL;
  sinogram2 = NULL;

  reconstruction1 = NULL;
  reconstruction2 = NULL;
}

//---------------------------------------------------------------------------

// void ReconAlgorithm::acknowledgements (LogFileClass *acknowledge_file)
// {
//   acknowledge_file->Message ("__________________________________________________________________");
//   acknowledge_file->Message ("ReconAlgorithm class");
//   acknowledge_file->Message ("");
//   acknowledge_file->Message ("Parent class definition for reconstruction algorithm classes.");
//   acknowledge_file->Message ("All reconstruction algorithms should be placed in a class derived");
//   acknowledge_file->Message ("from this one.  This should allow for the highest level of interopability");
//   acknowledge_file->Message ("with the least amount of maintenance");
//   acknowledge_file->Message ("Developed and Maintained by:");
//   acknowledge_file->Message ("       Brian Tieman");
//   acknowledge_file->Message ("       Argonne National Laboratory");
//   acknowledge_file->Message ("       tieman@aps.anl.gov");
//   acknowledge_file->Message ("");
//   acknowledge_file->Message ("8/20/2003  V1.0   BT  First version with acknowledgements");
//   acknowledge_file->Message ("4/23/2006  V1.13  BT  Modified handling of filter type to be done by");
//   acknowledge_file->Message ("		int rather than by string.");
//   acknowledge_file->Message ("");
//   acknowledge_file->Message ("");
//   acknowledge_file->Message ("");
//   acknowledge_file->Message ("__________________________________________________________________");
// }

//---------------------------------------------------------------------------

int ReconAlgorithm::numberOfSinogramsNeeded (void)
{
  return (num_sinograms_needed);
}

//---------------------------------------------------------------------------

int ReconAlgorithm::numberOfReconstructions (void)
{
  return (num_sinograms_needed);
}

//---------------------------------------------------------------------------

void ReconAlgorithm::setThetaList (float *theta_list, int theta_list_size)
{
  this->theta_list = theta_list;
  this->theta_list_size = theta_list_size;
}

//---------------------------------------------------------------------------

int ReconAlgorithm::getThetaListSize (void)
{
  return (theta_list_size);
}

//---------------------------------------------------------------------------

float *ReconAlgorithm::getThetaList (void)
{
  return (theta_list);
}

//---------------------------------------------------------------------------

void ReconAlgorithm::setSinogramDimensions (int x_dim, int y_dim)
{
  sinogram_x_dim = x_dim;
  sinogram_y_dim = y_dim;
}

//---------------------------------------------------------------------------

int ReconAlgorithm::getSinogramXDimension (void)
{
  return (sinogram_x_dim);
}

//---------------------------------------------------------------------------

int ReconAlgorithm::getSinogramYDimension (void)
{
  return (sinogram_y_dim);
}

//---------------------------------------------------------------------------

void ReconAlgorithm::setFilter (int filter_name)
{
  filter.setFilter(filter_name);
}

//---------------------------------------------------------------------------

int ReconAlgorithm::getFilter (void)
{
  return (filter.getFilter());
}

//---------------------------------------------------------------------------

void ReconAlgorithm::setSinoAndReconBuffers (int number, float *sinogram_address, float *reconstruction_address)
{
  if (number == 1)
    {
      sinogram1 = sinogram_address;
      reconstruction1 = reconstruction_address;
    }
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//Virtual placeholders
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

void ReconAlgorithm::init (void)
{
}

//---------------------------------------------------------------------------

void ReconAlgorithm::reconstruct (void)
{
}

//---------------------------------------------------------------------------

void ReconAlgorithm::destroy (void)
{
}

//---------------------------------------------------------------------------

