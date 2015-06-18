#ifndef FFT_H
#define FFT_H
/*_____________________________________________________________________________________*/

#define PI      3.1415926535897932385

/*_____________________________________________________________________________________*/

void initFFTMemoryStructures (void);
void destroyFFTMemoryStructures (void);

void four1 (float data[], unsigned long nn, int isign);
void fourn (float data[], unsigned long nn[], int ndim, int isign);

/*_____________________________________________________________________________________*/
#endif
