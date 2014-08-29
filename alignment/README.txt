The data in this directory can be used to automatically determine/adjust the 
tomography stage rotation axis pitch/roll. 

Pitch: rotation around X (inboard/outboard direction)
Roll: rotation around Z (beam direction)


Two folders 0_180 and 0_90_180 exist in the microCT directory. The files
in both folders share the same name format:

tilt_###_***_???.hdf. 

### is the roll angle:
	020 => roll angle is 0.2 deg
	002 => roll angle is 0.02 deg

*** is the pitch angle:
	020 => pitch angle is 0.2
	002 => pitch angle is 0.02

??? could be 1,2,3,4,5. 
	The last two images are the white field and the dark field images. 
	The first two (three) images are the 0, 180 (0, 90, 180) degree images.


One folders 0_180 exists in the nanoCT directory. The files:

    Pin_0deg.tif    Pin projection at 0 deg
    Pin_180deg.tif  Pin Projection at 180 deg



