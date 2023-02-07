#S-Transform



This directory contains the code to S-Transform vertical velocity data, and then retrieve the orientation and wavelength of the waves.

- nph_ndst.m is a matlab script written by Neil Hindley that I acquired off Corwin Wright's GitHub - I used this version: https://github.com/corwin365/MatlabFunctions/blob/3af8b4943874cc665bbbcd59db2482defe388cab/STransform/nph_ndst.m 

- s_transform_characteristics.py is a python script that takes the output of the S-transform code and returns wave characteristics.

- synthetic_convert.py is a python script to convert numpy arrays of vertical velocity into a format than can be read by the matlab script.