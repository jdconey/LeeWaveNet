##S-Transform



This directory contains the code to S-Transform vertical velocity data, and then retrieve the orientation and wavelength of the waves.

- nph_ndst.m is a matlab script written by Neil Hindley for running the S-transform: it is available from here: (https://doi.org/10.5281/zenodo.4721883).

- s_transform_characteristics.py is a python script that takes the output of the S-transform code and returns wave characteristics.

- synthetic_convert.py is a python script to convert numpy arrays of vertical velocity into a format than can be read by the matlab script.
