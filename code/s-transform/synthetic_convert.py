# -*- coding: utf-8 -*-
"""
Created on Fri May 20 12:13:02 2022

@author: mm16jdc
"""

#script to convert numpy arrays to matlab files to s-transform them.

import os 
import xarray
from scipy.io import savemat

import numpy as np

import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath



def load_xarray_vv(fname):
    '''
    read Met Office netCDF file of vertical velocities on pressure levels - 
    usually named e.g. 20210206T0900Z-PT0000H00M-wind_vertical_velocity_on_pressure_levels.nc
    The 8th height level is the 700 hPa level.
    
    returns xarray DataArray of cropped data on 700hPa level.
    '''
    xmin=275
    xmax=787
    ymin=250
    ymax=762
    cubes = xarray.open_dataset(fname)
    cubes_x = cubes['upward_air_velocity'][8,ymin:ymax,xmin:xmax]
    return cubes_x

def open_xarray(d):
    return d
def label_func(d):
    return d

base = 'C:/Users/mm16jdc/Documents/ukv_data/data5_test/'
dest_base =  'C:/Users/mm16jdc/Documents/ukv_data/data5_testn/'
fs = os.listdir(base+'data')

thresholds = [0.125]
for threshold in thresholds:
    if not os.path.isdir(dest_base+'S_transform3/'+str(threshold)+'/mat/'):
        os.makedirs(dest_base+'S_transform3/'+str(threshold)+'/mat/')
        os.makedirs(dest_base+'S_transform3/'+str(threshold)+'/ssd/')
    for f in fs:
        data2=np.load(base+'data/'+f)
        noise = np.random.normal(size=(512,512))
        data2 = data2 + threshold*noise
        
        
        
        mdic = {'vv':data2, 'label':'vv_Data'}
        savemat(dest_base+'S_transform3/'+str(threshold)+'/mat/'+f.replace('npy','mat'),mdic)

 
print('done')