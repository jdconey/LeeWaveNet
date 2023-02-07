# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:23:14 2023

@author: mm16jdc
"""


import scipy.io as sio
import numpy as np


def s_transform_orientation(strday,ls,c):
    F1 = sio.loadmat('C:/Users/mm16jdc/Documents/MATLAB/'+ls+'/F1_'+strday+'_'+c+'.mat')['F1']
    F2 = sio.loadmat('C:/Users/mm16jdc/Documents/MATLAB/'+ls+'/F2_'+strday+'_'+c+'.mat')['F2']

    orientation = np.arctan(F1**2/F2**2)
    orientation = 180*orientation/np.pi
    or2 = 90-orientation
    if or2<0:
        or2 = -or2
    return or2


def s_transform_wavelength(strday,ls,c):

    F1 = sio.loadmat('C:/Users/mm16jdc/Documents/MATLAB/'+ls+'/F1_'+strday+'_'+c+'.mat')['F1']
    F2 = sio.loadmat('C:/Users/mm16jdc/Documents/MATLAB/'+ls+'/F2_'+strday+'_'+c+'.mat')['F2']

    wavelength = (F1**2+F2**2)**(-0.5)
    return wavelength