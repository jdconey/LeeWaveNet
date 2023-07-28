# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:12:11 2023

@author: mm16jdc
"""
import pathlib

from fastai.vision.all import load_learner
import xarray as xr
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as cm

from cmaps import *


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def open_xarray(f):
    ds = xr.load_dataset(f)
    return ds

def label_func(f):
    label = np.load(f)
    return label
def label_func2(f):
    return label_func(f)

def open_np(f):
    return label_func(f)
    

def load_models():
    learn2 = load_learner('SEGMODEL.pkl')
    wavelength_model = load_learner('wavelength_model_0.125.pkl')
    angle_model_v3 = load_learner('orientation_pretrained_frozen_noise_0.25.pkl')
    amp_model = load_learner('amplitude5_0.0625.pkl')
    return {'segmentation':learn2,
            'wavelength':wavelength_model,
            'orientation':angle_model_v3,
            'amplitude':amp_model
            }

def predict(models_dict,ds,xcoord='projection_x_coordinate',ycoord='projection_y_coordinate',mask_nonwaves=True):
    arr = ds['upward_air_velocity'].values
    ds['segmentation'] = ((ycoord,xcoord),models_dict['segmentation'].predict(arr)[0].numpy())
    wavelength = models_dict['wavelength'].predict(torch.Tensor([arr,arr,arr]))[0][0].numpy()
    ds['wavelength']=((ycoord,xcoord),wavelength)
    orient = models_dict['orientation'].predict(torch.Tensor([arr,arr,arr]))[0]
    orient = 180/np.pi * np.arctan(orient[0]/orient[1])
    ds['orientation'] = ((ycoord,xcoord),orient)
    amplitude =  models_dict['amplitude'].predict(torch.Tensor(np.array([arr,arr,arr])))[0][0]
    ds['amplitude'] = ((ycoord,xcoord),amplitude)
    
    if mask_nonwaves:
        for char in ['amplitude','orientation','wavelength']:
            ds[char] = ds[char].where(ds['segmentation']==1)
    return ds
    
def quiver_orient(dataset,sep=32,xcoord='projection_x_coordinate',ycoord='projection_y_coordinate'):
    angle_rad = dataset['orientation'].values*(np.pi/180)
    new_x = np.zeros(int(512/sep))
    new_y = np.zeros(int(512/sep))
    i=0
    angle_rad2=np.zeros((int(512/sep),int(512/sep)))
    while i<len(angle_rad2):
        j=0
        while j<len(angle_rad2[i]):
            new_angle_rad = np.pi/2 - angle_rad[sep*i][sep*j]
            new_y[j] = dataset[ycoord][j*sep]
            angle_rad2[i][j]=new_angle_rad
            j=j+1
        new_x[i] = dataset[xcoord][i*sep]
        i=i+1
    alt_dataframe = xr.Dataset(data_vars = {'angle_rad':((ycoord,xcoord),angle_rad2)},
    coords = {xcoord:new_x,ycoord:new_y}
            )
    sf=2
    alt_dataframe['orient_u'] = ((ycoord,xcoord),sf*np.cos(angle_rad2))
    alt_dataframe['orient_v'] = ((ycoord,xcoord),sf*np.sin(angle_rad2))
    alt_dataframe['-orient_u'] = ((ycoord,xcoord),sf*-np.cos(angle_rad2))
    alt_dataframe['-orient_v'] = ((ycoord,xcoord),sf*-np.sin(angle_rad2))
    return alt_dataframe

def plot(ds,data='ukv'):
    if data=='ukv':
        with open('lee_waves_zenodo/data/projection/crs.pkl', 'rb') as projfile:
            proj=pickle.load(projfile)
        xcoord='projection_x_coordinate'
        ycoord='projection_y_coordinate'
    if data=='synthetic':
        proj=None
        xcoord='x'
        ycoord='y'
    fig = plt.figure(figsize=(13,10),layout='constrained')
    ax1 = fig.add_subplot(221,projection=proj)
    ax2 = fig.add_subplot(222,projection=proj)
    ax3 = fig.add_subplot(223,projection=proj)
    ax4 = fig.add_subplot(224,projection=proj)
    
    
    ds['upward_air_velocity'].plot.pcolormesh(cmap=vv_cmap,robust=False,rasterized=True,ax=ax1, vmin=-4.25,vmax=4.25,add_colorbar=True,
                cbar_kwargs={'label':'Upward Air Velocity (m s $^{-1}$)','shrink':0.6,'ticks':np.arange(-4,5,1),'extend':'neither'})
    
    ds['segmentation'].plot.contour(cmap=cm.ListedColormap(['black']),alpha=1,add_colorbar=False,rasterized=True,ax=ax1)
    
    ds['wavelength'].plot.pcolormesh(cmap='viridis',alpha=1,add_colorbar=True,rasterized=True,ax=ax2,
             cbar_kwargs={
                      'label':'Wavelength (km)',
                      'shrink':0.6,
                      'extend':'neither'}
                )
    #ax1.plot([220000,320000],[400000,400000],transform=proj,zorder=200,lw=3,c='k')
    #ax1.plot([210000,210000],[420000,380000],transform=proj,zorder=200,lw=2,c='k')
    #ax1.plot([330000,330000],[420000,380000],transform=proj,zorder=200,lw=2,c='k')
    #ax1.text(200000,320000,'100 km',transform=proj)
    
    ds['amplitude'].plot.pcolormesh(cmap=amp_cmap,vmin=0,vmax=4.25,robust=False,rasterized=True,ax=ax3,add_colorbar=True,
                    cbar_kwargs={'label':'Amplitude Prediction (m s $^{-1}$)','shrink':0.6,'ticks':np.arange(0,4.5,0.5),'extend':'neither'})
            
    ds['upward_air_velocity'].plot.pcolormesh(cmap=vv_cmap,robust=False,rasterized=True,ax=ax4, vmin=-4.25,vmax=4.25,add_colorbar=True,alpha=1,
                cbar_kwargs={'label':'Upward Air Velocity (m s $^{-1}$)','shrink':0.6,'ticks':np.arange(-4,5,1),'extend':'neither'})
    
       #     scale=60
    headlength=3
    headaxislength=2   #set these to 0 for no arrows

    width=0.004
    ds2= quiver_orient(ds,sep=16,xcoord=xcoord,ycoord=ycoord)        
            #now with arrows!!!!
    print(ds2)
    ds2.plot.quiver(xcoord,ycoord,'orient_u','orient_v',ax=ax4,
                              transform=proj,   
                                  width=width,#0.004,
                          pivot='tail',
                          headlength=headlength, headaxislength=headaxislength,
                                      add_guide=False,
                         )

    ds2.plot.quiver(xcoord,ycoord,'-orient_u','-orient_v',ax=ax4,
                              transform=proj,
                                  width=width, pivot='tail',
                          headlength=headlength, headaxislength=headaxislength,
                                      add_guide=False,
                                          )
    
    ax1.set_title('700 hPa Vertical Velocity and segmentation mask')
    ax2.set_title('Wavelength')
    ax3.set_title('Amplitude')
    ax4.set_title('Orientation (perpendicular to wave fronts)')
    if proj!=None:
        for ax in [ax1,ax2,ax3,ax4]:
            ax.coastlines('10m',alpha=0.5)
    if data=='ukv':
        forecast_time = str(ds['forecast_reference_time'].values)[:-10]+'Z'
        fig.suptitle('Lee Wave Data: Characteristics Prediction '+forecast_time)
    if data=='synthetic':
        fig.suptitle('Synthetic Wave Characteristic Prediction',y=.93)
            
    plt.savefig('example_characteristics.pdf',bbox_inches='tight')
    


leewaves = open_xarray('20210214T0900Z-PT0000H00M-wind_vertical_velocity_at_700hPa.nc')

#synthetic = np.load('37.npy')
#ds=xr.Dataset({'upward_air_velocity':(('y','x'),synthetic)},coords={'x':np.arange(0,512,1),'y':np.arange(0,512,1)})

models = load_models()

#output=predict(models,ds,xcoord='x',ycoord='y',mask_nonwaves=False)
#plot(output,data='synthetic')

output = predict(models,leewaves)
plot(output,data='ukv')