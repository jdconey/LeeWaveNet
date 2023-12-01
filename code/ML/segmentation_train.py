# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:43:08 2023

@author: mm16jdc@leeds.ac.uk
"""

import xarray as xr
from fastai.vision.all import *
import numpy as np
import pathlib
import os

#make a directory called models_out
os.mkdir('models_out')


root = pathlib.Path('~/data/train/')

codes={0:'no wave',255:'lee wave'}

def label_func(fn): 
    string = str(fn.stem)[:49]+"mask.png"
    return root/"masks_png"/string

def open_xarray(fname):
    x = xr.open_dataarray(fname)
    array = x.values
    return array

tfms = [Normalize.from_stats([0,0,0], [1,1,1]),
    Flip(),
    Zoom(max_zoom=20,p=0.5),Rotate(max_deg=360, p=0.9),
  
    ]

waves_ds = DataBlock(blocks = (ImageBlock, MaskBlock(codes)),
                  get_items = get_files,
                  get_x=open_xarray,
                  get_y = label_func,
                  splitter=RandomSplitter(),
                  batch_tfms=tfms,
                    )
dsets = waves_ds.datasets(root/'vertical_velocities')

dls = waves_ds.dataloaders(root/"vertical_velocities", path=root, bs=4)

learn2 = unet_learner(dls,resnet34,metrics=DiceMulti)
learn2.fine_tune(100,cbs=EarlyStoppingCallback(monitor='valid_loss', patience=5))
learn2.export('~/models_out/segmodel.pkl')