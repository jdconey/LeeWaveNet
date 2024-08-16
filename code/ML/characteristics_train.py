# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:43:08 2023

@author: mm16jdc@leeds.ac.uk
"""

import os
import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from fastai.vision.all import *
from fastai.data.transforms import get_files, RandomSplitter
import pathlib
#from fastai.vision.data import SegmentationDataLoaders, DataBlock, ImageBlock, MaskBlock
#from fastai.vision.learner import unet_learner, cnn_learner
#from fastai.vision.models import resnet34
from fastai.learner import Learner
import albumentations as A

import xarray as xr
from fastai.vision.all import *
import numpy as np

pathlib.PosixPath = pathlib.WindowsPath

root = pathlib.Path('C:/Users/mm16jdc/Documents/GitHub/LeeWaveNet/data/synthetic/train/')


def label_func(fn): 
    string = str(fn.stem)[:49]+"mask.png"
    return root/"mask_png"/string

def open_xarray(fname):
    x = xr.open_dataarray(fname)
    array = x.values
    return array


fnames = get_files(root/"data")

def open_np(fname):
    x = np.load(fname)
    
    noise = np.random.normal(size=(512,512))
    x = x + threshold*noise
    x2 = np.array([x,x,x])
    return torch.Tensor(x2)

def label_func_wl(fn): 
    string = str(fn.stem)[:49]+".npy"
    lbl = np.load(root/"wavelength"/string).astype('float')#.tolist()
    return lbl/1000

def label_func_amp(fn): 
    string = str(fn.stem)[:49]+".npy"
    lbl = np.load(root/"amplitude"/string).astype('float')#.tolist()
    return lbl

def label_func_or(fn): 
    string = str(fn.stem)[:49]+".npy"
    lbl = np.load(root/"orientation"/string).astype('float')#.tolist()
    lbl_rad = lbl*np.pi/180
    return np.array([np.sin(lbl_rad),np.cos(lbl_rad)])


def train(waves,characteristic):
    dls = waves.dataloaders(root/"data", path=root, bs=2)
    learn3 = load_learner('~/models/segmodel.pkl')
    learn3.model.layers[-2] = nn.Sequential(torch.nn.Conv2d(99, 50, kernel_size=(1, 1), stride=(1, 1)),torch.nn.ReLU(),torch.nn.Conv2d(50, 1, kernel_size=(1, 1), stride=(1, 1)))
    learn3.dls = dls
    learn3.loss_func =  MSELossFlat()
    learn3.unfreeze()
    learn3.freeze_to(-3)
    base_lr = 1e-4
   # base_lr = learn3.lr_find()[0]#1e-4
    
    print('lr',base_lr)
    lr_mult = 10
    learn3.unfreeze()
    learn3.freeze_to(-3)
    learn3.fit_one_cycle(100, slice(base_lr/lr_mult, base_lr),cbs=EarlyStoppingCallback(monitor='valid_loss', patience=5))
    learn3.export('~/models_out/'+characteristic+'_'+str(threshold)+'.pkl')

def train_orientation(waves):
    dls = waves.dataloaders(root/"data", path=root, bs=2)
    learn3 = load_learner('~/models/segmodel.pkl')
    m = learn3.model
    or_model = Learner(dls,m,loss_func=loss_func)
    base_lr = 1e-4

    print('lr',base_lr)
    lr_mult = 10
    or_model.unfreeze()
    or_model.freeze_to(-3)
    or_model.fit_one_cycle(100, slice(base_lr/lr_mult, base_lr),cbs=EarlyStoppingCallback(monitor='valid_loss', patience=5))
    or_model.export('~/models_out/orientation_'+str(threshold)+'.pkl')

def amplitude():
    global threshold
    threshold=0.0625

    waves = DataBlock(
    blocks=(DataBlock, DataBlock),
    get_items = get_files,
    get_x=open_np,
    get_y=label_func_amp,
    splitter=RandomSplitter(),
    batch_tfms=[Normalize.from_stats(*imagenet_stats)],
)
    
    train(waves,'amplitude')
    
def wavelength():
    global threshold
    threshold = 0.125

    waves = DataBlock(
    blocks=(DataBlock, DataBlock),
    get_items = get_files,
    get_x=open_np,
    get_y=label_func_wl,
    splitter=RandomSplitter(),
    batch_tfms=[Normalize.from_stats(*imagenet_stats)],
)
    
    train(waves,'wavelength')
    
def orientation():
    global threshold
    threshold = 0.25

    waves = DataBlock(
    blocks=(DataBlock, DataBlock),
    get_items = get_files,
    get_x=open_np,
    get_y=label_func_or,
    splitter=RandomSplitter(),
    batch_tfms=[Normalize.from_stats(*imagenet_stats)],
)
    
    train_orientation(waves)


