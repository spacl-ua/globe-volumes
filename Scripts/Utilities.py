#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:12:56 2019

@author: umapathy
Script containing all functions for generating training data
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import morphology
from skimage import measure
from keras import backend as K
import os, natsort, glob
import tensorflow as tf
from keras import optimizers
from keras.models import load_model
import dicom

'''
Load a nii file
'''
def load_orbitalCT(niiPath,opShape=(512,512)):
    xx = nib.load(niiPath)   
    img_r = np.rot90(xx.get_data())
     # Crop to required size
    img_r = myCrop3D(img_r,opShape)
    return img_r

'''
Pre-Processing for CT images with Hounsfield Unit as intensity values.  
'''

def preProcess_orbitalCT(img,opShape,WL=50,WW=100):
    xDim,yDim,zDim = img.shape
    img = HU_threshold(img,WL,WW)
    img = normalize_image(img) 
    img = myCrop3D(img,opShape)
    return img


def postProcess_orbitalCT(img):  
    for idx in range(img.shape[2]):
        img[:,:,idx] = morphology.binary_fill_holes(img[:,:,idx], structure = np.ones((10,10)))
    return img

def getVolumeStats(ipMask,pixdim):
    volume = np.count_nonzero(ipMask) * pixdim[0] * pixdim[1] * pixdim[2]
    return volume

def getPixelInfo(niiPath):
    xx = nib.load(niiPath)   
    return xx.header.get_zooms()


def findGlobes(OcularMask):
    conn = 3
    Eye_L = np.zeros(OcularMask.shape)
    Eye_R = np.zeros(OcularMask.shape)
    [labeled,num_labels] = measure.label(OcularMask, background=False, connectivity=conn,return_num='true')
    
    rp = measure.regionprops(labeled)
    if rp[0].centroid[1] > rp[1].centroid[1]:
        L_idx = 2
        R_idx = 1
    else:
        L_idx = 1
        R_idx = 2
    Eye_L[labeled == L_idx] = 1
    Eye_R[labeled == R_idx] = 1
    return Eye_L,Eye_R


def myCrop3D(ipImg,opShape):
    xDim,yDim = opShape
    zDim = ipImg.shape[2]
    min_val = np.min(ipImg)
    opImg = np.ones((xDim,yDim,zDim))
    opImg = opImg * min_val
    xPad = xDim - ipImg.shape[0]
    yPad = yDim - ipImg.shape[1]
    
    x_lwr = int(np.ceil(np.abs(xPad)/2))
    x_upr = int(np.floor(np.abs(xPad)/2))
    y_lwr = int(np.ceil(np.abs(yPad)/2))
    y_upr = int(np.floor(np.abs(yPad)/2))
    if xPad >= 0 and yPad >= 0:
        opImg[x_lwr:xDim - x_upr ,y_lwr:yDim - y_upr,:] = ipImg
    elif xPad < 0 and yPad < 0:
        xPad = np.abs(xPad)
        yPad = np.abs(yPad)
        opImg = ipImg[x_lwr: -x_upr ,y_lwr:- y_upr,:]
    elif xPad < 0 and yPad >= 0:
        xPad = np.abs(xPad)
        temp_opImg = ipImg[x_lwr: -x_upr,:,:]
        opImg[:,y_lwr:yDim - y_upr,:] = temp_opImg
    else:
        yPad = np.abs(yPad)
        temp_opImg = ipImg[:,y_lwr: -y_upr,:]
        opImg[x_lwr:xDim - x_upr,:,:] = temp_opImg
    return opImg
    


'''
Scale the image intensities to  [0,1] range

'''

def normalize_image(img):
    min_, max_ = float(np.min(img)), float(np.max(img))
    return (img - min_) / (max_ - min_)



'''
Enhance soft tissue contrasts using a window level WL and window width WW

'''

def HU_threshold(img,WL = 50,WW = 100):
    clip_lwr = WL - (WW/2)
    clip_upr = WL + (WW/2)
    opImg = np.clip(img, clip_lwr, clip_upr) 
    return opImg

def computeVolumeStats(ipMask,pixdim):
    volume = np.count_nonzero(ipMask) * pixdim[0] * pixdim[1] * pixdim[2]
    volume = volume * 0.001  #convert to mL
    return volume

def computeIGVD(ipMask,pixdim):
    ipMask_L,ipMask_R = findGlobes(ipMask)
    volume_L = np.count_nonzero(ipMask_L) * pixdim[0] * pixdim[1] * pixdim[2] * 0.001
    volume_R = np.count_nonzero(ipMask_R) * pixdim[0] * pixdim[1] * pixdim[2] * 0.001
    return volume_L - volume_R


def binary_dice_loss(y_true, y_pred):
    eps = 0.00001
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection_fg = K.sum(y_true_f * y_pred_f)
    intersection_bg = K.sum( (1.-y_true_f) * (1.- y_pred_f))
    union_fg = K.sum(y_true_f) + K.sum(y_pred_f)
    union_bg = K.sum(1.-y_true_f) + K.sum(1.- y_pred_f)
    term1 = intersection_fg / (union_fg + eps)
    term2 = intersection_bg / (union_bg + eps)
    loss = 1 - term1 - term2
    return loss

def predictGlobes(model, ipImg):
    ipImg = np.transpose(ipImg,[2,0,1])  
    ipImg = np.expand_dims(ipImg,axis=3)
    predImg = model.predict(ipImg)
    predImg = np.transpose(np.squeeze(predImg),[1,2,0])
    predImg_bin = np.zeros(predImg.shape)
    predImg_bin[predImg >= 0.5] = 1
    predImg_bin[predImg < 0.5] = 0
    return predImg_bin

def setTF_environment(visible_gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 1
    K.set_session(tf.Session(config=config))
    return


def loadSavedModel(modelLoadPath, visible_gpu= None):
    if visible_gpu is None:
        visible_gpu = '0'
    setTF_environment(visible_gpu)
    model = load_model(modelLoadPath)
    Adamopt = optimizers.Adam(lr = 1e-3)
    model.compile(loss=binary_dice_loss, optimizer=Adamopt,metrics=['accuracy'])
    return model

def loadDicomSeries(dcmDirectory, filepattern = "IM_*"):
    if not os.path.exists(dcmDirectory) or not os.path.isdir(dcmDirectory):
        raise ValueError("Given directory does not exist or is a file : "+str(dcmDirectory))
    DcmFiles = natsort.natsorted(glob.glob(os.path.join(dcmDirectory, filepattern)))
 
    FirstDcm = dicom.read_file(DcmFiles[0])
    ArraySize = (int(FirstDcm.Rows), int(FirstDcm.Columns), len(DcmFiles))
    ArrayDicom = np.zeros(ArraySize, dtype=FirstDcm.pixel_array.dtype)

    for DcmFileName in DcmFiles:
        temp = dicom.read_file(DcmFileName)
        ArrayDicom[:, :, DcmFiles.index(DcmFileName)] = temp.pixel_array
    return ArrayDicom
    
def getPixDims_Dicom(dcmDirectory, filepattern = "IM_*"):
    if not os.path.exists(dcmDirectory) or not os.path.isdir(dcmDirectory):
        raise ValueError("Given directory does not exist or is a file : "+str(dcmDirectory))
    DcmFiles = natsort.natsorted(glob.glob(os.path.join(dcmDirectory, filepattern)))
    FirstDcm = dicom.read_file(DcmFiles[0])
    xDim, yDim = np.round(FirstDcm.PixelSpacing,4)
    zDim = FirstDcm.SliceThickness
    return (xDim,yDim,zDim)


def loadDicomSeries_sorted(dcmDirectory, filepattern = "IM_*"):
    if not os.path.exists(dcmDirectory) or not os.path.isdir(dcmDirectory):
        raise ValueError("Given directory does not exist or is a file : "+str(dcmDirectory))
    DcmFiles = natsort.natsorted(glob.glob(os.path.join(dcmDirectory, filepattern)))
 
    FirstDcm = dicom.read_file(DcmFiles[0])
    ArraySize = (int(FirstDcm.Rows), int(FirstDcm.Columns), len(DcmFiles))
    ArrayDicom = np.zeros(ArraySize, dtype=FirstDcm.pixel_array.dtype)
    SliceLocation = np.zeros((len(DcmFiles)))
    for DcmFileName in DcmFiles:
        temp = dicom.read_file(DcmFileName)
        ArrayDicom[:, :, DcmFiles.index(DcmFileName)] = temp.pixel_array
        SliceLocation[DcmFiles.index(DcmFileName)] = int(temp.InstanceNumber)
    sorted_idx = np.argsort(SliceLocation)
    ArrayDicom = ArrayDicom[:,:,sorted_idx]
    return ArrayDicom
