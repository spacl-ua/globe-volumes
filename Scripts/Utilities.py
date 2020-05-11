"""
UNIVERSITY OF ARIZONA
Author: Lavanya Umapathy
Utilities for using EvaluationScript

If you use this CNN model in your work, please site the following:
Lavanya Umapathy, Blair Winegar, Lea MacKinnon, Michael Hill, Maria I Altbach, Joseph M Miller and Ali Bilgin, 
"Fully Automated Segmentation of Globes for Volume Quantification in CT Orbits Images Using Deep Learning", 
American Journal of Neuroradiology, June 2020.
"""

import numpy as np
import os, natsort, glob
import dicom
from scipy.ndimage import morphology
from skimage import measure
from skimage.transform import resize
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.models import load_model



'''
Load CT images from a specified dicom directory. Use the loadDicomSeries_sorted 
function to load CT images sorted on instance number
'''
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
    

'''
Image Pre-Processing. Soft tissue contrast is enhanced using a window level WL
and window width WW
'''

def preProcess_orbitalCT(img,opShape,WL=50,WW=100):
    xDim,yDim,zDim = img.shape
    img = HU_threshold(img,WL,WW)
    img = normalize_image(img) 
    img = resize(img,opShape)
    return img

    # Hounsfield Unit thresholding
def HU_threshold(img,WL = 50,WW = 100):
    clip_lwr = WL - (WW/2)
    clip_upr = WL + (WW/2)
    opImg = np.clip(img, clip_lwr, clip_upr) 
    return opImg
 
#Scale the image intensities to  [0,1] range
def normalize_image(img):
    min_, max_ = float(np.min(img)), float(np.max(img))
    return (img - min_) / (max_ - min_)

    
'''
Post-processing on predictions. Morphological operations are performed to fill 
any holes in the binary predictions. Adjust this function as needed.
'''

def postProcess_orbitalCT(img):  
    for idx in range(img.shape[2]):
        img[:,:,idx] = morphology.binary_fill_holes(img[:,:,idx], structure = np.ones((10,10)))
    return img

# Separate the left and the right globes from the predicted globe masks
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


def computeVolumeStats(ipMask,pixdim):
    volume = np.count_nonzero(ipMask) * pixdim[0] * pixdim[1] * pixdim[2]
    volume = volume * 0.001  #convert to mL
    return volume

def computeIGVD(ipMask,pixdim):
    ipMask_L,ipMask_R = findGlobes(ipMask)
    volume_L = np.count_nonzero(ipMask_L) * pixdim[0] * pixdim[1] * pixdim[2] * 0.001
    volume_R = np.count_nonzero(ipMask_R) * pixdim[0] * pixdim[1] * pixdim[2] * 0.001
    return volume_L - volume_R


'''
Functions to use the pretrained model.
'''

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
    # Reshape input to Sections x Height x Width x Channels
    ipImg = np.expand_dims(np.transpose(ipImg,[2,0,1]), axis=3)
    prediction = model.predict(ipImg)
    prediction = np.transpose(np.squeeze(prediction),[1,2,0])
    prediction = binarizePrediction(prediction,0.5)
    return prediction

def binarizePrediction(prediction,threshold):
    prediction_bin = np.zeros(prediction.shape)
    prediction_bin[prediction > threshold] = 1
    prediction_bin[prediction <= threshold] = 0
    return prediction_bin

def setTF_environment(visible_gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 1
    K.set_session(tf.Session(config=config))
    return

    # Model runs on the default GPU if no GPU is specified
def loadSavedModel(modelLoadPath, visible_gpu= None):
    if visible_gpu is None:
        visible_gpu = '0'
    setTF_environment(visible_gpu)
    model = load_model(modelLoadPath)
    Adamopt = optimizers.Adam(lr = 1e-3)
    model.compile(loss=binary_dice_loss, optimizer=Adamopt,metrics=['accuracy'])
    return model

'''
Compute z-score (extent of deviation of IGVD from a normal cohort). The values 
of mean and standard deviation of inter-globe volume difference used here are
from a cohort of normal subjects (n = 98) with no imaging or clinical evidence
of open-globe injuries
'''
def computeZScore(value):
    mean_ = -0.01  # in mL
    std_ = 0.33    # in mL
    zscore = (value - mean_) / std_
    return zscore
    
def computeGlobeStats(prediction,pixdim):
    prediction_left,prediction_right = findGlobes(prediction)
    totalGlobeVolume = computeVolumeStats(prediction,pixdim)
    IGVD = computeIGVD(prediction,pixdim)
    zscore = computeZScore(IGVD)
    # Print the total globe volume (in mL) (left + right)
    print('Predicted volume :',round(totalGlobeVolume,4), 'mL')
    # Print the Inter Globe Volume Difference, IGVD = V_L - V_R
    print('Inter Globe Volume Difference :',round(IGVD,4) ,'mL')
    # Print the IGVD deviation from a normal cohort
    print('Inter Globe Volume Difference :',round(zscore,4) ,'mL')
    return
