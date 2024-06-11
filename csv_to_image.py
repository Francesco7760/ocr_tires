import numpy as np
import pandas as pd
from numba import njit

from sklearn.impute import KNNImputer

from irfpy.ica.baseline import als

import pywt

import configparser as cp

config = cp.ConfigParser()
config.read(r'config.ini')

###############################
## READ CSV AND RETURN ARRAY ##
###############################
def read_array(PATH):

    ## lavora con csv
    df = pd.read_csv(PATH, sep=",",header=None)
    array = df.to_numpy()

    return np.delete(array, [0,1]),array[0],array[1]

################
## CROP IMAGE ##
################

def crop_array(ARRAY, LFT_LIMIT, RGT_LIMIT, POINTS_IN_PROFILE,NUM_PROFILE):
    
    NUM_COL = RGT_LIMIT - LFT_LIMIT
    
    array = ARRAY.reshape(NUM_PROFILE,POINTS_IN_PROFILE)
    array_cropped =  array[:,LFT_LIMIT:RGT_LIMIT]
    
    return array_cropped,NUM_COL

###################################
## IMPUTATE NULL VALUES OF ARRAY ##
###################################

def imputate (ARRAY):
    
    imputer = KNNImputer(n_neighbors=5, missing_values=np.nan, weights='distance')
    array_imputate = imputer.fit_transform(ARRAY)
    
    return array_imputate

#########################
## BASELINE CORRECTION ##
#########################

## baseline with polynimial interpolation
def baseline_profile_polyn(PROFILE):

    x = []
    for i in range(len(PROFILE)): x.append(i)

    coef = np.polyfit(x,PROFILE,3)
    poly_fn = np.poly1d(coef)
    baseline = poly_fn(x)
   
    return baseline

## baseline with ALS
def baseline_profile_als(PROFILE):

    baseline = als(PROFILE, lam=1e6, itermax=5)
    
    return baseline

## baseline with wavelet
def baseline_profile_wavelet(PROFILE):

    wavelet_type = 'db6'
    coeffs_wavelet = pywt.wavedec(PROFILE, wavelet_type, level = 7)
    baseline_coeffs = coeffs_wavelet.copy()

    # cancels every component with a degree greater than 0 
    for index,_ in enumerate(baseline_coeffs):
        if index != 0:
            baseline_coeffs[index] = 0*baseline_coeffs[index]

    baseline = pywt.waverec(baseline_coeffs, wavelet_type)

    return baseline

def baseline_correction(BASELINE_ENGINE, PROFILE):
    if str(BASELINE_ENGINE) == 'poly': baseline = baseline_profile_polyn(PROFILE)
    elif str(BASELINE_ENGINE) == 'als': baseline =  baseline_profile_wavelet(PROFILE)
    elif str(BASELINE_ENGINE) == 'wave': baseline = baseline_profile_als(PROFILE)
    else: print("Error: baseline engine wrong, set correct name in config.ini")
    
    return baseline

####################################################
## CONVERT MATRIX ACQUISITION IN GREY SCALE IMAGE ##
####################################################

## detect min, max e avg values on array
def min_max(ARRAY):
    ## min 
    MIN = np.min(ARRAY)
    ## max
    MAX = np.max(ARRAY)
    ## avg
    AVG = np.average(ARRAY)

    return MIN,MAX,AVG

# convert matrix in grey scale
@njit(target_backend='cuda')
def convert_matrix_in_grey_scale(ARRAY,MIN,MAX):

    array_min_max = []
    for row in ARRAY:
        array_row = []
        for i in row:   
            value = (((i)-MIN)/(MAX - MIN))*255
            array_row.append(value)
        
        array_min_max.append(array_row)

    return array_min_max
