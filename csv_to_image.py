import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, cuda, objmode
from sklearn.impute import SimpleImputer

import configparser as cp

config = cp.ConfigParser()
config.read(r'config.ini')

## read csv and return numpy's array and datframe
def read_array(PATH):

    ## lavora con csv
    print("--- start read ---")
    df = pd.read_csv(PATH, sep=",",header=None)
    array = df.to_numpy()

    print("--- end read ---")

    return array,df

## crops image from LFT_LIMIT to RIGHT_LIMIT
## contains the writing area
def crop_array(ARRAY, LFT_LIMIT, RGT_LIMIT, POINTS_IN_PROFILE,NUM_PROFILE):
    
    NUM_COL = RGT_LIMIT - LFT_LIMIT
    
    array = ARRAY.reshape(NUM_PROFILE,POINTS_IN_PROFILE)
    array_cropped =  array[:,LFT_LIMIT:RGT_LIMIT]
    
    return array_cropped,NUM_COL

## impute the null values of array 
def imputate (ARRAY):
    meanImputer = SimpleImputer(missing_values=np.nan, strategy='median')
    meanImputer.fit(ARRAY)
    ## new_df = meanImputer.transform(df)
    DF_avg = pd.DataFrame(meanImputer.transform(ARRAY)) 
    array_imputate = DF_avg.to_numpy()

    return array_imputate

## fit polynomial on profile
def plot_profile_polyn(PROFILE,DEG,X_ARRAY):

    coef = np.polyfit(x,PROFILE,DEG)
    poly1d_fn = np.poly1d(coef) - 2
    diff = PROFILE - poly1d_fn(x)
   
    return poly1d_fn(X_ARRAY), diff


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