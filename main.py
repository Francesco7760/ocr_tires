import configparser as cp
import numpy as np
import cv2
import os

from csv_to_image import read_array, crop_array, imputate, baseline_correction, min_max, convert_matrix_in_grey_scale
from detect_DOT import dot_bbox_detect, draw_bbox_text


config = cp.ConfigParser()
config.read(r'config.ini')

BIN_FILE_NAME = str(config.get('file_name','binary_file_name'))
CSV_FILE_NAME = str(config.get('file_name','csv_file_name'))
IMAGE_GREY_SCALE_NAME = str(config.get('file_name','image_grey_scale_name'))
THRESHOLD_IMAGE_NAME = str(config.get('file_name','trheshold_image_name'))
DOT_DETECT_NAME = str(config.get('file_name','dot_detect_name'))

POINTS_IN_PROFILE = int(config.get('general','points_profile'))
NUM_PROFILE = int(config.get('general','lines_count_acquisition'))

## read csv file, csv represents matrix rows x colums
array_csv,_,_ = read_array(PATH = os.path.join('.', 'csv_dir', CSV_FILE_NAME))

array_cropped, num_col = crop_array(ARRAY=array_csv,
                           LFT_LIMIT=int(config.get('general','left_limit')),
                           RGT_LIMIT=int(config.get('general','right_limit')),
                           POINTS_IN_PROFILE=int(config.get('general','points_profile')),
                           NUM_PROFILE=int(config.get('general','lines_count_acquisition')))

array_imputate = imputate(ARRAY=array_cropped)

INDEX_PROFILE = int(config.get('general', 'index_profile_zero'))

BASELINE = baseline_correction(str(config.get('general', 'baseline_engine'))
                               ,ARRAY=array_imputate[INDEX_PROFILE])

MATRIX_RELATIVE = []

for i,pr in enumerate(array_imputate): MATRIX_RELATIVE.append(pr - BASELINE*1)

MATRIX_RELATIVE = np.array(MATRIX_RELATIVE).reshape(NUM_PROFILE,num_col)

## get the grayscale image
MATRIX_GREY_SCALE = [] 
MIN,MAX,AVG = min_max(MATRIX_RELATIVE)

MATRIX_GREY_SCALE = convert_matrix_in_grey_scale(MATRIX_RELATIVE,MIN=AVG-5,MAX=AVG+5)
MATRIX_GREY_SCALE = np.array(MATRIX_GREY_SCALE, dtype=np.uint8).reshape(NUM_PROFILE,num_col)

## invert color, char black on background white
MATRIX_GREY_SCALE_INV = 255 - MATRIX_GREY_SCALE

## save image
cv2.imwrite(os.path.join('.','image_grey_dir',IMAGE_GREY_SCALE_NAME)), cv2.rotate(MATRIX_GREY_SCALE_INV, cv2.ROTATE_90_COUNTERCLOCKWISE)

## image increase constrast(beta) e bright(alpha)
MATRIX_GREY_SCALE_INV_INCREASE = np.clip(MATRIX_GREY_SCALE_INV*
                                         int(config.get('image_processing','beta_constrast'))+
                                         int(config.get('image_processing','alpha_comtrast')),0,255)

## threshold adaptive 
TRHESHOLD_ADAPTIVE = cv2.adaptiveThreshold(MATRIX_GREY_SCALE_INV_INCREASE,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,55,3)

cv2.imwrite(os.path.join('.','thresh_dir',THRESHOLD_IMAGE_NAME)), cv2.rotate(TRHESHOLD_ADAPTIVE, cv2.ROTATE_90_COUNTERCLOCKWISE)

text,bbox,stride,index = dot_bbox_detect()

### relative 
## top - left
relative_path_top_left = bbox[0]
x_relative_path_top_left,y_relative_path_top_left = bbox[0] 

## bottom - right 
realtive_path_bottom_right = bbox[2]
x_relative_path_bottom_right,y_relative_path_bottom_right = bbox[2] 

### assolute
## top - left 
x_assolute_path_top_left =  x_relative_path_top_left + stride*index
y_assolute_path_top_left = y_relative_path_top_left
## bottom - right
x_assolute_path_bottom_right =  x_relative_path_bottom_right + stride*index
y_assolute_path_bottom_right = y_relative_path_bottom_right

draw_bbox_text(IMAGE = MATRIX_GREY_SCALE_INV_INCREASE,TEXT = text,
               DOT_DETECT_PATH = os.path.join('.','dot_result_dir',DOT_DETECT_NAME),
               X_ASSOLUTE_PATH_TOP_LEFT = x_assolute_path_top_left, 
               Y_ASSOLUTE_PATH_TOP_LEFT = y_assolute_path_top_left,
               X_ASSOLUTE_PATH_BOTTOM_RIGHT = x_assolute_path_bottom_right,
               Y_ASSOLUTE_PATH_BOTTOM_RIGHT = y_assolute_path_bottom_right)