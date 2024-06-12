import configparser as cp
import numpy as np
import cv2
import os

from csv_to_image import read_array, crop_array, imputate, baseline_correction, min_max, convert_matrix_in_grey_scale
from detect_DOT import create_dirs, dot_text_bbox_detect_easyocr, draw_bbox_text


config = cp.ConfigParser()
config.read(r'config.ini')

###############################
## check if directory exists ##
###############################
## and create them if so
DOT_RESULT_DIR = os.path.join('.', 'dor_result_dir')
IMAGE_GREY_DIR = os.path.join('.', 'image_grey_dir')
THRESH_DIR = os.path.join('.', 'thresh_dir')
create_dirs([DOT_RESULT_DIR, IMAGE_GREY_DIR, THRESH_DIR])

####################################
## values extract from config.ini ##
####################################
## name of file to read or save
BIN_FILE_NAME = config.get('file_name','binary_file_name')
CSV_FILE_NAME = config.get('file_name','csv_file_name')
IMAGE_GREY_SCALE_NAME = config.get('file_name','image_grey_scale_name')
THRESHOLD_IMAGE_NAME = config.get('file_name','trheshold_image_name')
DOT_DETECT_NAME = config.get('file_name','dot_detect_name')
## parameters of acquisition
POINTS_IN_PROFILE = int(config.get('general','points_profile'))
NUM_PROFILE = int(config.get('general','lines_count_acquisition'))

###########################################################
## read csv file and create grey scale, threshold images ##
###########################################################
## read csv file
array_csv,_,_ = read_array(PATH = os.path.join('.', 'csv_dir', CSV_FILE_NAME))
## crop image between left e right limits
array_cropped, num_col = crop_array(ARRAY=array_csv,
                           LFT_LIMIT=int(config.get('general','left_limit')),
                           RGT_LIMIT=int(config.get('general','right_limit')),
                           POINTS_IN_PROFILE=int(config.get('general','points_profile')),
                           NUM_PROFILE=int(config.get('general','lines_count_acquisition')))


## imputate NaN values
array_imputate = imputate(ARRAY=array_cropped)

## create baseline 
INDEX_PROFILE = int(config.get('general', 'index_profile_zero'))
BASELINE = baseline_correction(config.get('general', 'baseline_engine')
                               ,ARRAY=array_imputate[INDEX_PROFILE])

## render image from baseline 
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
cv2.imwrite(os.path.join(IMAGE_GREY_DIR,IMAGE_GREY_SCALE_NAME)), cv2.rotate(MATRIX_GREY_SCALE_INV, cv2.ROTATE_90_COUNTERCLOCKWISE)

## image increase constrast(beta) e bright(alpha)
MATRIX_GREY_SCALE_INV_INCREASE = np.clip(MATRIX_GREY_SCALE_INV*
                                         int(config.get('image_processing','beta_constrast'))+
                                         int(config.get('image_processing','alpha_comtrast')),0,255)

## threshold adaptive 
TRHESHOLD_ADAPTIVE = cv2.adaptiveThreshold(MATRIX_GREY_SCALE_INV_INCREASE,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,55,3)

## save thresh image
cv2.imwrite(os.path.join(THRESH_DIR,THRESHOLD_IMAGE_NAME)), cv2.rotate(TRHESHOLD_ADAPTIVE, cv2.ROTATE_90_COUNTERCLOCKWISE)

#################################################
## extract DOT, its writings and its positions ##
#################################################
text,bbox,stride,index = dot_text_bbox_detect_easyocr()

## relative positions
## top - left
relative_path_top_left = bbox[0]
x_relative_path_top_left,y_relative_path_top_left = bbox[0] 
## bottom - right 
realtive_path_bottom_right = bbox[2]
x_relative_path_bottom_right,y_relative_path_bottom_right = bbox[2] 

## assolute positions
## top - left 
x_assolute_path_top_left =  x_relative_path_top_left + stride*index
y_assolute_path_top_left = y_relative_path_top_left
## bottom - right
x_assolute_path_bottom_right =  x_relative_path_bottom_right + stride*index
y_assolute_path_bottom_right = y_relative_path_bottom_right

###############################
## draw text and bbox of DOT ##
###############################
draw_bbox_text(IMAGE = MATRIX_GREY_SCALE_INV_INCREASE,TEXT = text,
               DOT_DETECT_PATH = os.path.join(DOT_RESULT_DIR,DOT_DETECT_NAME),
               X_ASSOLUTE_PATH_TOP_LEFT = x_assolute_path_top_left, 
               Y_ASSOLUTE_PATH_TOP_LEFT = y_assolute_path_top_left,
               X_ASSOLUTE_PATH_BOTTOM_RIGHT = x_assolute_path_bottom_right,
               Y_ASSOLUTE_PATH_BOTTOM_RIGHT = y_assolute_path_bottom_right)