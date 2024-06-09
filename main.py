import configparser as cp
import numpy as np
import cv2

from csv_to_image import read_array, crop_array, imputate, plot_profile_polyn, min_max, convert_matrix_in_grey_scale


config = cp.ConfigParser()
config.read(r'config.ini')

POINTS_IN_PROFILE = int(config.get('general','points_profile'))
NUM_PROFILE = int(config.get('general','lines_count_acquisition'))

#str(config.get('directory', 'binary_directory_path'))

'''
read csv 

convert trheshold
- read csv form dir
- crop area write
- imputate
- best profile zero
- fit array with profile zero
- normalized 0-255
- save image

tiles procedure
- split image in tiles 
- draw bbox of dot
- write OCR result
- draw position OCR

result OCR
'''

## read csv file, csv represents matrix rows x colums
array_csv = read_array(PATH = str(config.get('directory', 'csv_file_path')))

array_cropped, num_col = crop_array(ARRAY=array_csv,
                           LFT_LIMIT=int(config.get('general','left_limit')),
                           RGT_LIMIT=int(config.get('general','right_limit')),
                           POINTS_IN_PROFILE=int(config.get('general','points_profile')),
                           NUM_PROFILE=int(config.get('general','lines_count_acquisition')))

array_imputate = imputate(ARRAY=array_cropped)

## x is a useful array for the polynomial fit
x = []
for i in range(len(array_imputate[0])): x.append(i)

INDEX_PROFILE = int(config.get('general', 'index_profile_zero'))

POLY_ZERO, diff = plot_profile_polyn(ARRAY=array_imputate[INDEX_PROFILE],
                                     DEG=int(config.get('general', 'deg_polynomial')))

MATRIX_RELATIVE = []

for i,pr in enumerate(array_imputate): MATRIX_RELATIVE.append(pr - POLY_ZERO*1)

MATRIX_RELATIVE = np.array(MATRIX_RELATIVE).reshape(NUM_PROFILE,num_col)

## get the grayscale image
MATRIX_GREY_SCALE = [] 
MIN,MAX,AVG = min_max(MATRIX_RELATIVE)

MATRIX_GREY_SCALE = convert_matrix_in_grey_scale(MATRIX_RELATIVE,MIN=AVG-5,MAX=AVG+5)
MATRIX_GREY_SCALE = np.array(MATRIX_GREY_SCALE, dtype=np.uint8).reshape(NUM_PROFILE,num_col)

## invert color, char black on background white
MATRIX_GREY_SCALE_INV = 255 - MATRIX_GREY_SCALE

## save image
cv2.imwrite(str(config.get('files', 'image_grey_scale_path')), cv2.rotate(MATRIX_GREY_SCALE_INV, cv2.ROTATE_90_COUNTERCLOCKWISE))