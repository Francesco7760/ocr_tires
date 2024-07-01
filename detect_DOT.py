
import configparser as cp
import cv2
import numpy as np
import easyocr
import pytesseract
from pytesseract import Output
import os 

config = cp.ConfigParser()
config.read(r'config.ini')

reader = easyocr.Reader(['en'], gpu=True) 

font = cv2.FONT_HERSHEY_SIMPLEX

##########################################
## function to create dirs if not exit ##
##########################################
def create_dirs(ARRAY_PATH):
    for  i in ARRAY_PATH:
        if not os.path.exists(i):
            os.makedirs(i, exist_ok=True)

##################
## function OCR ##
##################
## tiling 
## easyOCR as engine OCR
def dot_text_bbox_detect_easyocr(IMAGE_SCAN,TILE_SIZE):
    _,cols_image_scan,_ = IMAGE_SCAN.shape

    stride = TILE_SIZE//2
    num_tiles = cols_image_scan//TILE_SIZE

    i = 0
    text = ""
    run = True
    
    while i <= num_tiles*2-1 and run == True:
        r = reader.readtext(IMAGE_SCAN[:,stride*i:TILE_SIZE+stride*i,:] ,
                                           allowlist = config.get('easyocr','char_allows_list'))
        
        for elem in r:
            #print(i)
            bbox_,text_,_ = elem
            if text_ == "DOT":  
                bbox,text = bbox_,text_
                x_relative_path_top_left,y_relative_path_top_left = bbox[0] 
                x_relative_path_bottom_right,y_relative_path_bottom_right = bbox[2] 
                index=i
                run = False
        i+=1

    X_ASSOLUTE_PATH_TOP_LEFT = x_relative_path_top_left + stride*index
    Y_ASSOLUTE_PATH_TOP_LEFT = y_relative_path_top_left
    X_ASSOLUTE_PATH_BOTTOM_RIGHT = x_relative_path_bottom_right + stride*index
    Y_ASSOLUTE_PATH_BOTTOM_RIGHT = y_relative_path_bottom_right

    return text,X_ASSOLUTE_PATH_TOP_LEFT,Y_ASSOLUTE_PATH_TOP_LEFT,X_ASSOLUTE_PATH_BOTTOM_RIGHT,Y_ASSOLUTE_PATH_BOTTOM_RIGHT


## tesseract as engine OCR
def dot_text_bbox_detect_tesseract(IMAGE_SCAN,TILE_SIZE):
    _,cols_image_scan,_ = IMAGE_SCAN.shape

    stride = TILE_SIZE//2
    num_tiles = cols_image_scan//TILE_SIZE

    i = 0
    
    while i <= num_tiles*2-1:
        d = pytesseract.image_to_data(IMAGE_SCAN[:,stride*i:TILE_SIZE+stride*i,:], config= 
                                      '--psm ' + config.get('tesseract','ts_psm') + 
                                      ' --oem ' + config.get('tesseract','ts_oem') + 
                                      ' -c ' + config.get('tesseract','tessedit_char_whitelist')
                                      ,output_type=Output.DICT)
        
        if 'DOT' in d['text']:
            j = d['text'].index('DOT')
            text = d['text'][j]
            (x, y, w, h) = (d['left'][j], d['top'][j], d['width'][j], d['height'][j])
            bbox = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
            x_relative_path_top_left,y_relative_path_top_left = bbox[0] 
            x_relative_path_bottom_right,y_relative_path_bottom_right = bbox[2] 
            index = i
            #run = False
            break
        i += 1

    X_ASSOLUTE_PATH_TOP_LEFT = x_relative_path_top_left + stride*index
    Y_ASSOLUTE_PATH_TOP_LEFT = y_relative_path_top_left
    X_ASSOLUTE_PATH_BOTTOM_RIGHT = x_relative_path_bottom_right + stride*index
    Y_ASSOLUTE_PATH_BOTTOM_RIGHT = y_relative_path_bottom_right

    return text,X_ASSOLUTE_PATH_TOP_LEFT,Y_ASSOLUTE_PATH_TOP_LEFT,X_ASSOLUTE_PATH_BOTTOM_RIGHT,Y_ASSOLUTE_PATH_BOTTOM_RIGHT


# edges detector
## easyOCR as engine OCR
def dot_text_bbox_detect_easyocr_edges(IMAGE_SCAN,IMAGE_ENHANCED):
    edges_pipeline = cv2.Canny(IMAGE_ENHANCED, 150, 255)
    contours, _ = cv2.findContours(edges_pipeline,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    padding_size = -3

    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        if ((w > 20) and (h > 30)):
            r = reader.readtext(IMAGE_SCAN[y-padding_size:y+h+padding_size,x-padding_size:x+w+padding_size], 
                                allowlist = config.get('easyocr','char_allows_list'))
            if(len(r) > 0 and r[0][1] == 'DOT'):
                bbox = r[0][0]
                text = r[0][1]
                x_relative_path_top_left,y_relative_path_top_left = bbox[0] 
                x_relative_path_bottom_right,y_relative_path_bottom_right = bbox[2] 

    X_ASSOLUTE_PATH_TOP_LEFT = x+x_relative_path_top_left
    Y_ASSOLUTE_PATH_TOP_LEFT = y+y_relative_path_top_left
    X_ASSOLUTE_PATH_BOTTOM_RIGHT = x+w+x_relative_path_bottom_right
    Y_ASSOLUTE_PATH_BOTTOM_RIGHT = y+h+y_relative_path_bottom_right

    return text,X_ASSOLUTE_PATH_TOP_LEFT,Y_ASSOLUTE_PATH_TOP_LEFT,X_ASSOLUTE_PATH_BOTTOM_RIGHT,Y_ASSOLUTE_PATH_BOTTOM_RIGHT


## tesseract as engine OCR
def dot_text_bbox_detect_tesseract_edges(IMAGE_SCAN,IMAGE_ENHANCED):
    edges_pipeline = cv2.Canny(IMAGE_ENHANCED, 150, 255)
    contours, _ = cv2.findContours(edges_pipeline,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    padding_size = -3

    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        if ((w > 20) and (h > 30)):
            d = pytesseract.image_to_data(IMAGE_SCAN[y-padding_size:y+h+padding_size,x-padding_size:x+w+padding_size], config= 
                                      '--psm ' + config.get('tesseract','ts_psm') + 
                                      ' --oem ' + config.get('tesseract','ts_oem') + 
                                      ' -c ' + config.get('tesseract','tessedit_char_whitelist')
                                      ,output_type=Output.DICT)
            if 'DOT' in d['text']:
                j = d['text'].index('DOT')
                text = d['text'][j]
                (x_rel, y_rel, w_rel, h_rel) = (d['left'][j], d['top'][j], d['width'][j], d['height'][j])
                bbox = np.array([[x_rel,y_rel],[x_rel+w_rel,y_rel],[x_rel+w_rel,y_rel+h_rel],[x_rel,y_rel+h_rel]])
                x_relative_path_top_left,y_relative_path_top_left = bbox[0] 
                x_relative_path_bottom_right,y_relative_path_bottom_right = bbox[2] 

    
    X_ASSOLUTE_PATH_TOP_LEFT = x+x_relative_path_top_left
    Y_ASSOLUTE_PATH_TOP_LEFT = y+y_relative_path_top_left
    X_ASSOLUTE_PATH_BOTTOM_RIGHT = x+w+x_relative_path_bottom_right
    Y_ASSOLUTE_PATH_BOTTOM_RIGHT = y+h+y_relative_path_bottom_right
    
    return text,X_ASSOLUTE_PATH_TOP_LEFT,Y_ASSOLUTE_PATH_TOP_LEFT,X_ASSOLUTE_PATH_BOTTOM_RIGHT,Y_ASSOLUTE_PATH_BOTTOM_RIGHT



#######################################
## draw rectangle and text above DOT ##
#######################################
def draw_bbox_text (IMAGE,TEXT,DOT_DETECT_PATH,X_ASSOLUTE_PATH_TOP_LEFT,Y_ASSOLUTE_PATH_TOP_LEFT,X_ASSOLUTE_PATH_BOTTOM_RIGHT,Y_ASSOLUTE_PATH_BOTTOM_RIGHT):

    ## draw bbox 
    cv2.rectangle(IMAGE, (X_ASSOLUTE_PATH_TOP_LEFT,Y_ASSOLUTE_PATH_TOP_LEFT), (X_ASSOLUTE_PATH_BOTTOM_RIGHT,Y_ASSOLUTE_PATH_BOTTOM_RIGHT), color=(0,255,0), thickness=2)
    image_scan_bbox = IMAGE.copy()
    ## write text 
    cv2.putText(image_scan_bbox, TEXT, (X_ASSOLUTE_PATH_TOP_LEFT-70,Y_ASSOLUTE_PATH_TOP_LEFT+80), font, 1,(0,0,255),2,cv2.LINE_AA)
    ## save image with bbox and text
    cv2.imwrite(DOT_DETECT_PATH, image_scan_bbox)
