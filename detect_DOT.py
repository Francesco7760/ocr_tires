
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
## easyOCR as engine OCR
def dot_text_bbox_detect_easyocr(IMAGE_SCAN,TILE_SIZE):
    _,cols_image_scan,_ = IMAGE_SCAN.shape
    TILES = []
    RESULTS_OCR = []

    stride = TILE_SIZE//2
    num_tiles = cols_image_scan//TILE_SIZE

    i = 0
    text = ""
    run = True
    
    while i <= num_tiles*2-1 and run == True:
        TILES.append(IMAGE_SCAN[:,stride*i:TILE_SIZE+stride*i,:])
        RESULTS_OCR.append(reader.readtext(TILES[i], 
                                           allowlist = config.get('easyocr','char_allows_list')))
        
        if len(RESULTS_OCR[i]) != 0: 
            for elem in RESULTS_OCR[i]:
                bbox,text,_ = elem
                if text == "DOT":  
                    index = i
                    run = False
        i += 1

    return text,bbox,stride,index

## tesseract as engine OCR
def dot_text_bbox_detect_tesseract(IMAGE_SCAN,TILE_SIZE):
    _,cols_image_scan,_ = IMAGE_SCAN.shape
    TILES = []

    stride = TILE_SIZE//2
    num_tiles = cols_image_scan//TILE_SIZE

    i = 0
    text = ""
    run = True
    
    while i <= num_tiles*2-1 and run == True:
        TILES.append(IMAGE_SCAN[:,stride*i:TILE_SIZE+stride*i,:])
        d = pytesseract.image_to_data(TILES[i], config= 
                                      '--psm ' + config.get('tesseract','ts_psm') + 
                                      ' --oem ' + config.get('tesseract','ts_oem') + 
                                      ' -c ' + config.get('tesseract','tessedit_char_whitelist')
                                      ,output_type=Output.DICT)
        
        n_boxes = len(d['level'])
        for elem in range(n_boxes):
            if d['text'][elem] == 'DOT':
                text = d['text'][elem]
                (x, y, w, h) = (d['left'][elem], d['top'][elem], d['width'][elem], d['height'][elem])
                bbox = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
                index = i
                run = False
        i += 1

    return text,bbox,stride,index

#######################################
## draw rectangle and text above DOT ##
#######################################
def draw_bbox_text (IMAGE,TEXT,DOT_DETECT_PATH,X_ASSOLUTE_PATH_TOP_LEFT,Y_ASSOLUTE_PATH_TOP_LEFT,X_ASSOLUTE_PATH_BOTTOM_RIGHT,Y_ASSOLUTE_PATH_BOTTOM_RIGHT):

    ## draw bbox 
    cv2.rectangle(IMAGE, (X_ASSOLUTE_PATH_TOP_LEFT,Y_ASSOLUTE_PATH_TOP_LEFT), (X_ASSOLUTE_PATH_BOTTOM_RIGHT,Y_ASSOLUTE_PATH_BOTTOM_RIGHT), color=(0,255,0), thickness=3)
    image_scan_bbox = IMAGE.copy()
    ## write text 
    cv2.putText(image_scan_bbox, TEXT, (X_ASSOLUTE_PATH_TOP_LEFT+10,Y_ASSOLUTE_PATH_TOP_LEFT-20), font, 3,(0,0,255),3,cv2.LINE_AA)
    ## save image with bbox and text
    cv2.imwrite(DOT_DETECT_PATH, image_scan_bbox)
