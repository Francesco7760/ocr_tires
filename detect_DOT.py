
import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['en'], gpu=True) 

font = cv2.FONT_HERSHEY_SIMPLEX

def dot_bbox_detect(IMAGE_SCAN,TILE_SIZE):
    _,cols_image_scan,_ = IMAGE_SCAN.shape
    TILES = []
    RESULTS_OCR = []

    stride = TILE_SIZE//2
    num_tiles = cols_image_scan//TILE_SIZE

    ## while ritorna il numero del tile contenente la scritta DOT
    ## indice = numero_tile - 1
    i = 0
    text = ""
    run = True
    
    while i <= num_tiles*2-1 and run == True:
        TILES.append(IMAGE_SCAN[:,stride*i:TILE_SIZE+stride*i,:])
        RESULTS_OCR.append(reader.readtext(TILES[i], allowlist = "QWERTYUIPASDOFGHJKLZXCVBNM123456789"))
        
        ## considera solo i risultati del OCR
        if len(RESULTS_OCR[i]) != 0: 
            for elem in RESULTS_OCR[i]:
                bbox,text,_ = elem
                print(elem)
                if text == "DOT": 
                    print(f'index tile {i} -- {text} -- bbox {bbox} ') 
                    index = i
                    run = False
        i += 1

    return text,bbox,stride,index


def draw_bbox_text (IMAGE,TEXT,DOT_DETECT_PATH,X_ASSOLUTE_PATH_TOP_LEFT,Y_ASSOLUTE_PATH_TOP_LEFT,X_ASSOLUTE_PATH_BOTTOM_RIGHT,Y_ASSOLUTE_PATH_BOTTOM_RIGHT):

    ## disegna bbox attorno a DOT
    cv2.rectangle(IMAGE, (X_ASSOLUTE_PATH_TOP_LEFT,Y_ASSOLUTE_PATH_TOP_LEFT), (X_ASSOLUTE_PATH_BOTTOM_RIGHT,Y_ASSOLUTE_PATH_BOTTOM_RIGHT), color=(0,255,0), thickness=3)
    image_scan_bbox = IMAGE.copy()
    ## scrivere l'OCR DOT 
    cv2.putText(image_scan_bbox, TEXT, (X_ASSOLUTE_PATH_TOP_LEFT+10,Y_ASSOLUTE_PATH_TOP_LEFT-20), font, 3,(0,0,255),3,cv2.LINE_AA)

    ## salva immagine con bbox e OCR
    cv2.imwrite(DOT_DETECT_PATH, image_scan_bbox)
