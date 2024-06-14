import cv2
from skimage.morphology import skeletonize
from skimage.util import invert
import numpy as np

kernel = np.ones((5,5),np.uint8)

## BINARY_IMAGE_MODE = 0 if chars ar black on background white
## BINARY_IMAGE_MODE = 1 if chars ar white on background black

## opening (remove noise to external char)
def opening(IMAGE_BINARY, KERNEL=kernel):
    return cv2.morphologyEx(cv2.cvtColor(IMAGE_BINARY, cv2.COLOR_BGR2GRAY)) , cv2.MORPH_OPEN, KERNEL


## closing
def closing(IMAGE_BINARY, KERNEL=kernel):
    return cv2.morphologyEx(cv2.cvtColor(IMAGE_BINARY, cv2.COLOR_BGR2GRAY), cv2.MORPH_CLOSE, KERNEL)

## dilation
def dilatation(IMAGE_BINARY,BINARY_IMAGE_MODE, N, KERNEL=kernel):
    if BINARY_IMAGE_MODE == 0:
        return invert(cv2.dilate(invert(cv2.cvtColor(IMAGE_BINARY, cv2.COLOR_BGR2GRAY)),KERNEL,iterations = N))
    elif BINARY_IMAGE_MODE == 1:
        return cv2.dilate(cv2.cvtColor(IMAGE_BINARY, cv2.COLOR_BGR2GRAY),KERNEL,iterations = N)
    else:
        print("Error to set BINARY_IMAGE_MODE, 0 or 1")

## erode img (increase thickness)
def erode(IMAGE_BINARY,BINARY_IMAGE_MODE, N, KERNEL=kernel):
    if BINARY_IMAGE_MODE == 0:
        return invert(cv2.erode(invert(cv2.cvtColor(IMAGE_BINARY, cv2.COLOR_BGR2GRAY)),KERNEL,iterations = N))
    elif BINARY_IMAGE_MODE == 1:
        return cv2.erode(cv2.cvtColor(IMAGE_BINARY, cv2.COLOR_BGR2GRAY),KERNEL,iterations = N)
    else:
        print("Error to set BINARY_IMAGE_MODE, 0 or 1")

## skeleton trasformation
def skeleton_image(IMAGE_BINARY, BINARY_IMAGE_MODE, METHOD = 'lee'):
    if BINARY_IMAGE_MODE == 0:
        return invert(skeletonize(invert(cv2.cvtColor(IMAGE_BINARY, cv2.COLOR_BGR2GRAY)), method=METHOD).view(np.uint8)*255)
    elif BINARY_IMAGE_MODE == 1:
        return skeletonize(cv2.cvtColor(IMAGE_BINARY, cv2.COLOR_BGR2GRAY), method=METHOD).view(np.uint8)*255
    else:
        print("Error to set BINARY_IMAGE_MODE, 0 or 1")


## pipeline_result = erode(dilatation(img, kernel=KERNEL, n=2),kernel=KERNEL, n=5)