import numpy as np
import cv2 as cv


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def CLAHEqualize(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    #assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for n in range(imgs.shape[0]):
        for dim in range(imgs.shape[1]):
            imgs[n,dim,:,:] = clahe.apply(np.array(imgs[n,dim,:,:], dtype=np.uint8))
    return imgs