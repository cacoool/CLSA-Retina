import numpy as np
import cv2 as cv


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    #assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    # new_imgs = np.empty(imgs.shape)
    for n in range(imgs.shape[0]):
        for dim in range(imgs.shape[1]):
            imgs[n,dim,:,:] = cv.LUT(np.array(imgs[n,dim,:,:], dtype = np.uint8), table)
    return imgs

# ALLO ELODIE