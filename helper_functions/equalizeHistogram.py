import numpy as np
import cv2 as cv


def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv.equalizeHist(np.array(imgs[i,0], dtype=np.uint8))
    return imgs_equalized