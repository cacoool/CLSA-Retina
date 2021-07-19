import cv2 as cv
import numpy as np
from helper_functions.adjustGamma import adjust_gamma
from helper_functions.CLAHEqualization import CLAHEqualize
from glob import glob
from tqdm import tqdm


def scaleRadius(img, scale):
    x = img[img.shape[0]//2, :, :].sum(1)
    r = (x > x.mean()/10).sum()/2
    s = scale*1.0/r
    return cv.resize(img, (0, 0), fx=s, fy=s)

def preprocess_graham(img, scale):
        a = scaleRadius(img, scale)
        b = np.zeros(a.shape)
        cv.circle(b, (a.shape[1]//2, a.shape[0]//2), int(scale*0.9), (1, 1, 1), -1, 8, 0)
        FOV_mask = cv.resize(b, (800, 800))
        FOV_mask = FOV_mask.transpose(2, 0, 1).astype(np.bool)
        aa = cv.addWeighted(a, 4, cv.GaussianBlur(a, (0, 0), scale/30), -4, 128)*b+128*(1-b)
        aa = cv.resize(aa, (800, 800))
        aa = np.expand_dims(aa.transpose(2, 0 ,1), 0)
        hyperparameters = {}
        hyperparameters['hyperX'] = True

        for n in range(aa.shape[0]):
            for d in range(aa.shape[1]):
                if np.max(aa[n, d]) != np.min(aa[n, d]):
                    imgs_mean = np.mean(aa[n, d, :, :][FOV_mask[n]])
                    imgs_std = np.std(aa[n, d, :, :][FOV_mask[n]])
                aa[n, d, :, :] = np.nan_to_num((aa[n, d, :, :] - imgs_mean) / imgs_std, 0) * FOV_mask[n]
                min = np.min(aa[n, d][FOV_mask[n]])
                max = np.max(aa[n, d][FOV_mask[n]])
                aa[n, d, :, :] = (((aa[n, d] - min) / (max - min) * FOV_mask[n])) * 255

        aa = CLAHEqualize(aa)
        aa = adjust_gamma(aa, 1.2)
        aa = aa/255.

        aa = (aa.squeeze()*FOV_mask*255).astype(np.uint8).transpose(1, 2, 0)
        # cv.imshow("test",aa)
        # cv.waitKeyEx()
        cv.imwrite("/hdd/CLSA/Baseline/2006020_EPolytechniqueM_FLesage_retinal_baseline_GoodAndUsable_Graham" + f[f.rfind('/'):], aa)


if __name__ == '__main__':
    for f in tqdm((glob("/hdd/CLSA/Baseline/2006020_EPolytechniqueM_FLesage_retinal_baseline_GoodAndUsable/*.jpg"))):
        preprocess_graham(cv.imread(f), 400)


