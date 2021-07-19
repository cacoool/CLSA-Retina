# ---------------------------------- Create Dataset  ------------------------------------------
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2 as cv
import os

variable = "CFA_GRAHAM_GoodAndUsable"

df_geno = pd.read_pickle("/hdd/retina/geno.pkl")
df_meta = pd.read_pickle("/hdd/retina/meta.pkl")
df_meta_phys = pd.read_pickle("/hdd/retina/meta_phys.pkl")
df = pd.read_spss("/home/dcorbin/Desktop/ALL_PP_df_CLSA_en_filtered_C_test_C.sav")
scoreList = []
scanList = []
# merge with genomics data
s1 = pd.merge(df, df_geno, how='inner', on=['entity_id'])
# merge with METADATA_phys
s2 = pd.merge(s1, df_meta_phys, how='inner', on=['entity_id'])
# merge with METADATA
s2 = pd.merge(s2, df_meta, how='inner', on=['entity_id'])
s2 = s2.fillna(0)
s2 = s2.sort_values(['rs429358', 'rs7412'], ascending=[True, False])
for index, row in s2.iterrows():
        ligne = np.array(row)
        ligne[15] = ligne[15].replace("crop_800_PP", "GoodAndUsable")
        if os.path.isfile(ligne[15]):
                scanList.append(ligne[15])
                ligne[15] = 0
                scoreList.append(ligne.astype(np.float32))

# one hot encoding of metadata
array_score = np.delete(np.array(scoreList), 1 , 1)
meta = np.concatenate([array_score[:, 39:40], array_score[:, 41:64], array_score[:, 66:]], axis=1).astype(np.uint8)
meta[meta > 10] = 0
size = np.int(meta.max(axis=0).sum() + len(meta.max(axis=0)))
metaEnc = np.zeros((meta.shape[0],size))
min_array = 0
for i in range(meta.shape[1]):
        a = meta[:, i]
        b = np.zeros((a.size, a.max()+1))
        b[np.arange(a.size), a] = 1
        max_array = a.max() + 1 + min_array
        metaEnc[:, min_array:max_array] = b
        min_array = max_array
meta = metaEnc.copy().astype(np.uint8)

new_scoreList = []
test = []
imgList = np.ones((len(scanList), 600, 600, 3), dtype=np.uint8)
for i, scan in enumerate(tqdm(scanList)):
        imgList[i] = cv.resize(np.array(Image.open(scan)), (600, 600))
        new_scoreList.append(scoreList[i])

with h5py.File("CLSA_" + variable + ".hdf5", "w") as f:
        f.create_dataset("img", data=np.array(imgList), dtype=np.array(imgList).dtype, compression="gzip")
        f.create_dataset("gt", data=np.array(new_scoreList).astype(np.float32), dtype=np.array(new_scoreList).astype(np.float32).dtype, compression="gzip")
        f.create_dataset("meta", data=meta, dtype=meta.dtype, compression="gzip")


