import pandas as pd
import math
import numpy as np

pd.set_option('use_inf_as_na', True)
CLSA_csv = pd.read_csv('data.csv', low_memory=False)
frame = {
    'entity_id': CLSA_csv['entity_id'],
    "SEX_ASK_COM" : CLSA_csv["SEX_ASK_COM"],
    "OWN_DWLG_COM" : CLSA_csv["OWN_DWLG_COM"],
    "ED_ELHS_COM" : CLSA_csv["ED_ELHS_COM"],
    "ED_HIGH_COM" : CLSA_csv["ED_HIGH_COM"],
    "ALC_FREQ_COM" : CLSA_csv["ALC_FREQ_COM"],
    "GEN_HLTH_COM" : CLSA_csv["GEN_HLTH_COM"],
    "GEN_MNTL_COM" : CLSA_csv["GEN_MNTL_COM"],
    "GEN_BRD_COM" : CLSA_csv["GEN_BRD_COM"],
    "GEN_MUSC_COM" : CLSA_csv["GEN_MUSC_COM"],
    "VIS_SGHT_COM" : CLSA_csv["VIS_SGHT_COM"],
    "VIS_AID_COM" : CLSA_csv["VIS_AID_COM"],
    "HRG_HRG_COM" : CLSA_csv["HRG_HRG_COM"],
    "SLE_HOUR_NB_COM" : CLSA_csv["SLE_HOUR_NB_COM"],
    "INJ_OCC_COM" : CLSA_csv["INJ_OCC_COM"],
    "INC_TOT_COM" : CLSA_csv["INC_TOT_COM"],
    "INC_PTOT_COM" : CLSA_csv["INC_PTOT_COM"],
    "ADL_NBRMIS_COM" : CLSA_csv["ADL_NBRMIS_COM"],
    "CCC_F2_COM" : CLSA_csv["CCC_F2_COM"],
    "ADL_DCLST_COM" : CLSA_csv["ADL_DCLST_COM"],
    "ADL_DMEA_COM" : CLSA_csv["ADL_DMEA_COM"],
    "ADL_DCLS_COM" : CLSA_csv["ADL_DCLS_COM"],
    "ALC_TTM_COM" : CLSA_csv["ALC_TTM_COM"],
    "SMK_DSTY_COM" : CLSA_csv["SMK_DSTY_COM"],
    "FAL_DSTA_COM" : CLSA_csv["FAL_DSTA_COM"],
    "HGT_HEIGHT_M_COM" : CLSA_csv["HGT_HEIGHT_M_COM"],
    "WGT_WEIGHT_KG_COM" : CLSA_csv["WGT_WEIGHT_KG_COM"],
    "CCC_HEART_COM" : CLSA_csv["CCC_HEART_COM"],
    "CCC_PVD_COM" : CLSA_csv["CCC_PVD_COM"],
    "CCC_MEMPB_COM" : CLSA_csv["CCC_MEMPB_COM"],
    "CCC_ALZH_COM" : CLSA_csv["CCC_ALZH_COM"],
    "DIA_DIAB_COM" : CLSA_csv["DIA_DIAB_COM"],
    "CCC_HBP_COM" : CLSA_csv["CCC_HBP_COM"],
    "CCC_CVA_COM" : CLSA_csv["CCC_CVA_COM"],
    "CCC_AMI_COM" : CLSA_csv["CCC_AMI_COM"],
    "CCC_TIA_COM" : CLSA_csv["CCC_TIA_COM"],
         }

df = pd.DataFrame(frame)

for col in df.columns:
    cnt = df[col].isnull().sum()
    print(col + ": " + str(cnt))

df["SEX_ASK_COM"][df["SEX_ASK_COM"] == "M"] = 1
df["SEX_ASK_COM"][df["SEX_ASK_COM"] == "F"] = 0

df.to_pickle("meta.pkl")