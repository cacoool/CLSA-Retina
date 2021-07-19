from pandas_plink import read_plink1_bin
import pandas as pd
import numpy as np
G = read_plink1_bin("clsa_gen_v3.bed", "clsa_gen_v3.bim", "clsa_gen_v3.fam", verbose=False)

id_rs429358 = list(G.snp.values).index('AX-95861335')
id_rs7412 = list(G.snp.values).index('AX-59878593')

variant_rs429358 = list(G.variant.values)[id_rs429358]
variant_rs7412 = list(G.variant.values)[id_rs7412]

geneotype_rs429358 = np.asarray(G.sel(variant=variant_rs429358).to_dataframe()['genotype'])
geneotype_rs7412 = np.asarray(G.sel(variant=variant_rs7412).to_dataframe()['genotype'])
iid = np.asarray(G.sel(variant=variant_rs429358).to_dataframe()['iid'])

df = pd.DataFrame(np.array([iid, geneotype_rs429358, geneotype_rs7412]).transpose(), columns=["ADM_GWAS3_COM", "rs429358", "rs7412"])

CLSA_csv = pd.read_csv('CLSA.csv', low_memory=False)
frame = {'entity_id': CLSA_csv['entity_id'],
         'ADM_GWAS3_COM': CLSA_csv['ADM_GWAS3_COM'],
         }
df2 = pd.DataFrame(frame)
df2 = df2.dropna()
df2['ADM_GWAS3_COM'] = df2['ADM_GWAS3_COM'].astype(np.uint16).astype(str)
df.to_pickle("geno.pck")
s1 = pd.merge(df, df2, how='inner', on=['ADM_GWAS3_COM'])
s1 = s1.dropna()

s1.to_pickle("apoe4.pkl")