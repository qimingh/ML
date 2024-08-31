# -*- coding: utf-8 -*-
"""
Created on Sun May 20 20:02:29 2018

@author: Minot
"""


import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
df = pd.read_csv('outputP-50.txt', delimiter=" ", header = None)
dfl = pd.read_csv('subset.01.csv')
dfl['SITEO2V'] = map(lambda x: x.replace("C50","1"), dfl['SITEO2V'])
dfl['Label'] = dfl['SRV_TIME_MON']
dfl['Label'] = np.where(dfl['Label'] <= 60, 0, dfl['Label'])
dfl['Label'] = np.where(dfl['Label'] > 60, 1, dfl['Label'])

print ("df.shape", df.shape)

for n_cluster in range(3, 10, 2):
    print ("#. n_cluster is: ", n_cluster)
    km = KModes(n_clusters= n_cluster , init='Huang', n_init=5, verbose=1)
    clusters = km.fit_predict(df)
#    print clusters
    appended_data = pd.DataFrame()
    dfl['cluster'] = clusters    
    for index in range(0, n_cluster):    
        newDf = dfl.loc[dfl['cluster'] == index]
        newDf = newDf.apply(lambda x:x.fillna(x.value_counts().index[0]))
        appended_data = appended_data.append(newDf,ignore_index=True)
    dropList = ['cluster']
    for item in dropList:
        appended_data.drop(item, axis=1, inplace=True)
    print ("appended_data.shape", appended_data.shape)
    from classifier import getRes
    getRes(appended_data)

