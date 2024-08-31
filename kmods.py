# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:55:51 2018

@author: Minot
"""

import pandas as pd
import numpy as np
from kmodes.kmodes import KModes

df = pd.read_csv('subset.04.csv')
df['SITEO2V'] = map(lambda x: x.replace("C5","5"), df['SITEO2V'])


for n_cluster in range(3, 10, 2):
    print ("#. n_cluster is: ", n_cluster)
    km = KModes(n_clusters= n_cluster , init='Huang', n_init=5, verbose=1)
    clusters = km.fit_predict(df)
#    print clusters
#    df['cluster'] = clusters
    df_train = df.copy()
    df_train['cluster'] = clusters    
    df_train['Label'] = df['SRV_TIME_MON']
    df_train['Label'] = np.where(df_train['Label'] <= 36, 0, df_train['Label'])
#    df_train['Label'] = np.where(np.logical_and(df_train['Label'] > 24 , df_train['Label'] <= 60), 1, df_train['Label'])
    df_train['Label'] = np.where(df_train['Label'] > 36, 2, df_train['Label'])

    for index in range(0, n_cluster):    
        newDf = df_train.loc[df_train['cluster'] == index]
        newDf = newDf.apply(lambda x:x.fillna(x.value_counts().index[0]))   
        from classifier import getRes
        getRes(newDf)

