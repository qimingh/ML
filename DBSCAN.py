# -*- coding: utf-8 -*-
"""
Created on Sat May 12 20:35:34 2018

@author: Minot
"""

import numpy as np
import pandas as pd

dfl = pd.read_csv('DBSCAN-SOM1500-65.csv')
df = pd.read_csv('subset.csv')
df2 = pd.read_csv('subset.04.csv')
df['SITEO2V'] = map(lambda x: x.replace("C5","5"), df['SITEO2V'])
df['Label'] = df['SRV_TIME_MON']
df['Label'] = np.where(df['Label'] <= 24, 0, df['Label'])
df['Label'] = np.where(np.logical_and(df['Label'] > 24 , df['Label'] <= 60), 1, df['Label'])
df['Label'] = np.where(df['Label'] > 60, 3, df['Label'])


# random categorical data
#data = np.random.choice(20, (100, 10))


dfl = dfl[['CASENUM','label']]
df = df.merge(dfl, on='CASENUM', how='left')
#df2 = df2.merge(df, on='CASENUM', how='left')


for label in range(-1, 9):
    print "label:", label
    newDf = df.loc[df['label'] == label]
    newDf = newDf.apply(lambda x:x.fillna(x.value_counts().index[0]))
    from classifier import getRes
    getRes(newDf)