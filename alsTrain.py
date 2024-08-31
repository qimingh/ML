# -*- coding: utf-8 -*-
"""
Created on Thu May 10 07:32:11 2018

@author: Minot
"""
import numpy as np
import pandas as pd

df = pd.read_csv('subset.01.csv')
data = pd.read_csv('outputAls.txt', delimiter="\t", header = None, names = ['Row','Col','Value'])

data1 = data.loc[data['Col'] == 12]
data1.index = data1.Row-1
data1['Value'] = np.where(data1['Value'] < 10.5, 0, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(10.5,20.5), 10, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(20.5,21.5), 20, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(21.5,22.5), 21, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(22.5,23.5), 22, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(23.5,24.5), 23, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(24.5,27), 24, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(27,35), 30, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(35,40.5), 40, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(40.5,41.5), 41, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(41.5,42.5), 42, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(42.5,43.5), 43, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(40.5,41.5), 41, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(41.5,42.5), 42, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(42.5,43.5), 43, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(43.5,44.5), 44, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(44.5,45.5), 45, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(45.5,46.5), 46, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(46.5,47.5), 47, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(47.5,48.5), 48, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(48.5,49.5), 49, data1['Value'])
data1['Value'] = np.where(data1['Value'].between(49.5,50.5), 50, data1['Value'])
data1['Value'] = np.where(data1['Value'] > 50.5, 51, data1['Value'])
#data1.Value = data1.Value.astype(int)
data2 = data.loc[data['Col'] == 6]
data2.index = data2.Row-1
#data2.Value = data2.Value.astype(int)
data3 = data.loc[data['Col'] == 24]
data3.index = data3.Row-1
data3['Value'] = np.where(data3['Value'] < 3, 0, data3['Value'])
data3['Value'] = np.where(data3['Value'].between(3, 8), 5, data3['Value'])
data3['Value'] = np.where(data3['Value'].between(8, 11.5), 11, data3['Value'])
data3['Value'] = np.where(data3['Value'].between(11.5, 13.5), 12, data3['Value'])
data3['Value'] = np.where(data3['Value'].between(13.5, 16.5), 15, data3['Value'])
data3['Value'] = np.where(data3['Value'].between(16.5, 19), 18, data3['Value'])
data3['Value'] = np.where(data3['Value'].between(19, 25), 20, data3['Value'])
data3['Value'] = np.where(data3['Value'].between(25, 40.5), 30, data3['Value'])
data3['Value'] = np.where(data3['Value']>40.5, 41, data3['Value'])
#data3.Value = data3.Value.astype(int)
data4 = data.loc[data['Col'] == 25]
data4.index = data4.Row-1
data4['Value'] = np.where(data4['Value'] < 5.5, 0, data4['Value'])
data4['Value'] = np.where(data4['Value'].between(5.5, 15), 10, data4['Value'])
data4['Value'] = np.where(data4['Value'].between(15, 25), 20, data4['Value'])
data4['Value'] = np.where(data4['Value'] > 25, 30, data4['Value'])
#data4.Value = data4.Value.astype(int)
data5 = data.loc[data['Col'] == 26]
data5.index = data5.Row-1
data5['Value'] = np.where(data5['Value'] < 5.5, 0, data5['Value'])
data5['Value'] = np.where(data5['Value'] > 5.5, 10, data5['Value'])
#data5.Value = data5.Value.astype(int)
data6 = data.loc[data['Col'] == 27]
data6.index = data6.Row-1
data6['Value'] = np.where(data6['Value'] < 5.5, 0, data6['Value'])
data6['Value'] = np.where(data6['Value'].between(5.5, 21), 10, data6['Value'])
data6['Value'] = np.where(data6['Value'].between(21, 32.5), 32, data6['Value'])
data6['Value'] = np.where(data6['Value'].between(32.5, 42), 33, data6['Value'])
data6['Value'] = np.where(data6['Value'] > 42 , 51, data6['Value'])
#data6.Value = data6.Value.astype(int)
#df['SURGPRIM'] = df['SURGPRIM'].combine_first(data['Value'])

df['SURGPRIM'] = df['SURGPRIM'].fillna(data1['Value'])
df["AGE_DX"] = df["AGE_DX"].replace([999],np.nan)
df['AGE_DX'] = df['AGE_DX'].fillna(data2['Value'])
df['YR_BRTH'] = df['DATE_yr']-df['AGE_DX']
df['ADJTM_6VALUE'] = df['ADJTM_6VALUE'].fillna(data3['Value'])
df['ADJNM_6VALUE'] = df['ADJNM_6VALUE'].fillna(data4['Value'])
df['ADJM_6VALUE'] = df['ADJM_6VALUE'].fillna(data5['Value'])
df['ADJAJCCSTG'] = df['ADJAJCCSTG'].fillna(data6['Value'])
#print df['SURGPRIM']
# filling finish

df['SITEO2V'] = map(lambda x: x.replace("C5","5"), df['SITEO2V'])

df['Label'] = df['SRV_TIME_MON']
df['Label'] = np.where(df['Label'] <= 60, 0, df['Label'])
df['Label'] = np.where(df['Label'] > 60, 1, df['Label'])

from classifier import getRes
getRes(df)