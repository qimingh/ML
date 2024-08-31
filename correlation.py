# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:58:18 2023

@author: qm7in
"""

### Merging surgery features 1973~1997 & 1998+
import pandas as pd
import numpy as np



# df = pd.read_csv("/Users/qm7in/Documents/Seer Breast Cancer Therapy 1.csv")
# df.index = df.index + 1
# df= df.replace({'RX Summ--Surg Prim Site (1998+)':{21 : 20, 22 : 20, 23 : 20, 24 : 20}})
# df= df.replace({'RX Summ--Surg Prim Site (1998+)':{41 : 40, 42 : 40, 43 : 40, 44 : 40, 45 : 40, 46 : 40, 47 : 40, 48 : 40, 49 : 40, 75 : 40}})
# df= df.replace({'RX Summ--Surg Prim Site (1998+)':{51 : 50, 52 : 50, 53 : 50, 54 : 50, 55 : 50, 56 : 50, 57 : 50, 58 : 50, 59 : 50, 63 : 50}})
# df= df.replace({'RX Summ--Surg Prim Site (1998+)':{61 : 60, 62 : 60, 64 : 60, 65 : 60, 66 : 60, 67 : 60, 68 : 60, 69 : 60, 73 : 60, 74 : 60}})
# df= df.replace({'RX Summ--Surg Prim Site (1998+)':{71 : 70, 72 : 70}})
# df= df.replace({'RX Summ--Surg Prim Site (1998+)':{19 : 10, 99 : 0}})
# df= df.replace({'RX Summ--Surg Prim Site (1998+)':{126 : np.NaN}})

# # df['Race/ethnicity']= np.where(df['Race/ethnicity'] > 2 & df['Race/ethnicity'] <98, 3, df['Race/ethnicity'])
# # df= df.replace({'Race/ethnicity':{99 : 4}})

# df= df.replace({'Site specific surgery (1973-1997 varying detail by year and site)':{1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0}})
# df= df.replace({'Site specific surgery (1973-1997 varying detail by year and site)':{18 : 10, 28 : 20, 38 : 30, 48 : 40, 58 : 50, 68 : 60, 78 : 70, 88 : 80, 98 : 90}})
# df= df.replace({'Site specific surgery (1973-1997 varying detail by year and site)':{126 : np.NaN}})

# #df['Race/ethnicity'] = np.where(df['Race/ethnicity'].between(3,98), 3, df['Race/ethnicity'])
# #df['Race/ethnicity'] = pd.cut(df.age, bins=[1,2,3,98], labels=[1,2,3,4])
# df= df.replace({'Primary Site':{500 : 10, 501 : 1, 502 : 2, 503 : 3, 504 : 4, 505 : 5, 506 : 6, 507 : 7, 508 : 8, 509 : 9}})
# df= df.replace({'ER Status Recode Breast Cancer (1990+)':{4 : np.NaN, 9 : np.NaN}})
# df= df.replace({'PR Status Recode Breast Cancer (1990+)':{4 : np.NaN, 9 : np.NaN}})
# df= df.replace({'Breast - Adjusted AJCC 6th Stage (1988-2015)':{0 : 1, 10 : 2, 32 : 3, 33 : 4, 51 : 5, 52 : 6, 53 : 7, 54 : 8, 70 : 9, 88 : 10, 99 : np.NaN, 126 : np.NaN}})

# df['surg combine'] = df['RX Summ--Surg Prim Site (1998+)'].combine_first(df['Site specific surgery (1973-1997 varying detail by year and site)'])
# df.drop(['Patient ID',
#          'RX Summ--Surg Prim Site (1998+)', 
#          'Site specific surgery (1973-1997 varying detail by year and site)',
#          'Scope of reg lymph nd surg (1998-2002)',
#          'RX Summ--Scope Reg LN Sur (2003+)', 
#          'RX Summ--Surg Oth Reg/Dis (2003+)', 
#          'Surgery of oth reg/dis sites (1998-2002)'], 1, inplace=True)

# print(" \nCount total NaN at each column in DataFrame : \n\n",df.isnull().sum())

df= pd.read_csv("Seer Breast Cancer Therapy combine 1.csv")
import seaborn as sns
import matplotlib.pyplot as plt


f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

#sns.heatmap(df.corr())
a = df.corr()['Survival months']
print (a)