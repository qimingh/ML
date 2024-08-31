import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn import preprocessing

df = pd.read_csv("/Users/qm7in/Documents/2016+.csv")
df.index = df.index + 1
df= df.replace({'RX Summ--Surg Prim Site (1998+)':{21 : 20, 22 : 20, 23 : 20, 24 : 20}})
df= df.replace({'RX Summ--Surg Prim Site (1998+)':{41 : 40, 42 : 40, 43 : 40, 44 : 40, 45 : 40, 46 : 40, 47 : 40, 48 : 40, 49 : 40, 75 : 40}})
df= df.replace({'RX Summ--Surg Prim Site (1998+)':{51 : 50, 52 : 50, 53 : 50, 54 : 50, 55 : 50, 56 : 50, 57 : 50, 58 : 50, 59 : 50, 63 : 50}})
df= df.replace({'RX Summ--Surg Prim Site (1998+)':{61 : 60, 62 : 60, 64 : 60, 65 : 60, 66 : 60, 67 : 60, 68 : 60, 69 : 60, 73 : 60, 74 : 60}})
df= df.replace({'RX Summ--Surg Prim Site (1998+)':{71 : 70, 72 : 70}})
df= df.replace({'RX Summ--Surg Prim Site (1998+)':{19 : 10, 99 : 0}})
df= df.replace({'RX Summ--Surg Prim Site (1998+)':{126 : np.NaN}})


df= df.replace({'Site specific surgery (1973-1997 varying detail by year and site)':{1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0}})
df= df.replace({'Site specific surgery (1973-1997 varying detail by year and site)':{18 : 10, 28 : 20, 38 : 30, 48 : 40, 58 : 50, 68 : 60, 78 : 70, 88 : 80, 98 : 90}})
df= df.replace({'Site specific surgery (1973-1997 varying detail by year and site)':{126 : np.NaN}})

df= df.replace({'Derived SEER Cmb Stg Grp (2016+)':{1 : 0, 5 : 10, 8 : 10, 14 : 32, 17 : 33, 19 : 51, 20 : 52, 21 : 53, 22 : 54, 25: 70, 32 : 88, 34 : 99, 126 : np.NaN}})

#print (df['RX Summ--Surg Prim Site (1998+)'].value_counts())
#print (df['Site specific surgery (1973-1997 varying detail by year and site)'].value_counts())
#print(" \nCount total NaN at each column in DataFrame : \n\n",
#      df.isnull().sum())
df1 = df[(df['SEER cause-specific death classification'] == 1)]
df2 = df[(df['SEER cause-specific death classification'] == 0) & (df['Survival months'] >= 60)]
df = df1.append(df2)
df['surg combine'] = df['RX Summ--Surg Prim Site (1998+)'].combine_first(df['Site specific surgery (1973-1997 varying detail by year and site)'])
df['stage combine'] = df['Breast - Adjusted AJCC 6th Stage (1988-2015)'].combine_first(df['Derived SEER Cmb Stg Grp (2016+)'])

df = df.loc[df['Record number recode'] == 1]
# df.to_csv("Seer pre-encode.csv",index=False)
df['Age at diagnosis']=df['Age at diagnosis'].astype('category').cat.codes + 1
df['Year of birth']=df['Year of birth'].astype('category').cat.codes + 1
df['Year of diagnosis']=df['Year of diagnosis'].astype('category').cat.codes + 1


df.drop(['Patient ID',
         'RX Summ--Surg Prim Site (1998+)', 
         'Site specific surgery (1973-1997 varying detail by year and site)',
         'Derived SEER Cmb Stg Grp (2016+)',
         'Breast - Adjusted AJCC 6th Stage (1988-2015)',
         'SEER other cause of death classification',
         'SEER cause-specific death classification',
         'Record number recode'], 1, inplace=True)

scaling = preprocessing.MinMaxScaler()
df[['Age at diagnosis','Year of birth','Year of diagnosis']] = scaling.fit_transform(df[['Age at diagnosis','Year of birth','Year of diagnosis']])

# print (df['stage combine'].value_counts())
# print(" \nCount total NaN at each column in a DataFrame : \n\n", df.isnull().sum())
# df.to_csv("Seer Breast Cancer Therapy combine 1.csv",index=False)

encoderSR=ce.OneHotEncoder(cols='SEER registry',handle_unknown='return_nan',return_df=True,use_cat_names=True)

encoderSRB=ce.BinaryEncoder(cols=['SEER registry'],return_df=True)

encoderMS=ce.OneHotEncoder(cols='Marital status at diagnosis',handle_unknown='return_nan',return_df=True,use_cat_names=True)

encoderMSB=ce.BinaryEncoder(cols=['Marital status at diagnosis'],return_df=True)

encoderRE=ce.OneHotEncoder(cols='Race/ethnicity',handle_unknown='return_nan',return_df=True,use_cat_names=True)

encoderREB=ce.BinaryEncoder(cols=['Race/ethnicity'],return_df=True)

encoderRO=ce.OneHotEncoder(cols='Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)',handle_unknown='return_nan',return_df=True,use_cat_names=True)

encoderROB=ce.BinaryEncoder(cols=['Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)'],return_df=True)

encoderPS=ce.OneHotEncoder(cols='Primary Site',handle_unknown='return_nan',return_df=True,use_cat_names=True)

encoderPSB=ce.BinaryEncoder(cols=['Primary Site'],return_df=True)

encoderLR=ce.OneHotEncoder(cols='Laterality',handle_unknown='return_nan',return_df=True,use_cat_names=True)

encoderLRB=ce.BinaryEncoder(cols=['Laterality'],return_df=True)

encoderRS=ce.OneHotEncoder(cols='Reason no cancer-directed surgery',handle_unknown='return_nan',return_df=True,use_cat_names=True)

encoderRSB=ce.BinaryEncoder(cols=['Reason no cancer-directed surgery'],return_df=True)

encoderHT=ce.OneHotEncoder(cols='Histology recode - broad groupings',handle_unknown='return_nan',return_df=True,use_cat_names=True)

encoderHTB=ce.BinaryEncoder(cols=['Histology recode - broad groupings'],return_df=True)

def f1(row):
    if row['ER Status Recode Breast Cancer (1990+)'] == 1:
        val1 = 1
    elif row['ER Status Recode Breast Cancer (1990+)'] == 2:
        val1 = -1
    else:
        val1 = 0
    return val1
def f2(row):
    if row['ER Status Recode Breast Cancer (1990+)'] == 4:
        val2 = 1
    elif row['ER Status Recode Breast Cancer (1990+)'] == 9:
        val2 = -1
    else:
        val2 = 0
    return val2

df['ER Status Recode Breast Cancer (1990+)_0'] = df.apply(f1, axis=1)
df['ER Status Recode Breast Cancer (1990+)_1'] = df.apply(f2, axis=1)

def f3(row):
    if row['PR Status Recode Breast Cancer (1990+)'] == 1:
        val1 = 1
    elif row['PR Status Recode Breast Cancer (1990+)'] == 2:
        val1 = -1
    else:
        val1 = 0
    return val1
def f4(row):
    if row['PR Status Recode Breast Cancer (1990+)'] == 4:
        val2 = 1
    elif row['PR Status Recode Breast Cancer (1990+)'] == 9:
        val2 = -1
    else:
        val2 = 0
    return val2

df['PR Status Recode Breast Cancer (1990+)_0'] = df.apply(f3, axis=1)
df['PR Status Recode Breast Cancer (1990+)_1'] = df.apply(f4, axis=1)

df.drop(['ER Status Recode Breast Cancer (1990+)',
         'PR Status Recode Breast Cancer (1990+)'], 1, inplace=True)

encoderSTO=ce.OrdinalEncoder(cols=['stage combine'],return_df=True,
                            mapping=[{'col':'stage combine',
'mapping':{'None': 0 , 0 : 1 , 10 : 2 , 32 : 3 , 33 : 4 , 51 : 5 , 52 : 6 , 53 : 7 , 54 : 8 , 70 : 9 , 88 : 10 , 99 : 11 , 126 : 12 }}])

encoderSC=ce.OneHotEncoder(cols='surg combine',handle_unknown='return_nan',return_df=True,use_cat_names=True)

encoderSCB=ce.BinaryEncoder(cols=['surg combine'],return_df=True)

encoderSTC=ce.OneHotEncoder(cols='stage combine',handle_unknown='return_nan',return_df=True,use_cat_names=True)

encoderSTCB=ce.BinaryEncoder(cols=['stage combine'],return_df=True)

#One hot encoding for Linear Regression, Naive Bayes or SOM
SR_encodedSR = encoderSR.fit_transform(df)
SR_encodedMS = encoderMS.fit_transform(SR_encodedSR)
SR_encodedRE = encoderRE.fit_transform(SR_encodedMS)
SR_encodedRO = encoderRO.fit_transform(SR_encodedRE)
SR_encodedPS = encoderPS.fit_transform(SR_encodedRO)
SR_encodedLR = encoderLR.fit_transform(SR_encodedPS)
SR_encodedRS = encoderRS.fit_transform(SR_encodedLR)
SR_encodedHT = encoderHT.fit_transform(SR_encodedRS)
SR_encodedSTO = encoderSTO.fit_transform(SR_encodedHT)
SR_encodedSC = encoderSC.fit_transform(SR_encodedSTO)
SR_encodedSTC = encoderSTC.fit_transform(SR_encodedSC)
SR_encodedSTC.to_csv("EncodedOH.csv",index=False)

#Binary encoding for MLP SVM
SR_encodedSRB = encoderSRB.fit_transform(df)
SR_encodedMSB = encoderMSB.fit_transform(SR_encodedSRB)
SR_encodedREB = encoderREB.fit_transform(SR_encodedMSB)
SR_encodedROB = encoderROB.fit_transform(SR_encodedREB)
SR_encodedPSB = encoderPSB.fit_transform(SR_encodedROB)
SR_encodedLRB = encoderLRB.fit_transform(SR_encodedPSB)
SR_encodedRSB = encoderRSB.fit_transform(SR_encodedLRB)
SR_encodedHTB = encoderHTB.fit_transform(SR_encodedRSB)
SR_encodedSTO = encoderSTO.fit_transform(SR_encodedHTB)
SR_encodedSCB = encoderSCB.fit_transform(SR_encodedSTO)
SR_encodedSTCB = encoderSTCB.fit_transform(SR_encodedSCB)
SR_encodedSTCB.to_csv("EncodedB.csv",index=False)
