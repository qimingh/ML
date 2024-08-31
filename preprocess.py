
import pandas as pd
import numpy as np
df1 = pd.read_csv('subset.01.csv')
df1.index = df1.index + 1
df1['SITEO2V'] = map(lambda x: x.replace("C50",""), df1['SITEO2V'])
df1["SITEO2V"] = df1["SITEO2V"].astype(str).astype(int)
df1["AGE_DX"] = df1["AGE_DX"].replace([999],np.nan)
df1["ADJM_6VALUE"] = df1["ADJM_6VALUE"].replace([99,0],[1,1])
df5 = df1[pd.notnull(df1['AGE_DX'])]
#print(df1['RACE'].value_counts())
#print(df1['REG']).value_counts()
df1= df1.drop('SRV_TIME_MON', 1)
df1 = df1.drop('CASENUM', 1)
null_data = df1[df1.isnull().any(axis=1)] #Every possible missing
# print (null_data)
df2 = df1[pd.notnull(df1['ADJTM_6VALUE'])] # ADJTM not missing, SURGPRIM can be missing

df3 = df1[pd.notnull(df1['SURGPRIM'])] # SURGPRIM no_missing
df4 = null_data[pd.isnull(null_data['SURGPRIM'])] # SURGPRIM missing
df6 = df1[pd.isnull(df1['AGE_DX'])]
null_data = null_data[pd.isnull(null_data['ADJTM_6VALUE'])] #ADJTM missing
# null_data = df1[df1.isnull().any(axis=1)]
#print df1

df = df1
df["REG_a"] = df["REG"].astype(int)
df['REG1'] = df["REG"]/df["REG"]
df['REG1'] = df['REG1'].astype(int)
REG = df[['REG1','REG_a','REG1']]
df["MAR_STAT_a"] = df["MAR_STAT"].astype(int)
df["MAR_STAT2"] = df["MAR_STAT"]/df["MAR_STAT"]*2
df["MAR_STAT2"] = df ["MAR_STAT2"].astype(int)
MAR_STAT = df[['MAR_STAT2','MAR_STAT_a','MAR_STAT2']]
df["RACE3"] = df["RACE"]/df["RACE"]*3
df["RACE3"] = df["RACE3"].astype(int)
RACE = df[['RACE3','RACE','RACE3']]
df["ORIGIN4"] = df["REG"]/df["REG"]*4
df["ORIGIN4"] = df["ORIGIN4"].astype(int)
ORIGIN = df[['ORIGIN4','ORIGIN','ORIGIN4']]
SEX = df["SEX"]
df["SEX5"] = df ["SEX"]/df["SEX"]*5
df["SEX5"] = df["SEX5"].astype(int)
SEX = df[['SEX5','SEX','SEX5']]

df["SEQ_NUM8"] = df["REG"]/df["REG"]*8
df["SEQ_NUM8"] = df["SEQ_NUM8"].astype(int)
SEQ_NUM = df[['SEQ_NUM8','SEQ_NUM','SEQ_NUM8']]
df["SITEO2V10"] = df["REG"]/df["REG"]*10
df["SITEO2V10"] = df["SITEO2V10"].astype(int)
SITEO2V = df[['SITEO2V10','SITEO2V','SITEO2V10']]


df["LATERAL11"] = df["LATERAL"]/df["LATERAL"]*11
df["LATERAL11"] = df["LATERAL11"].astype(int)
LATERAL = df[['LATERAL11','LATERAL','LATERAL11']]


df["NO_SURG13"] = df["REG"]/df["REG"]*13
df["NO_SURG13"] = df["NO_SURG13"].astype(int)
NO_SURG = df[['NO_SURG13','NO_SURG','NO_SURG13']]


df["RADIATN14"] = df["REG"]/df["REG"]*14
df["RADIATN14"] = df["RADIATN14"].astype(int)
RADIATN = df[['RADIATN14','RADIATN','RADIATN14']]

df["RAD_SURG15"] = df["REG"]/df["REG"]*15
df["RAD_SURG15"] = df["RAD_SURG15"].astype(int)
RAD_SURG = df[['RAD_SURG15','RAD_SURG','RAD_SURG15']]
df["NUMPRIMS16"] = df["NUMPRIMS"]/df["NUMPRIMS"]*16
df["NUMPRIMS16"] = df["NUMPRIMS16"].astype(int)
NUMPRIMS = df[['NUMPRIMS16','NUMPRIMS','NUMPRIMS16']]

df["FIRSTPRM17"] = df["REG"]/df["REG"]*17
df["FIRSTPRM17"] = df["FIRSTPRM17"].astype(int)
FIRSTPRM = df[['FIRSTPRM17','FIRSTPRM','FIRSTPRM17']]

df["HISTREC18"] = df["REG"]/df["REG"]*18
df["HISTREC18"] = df["HISTREC18"].astype(int)
HISTREC = df[['HISTREC18','HISTREC','HISTREC18']]

df["DTH_CLASS19"] = df["REG"]/df["REG"]*19
df["DTH_CLASS19"] = df["DTH_CLASS19"].astype(int)
DTH_CLASS = df[['DTH_CLASS19','DTH_CLASS','DTH_CLASS19']]

df["ERSTATUS20"] = df["ERSTATUS"]/df["ERSTATUS"]*20
df["ERSTATUS20"] = df["ERSTATUS20"].astype(int)
ERSTATUS = df[['ERSTATUS20','ERSTATUS','ERSTATUS20']]

df["PRSTATUS21"] = df["PRSTATUS"]/df["PRSTATUS"]*21
df["PRSTATUS21"] = df["PRSTATUS21"].astype(int)
PRSTATUS = df[['PRSTATUS21','PRSTATUS','PRSTATUS21']]
df["SRV_TIME_MON_FLAG22"] = df["REG"]/df["REG"]*22
df["SRV_TIME_MON_FLAG22"] = df["SRV_TIME_MON_FLAG22"].astype(int)
SRV_TIME_MON_FLAG = df[['SRV_TIME_MON_FLAG22','SRV_TIME_MON_FLAG','SRV_TIME_MON_FLAG22']]
df["BEHANAL23"] = df["BEHANAL"]/df["BEHANAL"]*23
df["BEHANAL23"] = df["BEHANAL23"].astype(int)
BEHANAL = df[['BEHANAL23','BEHANAL','BEHANAL23']]
df= df5

df["AGE_DX6"] = df["AGE_DX"]/df["AGE_DX"]*6
df["AGE_DX6"] = df["AGE_DX6"].astype(int)
AGE_DX = df[['AGE_DX6','AGE_DX','AGE_DX6']]
df= df3

df["SURGPRIM12"] = df["REG"]/df["REG"]*12
df["SURGPRIM12"] = df["SURGPRIM12"].astype(int)
SURGPRIM = df[['SURGPRIM12', 'SURGPRIM', 'SURGPRIM12']]
df= df2

df["ADJTM_6VALUE24"] = df["REG"]/df["REG"]*24
df["ADJTM_6VALUE24"] = df["ADJTM_6VALUE24"].astype(int)
ADJTM_6VALUE = df[['ADJTM_6VALUE24','ADJTM_6VALUE','ADJTM_6VALUE24']]

df["ADJNM_6VALUE25"] = df["REG"]/df["REG"]*25
df["ADJNM_6VALUE25"] = df["ADJNM_6VALUE25"].astype(int)
ADJNM_6VALUE = df[['ADJNM_6VALUE25','ADJNM_6VALUE','ADJNM_6VALUE25']]

df["ADJM_6VALUE26"] = df["REG"]/df["REG"]*26
df["ADJM_6VALUE26"] = df["ADJM_6VALUE26"].astype(int)
ADJM_6VALUE = df[['ADJM_6VALUE26','ADJM_6VALUE','ADJM_6VALUE26']]

df["ADJAJCCSTG27"] = df["REG"]/df["REG"]*27
df["ADJAJCCSTG27"] = df["ADJAJCCSTG27"].astype(int)
ADJAJCCSTG = df[['ADJAJCCSTG27','ADJAJCCSTG','ADJAJCCSTG27']]
with open('not_null.txt', 'a') as f:
    REG.to_csv(f, header=False, index=True, sep='\t')
    MAR_STAT.to_csv(f, header=False, index=True, sep='\t')
    RACE.to_csv(f, header=False, index=True, sep='\t')
    ORIGIN.to_csv(f, header=False, index=True, sep='\t')
    SEX.to_csv(f, header=False, index=True, sep='\t')
    AGE_DX.to_csv(f, header=False, index=True, sep='\t')
    SEQ_NUM.to_csv(f, header=False, index=True, sep='\t')
    SITEO2V.to_csv(f, header=False, index=True, sep='\t')
    LATERAL.to_csv(f, header=False, index=True, sep='\t')
    SURGPRIM.to_csv(f, header=False, index=True, sep='\t')
    NO_SURG.to_csv(f, header=False, index=True, sep='\t')
    RADIATN.to_csv(f, header=False, index=True, sep='\t')
    RAD_SURG.to_csv(f, header=False, index=True, sep='\t')
    NUMPRIMS.to_csv(f, header=False, index=True, sep='\t')
    FIRSTPRM.to_csv(f, header=False, index=True, sep='\t')
    HISTREC.to_csv(f, header=False, index=True, sep='\t')
    DTH_CLASS.to_csv(f, header=False, index=True, sep='\t')
    ERSTATUS.to_csv(f, header=False, index=True, sep='\t')
    PRSTATUS.to_csv(f, header=False, index=True, sep='\t')
    SRV_TIME_MON_FLAG.to_csv(f, header=False, index=True, sep='\t')
    BEHANAL.to_csv(f, header=False, index=True, sep='\t')
    ADJTM_6VALUE.to_csv(f, header=False, index=True, sep='\t')
    ADJNM_6VALUE.to_csv(f, header=False, index=True, sep='\t')
    ADJM_6VALUE.to_csv(f, header=False, index=True, sep='\t')
    ADJAJCCSTG.to_csv(f, header=False, index=True, sep='\t')

df = df6
df["AGE_DX6"] = df["REG"]/df["REG"]*6
df["AGE_DX6"] = df["AGE_DX6"].astype(int)
AGE_DX = df[['AGE_DX6','AGE_DX','AGE_DX6']]
df = df4
df["SURGPRIM12"] = df["REG"]/df["REG"]*12
df["SURGPRIM12"] = df["SURGPRIM12"].astype(int)
SURGPRIM = df4[['SURGPRIM12', 'SURGPRIM', 'SURGPRIM12']]

df = null_data
df["ADJTM_6VALUE24"] = df["REG"]/df["REG"]*24
df["ADJTM_6VALUE24"] = df["ADJTM_6VALUE24"].astype(int)
ADJTM_6VALUE = df[['ADJTM_6VALUE24','ADJTM_6VALUE','ADJTM_6VALUE24']]
df["ADJNM_6VALUE25"] = df["REG"]/df["REG"]*25
df["ADJNM_6VALUE25"] = df["ADJNM_6VALUE25"].astype(int)
ADJNM_6VALUE = df[['ADJNM_6VALUE25','ADJNM_6VALUE','ADJNM_6VALUE25']]
df["ADJM_6VALUE26"] = df["REG"]/df["REG"]*26
df["ADJM_6VALUE26"] = df["ADJM_6VALUE26"].astype(int)
ADJM_6VALUE = df[['ADJM_6VALUE26','ADJM_6VALUE','ADJM_6VALUE26']]
df["ADJAJCCSTG27"] = df["REG"]/df["REG"]*27
df["ADJAJCCSTG27"] = df["ADJAJCCSTG27"].astype(int)
ADJAJCCSTG = df[['ADJAJCCSTG27','ADJAJCCSTG','ADJAJCCSTG27']]
    
with open('null.txt', 'a') as f:
    AGE_DX.to_csv(f, header=False, index=True, sep='\t')
    SURGPRIM.to_csv(f, header=False, index=True, sep='\t')
    ADJTM_6VALUE.to_csv(f, header=False, index=True, sep='\t')
    ADJNM_6VALUE.to_csv(f, header=False, index=True, sep='\t')
    ADJM_6VALUE.to_csv(f, header=False, index=True, sep='\t')
    ADJAJCCSTG.to_csv(f, header=False, index=True, sep='\t')