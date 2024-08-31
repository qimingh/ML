import pandas as pd
import numpy as np


som_shape = (88, 66)

SOM = pd.read_csv("/Users/qm7in/.spyder-py3/x_tNBsample.csv")
dt = pd.read_csv("/Users/qm7in/.spyder-py3/XpDT-I.csv")
dt = dt.loc[(dt['dataset'] == 'test')]
nb = pd.read_csv("/Users/qm7in/.spyder-py3/XpNB-I.csv")
nb = nb.loc[(nb['dataset'] == 'test')]
nn = pd.read_csv("/Users/qm7in/.spyder-py3/XpNN-I.csv")
nn = nn.loc[(nn['dataset'] == 'test')]
rf = pd.read_csv("/Users/qm7in/.spyder-py3/XpRF-I.csv")
rf = rf.loc[(rf['dataset'] == 'test')]
ab = pd.read_csv("/Users/qm7in/.spyder-py3/XpAB-I.csv")
ab = ab.loc[(ab['dataset'] == 'test')]
lr = pd.read_csv("/Users/qm7in/.spyder-py3/XpLR-I.csv")
lr = lr.loc[(lr['dataset'] == 'test')]
svm = pd.read_csv("/Users/qm7in/.spyder-py3/XpSVM-I.csv")
svm = svm.loc[(svm['dataset'] == 'test')]
ensemble = pd.read_csv("/Users/qm7in/.spyder-py3/Ensemble.csv").iloc[: , 1:]
col = SOM['Col']
row = SOM['Row']

a = dt[["Class", "Y_predict"]]
a = pd.concat([row, a], axis=1)
a = pd.concat([col, a], axis=1)
a.columns=["Col", "Row", "Class", "dt"]
b = nb[["Y_predict"]]
c = nn[["Y_predict"]]
d = rf[["Y_predict"]]
e = ab[["Y_predict"]]
f = lr[["Y_predict"]]
g = svm[["Y_predict"]]
h = pd.concat([a, b], axis=1)
h = pd.concat([h, c], axis=1)
h = pd.concat([h, d], axis=1)
h = pd.concat([h, e], axis=1)
h = pd.concat([h, f], axis=1)
h = pd.concat([h, g], axis=1)
df = h
df = pd.DataFrame.dropna(df)
df.columns=["Col", "Row", "Class", "dt", "nb", "nn", "rf", "ab", "lr","svm"]
df = df#.head(43420)
print(df)

A=np.zeros((som_shape[0],som_shape[1]),dtype=object)
B=np.zeros((som_shape[0],som_shape[1]),dtype=object)
ensemble=np.array(ensemble)
for c in range(0, som_shape[1]):
    for r in range(0, som_shape[0]):
        if ensemble[r,c]==0:
            continue
        elif ensemble[r,c]=='Decision Tree':
            A[r,c]=1
        elif ensemble[r,c]=='Naive Bayes':
            B[r,c]=1
A=pd.DataFrame(A)
B=pd.DataFrame(B)


df['ensemble'] = 0
m = df["Col"]
n = df["Row"]

for row in range(ensemble.shape[1]):
      for col in range(ensemble.shape[0]):
        df.loc[((df['Col'] == col) & (df['Row'] == row)), 'ensemble'] = ensemble[col, row]
# idx=0
# for idx in range(df.shape[0]):
#     row,col=df.loc[idx, ['Row','Column']]
#     df.loc[idx, 'ensemble'] = ensemble[col,row]
print (df)
aa = pd.value_counts(df.ensemble)
print (aa)
df.drop(df.loc[df['ensemble']=='0'].index, inplace=True)
df.loc[(df['ensemble'] == 'Decision Tree'), 'ensemble'] = df['dt']
df.loc[(df['ensemble'] == 'Naive Bayes'), 'ensemble'] = df['nb']
df.loc[(df['ensemble'] == 'Neural Net'), 'ensemble'] = df['nn']
df.loc[(df['ensemble'] == 'Random Forest'), 'ensemble'] = df['rf']
df.loc[(df['ensemble'] == 'AdaBoost'), 'ensemble'] = df['ab']
df.loc[(df['ensemble'] == 'Logistic Regression'), 'ensemble'] = df['lr']
df.loc[(df['ensemble'] == 'SVM'), 'ensemble'] = df['svm']


acc = (df.ensemble == df.Class).sum()
print (df)
print ('Overall accuracy')
print ('accurate', acc)
print ('total', len(df))
print (acc/len(df))
