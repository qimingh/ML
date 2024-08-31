import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


som_shape = (88, 66)

SOM = pd.read_csv("/Users/qm7in/.spyder-py3/x_tNBsample.csv")
col = SOM['Col']
row = SOM['Row']
dt = pd.read_csv("XpDT-I.csv")
nb = pd.read_csv("XpNB-I.csv")
nn = pd.read_csv("XpNN-I.csv")
rf = pd.read_csv("XpRF-I.csv")
ab = pd.read_csv("XpAB-I.csv")
lr = pd.read_csv("XpLR-I.csv")
svm = pd.read_csv("XpSVM-I.csv")
ensemble = pd.read_csv("Ensemble.csv").iloc[: , 1:]
data = pd.read_csv("EncodedB.csv")


a = dt[["dataset","Class", "Y_predict"]]
a.columns=["dataset","Class", "dt"]
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
h = pd.concat([h, col], axis=1)
h = pd.concat([h, row], axis=1)
df = h
df.columns=["dataset","Class", "dt", 
            "nb", 
            "nn", 
            "rf", 
            "ab", 
            "lr",
            "svm",
            "Col", "Row"
            ]
df = pd.DataFrame.dropna(df)
df = df.loc[(df['dataset'] == 'valid')]

ensemble=np.array(ensemble)

df['ensemble'] = 0

for row in range(ensemble.shape[1]):
      for col in range(ensemble.shape[0]):
        df.loc[((df['Col'] == col) & (df['Row'] == row)), 'ensemble'] = ensemble[col, row]
# idx=0
# for idx in range(df.shape[0]):
#     row,col=df.loc[idx, ['Row','Column']]
#     df.loc[idx, 'ensemble'] = ensemble[col,row]
df = df.loc[~(df['ensemble']==0)]
# print ('df', df)
aa = pd.value_counts(df.ensemble)
# print ('aa', aa)


dd = pd.concat([data, df], axis=1, sort=False)
dd=pd.DataFrame.dropna(dd)
dd.drop(['dataset'], 1, inplace=True)
# print (dd)


X_dt = dd[(dd['ensemble'] == 'Decision Tree')]
X_nb = dd[(dd['ensemble'] == 'Naive Bayes')]
X_nn = dd[(dd['ensemble'] == 'Neural Net')]
X_rf = dd[(dd['ensemble'] == 'Random Forest')]
X_ab = dd[(dd['ensemble'] == 'AdaBoost')]
X_lr = dd[(dd['ensemble'] == 'Logistic Regression')]
X_svm = dd[(dd['ensemble'] == 'SVM')]
#print(X_nb)
X_611 = dd[(dd['Col'] < 2) & (dd['Row'] <3)]
X_587 = dd[(dd['Col'] == 7) & (dd['Row'] == 58)]
X_577 = dd[(dd['Col'] == 7) & (dd['Row'] == 57)]
X_597 = dd[(dd['Col'] == 7) & (dd['Row'] == 59)]
X_576 = dd[(dd['Col'] == 6) & (dd['Row'] == 57)]
X_586 = dd[(dd['Col'] == 6) & (dd['Row'] == 58)]
X_g1 = dd[(dd['Col'] < 9) & (dd['Col'] > 6) & (dd['Row'] < 59) & (dd['Row'] > 56)]
X_597 = dd[(dd['Col'] == 7) & (dd['Row'] == 59)]
X_g1 = pd.concat([X_g1, X_597])
X_4987 = dd[(dd['Col'] == 87) & (dd['Row'] == 49)]
X_5087 = dd[(dd['Col'] == 87) & (dd['Row'] == 50)]
X_5187 = dd[(dd['Col'] == 87) & (dd['Row'] == 51)]
X_4887 = dd[(dd['Col'] == 87) & (dd['Row'] == 48)]
X_4886 = dd[(dd['Col'] == 86) & (dd['Row'] == 48)]
X_4986 = dd[(dd['Col'] == 86) & (dd['Row'] == 49)]
X_g2 = dd[(dd['Col'] == 87) & (dd['Row'] < 52) & (dd['Row'] > 48)]
X_00 = dd[(dd['Col'] == 0) & (dd['Row'] == 0)]
X_873 = dd[(dd['Col'] == 73) & (dd['Row'] == 8)]
X_872 = dd[(dd['Col'] == 72) & (dd['Row'] < 9 ) & (dd['Row'] > 6 )]
X_874 = dd[(dd['Col'] == 74) & (dd['Row'] == 8)]
X_773 = dd[(dd['Col'] == 73) & (dd['Row'] == 7)]
X_360 = dd[(dd['Col'] == 60) & (dd['Row'] == 3)]
X_361 = dd[(dd['Col'] == 61) & (dd['Row'] == 3)]
X_460 = dd[(dd['Col'] == 60) & (dd['Row'] == 4)]
X_g3 = pd.concat([X_360, X_460])
X_g4 = pd.concat([X_873, X_872])
X_190 = dd[(dd['Col'] == 0) & (dd['Row'] == 19)]
X_191 = dd[(dd['Col'] == 1) & (dd['Row'] == 19)]
X_181 = dd[(dd['Col'] == 1) & (dd['Row'] == 18)]
X_201 = dd[(dd['Col'] == 1) & (dd['Row'] == 20)]
X_g5 = dd[(dd['Col'] < 2) & (dd['Row'] < 21 ) & (dd['Row'] > 18 )]

X_age35 = dd[(dd['Age at diagnosis'] < 35)]
X_age45 = dd[(dd['Age at diagnosis'] > 34) &(dd['Age at diagnosis'] < 45)]
X_age55 = dd[(dd['Age at diagnosis'] > 44) &(dd['Age at diagnosis'] < 55)]
X_age65 = dd[(dd['Age at diagnosis'] > 54) &(dd['Age at diagnosis'] < 65)]
X_age65p = dd[(dd['Age at diagnosis'] > 64)]
# aa = pd.value_counts(X_age35['Survival months'])
# aa.to_csv("age35.csv",index=True)

X_06 = dd[(dd['Col'] != 2)]
X_16 = dd[(dd['Col'] == 2) & (dd['Row'] != 1)]
X_0 = X_06.append(X_16)
X_1dt = dd[(dd['ensemble'] != 'Decision Tree')]
X_1nb = dd[(dd['ensemble'] != 'Naive Bayes')]
X_1nn = dd[(dd['ensemble'] != 'Neural Net')]
X_1rf = dd[(dd['ensemble'] != 'Random Forest')]
X_1ab = dd[(dd['ensemble'] != 'AdaBoost')]
X_1lr = dd[(dd['ensemble'] != 'Logistic Regression')]
X_1svm = dd[(dd['ensemble'] != 'SVM')]
from scipy.spatial.distance import euclidean
# print ("g4", X_g4)

dd.drop(['nn','nb','rf','ab','lr','svm','Col','Row','ensemble'], 1, inplace=True)
X_cg4 = X_873.loc[X_873['dt'] == X_873['Class']]
X_ig4 = X_873.loc[X_873['dt'] != X_873['Class']]
print ("correct")
print (X_cg4)
print ("incorrect")
print (X_ig4)
for i in dd.columns:
    aa1 = pd.value_counts(X_cg4[i])
    aa = np.average(aa1.index, weights=aa1)
    print (i)
    print ("Correct",aa)
    ab1 = pd.value_counts(X_ig4[i])
    ab = np.average(ab1.index, weights=ab1)
    print ("Incorrect",ab)
    # dist = euclidean(aa, ab)
    # print("dist",dist)
    results, edges = np.histogram(X_cg4[i], bins=20, normed=True)
    binWidth = edges[1] - edges[0]
    results1, edges1 = np.histogram(X_ig4[i], bins=20, normed=True)
    binWidth1 = edges1[1] - edges1[0]
    # results2, edges2 = np.histogram(X_717[i], bins=20, normed=True)
    # binWidth2 = edges2[1] - edges2[0]
    plt.bar(edges[:-1], results*binWidth, binWidth, alpha=0.5, label='Correct')
    plt.bar(edges1[:-1], results1*binWidth1, binWidth1, alpha=0.5, label='Incorrect')
    # plt.bar(edges2[:-1], results2*binWidth2, binWidth2, alpha=0.5, label='(7,17)')
    plt.ylabel('Probability')
    plt.xlabel(i);
    plt.legend(loc='upper right')
    plt.show()
    # h=stats.mannwhitneyu(X_age35[i], dd[i])
    # print (i, h)