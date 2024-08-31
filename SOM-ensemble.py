# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:41:24 2022

@author: qm7in
"""

### Merging surgery features 1973~1997 & 1998+
import pandas as pd
import numpy as np
import csv
with open ("EncodedB.csv") as f:
    reader = csv.reader(f)
    i = next(reader)

#print(i)

df = pd.read_csv("EncodedB.csv")
df.index = df.index + 1
#df = df.sample(5000)

###  Classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

names = ["MLP" ]
clf = MLPClassifier(activation='relu',hidden_layer_sizes=[10], random_state=9)

ds = df#.sample(50000)
Y = np.array(ds['Survival months'])
ds['Label'] = ds['Survival months']
ds['Label'] = np.where(ds['Survival months'] < 60, 0, ds['Label'])
ds['Label'] = np.where(ds['Survival months'] >= 60, 1, ds['Label'])
Y = np.array(ds['Label'])
Y = np.resize(Y, (len(df)))

dropList = ['Survival months', 'Label']
for item in dropList:
    ds.drop(item, axis=1, inplace=True)
data = ds

scaling = preprocessing.MinMaxScaler()
X = scaling.fit_transform(ds)
#X = ds
#print ("X.shape:", X.shape)
result=[]
import matplotlib.pyplot as plt
from myminisom import MiniSom
import pickle
som_shape = [#(4,5), (6,8), (8,10), (10,13), (12,16), (14,18), (16,21), (18,23), (20, 26), (22,29), (24,32), (26,35), (28,37), (30,40), (32,43), (34,45), (36,48), (38,51), (40,53), (42,56), (44,59), (54,72), (66,88)
             (78,104)]
#som_shape = [(3,4), (5,7), (7,9), (9,12), (11,14), (13,17), (15,20), (17,22), (19, 25)]
som_data = X
for shape in som_shape:
    lst=[shape]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.4, random_state=1)
    
    
    # iterate over classifiers
    
    
    #
    clf.fit(X_train, Y_train)
    score_train = clf.score(X_train, Y_train)
    #      print(name,score_train)
    predicted_train = clf.predict(X_train)
    matrix = confusion_matrix(Y_train, predicted_train)
    class_names = ["0", "1"]
    Confusion = pd.DataFrame(matrix, index=class_names, columns=class_names)
    #      print("Train Report:")
    classification_train = classification_report(Y_train, predicted_train, labels= [0, 1])
    #      print(Confusion)
    #      print(classification_train)
    #for number in (2, 4, 6, 8, 10):
    for number in (1, 3, 5, 7, 9):
    #
    
        som = MiniSom(shape[0], shape[1], som_data.shape[1], sigma=shape[0]/2, learning_rate=.5, activation_distance='euclidean',
        neighborhood_function='gaussian', topology='hexagonal', random_seed=number)
        som.pca_weights_init(som_data)
        som.train_random(som_data, 700000, verbose=False)
        # with open('somsample.p', 'wb') as outfile:
        #       pickle.dump(som, outfile)
        # with open('somsample.p', 'rb') as infile:
        #       som = pickle.load(infile)
        winner_coordinates = np.array([som.winner(x) for x in X]).T
        #print('cluster',winner_coordinates)
        row0 = pd.DataFrame(winner_coordinates[0])
        col0 = pd.DataFrame(winner_coordinates[1])
        predicted = clf.predict(X)
          
    #
        Y_test = pd.DataFrame(Y)
        Y_predict = pd.DataFrame(predicted)
        X_t = pd.DataFrame(X)
        X_t = pd.concat([X_t, row0], axis=1)
        X_t = pd.concat([X_t, col0], axis=1)
        X_t = pd.concat([X_t, Y_test], axis=1)
        X_t = pd.concat([X_t, Y_predict], axis=1)
        X_t.columns = ['SEER registry_0',
                        'SEER registry_1',
                        'SEER registry_2',
                        'SEER registry_3',
                        'SEER registry_4',
                        'Marital status at diagnosis_0', 
                        'Marital status at diagnosis_1',
                        'Marital status at diagnosis_2',
                        'Race/ethnicity_0',
                        'Race/ethnicity_1',
                        'Race/ethnicity_2',
                        'Race/ethnicity_3',
                        'Race/ethnicity_4', 
                        'Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)_0',
                        'Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)_1',
                        'Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)_2', 
                        'Sex', 
                        'Age at diagnosis', 
                        'Year of birth', 
                        'Sequence number', 
                        'Year of diagnosis', 
                        'Primary Site_0',
                        'Primary Site_1', 
                        'Primary Site_2', 
                        'Primary Site_3', 
                        'Laterality_0', 
                        'Laterality_1', 
                        'Laterality_2', 
                        'Reason no cancer-directed surgery_0', 
                        'Reason no cancer-directed surgery_1', 
                        'Reason no cancer-directed surgery_2', 
                        'Reason no cancer-directed surgery_3', 
                        'Histology recode - broad groupings_0', 
                        'Histology recode - broad groupings_1', 
                        'Histology recode - broad groupings_2', 
                        'Histology recode - broad groupings_3', 
                        'Histology recode - broad groupings_4', 
                        'surg combine_0', 
                        'surg combine_1', 
                        'surg combine_2',
                        'surg combine_3', 
                        'stage combine_0', 
                        'stage combine_1', 
                        'stage combine_2', 
                        'stage combine_3', 
                        'ER Status Recode Breast Cancer (1990+)_0', 
                        'ER Status Recode Breast Cancer (1990+)_1', 
                        'PR Status Recode Breast Cancer (1990+)_0', 
                        'PR Status Recode Breast Cancer (1990+)_1', 
                        'Row', 
                        'Col', 
                        'Class', 
                        'Y_predict']
        #print(X_t.size)
        #X_t.to_csv("X_tNBsample.csv",index=True)
    
    
    
        col = X_t['Col']
        row = X_t['Row']
    
    
    
        Xa = pd.read_csv("XpDT-I.csv")
        Xa = Xa.loc[(Xa['dataset'] == 'valid')]
        X_ta = Xa[["Class", "Y_predict"]]
        X_ta = pd.concat([X_ta, col], axis=1)
        X_ta = pd.concat([X_ta, row], axis=1)
        X_ta = pd.DataFrame.dropna(X_ta)
        
        ma = X_ta[["Row", "Col", "Class", "Y_predict"]]
        
        A = np.zeros((shape[1],shape[0]),dtype=float)
        #fig, ax = plt.subplots(dpi=300)
        for c in range(0, shape[1]):
            X_c= (ma.loc[ma["Col"] == c])
            for r in range(0, shape[0]):
                X_r= (X_c.loc[X_c["Row"] == r])
                total = len(X_r)
                c0 = (X_r.Y_predict == X_r.Class).sum()
                c1 = total - c0
                if total==0:
                    continue
                else:
                    a = c0/total
        
                A[c,r] = a
        A=pd.DataFrame(A)
        #print ('DT')
        #print (A)
        A.to_csv("AccuracyDT.csv",index=True)
        # plt.colorbar()
        # plt.title('Incorrect (NB)')
        # plt.show()
        
        Xb = pd.read_csv("XpNB-I.csv")
        Xb = Xb.loc[(Xb['dataset'] == 'valid')]
        X_tb = Xb[["Class", "Y_predict"]]
        X_tb = pd.concat([X_tb, col], axis=1)
        X_tb = pd.concat([X_tb, row], axis=1)
        X_tb = pd.DataFrame.dropna(X_tb)
        mb = X_tb[["Row", "Col", "Class", "Y_predict"]]
        
        B = np.zeros((shape[1],shape[0]),dtype=float)
        #fig, ax = plt.subplots(dpi=300)
        for c in range(0, shape[1]):
            X_c= (mb.loc[mb["Col"] == c])
            for r in range(0, shape[0]):
                X_r= (X_c.loc[X_c["Row"] == r])
                total = len(X_r)
                c0 = (X_r.Y_predict == X_r.Class).sum()
                c1 = total - c0
                if total==0:
                    continue
                else:
                    a = c0/total
                B[c,r] = a
        B=pd.DataFrame(B)
        #print ('NB')
        #print (B)
        B.to_csv("AccuracyNB.csv",index=True)
        
        Xc = pd.read_csv("XpNN-I.csv")
        Xc = Xc.loc[(Xc['dataset'] == 'valid')]
        X_tc = Xc[["Class", "Y_predict"]]
        X_tc = pd.concat([X_tc, col], axis=1)
        X_tc = pd.concat([X_tc, row], axis=1)
        X_tc = pd.DataFrame.dropna(X_tc)
        mc = X_tc[["Row", "Col", "Class", "Y_predict"]]
        
        C = np.zeros((shape[1],shape[0]),dtype=float)
        #fig, ax = plt.subplots(dpi=300)
        for c in range(0, shape[1]):
            X_c= (mc.loc[mc["Col"] == c])
            for r in range(0, shape[0]):
                X_r= (X_c.loc[X_c["Row"] == r])
                total = len(X_r)
                c0 = (X_r.Y_predict == X_r.Class).sum()
                c1 = total - c0
                if total==0:
                    continue
                else:
                    a = c0/total
                C[c,r] = a
        C=pd.DataFrame(C)
        #print ('NN',C)
        C.to_csv("AccuracyNN.csv",index=True)
        
        Xd = pd.read_csv("XpRF-I.csv")
        Xd = Xd.loc[(Xd['dataset'] == 'valid')]
        X_td = Xd[["Class", "Y_predict"]]
        X_td = pd.concat([X_td, col], axis=1)
        X_td = pd.concat([X_td, row], axis=1)
        X_td = pd.DataFrame.dropna(X_td)
        md = X_td[["Row", "Col", "Class", "Y_predict"]]
        
        D = np.zeros((shape[1],shape[0]),dtype=float)
        #fig, ax = plt.subplots(dpi=300)
        for c in range(0, shape[1]):
            X_c= (md.loc[md["Col"] == c])
            for r in range(0, shape[0]):
                X_r= (X_c.loc[X_c["Row"] == r])
                total = len(X_r)
                c0 = (X_r.Y_predict == X_r.Class).sum()
                c1 = total - c0
                if total==0:
                    continue
                else:
                    a = c0/total
                D[c,r] = a
        D=pd.DataFrame(D)
        #print ('RF',D)
        D.to_csv("AccuracyRF.csv",index=True)
        
        Xe = pd.read_csv("XpAB-I.csv")
        Xe = Xe.loc[(Xe['dataset'] == 'valid')]
        X_te = Xe[["Class", "Y_predict"]]
        X_te = pd.concat([X_te, col], axis=1)
        X_te = pd.concat([X_te, row], axis=1)
        X_te = pd.DataFrame.dropna(X_te)
        me = X_te[["Row", "Col", "Class", "Y_predict"]]
        
        E = np.zeros((shape[1],shape[0]),dtype=float)
        #fig, ax = plt.subplots(dpi=300)
        for c in range(0, shape[1]):
            X_c= (me.loc[me["Col"] == c])
            for r in range(0, shape[0]):
                X_r= (X_c.loc[X_c["Row"] == r])
                total = len(X_r)
                c0 = (X_r.Y_predict == X_r.Class).sum()
                c1 = total - c0
                if total==0:
                    continue
                else:
                    a = c0/total
                E[c,r] = a
        E=pd.DataFrame(E)
        #print ('AB')
        #print (E)
        E.to_csv("AccuracyAB.csv",index=True)
        
        Xf = pd.read_csv("XpLR-I.csv")
        Xf = Xf.loc[(Xf['dataset'] == 'valid')]
        X_tf = Xf[["Class", "Y_predict"]]
        X_tf = pd.concat([X_tf, col], axis=1)
        X_tf = pd.concat([X_tf, row], axis=1)
        X_tf = pd.DataFrame.dropna(X_tf)
        mf = X_tf[["Row", "Col", "Class", "Y_predict"]]
        
        F = np.zeros((shape[1],shape[0]),dtype=float)
        for c in range(0, shape[1]):
            X_c= (mf.loc[mf["Col"] == c])
            for r in range(0, shape[0]):
                X_r= (X_c.loc[X_c["Row"] == r])
                total = len(X_r)
                c0 = (X_r.Y_predict == X_r.Class).sum()
                c1 = total - c0
                if total==0:
                    continue
                else:
                    a = c0/total
                F[c,r] = a
        F=pd.DataFrame(F)
        #print ('LR',F)
        F.to_csv("AccuracyLR.csv",index=True)
        
        Xg = pd.read_csv("XpSVM-I.csv")
        Xg = Xg.loc[(Xg['dataset'] == 'valid')]
        X_tg = Xg[["Class", "Y_predict"]]
        X_tg = pd.concat([X_tg, col], axis=1)
        X_tg = pd.concat([X_tg, row], axis=1)
        X_tg = pd.DataFrame.dropna(X_tg)
        mg = X_tg[["Row", "Col", "Class", "Y_predict"]]
        
        G = np.zeros((shape[1],shape[0]),dtype=float)
        for c in range(0, shape[1]):
            X_c= (mf.loc[mf["Col"] == c])
            for r in range(0, shape[0]):
                X_r= (X_c.loc[X_c["Row"] == r])
                total = len(X_r)
                c0 = (X_r.Y_predict == X_r.Class).sum()
                c1 = total - c0
                if total==0:
                    continue
                else:
                    a = c0/total
                G[c,r] = a
        G=pd.DataFrame(G)
        #print ('SVM',G)
        G.to_csv("AccuracySVM.csv",index=True)
        
        K = np.zeros((shape[1],shape[0]),dtype=object)
        #fig, ax = plt.subplots(dpi=300)
        for c in range(0, shape[1]):
            X_c= (ma.loc[ma["Col"] == c])
            X_b= (mb.loc[mb["Col"] == c])
            X_d= (mc.loc[mc["Col"] == c])
            X_e= (md.loc[md["Col"] == c])
            X_f= (me.loc[me["Col"] == c])
            X_g= (mf.loc[mf["Col"] == c])
            X_h= (mg.loc[mg["Col"] == c])
            for r in range(0, shape[0]):
                X_s= (X_c.loc[X_c["Row"] == r])
                X_r= (X_b.loc[X_b["Row"] == r])
                X_q= (X_d.loc[X_d["Row"] == r])
                X_p= (X_e.loc[X_e["Row"] == r])
                X_o= (X_f.loc[X_f["Row"] == r])
                X_n= (X_g.loc[X_g["Row"] == r])
                X_m= (X_h.loc[X_h["Row"] == r])
                total = len(X_s)
        # #
        
                c0 = (X_s.Y_predict == X_s.Class).sum()
                c1 = (X_r.Y_predict == X_r.Class).sum()
                c2 = (X_q.Y_predict == X_q.Class).sum()
                c3 = (X_p.Y_predict == X_p.Class).sum()
                c4 = (X_o.Y_predict == X_o.Class).sum()
                c5 = (X_n.Y_predict == X_n.Class).sum()
                c6 = (X_m.Y_predict == X_m.Class).sum()
                max_value = max(c0,
                                c1,
                                c2,
                                c3,
                                c4,
                                c5,
                                c6
                                )
                if max_value==0:
                    continue
                elif (max_value == c3):
                    d = c3/total
                    d = 'Random Forest'
                    #ax.scatter(c, r, s=5, color='Green', label='RF')
                elif (max_value == c4):
                    d = c4/total
                    d = 'AdaBoost'
                    #ax.scatter(c, r, s=5, color='Black', label='AB')
                elif (max_value == c0):
                    d = c0/total
                    d = 'Decision Tree'
                    #ax.scatter(c, r, s=5, color='Blue', label='DT')
                elif (max_value == c2):
                    d = c2/total
                    d = 'Neural Net'
                    #ax.scatter(c, r, s=5, color='Yellow', label='NN')
                elif (max_value == c5):
                    d = c5/total
                    d = 'Logistic Regression'
                    #ax.scatter(c, r, s=5, color='Purple', label='LR')
                elif (max_value == c6):
                    d = c6/total
                    d = 'SVM'
                    #ax.scatter(c, r, s=5, color='Orange', label='SVM')
                else:
                    d = c1/total
                    d = 'Naive Bayes'
                    #ax.scatter(c, r, s=5, color='Red', label='NB')
                K[c,r] = d
        K =pd.DataFrame(K)
        #print (K)
        
    #    plt.title('Ensemble')
    #    plt.show()
        
        
        dt = pd.read_csv("/Users/qm7in/.spyder-py3/XpDT-I.csv")
        dt = dt.loc[(dt['dataset'] == 'valid')]
        nb = pd.read_csv("/Users/qm7in/.spyder-py3/XpNB-I.csv")
        nb = nb.loc[(nb['dataset'] == 'valid')]
        nn = pd.read_csv("/Users/qm7in/.spyder-py3/XpNN-I.csv")
        nn = nn.loc[(nn['dataset'] == 'valid')]
        rf = pd.read_csv("/Users/qm7in/.spyder-py3/XpRF-I.csv")
        rf = rf.loc[(rf['dataset'] == 'valid')]
        ab = pd.read_csv("/Users/qm7in/.spyder-py3/XpAB-I.csv")
        ab = ab.loc[(ab['dataset'] == 'valid')]
        lr = pd.read_csv("/Users/qm7in/.spyder-py3/XpLR-I.csv")
        lr = lr.loc[(lr['dataset'] == 'valid')]
        svm = pd.read_csv("/Users/qm7in/.spyder-py3/XpSVM-I.csv")
        svm = svm.loc[(svm['dataset'] == 'valid')]
        
        col = X_t['Col']
        row = X_t['Row']
        
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
        #print(df)
        
        A=np.zeros((shape[1],shape[0]),dtype=object)
        B=np.zeros((shape[1],shape[0]),dtype=object)
        K=np.array(K)
        for c in range(0, shape[0]):
            for r in range(0, shape[1]):
                if K[r,c]==0:
                    continue
                elif K[r,c]=='Decision Tree':
                    A[r,c]=1
                elif K[r,c]=='Naive Bayes':
                    B[r,c]=1
        A=pd.DataFrame(A)
        B=pd.DataFrame(B)
        
        
        df['ensemble'] = 0
        m = df["Col"]
        n = df["Row"]
        
        for roww in range(K.shape[1]):
              for coll in range(K.shape[0]):
                df.loc[((df['Col'] == coll) & (df['Row'] == roww)), 'ensemble'] = K[coll, roww]
        
        
        df.drop(df.loc[df['ensemble']==0].index, inplace=True)
        #print (df)
        aa = pd.value_counts(df.ensemble)
        #print (aa)
        
        
        df.loc[(df['ensemble'] == 'Decision Tree'), 'ensemble'] = df['dt']
        df.loc[(df['ensemble'] == 'Naive Bayes'), 'ensemble'] = df['nb']
        df.loc[(df['ensemble'] == 'Neural Net'), 'ensemble'] = df['nn']
        df.loc[(df['ensemble'] == 'Random Forest'), 'ensemble'] = df['rf']
        df.loc[(df['ensemble'] == 'AdaBoost'), 'ensemble'] = df['ab']
        df.loc[(df['ensemble'] == 'Logistic Regression'), 'ensemble'] = df['lr']
        df.loc[(df['ensemble'] == 'SVM'), 'ensemble'] = df['svm']
        
        
        acc = (df.ensemble == df.Class).sum()
        #print (df)
        #print ('Overall accuracy')
        print ('accurate', acc)
        print ('total', len(df))
        accuracy = acc/len(df)
        lst.append(accuracy)
        print (lst)
    result.append(lst)
    print (result)
