

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
SOM = pd.read_csv("/Users/qm7in/.spyder-py3/x_tNBsample.csv")
col = SOM['Col']
row = SOM['Row']

som_shape = (88, 66)

X = pd.read_csv("XpDT-I.csv")
X = X.loc[(X['dataset'] == 'valid')]
X_t = X[["Class", "Y_predict"]]
X_t = pd.concat([X_t, col], axis=1)
X_t = pd.concat([X_t, row], axis=1)
X_t = pd.DataFrame.dropna(X_t)
# X_0= X_t[(X_t['Class'] == 0)]
# X_00= X_t[(X_t['Class'] == 0) & (X_t['Y_predict'] == 0)]
# X_01= X_t[(X_t['Class'] == 0) & (X_t['Y_predict'] == 1)]
# X_1= X_t[(X_t['Class'] == 1)]
# X_10= X_t[(X_t['Class'] == 1) & (X_t['Y_predict'] == 1)]     
# X_11= X_t[(X_t['Class'] == 1) & (X_t['Y_predict'] == 0)]
# x0 = np.array(X_0["Row"])
# y0 = np.array(X_0["Col"])
# x00 = np.array(X_00["Row"])
# y00 = np.array(X_00["Col"])
# x01 = np.array(X_01["Row"])
# y01 = np.array(X_01["Col"])
# x1 = np.array(X_1["Row"])
# y1 = np.array(X_1["Col"])
# x10 = np.array(X_10["Row"])
# y10 = np.array(X_10["Col"])
# x11 = np.array(X_11["Row"])
# y11 = np.array(X_11["Col"])

# X_01 =pd.DataFrame(X_01)
# X_11 =pd.DataFrame(X_11)
# X_w = pd.concat(X_01, X_11)
# X_w = X_w.to_np
# print (X_01)
ma = X_t[["Row", "Col", "Class", "Y_predict"]]

A = np.zeros((som_shape[0],som_shape[1]),dtype=float)
#fig, ax = plt.subplots(dpi=300)
for c in range(0, som_shape[0]):
    X_c= (ma.loc[ma["Col"] == c])
    for r in range(0, som_shape[1]):
        X_r= (X_c.loc[X_c["Row"] == r])
        total = len(X_r)
        c0 = (X_r.Y_predict == X_r.Class).sum()
        c1 = total - c0
        if total==0:
            continue
        # elif (c0<c1):
        #     continue
        # elif (c0>c1):
        #    a = c0/total
        else:
            a = c0/total
        #    continue
        #plt.scatter(c, r, s=2, c=a, cmap='Greys', vmin=0, vmax=1)
        A[c,r] = a
A=pd.DataFrame(A)
print ('DT')
print (A)
#A.to_csv("AccuracyDT.csv",index=True)
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

B = np.zeros((som_shape[0],som_shape[1]),dtype=float)
#fig, ax = plt.subplots(dpi=300)
for c in range(0, som_shape[0]):
    X_c= (mb.loc[mb["Col"] == c])
    for r in range(0, som_shape[1]):
        X_r= (X_c.loc[X_c["Row"] == r])
        total = len(X_r)
        c0 = (X_r.Y_predict == X_r.Class).sum()
        c1 = total - c0
        if total==0:
            continue
        # elif (c0<c1):
        #     continue
        # elif (c0>c1):
        #    a = c0/total
        else:
            a = c0/total
        #    continue
        #plt.scatter(c, r, s=2, c=a, cmap='Greys', vmin=0, vmax=1)
        B[c,r] = a
B=pd.DataFrame(B)
print ('NB')
print (B)
#B.to_csv("AccuracyNB.csv",index=True)

Xc = pd.read_csv("XpNN-I.csv")
Xc = Xc.loc[(Xc['dataset'] == 'valid')]
X_tc = Xc[["Class", "Y_predict"]]
X_tc = pd.concat([X_tc, col], axis=1)
X_tc = pd.concat([X_tc, row], axis=1)
X_tc = pd.DataFrame.dropna(X_tc)
mc = X_tc[["Row", "Col", "Class", "Y_predict"]]

C = np.zeros((som_shape[0],som_shape[1]),dtype=float)
#fig, ax = plt.subplots(dpi=300)
for c in range(0, som_shape[0]):
    X_c= (mc.loc[mc["Col"] == c])
    for r in range(0, som_shape[1]):
        X_r= (X_c.loc[X_c["Row"] == r])
        total = len(X_r)
        c0 = (X_r.Y_predict == X_r.Class).sum()
        c1 = total - c0
        if total==0:
            continue
        # elif (c0<c1):
        #     continue
        # elif (c0>c1):
        #    a = c0/total
        else:
            a = c0/total
        #    continue
        #plt.scatter(c, r, s=2, c=a, cmap='Greys', vmin=0, vmax=1)
        C[c,r] = a
C=pd.DataFrame(C)
print ('NN',C)
#C.to_csv("AccuracyNN.csv",index=True)

Xd = pd.read_csv("XpRF-I.csv")
Xd = Xd.loc[(Xd['dataset'] == 'valid')]
X_td = Xd[["Class", "Y_predict"]]
X_td = pd.concat([X_td, col], axis=1)
X_td = pd.concat([X_td, row], axis=1)
X_td = pd.DataFrame.dropna(X_td)
md = X_td[["Row", "Col", "Class", "Y_predict"]]

D = np.zeros((som_shape[0],som_shape[1]),dtype=float)
#fig, ax = plt.subplots(dpi=300)
for c in range(0, som_shape[0]):
    X_c= (md.loc[md["Col"] == c])
    for r in range(0, som_shape[1]):
        X_r= (X_c.loc[X_c["Row"] == r])
        total = len(X_r)
        c0 = (X_r.Y_predict == X_r.Class).sum()
        c1 = total - c0
        if total==0:
            continue
        # elif (c0<c1):
        #     continue
        # elif (c0>c1):
        #    a = c0/total
        else:
            a = c0/total
        #    continue
        #plt.scatter(c, r, s=2, c=a, cmap='Greys', vmin=0, vmax=1)
        D[c,r] = a
D=pd.DataFrame(D)
print ('RF',D)
#D.to_csv("AccuracyRF.csv",index=True)

Xe = pd.read_csv("XpAB-I.csv")
Xe = Xe.loc[(Xe['dataset'] == 'valid')]
X_te = Xe[["Class", "Y_predict"]]
X_te = pd.concat([X_te, col], axis=1)
X_te = pd.concat([X_te, row], axis=1)
X_te = pd.DataFrame.dropna(X_te)
me = X_te[["Row", "Col", "Class", "Y_predict"]]

E = np.zeros((som_shape[0],som_shape[1]),dtype=float)
#fig, ax = plt.subplots(dpi=300)
for c in range(0, som_shape[0]):
    X_c= (me.loc[me["Col"] == c])
    for r in range(0, som_shape[1]):
        X_r= (X_c.loc[X_c["Row"] == r])
        total = len(X_r)
        c0 = (X_r.Y_predict == X_r.Class).sum()
        c1 = total - c0
        if total==0:
            continue
        # elif (c0<c1):
        #     continue
        # elif (c0>c1):
        #    a = c0/total
        else:
            a = c0/total
        #    continue
        #plt.scatter(c, r, s=2, c=a, cmap='Greys', vmin=0, vmax=1)
        E[c,r] = a
E=pd.DataFrame(E)
print ('AB')
print (E)
#E.to_csv("AccuracyAB.csv",index=True)

Xf = pd.read_csv("XpLR-I.csv")
Xf = Xf.loc[(Xf['dataset'] == 'valid')]
X_tf = Xf[["Class", "Y_predict"]]
X_tf = pd.concat([X_tf, col], axis=1)
X_tf = pd.concat([X_tf, row], axis=1)
X_tf = pd.DataFrame.dropna(X_tf)
mf = X_tf[["Row", "Col", "Class", "Y_predict"]]

F = np.zeros((som_shape[0],som_shape[1]),dtype=float)
#fig, ax = plt.subplots(dpi=300)
for c in range(0, som_shape[0]):
    X_c= (mf.loc[mf["Col"] == c])
    for r in range(0, som_shape[1]):
        X_r= (X_c.loc[X_c["Row"] == r])
        total = len(X_r)
        c0 = (X_r.Y_predict == X_r.Class).sum()
        c1 = total - c0
        if total==0:
            continue
        # elif (c0<c1):
        #     continue
        # elif (c0>c1):
        #    a = c0/total
        else:
            a = c0/total
        #    continue
        #plt.scatter(c, r, s=2, c=a, cmap='Greys', vmin=0, vmax=1)
        F[c,r] = a
F=pd.DataFrame(F)
print ('LR',F)
#F.to_csv("AccuracyLR.csv",index=True)

Xg = pd.read_csv("XpSVM-I.csv")
Xg = Xg.loc[(Xg['dataset'] == 'valid')]
X_tg = Xg[["Class", "Y_predict"]]
X_tg = pd.concat([X_tg, col], axis=1)
X_tg = pd.concat([X_tg, row], axis=1)
X_tg = pd.DataFrame.dropna(X_tg)
mg = X_tg[["Row", "Col", "Class", "Y_predict"]]

G = np.zeros((som_shape[0],som_shape[1]),dtype=float)
fig, ax = plt.subplots(dpi=300)
for c in range(0, som_shape[0]):
    X_c= (mf.loc[mf["Col"] == c])
    for r in range(0, som_shape[1]):
        X_r= (X_c.loc[X_c["Row"] == r])
        total = len(X_r)
        c0 = (X_r.Y_predict == X_r.Class).sum()
        c1 = total - c0
        if total==0:
            continue
        # elif (c0<c1):
        #     continue
        # elif (c0>c1):
        #    a = c0/total
        else:
            a = c0/total
        #    continue
        #plt.scatter(c, r, s=2, c=a, cmap='Greys', vmin=0, vmax=1)
        G[c,r] = a
G=pd.DataFrame(G)
print ('SVM',G)
#G.to_csv("AccuracySVM.csv",index=True)

K = np.zeros((som_shape[0],som_shape[1]),dtype=object)
L = np.zeros((som_shape[0],som_shape[1]),dtype=object)
M = np.zeros((som_shape[0],som_shape[1]),dtype=object)
s = 0
for c in range(0, som_shape[0]):
    X_c= (ma.loc[ma["Col"] == c])
    X_b= (mb.loc[mb["Col"] == c])
    X_d= (mc.loc[mc["Col"] == c])
    X_e= (md.loc[md["Col"] == c])
    X_f= (me.loc[me["Col"] == c])
    X_g= (mf.loc[mf["Col"] == c])
    X_h= (mg.loc[mg["Col"] == c])
    for r in range(0, som_shape[1]):
        X_s= (X_c.loc[X_c["Row"] == r])
        X_r= (X_b.loc[X_b["Row"] == r])
        X_q= (X_d.loc[X_d["Row"] == r])
        X_p= (X_e.loc[X_e["Row"] == r])
        X_o= (X_f.loc[X_f["Row"] == r])
        X_n= (X_g.loc[X_g["Row"] == r])
        X_m= (X_h.loc[X_h["Row"] == r])
        total = len(X_s)
# #
#         c0 = (X_s.Class == 0).sum()
#         c1 = (X_s.Class == 1).sum()
#         if c0 == 0:
#             continue
#         elif c1 == 0:
#             continue
#         elif abs(c0-c1)/max(c0, c1) < 0.2:
#             d = abs(c0-c1)/max(c0, c1)
#             ax.scatter(c, r, c=d, cmap='Greys', vmin=0, vmax=0.2)
        c0 = (X_s.Y_predict == X_s.Class).sum()
        c1 = (X_r.Y_predict == X_r.Class).sum()
        c2 = (X_q.Y_predict == X_q.Class).sum()
        c3 = (X_p.Y_predict == X_p.Class).sum()
        c4 = (X_o.Y_predict == X_o.Class).sum()
        c5 = (X_n.Y_predict == X_n.Class).sum()
        c6 = (X_m.Y_predict == X_m.Class).sum()
        max_value = max(c0,c1,c2,c3,c4,c5,c6)
        if max_value==0:
            continue
        elif (max_value == c3):
            p = c3/total
            d = 'Random Forest'
            S = c3
            ax.scatter(r, c, s=5, color='Green', label='RF')
        elif (max_value == c4):
            p = c4/total
            d = 'AdaBoost'
            S = c4
            ax.scatter(r, c, s=5, color='Black', label='AB')
        elif (max_value == c0):
            p = c0/total
            d = 'Decision Tree'
            S = c0
            ax.scatter(r, c, s=5, color='Blue', label='DT')
        elif (max_value == c2):
            p = c2/total
            d = 'Neural Net'
            S = c2
            ax.scatter(r, c, s=5, color='Yellow', label='NN')
        elif (max_value == c5):
            p = c5/total
            d = 'Logistic Regression'
            S = c5
            ax.scatter(r, c, s=5, color='Purple', label='LR')
        elif (max_value == c6):
            p = c6/total
            d = 'SVM'
            S = c6
            ax.scatter(r, c, s=5, color='Orange', label='SVM')
        else:
            p = c1/total
            d = 'Naive Bayes'
            S = c1
            ax.scatter(r, c, s=5, color='Red', label='NB')
        K[c,r] = d
        L[c,r] = p
        if S > s:
            s = S
        M[c,r] = S
K =pd.DataFrame(K)
L =pd.DataFrame(L)
M =pd.DataFrame(M)
print ('K', K)
print ('L', L)
print ('M', M)
# K.to_csv("Ensemble.csv",index=True)
# L.to_csv("Ensemble%.csv",index=True)
# M.to_csv("Ensembled.csv",index=True)
# fig, ax = plt.subplots(dpi=300)
# for c in range(0, som_shape[0]):
#     X_c= (ma.loc[ma["Col"] == c])
#     for r in range(0, som_shape[1]):
#         X_r= (X_c.loc[X_c["Row"] == r])
#         total = len(X_r)
#         c0 = (X_r.Class == 0).sum()
#         c1 = (X_r.Class == 1).sum()
#         if total == 0:
#             continue
#         elif c0>c1:
#             z = c0/total
#             ax.scatter(c, r, s=5, c=z, cmap='Greys', vmin=0, vmax=1)
#         else:
#             z = c1/total
#             ax.scatter(c, r, s=5, c=z, cmap='Greys', vmin=0, vmax=1)
plt.title('Class Distribution')
plt.show()


# plt.hist2d(x0, y0, bins=(som_shape[0], som_shape[1]), cmap=plt.cm.Reds)
# plt.colorbar()
# plt.title('Class 0')
# plt.show()

# plt.hist2d(x00, y00, bins=(som_shape[0], som_shape[1]), cmap=plt.cm.Reds)
# plt.colorbar()
# plt.title('Class 0 Predicted 0')
# plt.show()

# plt.hist2d(x01, y01, bins=(som_shape[0], som_shape[1]), cmap=plt.cm.Reds)
# plt.colorbar()
# plt.title('Class 0 Predicted 1 DT')
# plt.show()

# plt.hist2d(x1, y1, bins=(som_shape[0], som_shape[1]), cmap=plt.cm.BuPu)
# plt.colorbar()
# plt.title('Class 1')
# plt.show()

# plt.hist2d(x10, y10, bins=(som_shape[0], som_shape[1]), cmap=plt.cm.BuPu)
# plt.colorbar()
# plt.title('Class 1 Predicted 1')
# plt.show()

# plt.hist2d(x11, y11, bins=(som_shape[0], som_shape[1]), cmap=plt.cm.BuPu)
# plt.colorbar()
# plt.title('Class 1 Predicted 0 DT')
# plt.show()

