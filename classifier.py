# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 15:57:10 2017

@author: Minot
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

#replace strrings with int
h = .02  # step size in the mesh

names = ["RBF SVM", 
        "Random Forrest"
        ,  "Neural Net"
        ]

classifiers = [
    SVC(gamma=2, C=1),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    ,    MLPClassifier(alpha=1),
               ]

def getRes(df):
    
    Y = np.array(df['Label'])
    dropList = ['Label', 'SRV_TIME_MON', 'CASENUM']
    for item in dropList:
        df.drop(item, axis=1, inplace=True)
    X = np.array(df)
# X = np.array(df.drop(['Label'], 1))
# X = np.array(df.drop(['SRV_TIME_MON'], 1))
# X = np.array(df.drop(['CASENUM'], 1))

    X = preprocessing.scale(X)
    print ("X.shape:", X.shape)
#    print "Y.value:", Counter(Y)

    linearly_separable = (X, Y)
    datasets = [
                #        make_moons(noise=0.3, random_state=0),
                #           make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

    #figure = plt.figure(figsize=(27, 9))
    # iterate over datasets

    for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)

#         x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#         y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#         xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                              np.arange(y_min, y_max, h))
        from sklearn.model_selection import cross_val_score
        clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        score = cross_val_score(clf, X, Y, cv = 5)
        print (score)
        
        # iterate over classifiers
        for name, clf in zip(names, classifiers):
        # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            #print score
            print (name, score)
#            score1 = score + score1
#        mean = float(score1/3)
#        print "mean", mean
