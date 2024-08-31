# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:55:51 2018

@author: Minot
"""


import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

df = pd.read_csv('subset.csv')
df['SITEO2V'] = map(lambda x: x.replace("C5","5"), df['SITEO2V'])
df['Label'] = df['SRV_TIME_MON']
df['Label'] = np.where(df['Label'] <= 45, 0, df['Label'])
df['Label'] = np.where(df['Label'] > 45, 1, df['Label'])
# random categorical data

# random categorical data
#data = np.random.choice(20, (100, 10))
km = KModes(n_clusters=3, init='Huang', n_init=5, verbose=1)

clusters = km.fit_predict(df)
print clusters
df['cluster'] = clusters
df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
df['SITEO2V'] = map(lambda x: x.replace("C5","5"), df['SITEO2V'])

df0 = df.loc[df['cluster'] == 0]
df1 = df.loc[df['cluster'] == 1]
df2 = df.loc[df['cluster'] == 2]
df = df0
# Print the cluster centroids
#print(km.cluster_centroids_)

            #replace strrings with int
h = .02  # step size in the mesh

names = ["RBF SVM", 
         "Random Forrest", "Neural Net"]

classifiers = [
    SVC(gamma=2, C=1),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),]

#X, y = readCSV(n_features=2, n_redundant=0, n_informative=2,
#                           random_state=1, n_clusters_per_class=1)
#[['REG', 'MAR_STAT', 'RACE', 'ORIGIN', 'SEX', 'AGE_DX', 'YR_BRTH','SEQ_NUM','DATE_yr','SITEO2V','LATERAL','SURGPRIM','NO_SURG','RAD_SURG','NUMPRIMS','FIRSTPRM','HISTREC','ERSTATUS','PRSTATUS','SRV_TIME_MON_FLAG','BEHANAL','SRV_TIME_MON','ADJTM_6VALUE','ADJNM_6VALUE','ADJM_6VALUE','ADJAJCCSTG']]
#try pandas



Y = np.array(df['Label'])
print df['Label'].value_counts()
X = np.array(df.drop(['Label'], 1))
X = np.array(df.drop(['SRV_TIME_MON'], 1))
X = preprocessing.scale(X)
print len(X)
print df.shape, X.size, Y.size
rng = np.random.RandomState(2)
#X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, Y)

datasets = [
#        make_moons(noise=0.3, random_state=0),
 #           make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

#figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
#    cm = plt.cm.RdBu
#    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
#    if ds_cnt == 0:
#        ax.set_title("Input data")
    # Plot the training points
#    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
#               edgecolors='k')
    # and testing points
#    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
 #              edgecolors='k')
 #   ax.set_xlim(xx.min(), xx.max())
 #   ax.set_ylim(yy.min(), yy.max())
  #  ax.set_xticks(())
   # ax.set_yticks(())
    #i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
 #       ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print score
        i += 1
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].

'''        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1
print score
plt.tight_layout()
plt.show()'''

df = df1

Y = np.array(df['Label'])
print df['Label'].value_counts()
X = np.array(df.drop(['Label'], 1))
X = np.array(df.drop(['SRV_TIME_MON'], 1))
X = preprocessing.scale(X)
print len(X)
print df.shape, X.size, Y.size
rng = np.random.RandomState(2)
#X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, Y)

datasets = [
#        make_moons(noise=0.3, random_state=0),
 #           make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

#figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
#    cm = plt.cm.RdBu
#    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
#    if ds_cnt == 0:
#        ax.set_title("Input data")
    # Plot the training points
#    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
#               edgecolors='k')
    # and testing points
#    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
 #              edgecolors='k')
 #   ax.set_xlim(xx.min(), xx.max())
 #   ax.set_ylim(yy.min(), yy.max())
  #  ax.set_xticks(())
   # ax.set_yticks(())
    #i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
 #       ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print score
        i += 1
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].

'''        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1
print score
plt.tight_layout()
plt.show()'''

df = df2

Y = np.array(df['Label'])
print df['Label'].value_counts()
X = np.array(df.drop(['Label'], 1))
X = np.array(df.drop(['SRV_TIME_MON'], 1))
X = preprocessing.scale(X)
print len(X)
print df.shape, X.size, Y.size
rng = np.random.RandomState(2)
#X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, Y)

datasets = [
#        make_moons(noise=0.3, random_state=0),
 #           make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

#figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
#    cm = plt.cm.RdBu
#    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
#    if ds_cnt == 0:
#        ax.set_title("Input data")
    # Plot the training points
#    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
#               edgecolors='k')
    # and testing points
#    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
 #              edgecolors='k')
 #   ax.set_xlim(xx.min(), xx.max())
 #   ax.set_ylim(yy.min(), yy.max())
  #  ax.set_xticks(())
   # ax.set_yticks(())
    #i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
 #       ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print score
        i += 1
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].

'''        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1
print score
plt.tight_layout()
plt.show()'''