### Merging surgery features 1973~1997 & 1998+
import pandas as pd
import numpy as np

df = pd.read_csv("/Users/qm7in/Documents/Seer Breast Cancer Therapy 1.csv")
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

#print (df['RX Summ--Surg Prim Site (1998+)'].value_counts())
#print (df['Site specific surgery (1973-1997 varying detail by year and site)'].value_counts())
#print(" \nCount total NaN at each column in DataFrame : \n\n",
#      df.isnull().sum())
df1 = df[(df['SEER cause-specific death classification'] == 1)]
df2 = df[(df['SEER cause-specific death classification'] == 0) & (df['Survival months'] >= 60)]
df = df1.append(df2)
df['surg combine'] = df['RX Summ--Surg Prim Site (1998+)'].combine_first(df['Site specific surgery (1973-1997 varying detail by year and site)'])
df.drop(['Patient ID',
         'RX Summ--Surg Prim Site (1998+)', 
         'Site specific surgery (1973-1997 varying detail by year and site)',
         'Scope of reg lymph nd surg (1998-2002)',
         'RX Summ--Reg LN Examined (1998-2002)',
         'RX Summ--Scope Reg LN Sur (2003+)', 
         'Type of Reporting Source',
         'RX Summ--Surg Oth Reg/Dis (2003+)', 
         'Surgery of oth reg/dis sites (1998-2002)',
         'SEER other cause of death classification',
         'SEER cause-specific death classification'], 1, inplace=True)
# print (df['surg combine'].value_counts())
# print(" \nCount total NaN at each column in a DataFrame : \n\n", df.isnull().sum())
# df.to_csv("Seer Breast Cancer Therapy-surg combine 1.csv",index=False)

# df1= df.loc[df['Age at diagnosis'] <= 35]
# print (df1.size)
# print (df1['ER Status Recode Breast Cancer (1990+)'].value_counts())


### Correlation
import seaborn as sns
import matplotlib.pyplot as plt

# df1= df.loc[df['Year of diagnosis'] < 192]
# #df1= df.loc[(df['Year of diagnosis'] >= 192) & (df['Year of diagnosis'] < 200)]
# #df1= df.loc[df['Year of diagnosis'] >= 200]
# # play with the figsize until the plot is big enough to plot all the columns
# # of your dataset, or the way you desire it to look like otherwise
# f, ax = plt.subplots(figsize=(10, 8))
# corr = df.corr()
# sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
#             square=True, ax=ax)
# plt.show()
# #sns.heatmap(df1.corr())
# a = df.corr()['Survival months']
# print (a)
# data = df

# data.describe()

# print(data.head())

# data.hist(bins=20, xlabelsize=0, ylabelsize=0, figsize=(20,20))

# plt.savefig('./Distribution.png')
# plt.figure(2,figsize=(20,15))

# data.corr()['Survival months'].sort_values(ascending = False).plot(kind='bar')

# plt.savefig('./CorrToSurvivalMonths.png')

###  Classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics


names = ["RBF SVM", 
          #"Decision Tree", 
          #"Neural Net", 
          #"Random Forest",
          #"AdaBoost",
          #"Naive Bayes"
          ]

classifiers = [SVC(C=10, kernel=('linear')),
              #DecisionTreeClassifier(max_depth=6),
              #MLPClassifier(activation= "logistic", hidden_layer_sizes= [10,10,10]),
              #RandomForestClassifier(max_depth=3, min_samples_split=5, n_estimators=9),
              #AdaBoostClassifier(),
              #GaussianNB()
              ]

ds = df#.sample(n=500)
Y = np.array(ds['Survival months'])
ds['Label'] = ds['Survival months']
ds['Label'] = np.where(ds['Survival months'] < 60, 0, ds['Label'])
ds['Label'] = np.where(ds['Survival months'] >= 60, 1, ds['Label'])
Y = np.array(ds['Label'])
#Y = np.resize(Y, (500))
Y = np.resize(Y, (703169,1))

dropList = ['Survival months', 'Label']
for item in dropList:
    ds.drop(item, axis=1, inplace=True)
data = ds

print (data.head()) 

scaling = preprocessing.MinMaxScaler()
X = scaling.fit_transform(ds)
#X = ds
print ("X.shape:", X.shape)



from scipy.stats import gaussian_kde
from myminisom import MiniSom
import mpl_scatter_density
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import pickle
som_shape = (100, 100)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.4, random_state=7)


        # from sklearn.model_selection import cross_val_score
        # clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        # score = cross_val_score(clf, X, Y, cv = 5)
        # print (score)
        
        # iterate over classifiers

for name, clf in zip(names, classifiers):
      clf.fit(X_train, Y_train)
      score_train = clf.score(X_train, Y_train)
      print(name,score_train)
      fig, ax = plt.subplots()
      model_displays = {}
      predicted_train = clf.predict(X_train)
      matrix = confusion_matrix(Y_train, predicted_train)
      class_names = ["0", "1"]
      Confusion = pd.DataFrame(matrix, index=class_names, columns=class_names)
      print("Train Report:")
      classification_train = classification_report(Y_train, predicted_train, labels= [0, 1])
      print(Confusion)
      print(classification_train)

      model_displays[name] = metrics.plot_roc_curve(clf, X_train, Y_train, ax=ax, name='train')
      
      score = clf.score(X_test, Y_test)
      print (name, score)
      predicted_test = clf.predict(X_test)
      matrix = confusion_matrix(Y_test, predicted_test)
      class_names = ["0", "1"]
      Confusion = pd.DataFrame(matrix, index=class_names, columns=class_names)
      print("Test Report:")
      classification_test = classification_report(Y_test, predicted_test, labels= [0, 1])
      print(Confusion)
      print(classification_test)
      model_displays[name] = metrics.plot_roc_curve(clf, X_test, Y_test, ax=ax, name='test')
      
      plt.title('SVM ROC curve')
      plt.show()
# #
#       som_data = X_train
#       som = MiniSom(som_shape[0], som_shape[1], som_data.shape[1], sigma=som_shape[0]/2, learning_rate=.5,
#               neighborhood_function='gaussian', random_seed=10)
#       som.pca_weights_init(som_data)
#       som.train_random(som_data, 7000000, verbose=False)
#       with open('som100.p', 'wb') as outfile:
#           pickle.dump(som, outfile)
      with open('som100.p', 'rb') as infile:
          som = pickle.load(infile)
      winner_coordinates = np.array([som.winner(x) for x in X_test]).T
      print('cluster',winner_coordinates)
      row = pd.DataFrame(winner_coordinates[0])
      col = pd.DataFrame(winner_coordinates[1])
      
#
      Y_t = pd.DataFrame(Y_test)
      Y_p = pd.DataFrame(predicted_test)
      X_t = pd.DataFrame(X_test)
      X_t = pd.concat([X_t, row], axis=1)
      X_t = pd.concat([X_t, col], axis=1)
      X_t = pd.concat([X_t, Y_t], axis=1)
      X_t = pd.concat([X_t, Y_p], axis=1)
      X_t.columns = ['SEER registry', 
                      'Marital status at diagnosis', 
                      'Race/ethnicity', 
                      'Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)', 
                      'Sex', 
                      'Age at diagnosis', 
                      'Year of birth', 
                      'Sequence number', 
                      'Year of diagnosis', 
                      'Primary Site', 
                      'Laterality', 
                      'Reason no cancer-directed surgery', 
                      'Record number recode', 
                      'Histology recode - broad groupings', 
                      'ER Status Recode Breast Cancer (1990+)', 
                      'PR Status Recode Breast Cancer (1990+)', 
                      'Breast - Adjusted AJCC 6th Stage (1988-2015)', 
                      'Surgery combine', 
                      'Row', 
                      'Col', 
                      'Y_test', 
                      'Y_predict']
      print(X_t)
      X_t.to_csv("X_tsvm.csv",index=True)

#       X_00= X_t[X_t['Y_test'] < 1]
#       X_01= X_t[(X_t['Y_test'] == 0) & (X_t['Y_predict'] == 1)]
#       X_10= X_t[X_t['Y_test'] == 1]     
#       X_11= X_t[(X_t['Y_test'] == 1) & (X_t['Y_predict'] == 0)]

#       x00 = np.array(X_00["Row"])
#       y00 = np.array(X_00["Col"])
#       x01 = np.array(X_01["Row"])
#       y01 = np.array(X_01["Col"])
#       x10 = np.array(X_10["Row"])
#       y10 = np.array(X_10["Col"])
#       x11 = np.array(X_11["Row"])
#       y11 = np.array(X_11["Col"])
# #      
#       ma = X_t[["Row", "Col", "Y_test"]]
#       plt.figure()
#       for r in range(0, som_shape[0]):
#           X_r= (ma.loc[ma["Row"] == r])
#           for c in range(0, som_shape[0]):
#               X_c= (X_r.loc[X_r["Col"] == c])
#               aa = (X_c.Y_test == 0).sum()
#               bb = (X_c.Y_test == 1).sum()
#               if aa>bb:
#                   plt.scatter(r, c, color='Red', label='CLass 0')
#               elif aa<bb:
#                   plt.scatter(r, c, color='Blue', label='CLass 1')
#               elif aa==bb & aa>0:
#                   plt.scatter(r, c, color='Yellow', label='CLass 0 = Class 1')
#               else:
#                   plt.scatter(r, c, color='White')
#       plt.show()


#       x0y0 = np.vstack([x00,y00])
#       z0 = gaussian_kde(x0y0)(x0y0)
#       idx = z0.argsort()
#       x00, y00, z0 = x00[idx], y00[idx], z0[idx]
#       fig = plt.figure()
#       ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
#       ax.scatter(x00, y00, c=z0, s=100)
#       ax.title.set_text('Classes 0')
#       white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
#     (0, '#ffffff'),
#     (1e-20, '#440053'),
#     (0.2, '#404388'),
#     (0.4, '#2a788e'),
#     (0.6, '#21a784'),
#     (0.8, '#78d151'),
#     (1, '#fde624'),
# ], N=256)
#       density = ax.scatter_density(x00, y00, cmap=white_viridis)
#       fig.colorbar(density, label='Number of points per pixel')
#       plt.show()
      
# #
#       x0y1 = np.vstack([x01,y01])
#       z1 = gaussian_kde(x0y1)(x0y1)
#       idx = z1.argsort()
#       x01, y01, z1 = x01[idx], y01[idx], z1[idx]
#       fig = plt.figure()
#       ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
#       ax.scatter(x01, y01, c=z1, s=100)
#       ax.title.set_text('Classes 0 predicted 1')
#       white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
#     (0, '#ffffff'),
#     (1e-20, '#440053'),
#     (0.2, '#404388'),
#     (0.4, '#2a788e'),
#     (0.6, '#21a784'),
#     (0.8, '#78d151'),
#     (1, '#fde624'),
# ], N=256)
#       density = ax.scatter_density(x01, y01, cmap=white_viridis)
#       fig.colorbar(density, label='Number of points per pixel')
#       plt.show()

# # #      

# #       for i in X_t.columns:
# #           results, edges = np.histogram(X_00[i], bins=30, normed=True)
# #           binWidth = edges[1] - edges[0]
# #           plt.bar(edges[:-1], results*binWidth, binWidth, alpha=0.5, label='Class 0')
# #           results1, edges1 = np.histogram(X_01[i], bins=30, normed=True)
# #           binWidth1 = edges1[1] - edges1[0]
# #           plt.bar(edges1[:-1], results1*binWidth1, binWidth1, alpha=0.5, label='Predicted 1')
# #           plt.ylabel('Probability')
# #           plt.xlabel(i);
# #           plt.legend(loc='upper right')
# #           plt.show()
# #           h=stats.mannwhitneyu(pd.value_counts(X_00[i]), pd.value_counts(X_01[i]))
# #           print (i, h)

# #
#       x1y0 = np.vstack([x10,y10])
#       z2 = gaussian_kde(x1y0)(x1y0)
#       idx = z2.argsort()
#       x10, y10, z2 = x10[idx], y10[idx], z2[idx]
#       fig = plt.figure()
#       ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
#       ax.scatter(x10, y10, c=z2, s=100)
#       ax.title.set_text('Classes 1')
#       white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
#     (0, '#ffffff'),
#     (1e-20, '#440053'),
#     (0.2, '#404388'),
#     (0.4, '#2a788e'),
#     (0.6, '#21a784'),
#     (0.8, '#78d151'),
#     (1, '#fde624'),
# ], N=256)
#       density = ax.scatter_density(x10, y10, cmap=white_viridis)
#       fig.colorbar(density, label='Number of points per pixel')
#       plt.show()

# #
#       x1y1 = np.vstack([x11,y11])
#       z3 = gaussian_kde(x1y1)(x1y1)
#       idx = z3.argsort()
#       x11, y11, z3 = x11[idx], y11[idx], z3[idx]
#       fig = plt.figure()
#       ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
#       ax.scatter(x11, y11, c=z3, s=100)
#       ax.title.set_text('Classes 1 predicted 0')
#       white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
#     (0, '#ffffff'),
#     (1e-20, '#440053'),
#     (0.2, '#404388'),
#     (0.4, '#2a788e'),
#     (0.6, '#21a784'),
#     (0.8, '#78d151'),
#     (1, '#fde624'),
# ], N=256)
#       density = ax.scatter_density(x11, y11, cmap=white_viridis)
#       fig.colorbar(density, label='Number of points per pixel')
#       plt.show()
#       for i in X_t.columns:
#           results2, edges2 = np.histogram(X_10[i], bins=30, normed=True)
#           binWidth2 = edges2[1] - edges2[0]
#           plt.bar(edges2[:-1], results2*binWidth2, binWidth2, alpha=0.5, label='Class 1')
#           results3, edges3 = np.histogram(X_11[i], bins=30, normed=True)
#           binWidth3 = edges3[1] - edges3[0]
#           plt.bar(edges3[:-1], results3*binWidth3, binWidth3, alpha=0.5, label='Predicted 0')
#           plt.ylabel('Probability')
#           plt.xlabel(i);
#           plt.legend(loc='upper right')
#           plt.show()
#           h=stats.ks_2samp(pd.value_counts(X_10[i]), pd.value_counts(X_11[i]))
#           print (i, h)


# gs = GridSearchCV(estimator=SVC(),
#               param_grid={'C': np.arange(1, 10), 'kernel': ('linear', 'rbf')})

# gs = GridSearchCV(estimator=DecisionTreeClassifier(),
#               param_grid={'criterion': ('gini', 'entropy'), 'max_depth': np.arange(5, 25)})

# gs = GridSearchCV(estimator=MLPClassifier(),
#               param_grid={'hidden_layer_sizes': np.arange(1, 100), 'activation': ('logistic', 'tanh', 'relu')}, cv=5)

# gs = GridSearchCV(estimator=RandomForestClassifier(),
#               param_grid={'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}, cv=5)

# clf = DecisionTreeClassifier(random_state=0)
# param_grid = {'criterion': ('gini', 'entropy'), 'max_depth': np.arange(5, 25)}
# search = HalvingGridSearchCV(clf, param_grid, random_state=0).fit(X, Y)
# print (search.best_params_)

# clf = RandomForestClassifier(random_state=0)
# param_grid = {"max_depth": [3, None], "min_samples_split": [5, 10]}
# search = HalvingGridSearchCV(clf, param_grid, resource='n_estimators', max_resources=10, random_state=0).fit(X, Y)
# print (search.best_params_)

# gs.fit(X, Y)
# # summarize results
# print("Best: %f using %s" % (gs.best_score_, gs.best_params_))


