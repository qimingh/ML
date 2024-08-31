### Merging surgery features 1973~1997 & 1998+
import pandas as pd
import numpy as np
import csv
with open ("EncodedB.csv") as f:
    reader = csv.reader(f)
    i = next(reader)

print(i)

df = pd.read_csv("EncodedB.csv")
df.index = df.index + 1

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
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics


names = [
          #"RBF SVM", 
          #"Decision Tree", 
          "MLP", 
          #"Random Forest",
          #"AdaBoost",
          #"Naive Bayes",
          #"Logistic Regression"
          ]

classifiers = [
              #SVC(C=10, kernel=('rbf'), gamma=0.005, class_weight={1:2.0, 0:1}, probability=True),
              #DecisionTreeClassifier(criterion='entropy', max_depth=10),
              MLPClassifier(activation='relu',hidden_layer_sizes=[10], random_state=1),
              #RandomForestClassifier(max_depth=15, n_estimators=1000, min_samples_split=2),
              #AdaBoostClassifier(),
              #GaussianNB(),
              #LogisticRegression()
              ]

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
#X = scaling.fit_transform(ds)
X = ds
print ("X.shape:", X.shape)
print (X)



from scipy.stats import gaussian_kde
from myminisom import MiniSom
import mpl_scatter_density
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import pickle
som_shape = (66, 88)
import shap



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.4, random_state=7)


        # from sklearn.model_selection import cross_val_score
        # clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        # score = cross_val_score(clf, X, Y, cv = 5)
        # print (score)
        
        # iterate over classifiers

for name, clf in zip(names, classifiers):
#
      model = clf.fit(X_train, Y_train)
      clf.fit(X_train, Y_train)
      score_train = clf.score(X_train, Y_train)
      print(name,score_train)
      explainer = shap.Explainer(model.predict, X_test, nsamples=50)
      shap_values = explainer.shap_values(X_test)
      shap.plots.beeswarm(shap_values)
      #shap.plots.waterfall(shap_values[0])
      #shap.plots.bar(shap_values)
      shap.summary_plot(shap_values, plot_type='violin')
      # predicted_train = clf.predict(X_train)
      # matrix = confusion_matrix(Y_train, predicted_train)
      # class_names = ["0", "1"]
      # Confusion = pd.DataFrame(matrix, index=class_names, columns=class_names)
      # print("Train Report:")
      # classification_train = classification_report(Y_train, predicted_train, labels= [0, 1])
      # print(Confusion)
      # print(classification_train)

#       model_displays[name] = metrics.plot_roc_curve(clf, X_train, Y_train, ax=ax, name='train')
      
#       score = clf.score(X_test, Y_test)
#       print (name, score)
#       predicted_test = clf.predict(X_test)
#       matrix = confusion_matrix(Y_test, predicted_test)
#       class_names = ["0", "1"]
#       Confusion = pd.DataFrame(matrix, index=class_names, columns=class_names)
#       print("Test Report:")
#       classification_test = classification_report(Y_test, predicted_test, labels= [0, 1])
#       print(Confusion)
#       print(classification_test)
#       model_displays[name] = metrics.plot_roc_curve(clf, X_test, Y_test, ax=ax, name='test')
#       plt.title('SVM ROC curve')
#       plt.show()

#
#       som_data = X
#       som = MiniSom(som_shape[0], som_shape[1], som_data.shape[1], sigma=som_shape[0]/2, learning_rate=.5, activation_distance='euclidean',
#         neighborhood_function='gaussian', topology='hexagonal', random_seed=10)
#       # som.pca_weights_init(som_data)
#       # som.train_random(som_data, 700000, verbose=False)
#       # with open('somsample.p', 'wb') as outfile:
#       #     pickle.dump(som, outfile)
#       with open('somsample.p', 'rb') as infile:
#           som = pickle.load(infile)
#       winner_coordinates = np.array([som.winner(x) for x in X]).T
#       #print('cluster',winner_coordinates)
#       row = pd.DataFrame(winner_coordinates[0])
#       col = pd.DataFrame(winner_coordinates[1])
#       predicted = clf.predict(X)
      
# #
#       Y_test = pd.DataFrame(Y)
#       Y_predict = pd.DataFrame(predicted)
#       X_t = pd.DataFrame(X)
#       X_t = pd.concat([X_t, row], axis=1)
#       X_t = pd.concat([X_t, col], axis=1)
#       X_t = pd.concat([X_t, Y_test], axis=1)
#       X_t = pd.concat([X_t, Y_predict], axis=1)
#       X_t.columns = ['SEER registry_0',
#                       'SEER registry_1',
#                       'SEER registry_2',
#                       'SEER registry_3',
#                       'SEER registry_4',
#                       'Marital status at diagnosis_0', 
#                       'Marital status at diagnosis_1',
#                       'Marital status at diagnosis_2',
#                       'Race/ethnicity_0',
#                       'Race/ethnicity_1',
#                       'Race/ethnicity_2',
#                       'Race/ethnicity_3',
#                       'Race/ethnicity_4', 
#                       'Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)_0',
#                       'Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)_1',
#                       'Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)_2', 
#                       'Sex', 
#                       'Age at diagnosis', 
#                       'Year of birth', 
#                       'Sequence number', 
#                       'Year of diagnosis', 
#                       'Primary Site_0',
#                       'Primary Site_1', 
#                       'Primary Site_2', 
#                       'Primary Site_3', 
#                       'Laterality_0', 
#                       'Laterality_1', 
#                       'Laterality_2', 
#                       'Reason no cancer-directed surgery_0', 
#                       'Reason no cancer-directed surgery_1', 
#                       'Reason no cancer-directed surgery_2', 
#                       'Reason no cancer-directed surgery_3', 
#                       'Histology recode - broad groupings_0', 
#                       'Histology recode - broad groupings_1', 
#                       'Histology recode - broad groupings_2', 
#                       'Histology recode - broad groupings_3', 
#                       'Histology recode - broad groupings_4', 
#                       'surg combine_0', 
#                       'surg combine_1', 
#                       'surg combine_2',
#                       'surg combine_3', 
#                       'stage combine_0', 
#                       'stage combine_1', 
#                       'stage combine_2', 
#                       'stage combine_3', 
#                       'ER Status Recode Breast Cancer (1990+)_0', 
#                       'ER Status Recode Breast Cancer (1990+)_1', 
#                       'PR Status Recode Breast Cancer (1990+)_0', 
#                       'PR Status Recode Breast Cancer (1990+)_1', 
#                       'Row', 
#                       'Col', 
#                       'Class', 
#                       'Y_predict']
#       print(X_t)
#      X_t.to_csv("X_tNBsample.csv",index=True)


#       X_00= X_t[X_t['Class'] < 1]
#       X_01= X_t[(X_t['Class'] == 0) & (X_t['Y_predict'] == 1)]
#       X_10= X_t[X_t['Class'] == 1]     
#       X_11= X_t[(X_t['Class'] == 1) & (X_t['Y_predict'] == 0)]

#       x00 = np.array(X_00["Row"])
#       y00 = np.array(X_00["Col"])
#       x01 = np.array(X_01["Row"])
#       y01 = np.array(X_01["Col"])
#       x10 = np.array(X_10["Row"])
#       y10 = np.array(X_10["Col"])
#       x11 = np.array(X_11["Row"])
#       y11 = np.array(X_11["Col"])
# #      
#       ma = X_t[["Row", "Col", "Class"]]
#       plt.figure()
#       for r in range(0, som_shape[0]):
#           X_r= (ma.loc[ma["Row"] == r])
#           for c in range(0, som_shape[1]):
#               X_c= (X_r.loc[X_r["Col"] == c])
#               aa = (X_c.Class == 0).sum()
#               bb = (X_c.Class == 1).sum()
#               if aa>bb:
#                   plt.scatter(c, r, color='Red', label='CLass 0')
#               elif aa<bb:
#                   plt.scatter(c, r, color='Blue', label='CLass 1')
#               elif aa==bb & aa>0:
#                   plt.scatter(c, r, color='Yellow', label='CLass 0 = Class 1')
#               else:
#                   plt.scatter(c, r, color='White')
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
#               param_grid={'C':[1, 10], 'kernel': ('linear', 'rbf')}, cv=5)

# gs = GridSearchCV(estimator=LogisticRegression(),
#               param_grid={'penalty': ('l1', 'l2'), "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'solver': ('liblinear', 'saga')}, cv=5)

# gs = GridSearchCV(estimator=DecisionTreeClassifier(),
#               param_grid={'criterion': ('gini', 'entropy'), 'max_depth': np.arange(5, 100)}, cv=5)

# gs = GridSearchCV(estimator=MLPClassifier(random_state=1),
#               param_grid={'activation': ('identity', 'logistic', 'tanh', 'relu'), 'hidden_layer_sizes': [(500, 400, 300, 200, 100), (400, 400, 400, 400, 400), (300, 300, 300, 300, 300), (200, 200, 200, 200, 200)]}, n_jobs=-1, cv=5)
          
# gs = GridSearchCV(estimator=RandomForestClassifier(),
#               param_grid={'max_depth': [5, 8, 10, 15], 'min_samples_split': [2, 5, 10]}, cv=5)

# clf = SVC(gamma='scale')
# param_grid={'kernel': ('linear', 'rbf'), 'C': [1, 10, 100]}
# search = HalvingGridSearchCV(clf, param_grid, factor=2).fit(X_train, Y_train)
# print (search.best_params_)

# clf = LogisticRegression()
# param_grid={'penalty': ('l1', 'l2'), "C": [0.01, 0.1, 1, 2, 10, 100], 'solver': ('liblinear', 'saga')}
# search = HalvingGridSearchCV(clf, param_grid, factor=2, random_state=0).fit(X_train, Y_train)
# print (search.best_params_)

# clf = DecisionTreeClassifier(random_state=0)
# param_grid = {'criterion': ('gini', 'entropy'), 'max_depth': np.arange(5, 25)}
# search = HalvingGridSearchCV(clf, param_grid, random_state=0).fit(X_train, Y_train)
# print (search.best_params_)

# clf = RandomForestClassifier(n_estimators=1000)
# param_grid = {"max_depth": [5, 10, 15, 20], "min_samples_split": np.arange(2, 11), "criterion": ["gini", "entropy"],}
# search = HalvingGridSearchCV(clf, param_grid, factor=2, random_state=0).fit(X_train, Y_train)
# print (search.best_params_)

# gs.fit(X_train, Y_train)
# # summarize results
# print("Best: %f using %s" % (gs.best_score_, gs.best_params_))


