from minisom import MiniSom

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D


data = pd.read_csv('/Users/qm7in/.spyder-py3/x_tNBsample.csv')
ensemble1 = pd.read_csv('/Users/qm7in/.spyder-py3/ensemble.csv').iloc[: , 1:]
ensemble = ensemble1.transpose()
ensemble = np.array(ensemble)
ratio = pd.read_csv('/Users/qm7in/.spyder-py3/ensemble%.csv').iloc[: , 1:]
ratio = np.array(ratio)
ratio2 = pd.read_csv('/Users/qm7in/.spyder-py3/ensembles%.csv').iloc[: , 1:]
ratio2 = np.array(ratio2)
data['ensemble'] = 0
size = pd.read_csv('/Users/qm7in/.spyder-py3/ensembled.csv').iloc[: , 1:]
size = np.array(size)

for row in range(ensemble.shape[1]):
      for col in range(ensemble.shape[0]):
        data.loc[((data['Col'] == col) & (data['Row'] == row)), 'ensemble'] = ensemble[col, row]

data= data.replace({'ensemble':{'Random Forest' : 1, 'AdaBoost' : 2, 'Decision Tree' : 3, 'Neural Net' : 4, 'Logistic Regression' : 5, 'SVM' : 6, 'Naive Bayes' : 7}})

    
data = data.loc[~(data['ensemble']==0)]
#print (data)
t = data['ensemble'].values
t = np.array(t, dtype=np.int64)
data = data[data.columns[:-1]]

print (t)
# data normalization
# df = pd.read_csv("EncodedB.csv")
# df.index = df.index + 1

# from sklearn import preprocessing

# ds = df#.sample(50000)
# Y = np.array(ds['Survival months'])
# ds['Label'] = ds['Survival months']
# ds['Label'] = np.where(ds['Survival months'] < 60, 0, ds['Label'])
# ds['Label'] = np.where(ds['Survival months'] >= 60, 1, ds['Label'])
# Y = np.array(ds['Label'])
# Y = np.resize(Y, (len(df)))


# dropList = ['Survival months', 'Label']
# for item in dropList:
#     ds.drop(item, axis=1, inplace=True)
# data = ds


# scaling = preprocessing.MinMaxScaler()
# X = scaling.fit_transform(ds)
# X = ds
# print ("X.shape:", X.shape)


som_shape = (66, 88)


import pickle

# som_data = X
som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=som_shape[0]/2, learning_rate=.5, activation_distance='euclidean',
        neighborhood_function='gaussian', topology='hexagonal', random_seed=10)
# som.pca_weights_init(som_data)
# som.train_random(som_data, 700000, verbose=False)
# with open('somsample.p', 'wb') as outfile:
#     pickle.dump(som, outfile)
with open('somsample.p', 'rb') as infile:
    som = pickle.load(infile)

xx, yy = som.get_euclidean_coordinates()
umatrix = ratio.T
umatrix2 = ratio2.T
#umatrix = som.distance_map()
vmatrix = size.T
weights = som.get_weights()

f = plt.figure(figsize=(66,88))
ax = f.add_subplot(111)

ax.set_aspect('equal')

ensemble1 = ensemble1.replace({'ensemble':{'Random Forest' : 1, 'AdaBoost' : 2, 'Decision Tree' : 3, 'Neural Net' : 4, 'Logistic Regression' : 5, 'SVM' : 6, 'Naive Bayes' : 7}})
ensemble1 = ensemble1.transpose()
vmax = 0

# iteratively add hexagons
for i in range(-1, weights.shape[0]):
    for j in range(-1, weights.shape[1]):
        if vmax < vmatrix[i,j]:
            vmax = vmatrix[i,j]
print (vmax)
for i in range(-1, weights.shape[0]):
    for j in range(-1, weights.shape[1]):
        wy = yy[(i, j)] * np.sqrt(3) / 2
        hex = RegularPolygon((xx[(i, j)], wy), 
                              numVertices=6, 
                              radius=.95 / np.sqrt(3),
                              #facecolor=cm.Blues(umatrix[i, j]), 
                              facecolor=cm.Blues(vmatrix[i, j]/vmax), 
                              alpha=1, 
                              edgecolor='gray')
        ax.add_patch(hex)

markers = ['o', '+', 'x', '<', '>', '*', '^']
colors = ['C8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

# for row in range(ensemble.shape[1]):
#       for col in range(ensemble.shape[0]):
#         data.loc[((data['Col'] == col) & (data['Row'] == row)), 'ensemble'] = ensemble[col, row]
#ensemble1 = ensemble1.transpose()
ensemble1 = ensemble1.stack().reset_index().rename(columns={'level_0':'Row','level_1':'Col', 0:'Clf'})
ensemble1 = ensemble1.replace({'Clf':{'Random Forest' : 1, 'AdaBoost' : 2, 'Decision Tree' : 3, 'Neural Net' : 4, 'Logistic Regression' : 5, 'SVM' : 6, 'Naive Bayes' : 7}})
t = ensemble1['Clf'].values
t = np.array(t, dtype=np.int64)
#ensemble1 = ensemble1[['Row', 'Col', 'Clf']]
ensemble1 = ensemble1[ensemble1.columns[:-1]]


ensemble1 = ensemble1.astype(int)
tuples = [tuple(x) for x in ensemble1.to_numpy()]


for cnt, x in enumerate(tuples):
    # getting the winner
    w = x
    # place a marker on the winning position for the sample xx
    wx, wy = som.convert_map_to_euclidean(w) 
    wy = wy * np.sqrt(3) / 2
    if t[cnt]-1 < 0:
        continue
    plt.plot(wx, wy, 
              markers[t[cnt]-1], 
              markerfacecolor='None',
              markeredgecolor=colors[t[cnt]-1], 
              markersize=12, 
              markeredgewidth=2)

legend_elements = [Line2D([0], [0], marker='o', color='C8', label='RF',
                    markerfacecolor='w', markersize=15, linestyle='None', markeredgewidth=2),
                    Line2D([0], [0], marker='+', color='C1', label='AB',
                    markerfacecolor='w', markersize=15, linestyle='None', markeredgewidth=2),
                    Line2D([0], [0], marker='x', color='C2', label='DT',
                    markerfacecolor='w', markersize=15, linestyle='None', markeredgewidth=2),
                    Line2D([0], [0], marker='<', color='C3', label='MLP',
                    markerfacecolor='w', markersize=15, linestyle='None', markeredgewidth=2),
                    Line2D([0], [0], marker='>', color='C4', label='LR',
                    markerfacecolor='w', markersize=15, linestyle='None', markeredgewidth=2),
                    Line2D([0], [0], marker='*', color='C5', label='SVM',
                    markerfacecolor='w', markersize=15, linestyle='None', markeredgewidth=2),
                    Line2D([0], [0], marker='^', color='C6', label='NB',
                    markerfacecolor='w', markersize=15, linestyle='None', markeredgewidth=2)]
ax.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1), loc='upper left', 
          borderaxespad=0., ncol=4, fontsize=32)

xrange = np.arange(weights.shape[0])
yrange = np.arange(weights.shape[1])
plt.xticks(xrange-.5, xrange)
plt.yticks(yrange * np.sqrt(3) / 2, yrange)
plt.axis('off')
norm= matplotlib.colors.Normalize(vmin=0,
                                  #vmax=1
                                  vmax=vmax
                                  )
divider = make_axes_locatable(ax)
ax_cb = divider.append_axes("right",size="1%", pad=0.04)    
cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Blues, 
                            norm = norm,
                            orientation='vertical', alpha=1)
cb1.ax.get_yaxis().labelpad = 10
cb1.ax.tick_params(labelsize=20)
cb1.ax.set_ylabel(#'Sample Ratio',
                  'Sample Size',
                  #'distance from neurons in the neighbourhood',
                  rotation=270, fontsize=28, weight='bold')
plt.gcf().add_axes(ax_cb)

#plt.savefig('som_seed_hex.png')

plt.show()
