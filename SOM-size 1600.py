import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


d = {'col': [8, 24, 40, 56, 72, 88, 104, 120, 136, 152, 168, 184, 200, 216, 248, 264, 280, 360, 424, 800, 1066, 1333, 1866, 2166, 2666], 
     'col2': [0.894865386, 0.897308859542964, 0.899898886327425, 0.901997393589068, 0.904092169192738, 0.90571340297549, 0.907603972245572, 0.908675288429603, 0.909934946321321, 0.911455236620383, 0.91237726352455, 0.913481163174812, 0.914594978112947, 0.915283622420754, 0.917828781369949, 0.918742399313255, 0.919544546885462, 0.923007928820005, 0.926265160719806, 0.935383835090388, 0.935399616924952, 0.944321, 0.948267, 0.9443, 0.954055]}

# plt.plot(d["col"], d["col2"])
# plt.xlim(1,2700)
# plt.show()

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



SOM = pd.read_csv("/Users/qm7in/.spyder-py3/2000_2666_1_3.csv", header = None)
SOM = SOM.T
SOM = SOM.set_axis({"Row", "Col"}, axis=1)
#print (SOM)
col = SOM['Col']
row = SOM['Row']

som_shape = (2666, 2000)

winner_coordinates = np.genfromtxt('2000_2666_1_3.csv', delimiter=',').astype('int64')
df = pd.read_csv("EncodedB.csv")
maxx = pd.DataFrame(winner_coordinates[0])
maxy = pd.DataFrame(winner_coordinates[1])
df = pd.concat([df, maxx], axis=1)
df = pd.concat([df, maxy], axis=1)
dropList = ['Survival months']
O=df[['Survival months']]
O.loc[(O['Survival months'] >120) ,'Survival months']=120
for item in dropList:
    df.drop(item, axis=1, inplace=True)
df = df
df.columns = ['SEER registry_0',
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
                        'Col', ]

# X_age7 = df[(df['Age at diagnosis'] > 0.7 )]
# print (X_age7)
# x_v1 = X_age7["Row"].to_numpy()
# y_v1 = X_age7["Col"].to_numpy()
# print (x_v1)

# X_age3 = df[(df['Age at diagnosis'] < 0.3 )]
# print (X_age3)
# x_v = X_age3["Row"].to_numpy()
# y_v = X_age3["Col"].to_numpy()
# print (x_v)

# X_age5 = df[(df['Age at diagnosis'] < 0.55 ) & (df['Age at diagnosis'] > 0.45 )]
# print (X_age5)
# x_v2 = X_age5["Row"].to_numpy()
# y_v2 = X_age5["Col"].to_numpy()
# print (x_v2)

# ma = df[["Age at diagnosis", "Row", "Col"]]
# x_values = ma["Row"].to_numpy()
# y_values = ma["Col"].to_numpy()
# plt.figure(dpi=1200)
# plt.rcParams["legend.markerscale"] = 36
# plt.scatter(x_v, y_v, s=0.01, c='green', label='Age<30')
# plt.scatter(x_v2, y_v2, s=0.01, c='yellow', marker='v', label='45<Age<55')
# plt.scatter(x_v1, y_v1, s=0.01, c='red', marker='s', label='Age>70')

# plt.title('Age')
# plt.legend(loc='lower center')
# plt.axis('off')
# plt.show()

# X_stg1 = df[(df['stage combine_0'] == 0 ) & (df['stage combine_1'] == 0 ) & (df['stage combine_2'] == 1 ) & (df['stage combine_3'] == 0 )]
# print (X_stg1)
# x_v = X_stg1["Row"].to_numpy()
# y_v = X_stg1["Col"].to_numpy()
# print (x_v)

# X_stg4 = df[(df['stage combine_0'] == 0 ) & (df['stage combine_1'] == 1 ) & (df['stage combine_2'] == 1 ) & (df['stage combine_3'] == 1 )]
# print (X_stg4)
# x_v4 = X_stg4["Row"].to_numpy()
# y_v4 = X_stg4["Col"].to_numpy()
# print (x_v4)


# X_stg21 = df[(df['stage combine_0'] == 1 ) & (df['stage combine_1'] == 0 ) & (df['stage combine_2'] == 0 ) & (df['stage combine_3'] == 0 )]
# X_stg22 = df[(df['stage combine_0'] == 0 ) & (df['stage combine_1'] == 1 ) & (df['stage combine_2'] == 0 ) & (df['stage combine_3'] == 0 )]
# X_stg2 = X_stg21.append(X_stg22, ignore_index=True)
# print (X_stg2)
# x_v2 = X_stg2["Row"].to_numpy()
# y_v2 = X_stg2["Col"].to_numpy()
# print (x_v2)

# ma = df[["Row", "Col"]]
# x_values = ma["Row"].to_numpy()
# y_values = ma["Col"].to_numpy()
# plt.figure(dpi=1200)
# plt.rcParams["legend.markerscale"] = 36
# plt.scatter(x_values, y_values, s=0.01, c='grey', alpha=0.5)
# plt.scatter(x_v, y_v, s=0.01, c='blue', label='Stage I')
# plt.scatter(x_v2, y_v2, s=0.01, c='brown', marker='v', label='Stage II')
# plt.scatter(x_v4, y_v4, s=0.01, c='yellow', marker='s', label='Stage IV')
# plt.legend(loc='lower center')
# plt.title('Stage')
# plt.axis('off')
# plt.show()

# X_ER1 = df[(df['ER Status Recode Breast Cancer (1990+)_0'] == 1 )]
# print (X_ER1)
# x_v1 = X_ER1["Row"].to_numpy()
# y_v1 = X_ER1["Col"].to_numpy()
# print (x_v1)

# X_ER2 = df[(df['ER Status Recode Breast Cancer (1990+)_0'] == -1 )]
# print (X_ER2)
# x_v2 = X_ER2["Row"].to_numpy()
# y_v2 = X_ER2["Col"].to_numpy()
# print (x_v2)

# X_ER3 = df[(df['ER Status Recode Breast Cancer (1990+)_0'] == 0 ) & (df['ER Status Recode Breast Cancer (1990+)_1'] == 0 )]
# print (X_ER3)
# x_v3 = X_ER3["Row"].to_numpy()
# y_v3 = X_ER3["Col"].to_numpy()
# print (x_v3)

# ma = df[["Row", "Col"]]
# x_values = ma["Row"].to_numpy()
# y_values = ma["Col"].to_numpy()
# plt.figure(dpi=1200)
# plt.rcParams["legend.markerscale"] = 36
# plt.scatter(x_values, y_values, s=0.01, c='grey', alpha=0.5)
# plt.scatter(x_v1, y_v1, s=0.01, c='purple', label='positive')
# plt.scatter(x_v2, y_v2, s=0.01, c='green', marker='v', label='negative')
# plt.scatter(x_v3, y_v3, s=0.01, c='orange', marker='s', label='borderline')
# plt.legend(loc='lower center')
# plt.title('ER status')
# plt.axis('off')
# plt.show()

# ma = df[["Row", "Col"]]
# X = pd.read_csv("XpDT-I.csv")
# X_t = X[["Class", "Y_predict"]]
# ma = pd.concat([X_t, df], axis=1)
# ma = pd.concat([ma, O], axis=1)
# ma1 = ma[(ma['Class'] == 0 )]
# x_values = ma1["Row"].to_numpy()
# y_values = ma1["Col"].to_numpy()
# ma2 = ma[(ma['Class'] == 1 )]
# x_values1 = ma2["Row"].to_numpy()
# y_values1 = ma2["Col"].to_numpy()
# x_values0 = ma["Row"].to_numpy()
# y_values0 = ma["Col"].to_numpy()

# o = ma["Survival months"]
# divisor = 12
# result = [x//divisor for x in o]
# plt.figure(dpi=1200)
# # plt.rcParams["legend.markerscale"] = 36
# # plt.scatter(x_v1, y_v1, s=0.01, c='red')
# # plt.scatter(x_v0, y_v0, s=0.01, c='green')
# # plt.scatter(x_values1, y_values1, s=0.01, c='green', label='Class 1', alpha=0.3)
# # plt.scatter(x_values, y_values, s=0.01, c='red', marker='+', label='Class 0', alpha=0.3)
# plt.scatter(x_values0, y_values0, c=result, s=0.005, cmap="RdYlGn")
# plt.colorbar()
# # plt.legend(loc='lower center')
# plt.title('Survival Years')
# plt.axis('off')
# plt.show()

X_LR1 = df[(df['Laterality_0'] == 0 ) & (df['Laterality_2'] == 0 )]
print (X_LR1)
x_v1 = X_LR1["Row"].to_numpy()
y_v1 = X_LR1["Col"].to_numpy()
print (x_v1)

X_LR2 = df[(df['Laterality_0'] == 0 ) & (df['Laterality_1'] == 0 )]
print (X_LR2)
x_v2 = X_LR2["Row"].to_numpy()
y_v2 = X_LR2["Col"].to_numpy()
print (x_v2)

X_LR3 = df[(df['Laterality_0'] == 1 )]
print (X_LR3)
x_v3 = X_LR3["Row"].to_numpy()
y_v3 = X_LR3["Col"].to_numpy()
print (x_v3)

ma = df[["Row", "Col"]]
x_values = ma["Row"].to_numpy()
y_values = ma["Col"].to_numpy()
plt.figure(dpi=1200)
plt.rcParams["legend.markerscale"] = 36
plt.scatter(x_values, y_values, s=0.01, c='grey', alpha=0.5)
plt.scatter(x_v1, y_v1, s=0.01, c='red', label='Right')
plt.scatter(x_v2, y_v2, s=0.01, c='blue', marker='v', label='Left')
plt.scatter(x_v3, y_v3, s=0.01, c='yellow', marker='s', label='Bilateral')
plt.legend(loc='lower center')
plt.title('Laterality')
plt.axis('off')
plt.show()