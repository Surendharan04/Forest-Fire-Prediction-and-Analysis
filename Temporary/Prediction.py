import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
from scipy import stats
from sklearn import svm


import os


import sys
import sklearn
#print(sys.version)
#print("numpy: ", np.__version__)
#print("sklearn: ", sklearn.__version__)

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 2000)

path = 'C:/Users/SURENDHARAN/OneDrive/Desktop/Programs/FinalYearProject/'
dfmain = pd.read_csv(path + 'forestfires.csv', header=0)
#print(dfmain.shape)
dfmain.head(2)

dfmain['fire_scale'] = dfmain['area'].apply(lambda x: 'no_fire' if (x==0) else 'small_fire' if ((x>0)&(x<2)) else 'large_fire')
#print(dfmain.shape) 
dfmain.head(2)

plt.hist(dfmain[(dfmain['area']>0)&(dfmain['area']<20)].area, bins=50)
#plt.show()

plt.hist(np.log(dfmain[(dfmain['area']>0)&(dfmain['area']<20)].area + 1), bins=50)
#plt.show()

t = dfmain.groupby(['month'])['month'].count()
plt.bar(t.index, t)

dfmain.groupby(['month', 'fire_scale'])['fire_scale'].count()

t = dfmain.groupby(['day'])['day'].count()
plt.bar(t.index, t)

d = dfmain[dfmain['area']>0].copy()
print(d.shape)
for m in d['month'].unique():
    if((m!='aug')&(m!='sep')):
        temp = d[d['month']==m].sample(300, replace=True)
        d = pd.concat([d, temp], axis=0)

#print(d.shape)

# plot
t = d.groupby(['month'])['month'].count()
plt.bar(t.index, t)

#d = dfmain[dfmain['area']>0].copy()
X = d.drop(['area', 'fire_scale'], axis=1)
y = d['area']
X = pd.get_dummies(X, ['month', 'day'])
X.head(2)

x_cols_for_scaling = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
x_train, x_test, y_train, y_test = train_test_split(X, np.log(y+1), shuffle=True)
#print(x_train.shape, x_test.shape)

x_train_orig = x_train.loc[:, x_cols_for_scaling]
x_train_cat = x_train.drop(x_cols_for_scaling, axis=1)

x_test_orig = x_test.loc[:, x_train_orig.columns]
x_test_cat = x_test.loc[:, x_train_cat.columns]

scl=preprocessing.StandardScaler()
scl.fit(x_train_orig)

x_train_orig = scl.transform(x_train_orig)
x_test_orig = scl.transform(x_test_orig)

# Combine
x_train = np.concatenate([x_train_orig, np.array(x_train_cat)], axis=1)
x_test = np.concatenate([x_test_orig, np.array(x_test_cat)], axis=1)

#print(x_train.shape, x_test.shape)

reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)

# Predict
y_pred = reg.predict(x_test)

# Score
mse = metrics.mean_squared_error(y_test, y_pred)
print('mse: ', np.round(mse, 4))

mae = metrics.mean_absolute_error(y_test, y_pred)
print('mae: ', np.round(mae, 4))

r2 = metrics.r2_score(y_test, y_pred)
print('r2: ', np.round(r2, 4))


# Plot
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('y_test')
plt.ylabel('y_pred')

reg = svm.SVR(C=1, kernel='rbf', gamma='auto', max_iter=5e4, cache_size=1000)
reg.fit(x_train, y_train)

# Predict
y_pred = reg.predict(x_test)

# Score
mse = metrics.mean_squared_error(y_test, y_pred)
print('mse: ', np.round(mse, 4))

mae = metrics.mean_absolute_error(y_test, y_pred)
print('mae: ', np.round(mae, 4))

r2 = metrics.r2_score(y_test, y_pred)
print('r2: ', np.round(r2, 4))


# Plot
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('y_test')
plt.ylabel('y_pred')

dfc = dfmain.copy()
dfc.fire_scale.value_counts()

#print(dfc.shape)
for m in dfc['month'].unique():
    if((m!='aug')&(m!='sep')):
        temp = dfc[dfc['month']==m].sample(300, replace=True)
        dfc = pd.concat([dfc, temp], axis=0)

#print(dfc.shape)

#d = dfmain[dfmain['area']>0].copy()
Xc = dfc.drop(['area', 'fire_scale'], axis=1)
yc = dfc['fire_scale']
Xc = pd.get_dummies(Xc, ['month', 'day'])
Xc.head(2)

x_cols_for_scaling = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
# Zero var
#print(Xc.shape)
Xc = Xc[Xc.columns[(Xc.var(axis=0)>0).values]]
#print(Xc.shape)

# Collinearity
#print(Xc.shape)
x_corr = Xc.corr()**2

x_upper = x_corr.where(np.triu(np.ones(x_corr.shape), k=1).astype(np.bool))

# r2>0.7
drop_col = [col for col in x_upper.columns if any(x_upper[col] > 0.70)]

Xc = Xc.drop(drop_col, axis=1)
#print(Xc.shape)

x_train, x_test, y_train, y_test = train_test_split(Xc, yc, 
                                                    shuffle=True, stratify=yc)
#print(x_train.shape, x_test.shape)

x_train_orig = x_train.loc[:, x_cols_for_scaling]
x_train_cat = x_train.drop(x_cols_for_scaling, axis=1)

x_test_orig = x_test.loc[:, x_train_orig.columns]
x_test_cat = x_test.loc[:, x_train_cat.columns]

scl=preprocessing.StandardScaler()
scl.fit(x_train_orig)

x_train_orig = scl.transform(x_train_orig)
x_test_orig = scl.transform(x_test_orig)

# Combine
x_train = np.concatenate([x_train_orig, np.array(x_train_cat)], axis=1)
x_test = np.concatenate([x_test_orig, np.array(x_test_cat)], axis=1)

#print(x_train.shape, x_test.shape)

clf = linear_model.LogisticRegression(max_iter=1e7, penalty='elasticnet', solver='saga', l1_ratio=0.75)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acu = metrics.accuracy_score(y_test, y_pred)
roundoff = np.round(acu, 2) * 100
print("Logistic Regression Prediction accuracy: ", roundoff, "%")

cm = metrics.confusion_matrix(y_test, y_pred)
cm

clf = svm.SVC(C=1, kernel='rbf', gamma='auto', max_iter=5e4, cache_size=1000)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acu = metrics.accuracy_score(y_test, y_pred)
roundoff = np.round(acu, 2) * 100
print("SVM Prediction Accuracy: ", roundoff, "%")

cm = metrics.confusion_matrix(y_test, y_pred)
