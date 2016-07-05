
# coding: utf-8

# In[ ]:

# %matplotlib inline
#%load_ext Cython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import gc
from operator import itemgetter

from sklearn.cross_validation import train_test_split, StratifiedKFold

# Univariate feature selection
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe
from sklearn.feature_selection import chi2, f_classif

# Recursive feature elimination
from sklearn.feature_selection import RFE, RFECV

# Feature selection using SelectFromModel

# L1-based feature selection
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Randomized sparse models
from sklearn.linear_model import RandomizedLogisticRegression, lasso_stability_path

# Tree-based feature selection
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

# XGBoost
#import xgboost as xgb
#from xgboost.sklearn import XGBClassifier

# http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import plot_sequential_feature_selection as plot_sfs

from scipy.optimize import basinhopping

def generateTrainSet(dat, exclude, target):
    not_features = exclude+target
    
    test_mask  = dat['returnQuantity']==255
    train_mask = dat['returnQuantity']!=255 
    
    X_tr = np.asfortranarray(dat.loc[train_mask,[col for col in dat.columns if col not in not_features]], dtype=np.float32)
    y_tr = np.asfortranarray(dat.loc[train_mask,[col for col in dat.columns if col in target]], dtype=np.float32)
    
    return X_tr, np.ravel(y_tr)

def generateTrainTestSplit(dat, exclude, target, test_percent=0.1):
    not_features = exclude+target
    
    test_mask  = dat['returnQuantity']==255
    train_mask = dat['returnQuantity']!=255
    
    # split the data set into training and test
    X_tr, X_va, y_tr, y_va = train_test_split(
        dat.loc[train_mask,[col for col in dat.columns if col not in not_features]],
        dat.loc[train_mask,[col for col in dat.columns if col in target]], 
        test_size=test_percent, 
        stratify=dat.loc[train_mask,[col for col in dat.columns if col in target]])
                         
    X_te = dat.loc[test_mask,[col for col in dat.columns if col not in not_features]]
    y_te = dat.loc[test_mask,[col for col in dat.columns if col in target]]
    
    X_tr = np.asfortranarray(X_tr.values, dtype=np.float32)
    X_va = np.asfortranarray(X_va.values, dtype=np.float32)
    y_tr = np.ravel(np.asfortranarray(y_tr.values, dtype=np.float32))
    y_va = np.ravel(np.asfortranarray(y_va.values, dtype=np.float32))
    
    return X_tr, X_va, X_te, np.ravel(y_tr), np.ravel(y_va), np.ravel(y_te)

def calcAbsError(estimator, X_va, y_va):
    # calculate the sum of absolute prediction error divided by the number of predictions
    y_pr = np.int64(estimator.predict(X_va))
    return -np.divide(
        np.float64(
            np.sum(
                np.absolute(
                    np.int64(y_va)-y_pr
                )
            )
        ),
        len(y_va)
    )

def loadObj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def saveObj(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

        
        
# Load data
# ..//..//..//Downloads//data_set

try: # check if data object is in memory
    data
except NameError: # if not LOAD it
    data = loadObj('data_set')
else: # else if it is not None WRITE it to disk
    if data is not None:
        saveObj('data_set', data)
    else: # if it is None LOAD it from disk
        data = loadObj('data_set')

print('{0:.0f} MB'.format(data.memory_usage().sum()/(1024*1024)))

# create test and training mask
test_mask = np.ravel(data.loc[:,['returnQuantity']]==255)
train_mask = np.ravel(data.loc[:,['returnQuantity']]!=255)


feature_columns = [col for col in data.columns if col not in ['returnQuantity', 'returnBin']]

# Split training set from the whole dataset

# X_tr, y_tr = generateTrainSet(data, ['returnBin'], ['returnQuantity'])
X_tr, X_va, X_te, y_tr, y_va, y_te = generateTrainTestSplit(data, ['returnBin'], ['returnQuantity'])
del data
gc.collect()


estimator = RandomForestClassifier(n_estimators=224,
                                   max_features=3,
                                   max_depth=8,
                                   n_jobs=16)
# estimator = RandomForestClassifier(n_estimators=8,
#                                    max_features=3,
#                                    max_depth=4,
#                                    n_jobs=2)
# X_tr = X_tr[0:500000]
# y_tr = y_tr[0:500000]


# In[ ]:

subsets = loadObj('Subsets001')

subset_sizes = [12,18,24,30]
subset_names = []
for key, values in subsets.items():
    subset_names.append(key)

results = np.zeros((len(subset_sizes),len(subsets)))
results = pd.DataFrame(results)
results.columns = subset_names
results['index'] = subset_sizes
results.set_index('index', drop=True, inplace=True)
results.index.names = [None]


# In[ ]:

for i,num in enumerate(subset_sizes):
    j = 0
    for key, value in subsets.items():
        if value:
            if key == 'SFFS' or key == 'SFBS':
                mask = value[num-1]
            else:
                mask = []
                for k,col in enumerate(feature_columns):
                    if col in value[0:num]:
                        mask.append(k)
            print(key, mask)
            estimator.fit(X_tr[:,mask], y_tr)
            results.iloc[i,j] = calcAbsError(estimator,X_va[:,mask], y_va)
        j += 1

saveObj('Results001',results)

