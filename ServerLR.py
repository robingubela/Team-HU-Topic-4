
# coding: utf-8

# In[1]:

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import gc

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

# http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from scipy.optimize import basinhopping


# In[2]:

def generateTrainSet(dat, exclude, target):
    not_features = exclude+target
    
    test_mask  = dat['returnQuantity']==255
    train_mask = dat['returnQuantity']!=255 
    
    X_tr = np.asfortranarray(dat.loc[train_mask,[col for col in dat.columns if col not in not_features]], dtype=np.float32)
    y_tr = np.asfortranarray(dat.loc[train_mask,[col for col in dat.columns if col in target]], dtype=np.float32)
    
    return X_tr, np.ravel(y_tr)

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


# In[3]:

try: # check if data object is in memory
    data
except NameError: # if not LOAD it
    with open('data_set', 'rb') as f:
        data = pickle.load(f)
else: # else if it is not None WRITE it to disk
    if data is not None:
        with open('data_set', 'wb') as f:
            pickle.dump(data, f)
    else: # if it is None LOAD it from disk
        with open('data_set', 'rb') as f:
            data = pickle.load(f)

print('{0:.0f} MB'.format(data.memory_usage().sum()/(1024*1024)))

X_tr, y_tr = generateTrainSet(data, ['returnBin'], ['returnQuantity'])
del data
gc.collect()


# In[ ]:

lr = LogisticRegression(penalty='l1', 
                        dual=False, 
                        tol=0.0001, 
                        C=1.0, 
                        class_weight='balanced', 
                        solver='liblinear', 
                        multi_class='ovr', 
                        verbose=0,  
                        n_jobs=8)

lr.fit(X_tr, y_tr)

with open('LR001', 'wb') as f:
    pickle.dump(lr, f)

