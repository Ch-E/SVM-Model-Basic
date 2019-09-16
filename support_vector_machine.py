# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 22:08:14 2019

@author: Charl
"""

import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import time as time

#**********************************Train & Test dataset**************************************
train = pd.read_csv('D:/Documents/UFS/5th Year/Honours Project/Data Sources/train_V2.csv', nrows=1000)
train.head()

test = pd.read_csv('D:/Documents/UFS/5th Year/Honours Project/Data Sources/test_V2.csv')
test.head()

train.isnull().sum().sum()
test.isnull().sum().sum()

train.winPlacePerc.fillna(1,inplace=True)
train.loc[train['winPlacePerc'].isnull()]

train["distance"] = train["rideDistance"]+train["walkDistance"]+train["swimDistance"]
train["skill"] = train["headshotKills"]+train["roadKills"]
train.drop(['rideDistance','walkDistance','swimDistance','headshotKills','roadKills'],inplace=True,axis=1)
print(train.shape)
train.head()

test["distance"] = test["rideDistance"]+test["walkDistance"]+test["swimDistance"]
test["skill"] = test["headshotKills"]+test["roadKills"]
test.drop(['rideDistance','walkDistance','swimDistance','headshotKills','roadKills'],inplace=True,axis=1)
print(test.shape)
test.head()

predictors = [  "kills",
                "maxPlace",
                "numGroups",
                "distance",
                "boosts",
                "heals",
                "revives",
                "killStreaks",
                "weaponsAcquired",
                "winPoints",
                "skill",
                "assists",
                "damageDealt",
                "DBNOs",
                "killPlace",
                "killPoints",
                "vehicleDestroys",
                "longestKill"
               ]

X = train[predictors]
X.head()

y = train['winPlacePerc']
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#**********************************Train & Test dataset**************************************
#%% hyperparameter tuning
from sklearn.model_selection import GridSearchCV

def log(x):
    # can be used to write to log file
    print(x)

# Utility function to report best scores (from scikit-learn.org)
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            log("Model with rank: {0}".format(i))
            log("Mean validation score: {0:.5f} (std: {1:.5f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            log("Parameters: {0}".format(results['params'][candidate]))
            log("")


#Function to determine the best fit (from scikit-learn.org)

def best_fit(clf, X_train, y_train):
    
    param_grid = {
                    'degree': np.arange(1,7)
                 }

    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10, n_jobs=8)

    import time as ttt
    now = time()
    log(ttt.ctime())
    
    grid_search.fit(X_train, y_train)
    
    report(grid_search.cv_results_, n_top=10)
    
    log(100*"-")
    log(ttt.ctime())
    log("Search (3-fold cross validation) took %.5f seconds for %d candidate parameter settings." 
        % (time() - now, len(grid_search.cv_results_['params'])))
    log('')
    log("The best parameters are %s with a score of %0.5f"
        % (grid_search.best_params_, grid_search.best_score_))
    
    return grid_search
#%%
from sklearn import svm

SVM = svm.SVR(gamma='scale')
#best_fit(svm.SVR, X_train, y_train)

SVM.fit(X_train, y_train)

prediction = SVM.predict(X_test)

#Metrics
from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error:")
print(mean_absolute_error(y_test, prediction))
print("")
#%%
#**********************************Submission**************************************
test_id = test["Id"]
submit = pd.DataFrame({'Id': test_id, "winPlacePerc": y_test} , columns=['Id', 'winPlacePerc'])
print(submit.head())

submit.to_csv("submission.csv", index = False)
#**********************************Submission**************************************
#%%

