# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 16:36:05 2019

@author: Charl
"""

import pandas as pd

import matplotlib.pyplot as plt 

data = pd.read_csv('D:/Documents/UFS/5th Year/Honours Project/Data Sources/train_V2.csv', nrows = 70000)

x = data.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27]].values
y = data.iloc[:, 28].values

from sklearn.preprocessing import LabelEncoder
LC=LabelEncoder()
x[:,15]=LC.fit_transform(x[:,15])

from sklearn import svm

SVM = svm.SVR(gamma='scale')
SVM.fit(x, y)

test = pd.read_csv('D:/Documents/UFS/5th Year/Honours Project/Data Sources/test_V2.csv')
x_test = test.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27]].values

result = SVM.predict(x_test)
print(result)

plt.plot(result)
plt.show()

submission = pd.DataFrame.from_dict(data={'Id': test['Id'], 'winPlacePerc': result})
submission.head()
submission.to_csv('submission.csv', index=False)