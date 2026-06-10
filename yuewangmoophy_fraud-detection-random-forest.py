import numpy as np 
import pandas as pd 
from shutil import copyfile
from sklearn.ensemble import RandomForestClassifier


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
copyfile(src = "../input/158333a1/a1_wrangle.py", dst = "../working/a1_wrangle.py")
import a1_wrangle as a1w


 X_train, y_train, X_test, submission = a1w.quick_wrangle()


y_train.head()


X_train.head()


X_test.head()


for n in range(100, 1501, 200):
    clf = RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0, n_jobs=4)
    clf.fit(X_train, y_train)
    submission['isFraud'] = clf.predict_proba(X_test)[:,1]
    submission.to_csv('RandomForest' + str(n) + '.csv')

