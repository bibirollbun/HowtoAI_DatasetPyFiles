#TPOT AutoML tool
from tpot import TPOTClassifier


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np # linear algebra
np.set_printoptions(precision=2)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame


# Preprocessing, modelling and evaluating
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.preprocessing import Imputer,MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error, mean_absolute_error
from sklearn import svm

## Hyperopt modules
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from functools import partial


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


df_trans = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
df_test_trans = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

df_id = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
df_test_id = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')

df_train = df_trans.merge(df_id, how='left', left_index=True, right_index=True)
df_test = df_test_trans.merge(df_test_id, how='left', left_index=True, right_index=True)



one_value_cols = [col for col in df_train.columns if df_train[col].nunique() <= 1]
one_value_cols_test = [col for col in df_test.columns if df_test[col].nunique() <= 1]


many_null_cols = [col for col in df_train.columns if df_train[col].isnull().sum() / df_train.shape[0] > 0.9]
many_null_cols_test = [col for col in df_test.columns if df_test[col].isnull().sum() / df_test.shape[0] > 0.9]


%%time
big_top_value_cols = [col for col in df_train.columns if df_train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
big_top_value_cols_test = [col for col in df_test.columns if df_test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
cols_to_drop = list(set(many_null_cols + many_null_cols_test +
                        big_top_value_cols +
                        big_top_value_cols_test +
                        one_value_cols+ one_value_cols_test))
len(cols_to_drop)


cols_to_drop.remove('isFraud')

df_train = df_train.drop(cols_to_drop, axis=1)
df_test = df_test.drop(cols_to_drop, axis=1)


cat_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
            'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4','P_emaildomain',
            'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']


%%time
for col in cat_cols:
    if col in df_train.columns:
        le = preprocessing.LabelEncoder()
        le.fit(list(df_train[col].astype(str).values) + list(df_test[col].astype(str).values))
        df_train[col] = le.transform(list(df_train[col].astype(str).values))
        df_test[col] = le.transform(list(df_test[col].astype(str).values))   



%%time
threshold = 0.97
    
# Absolute value correlation matrix
corr_matrix = df_train[df_train['isFraud'].notnull()].corr().abs()

# Getting the upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove.' % (len(to_drop)))
df_train = df_train.drop(columns = to_drop)
df_test = df_test.drop(columns = to_drop)


%%time
X_train = df_train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT'], axis=1)
y_train = df_train.sort_values('TransactionDT')['isFraud']
X_test = df_test.sort_values('TransactionDT').drop(['TransactionDT'], axis=1)


del df_train
df_test = df_test[["TransactionDT"]]

import gc
gc.collect()


#%%time
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.25)
#pipeline_optimizer = TPOTClassifier()
#pipeline_optimizer = TPOTClassifier(generations=3, population_size=10, cv=3,
#                                    random_state=30, verbosity=2,config_dict="TPOT light")
#pipeline_optimizer.fit(X_train, y_train)
#print(pipeline_optimizer.score(X_val, y_val))
#pipeline_optimizer.export('IEEE_Frauds_tpot_exported_pipeline_light.py')



%%time
features = X_train
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, y_train, random_state=30)

imputer = SimpleImputer(strategy="median", missing_values=np.NaN)
imputer.fit(training_features)
training_features = training_features.fillna(training_features.median())#imputer.transform(training_features)
testing_features = testing_features.fillna(testing_features.median())#imputer.transform(testing_features)

# Average CV score on the training set was:0.9739357203229231
exported_pipeline = make_pipeline(
    MaxAbsScaler(),
    DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_leaf=8, min_samples_split=11)
)

exported_pipeline.fit(training_features, training_target)



score_from_training = exported_pipeline.score(testing_features,testing_target)
print("Score= " + str(score_from_training))


testing_feature = X_test
testing_features = testing_feature.fillna(testing_feature.median())#imputer.transform(testing_features)


#------------------ Predict for Submission ---------------------------
results = exported_pipeline.predict(testing_features)


dfIsFraud = pd.DataFrame(data={'TransactionID':X_test.index.values, 'isFraud':results})



dfIsFraud.to_csv('submission_TPOT_DecisionTreeClassifier.csv',index = False)

