# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier,Pool
from sklearn.metrics import roc_curve, auc
from IPython.display import display
import matplotlib.patches as patch
import matplotlib.pyplot as plt
from sklearn.svm import NuSVR
from scipy.stats import norm
from sklearn import svm
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import time
import glob
import sys
import os
import gc


print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))


print(os.listdir("../input/"))


print(os.listdir("../input/santander-customer-transaction-prediction"))


# 데이터 불러오기 Pandas
train= pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')


#데이터 확인
train.shape, test.shape


print(train.info())


print(test.info())


%%time
test.describe()


def missing_check(data):
    tf=data.isna().sum().any()
    if tf==True:
        total = data.isnull().sum()
        percent = (data.isnull().sum()/data.isnull().count()*100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        for col in data.columns:
            dtype = str(data[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)


missing_check(train)


missing_check(test)


sns.countplot(train.target)
plt.title("")


#수치
train['target'].value_counts()


#비율
train.target.value_counts() *100/ train.target.count()


#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist


test, NAlist = reduce_mem_usage(test)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)


train, NAlist = reduce_mem_usage(train)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)


train.info()


test.info()


columns=["target","ID_code"]
X = train.drop(columns,axis=1)
y = train["target"]


X_test  = test.drop("ID_code",axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,test_size=0.5, random_state=1)
rfc = RandomForestClassifier(random_state=0).fit(X_train, y_train)


rfc


import eli5
from eli5.sklearn import PermutationImportance

perm_imp = PermutationImportance(rfc, random_state=1).fit(X_test, y_test)


eli5.show_weights(perm_imp, feature_names = X_test.columns.tolist(), top=200)


Dec_tree = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(X_train, y_train)


features = [c for c in train.columns if c not in ['ID_code', 'target']]


from sklearn import tree
import graphviz
tree_graph = tree.export_graphviz(Dec_tree, out_file=None, feature_names=features)
graphviz.Source(tree_graph)


from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=Dec_tree, dataset= X_train, model_features=features, feature='var_110')

# plot it
pdp.pdp_plot(pdp_goals, 'var_110')
plt.show()


from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=Dec_tree, dataset= X_train, model_features=features, feature='var_81')

# plot it
pdp.pdp_plot(pdp_goals, 'var_81')
plt.show()


logit_clf = LogisticRegression(random_state=42).fit(X_train,y_train)
logit_clf


plt.figure(figsize=(10, 10))
fpr, tpr, thr = roc_curve(y_train, logit_clf.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic Plot', fontsize=20, y=1.05)
auc(fpr, tpr)


cross_val_score(logit_clf, X_train, y_train, scoring='roc_auc', cv=10).mean()


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(X_train, y_train)


plt.figure(figsize=(6, 6))
fpr, tpr, thr = roc_curve(y_train, lda_clf.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic Plot', fontsize=20, y=1.05)
auc(fpr, tpr)


cross_val_score(lda_clf, X_train, y_train, scoring='roc_auc', cv=10).mean()


qda_clf = QuadraticDiscriminantAnalysis()
qda_clf.fit(X_train, y_train)


plt.figure(figsize=(6, 6))
fpr, tpr, thr = roc_curve(y_train, qda_clf.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic Plot', fontsize=20, y=1.05)
auc(fpr, tpr)


cross_val_score(qda_clf, X_train, y_train, scoring='roc_auc', cv=10).mean()


from sklearn.preprocessing import StandardScaler
standardized_train = StandardScaler().fit_transform(train.set_index(['ID_code','target']))
standardized_test = StandardScaler().fit_transform(test.set_index(['ID_code']))
standardized_test = pd.DataFrame(standardized_test, columns=test.set_index(['ID_code']).columns)
standardized_test = standardized_test.join(test[['ID_code']])


X_test = standardized_test.set_index('ID_code').values.astype('float64')
submission = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')

logit_pred = logit_clf.predict_proba(X_test)[:,1]
lda_pred = lda_clf.predict_proba(X_test)[:,1]
qda_pred = qda_clf.predict_proba(X_test)[:,1]


submission = \
submission.join(pd.DataFrame(qda_pred, columns=['target1'])).join(pd.DataFrame(logit_pred, columns=['target2'])).\
join(pd.DataFrame(lda_pred, columns=['target3']))


submission['target'] = (submission.target1 + submission.target2 + submission.target3) / 3


submission.head()


del submission['target1']
del submission['target2']
del submission['target3']


submission.head()


submission.to_csv('logit_lda_qda_mean_ensemble.csv', index=False)

