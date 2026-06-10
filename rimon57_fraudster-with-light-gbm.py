# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,KFold,train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from category_encoders import CatBoostEncoder, TargetEncoder

from sklearn.linear_model import SGDClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from hyperopt.pyll.stochastic import sample
from hyperopt import STATUS_OK
import csv
import ast
from timeit import default_timer as timer


from time import time
import datetime

from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve,classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


#df_id_tr=pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
#df_id_ts=pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")
df_tran_tr=pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
df_tran_ts=pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")
#df_sample=pd.read_csv("/kaggle/input/ieee-fraud-detection/sample_submission.csv")


#Exploring data about the nature and types with memory consumption
df_tran_tr.info(verbose=True, null_counts=True, memory_usage='deep')


df_tran_ts.info(verbose=True, null_counts=True, memory_usage='deep')


df_tran_tr.describe()


# #making a joint dataframe: tran+id
# df_joint_tr=pd.merge(df_tran_tr,df_id_tr,on='TransactionID')
# df_joint_ts=pd.merge(df_tran_ts, df_id_ts, on="TransactionID")


%matplotlib notebook
%matplotlib inline
plt.figure(figsize=(18,9))
train_full_num = df_tran_tr.filter(regex='isFraud|TransactionDT|TransactionAmt|dist|C|D')
sns.heatmap(train_full_num.isnull(), cbar= False)


%matplotlib notebook
%matplotlib inline
plt.figure(figsize=(18,9))
train_full_num = df_tran_tr.filter(regex='M')
sns.heatmap(train_full_num.isnull(), cbar= False)


# #this can be avoided as most often this gives memory error
# %matplotlib notebook
# %matplotlib inline
# train_full_Vesta = df_tran_tr.filter(regex='V')
# plt.figure(figsize=(18,9))
# sns.heatmap(train_full_Vesta.isnull(), cbar= False)


# %matplotlib notebook
# %matplotlib inline
# train_full_id = df_id_tr.filter(regex='id')
# plt.figure(figsize=(18,9))
# sns.heatmap(train_full_id.isnull(), cbar= False)


#balanced/imbalanced?
%matplotlib notebook
%matplotlib inline
plt.figure(figsize=(10,5))
sns.countplot(x='isFraud',data=df_tran_tr)


# pd.options.display.max_rows=500
# df_joint_tr.info()


#checking each feature count relation with fraud
df_tran_tr.groupby('ProductCD')['isFraud'].value_counts(normalize=True).unstack()[1].sort_values(ascending=False)


df_tran_tr.groupby('P_emaildomain')['isFraud'].value_counts(normalize=True).unstack().fillna(0)[1].sort_values(ascending=False)


df_tran_tr.groupby('R_emaildomain')['isFraud'].value_counts(normalize=True).unstack().fillna(0)[1].sort_values(ascending=False)


df_tran_tr.groupby('card4')['isFraud'].value_counts(normalize=True).unstack()[1].sort_values(ascending=False)


df_tran_tr.groupby('card6')['isFraud'].value_counts(normalize=True).unstack().fillna(0)[1].sort_values(ascending=False)


df_tran_tr.groupby('M8')['isFraud'].value_counts(normalize=True).unstack().dropna()[1].sort_values(ascending=False)


df_tran_tr.groupby('M9')['isFraud'].value_counts(normalize=True).unstack().dropna()[1].sort_values(ascending=False)


df_tran_tr.groupby('M7')['isFraud'].value_counts(normalize=True).unstack().dropna()[1].sort_values(ascending=False)


%matplotlib notebook
%matplotlib inline
def cor_heat(df):
    cor=df.corr()
    plt.figure(figsize=(20,10),dpi=100)
    sns.heatmap(data=cor,annot=True,square=True,linewidths=0.1,cmap='YlGnBu')
    plt.title("Pearson Co-relation: Heat Map")
cor_heat(df_tran_tr.filter(regex='C|isFraud'))


%matplotlib notebook
%matplotlib inline
def cor_heat(df):
    cor=df.corr()
    plt.figure(figsize=(20,7),dpi=100)
    sns.heatmap(data=cor,annot=True,square=True,linewidths=0.1,cmap='YlGnBu')
    plt.title("Pearson Co-relation: Heat Map")
cor_heat(df_tran_tr.filter(regex='Tran|isFraud'))


pd.set_option('display.max_rows',400)
#using absolute to figure out features with higher co-relation irrespective of their +/- value
abs(df_tran_tr.filter(regex='V|isFraud').fillna(0).corr())['isFraud'].sort_values(ascending=False)


abs(df_tran_tr.filter(regex='D[0-9]|isFraud').fillna(0).corr())['isFraud'].sort_values(ascending=False)


abs(df_tran_tr.filter(regex='C|isFraud').fillna(0).corr())['isFraud'].sort_values(ascending=False)[0:11]


abs(df_tran_tr.filter(regex='add|isFraud').fillna(0).corr())['isFraud'].sort_values(ascending=False)


abs(df_tran_tr.filter(regex='dist|isFraud').fillna(0).corr())['isFraud'].sort_values(ascending=False)


abs(df_tran_tr.filter(regex='card1|card2|card3|card5|isFraud').fillna(0).corr())['isFraud'].sort_values(ascending=False)


#This code will save your memory which will be required to run SVC/KNN within allocated memory and grid_search
## This cell is for advanced model building with D features selected from their null counts and correlation values
df_tran_tr=df_tran_tr.filter(regex='V|addr|C2|C8|C12|C1|C4|C10|C11|C6|C7|Tran|card1|card3|card4|card5|card6|ProductCD|P_emaildomain|R_emaildomain|isFraud|D[0-9]')
#dropping these columns due to their very low correlation with target
df_tran_tr=df_tran_tr.drop(['D7','D13','D8'],axis=1)
#df_tran_tr.info(verbose=True, null_counts=True)
df_tran_ts=df_tran_ts.filter(regex='V|addr|C2|C8|C12|C1|C4|C10|C11|C6|C7|Tran|card1|card3|card4|card5|card6|ProductCD|P_emaildomain|R_emaildomain|D[0-9]')
df_tran_ts=df_tran_ts.drop(['D7','D13','D8'],axis=1)
print("Train data set shape: {0}".format(df_tran_tr.shape))
print("Test data set shape: {0}".format(df_tran_ts.shape))


manual_imputation= False
#manual_imputation=True


if manual_imputation==True:
    # use it when you don't want auto imputation for missing values
    #imputing D features missing values with median
    for p in df_tran_tr.filter(regex='D[0-9]'):
        df_tran_tr[p]=df_tran_tr[p].fillna(df_tran_tr[p].median())

    for q in df_tran_ts.filter(regex='D[0-9]'):
        df_tran_ts[q]=df_tran_ts[q].fillna(df_tran_ts[q].median())


if manual_imputation==True:
    # use it when you don't want auto imputation for missing values
    #imputing V features missing values with median
    for x in df_tran_tr.filter(regex='V'):
        df_tran_tr[x]=df_tran_tr[x].fillna(df_tran_tr[x].median())

    for y in df_tran_ts.filter(regex='V'):
        df_tran_ts[y]=df_tran_ts[y].fillna(df_tran_ts[y].median())


# #checking the co-relation after imputations
# pd.set_option('display.max_rows',400)
# abs(df_tran_tr.filter(regex='V|isFraud').fillna(0).corr())['isFraud'].sort_values(ascending=False)


if manual_imputation==True:
    # use it when you don't want auto imputation for missing values
    #filling numerical card features with median
    for a in df_tran_tr.filter(regex='card1|card3|card5'):
        df_tran_tr[a]=df_tran_tr[a].fillna(df_tran_tr[a].median())

    for b in df_tran_ts.filter(regex='card1|card3|card5'):
        df_tran_ts[b]=df_tran_ts[b].fillna(df_tran_ts[b].median())


if manual_imputation==True:
    #filling null C values in test tran dataset with median
    for c in df_tran_ts.filter(regex='C2|C8|C12|C1|C4|C10|C11|C6|C7'):
        df_tran_ts[c]=df_tran_ts[c].fillna(df_tran_ts[c].median())


if manual_imputation==True:
    # use it when you don't want auto imputation for missing values
    #filling null addr values with median
    df_tran_tr.addr1=df_tran_tr.addr1.fillna(df_tran_tr.addr1.median())
    df_tran_tr.addr2=df_tran_tr.addr2.fillna(df_tran_tr.addr2.median())

    df_tran_ts.addr1=df_tran_ts.addr1.fillna(df_tran_ts.addr1.median())
    df_tran_ts.addr2=df_tran_ts.addr2.fillna(df_tran_ts.addr2.median())


cat_encoding="catboost"
#cat_encoding="label"


if cat_encoding=="label":
    # If you want to R_emaildomain in your model
    #dropping null values for R_emaildomain; this is a very critical indicator to fruad; so have to utilize it
    ##df_tran_tr=df_tran_tr.dropna(subset=['R_emaildomain'])
    df_tran_tr['R_emaildomain']=df_tran_tr['R_emaildomain'].fillna('gmail.com')
    #as we can't drop any row from test data, filling it with mode
    df_tran_ts['R_emaildomain']=df_tran_ts['R_emaildomain'].fillna('gmail.com')


if cat_encoding=="label":
    #df_tran_tr=df_tran_tr.dropna(subset=['P_emaildomain'])
    df_tran_tr['P_emaildomain']=df_tran_tr['P_emaildomain'].fillna('gmail.com')
    #as we can't drop any row from test data, filling it with mode
    df_tran_ts['P_emaildomain']=df_tran_ts['P_emaildomain'].fillna('gmail.com')


if cat_encoding=="label":
    #checking max present card4 type to fill in null values
    df_tran_tr.groupby('card4')['isFraud'].value_counts().unstack()
    print(df_tran_tr.card4.mode())
    print(df_tran_ts.card4.mode())


if cat_encoding=="label":
    df_tran_tr.groupby('card6')['isFraud'].value_counts().unstack()
    print(df_tran_tr.card6.mode())
    print(df_tran_ts.card6.mode())


if cat_encoding=="label":
    df_tran_tr.card4=df_tran_tr.card4.fillna('visa')
    df_tran_tr.card6=df_tran_tr.card6.fillna('debit')

    df_tran_ts.card4=df_tran_ts.card4.fillna('visa')
    df_tran_ts.card6=df_tran_ts.card6.fillna('debit')


# for x in df_tran_tr.filter(regex='M'):
#     df_tran_tr[x]=df_tran_tr[x].fillna(df_tran_tr[x].value_counts().index[0])
# for y in df_tran_ts.filter(regex='M'):
#     df_tran_ts[y]=df_tran_ts[y].fillna(df_tran_ts[y].value_counts().index[0])


if cat_encoding=="label":
    #Label Encoding R_emaildomain
    df_tran_tr.R_emaildomain=LabelEncoder().fit_transform(df_tran_tr.R_emaildomain)
    df_tran_ts.R_emaildomain=LabelEncoder().fit_transform(df_tran_ts.R_emaildomain)

    #Label Encoding P_emaildomain
    df_tran_tr.P_emaildomain=LabelEncoder().fit_transform(df_tran_tr.P_emaildomain)
    df_tran_ts.P_emaildomain=LabelEncoder().fit_transform(df_tran_ts.P_emaildomain)

    #Label Encoding ProductCD
    df_tran_tr.ProductCD=LabelEncoder().fit_transform(df_tran_tr.ProductCD)
    df_tran_ts.ProductCD=LabelEncoder().fit_transform(df_tran_ts.ProductCD)

    #Label encoding card features
    df_tran_tr.card4=LabelEncoder().fit_transform(df_tran_tr.card4)
    df_tran_tr.card6=LabelEncoder().fit_transform(df_tran_tr.card6)
    df_tran_ts.card4=LabelEncoder().fit_transform(df_tran_ts.card4)
    df_tran_ts.card6=LabelEncoder().fit_transform(df_tran_ts.card6)


    # for z in df_tran_tr.filter(regex="M"):
    #     df_tran_tr[z]=LabelEncoder().fit_transform(df_tran_tr[z])

    # for a in df_tran_ts.filter(regex="M"):
    #     df_tran_ts[a]=LabelEncoder().fit_transform(df_tran_ts[a])


if cat_encoding=="catboost":
    cat_features=['R_emaildomain','P_emaildomain','ProductCD','card4','card6']

    cbe=CatBoostEncoder(cols=cat_features)
    # X= df_tran_tr.drop(['isFraud'],axis=1)
    # y= df_tran_tr[['isFraud']]
    cbe.fit(df_tran_tr[cat_features],df_tran_tr[['isFraud']])

    # #Train & Test Set transforming
    df_tran_tr=df_tran_tr.join(cbe.transform(df_tran_tr[cat_features]).add_suffix('_target'))
    df_tran_tr.drop(['R_emaildomain','P_emaildomain','ProductCD','card4','card6'],axis=1,inplace=True)

    df_tran_ts=df_tran_ts.join(cbe.transform(df_tran_ts[cat_features]).add_suffix('_target'))
    df_tran_ts.drop(['R_emaildomain','P_emaildomain','ProductCD','card4','card6'],axis=1,inplace=True)



#don't do this before category encoding
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


#Reducing memory without any data loss
df_tran_tr=reduce_mem_usage(df_tran_tr)
df_tran_ts=reduce_mem_usage(df_tran_ts)


X=df_tran_tr.drop(['isFraud'],axis=1)
y=df_tran_tr[['isFraud']]
## as this is an imbalanced problem, we need to stratify the splitting so that training is well distributed
#no need for LGB hyper model selection
# X_train, X_test, y_train, y_test = train_test_split(df_tran_tr.drop(['isFraud'],axis=1), df_tran_tr[['isFraud']], test_size=0.2, random_state=0,stratify=df_tran_tr[['isFraud']])


# this gives memory error; so I will be trying to iterate manually with parameter values
# param_test = {'n_estimators':np.arange(2,100,1), 
#               'max_depth':np.arange(3,10,1),
#               'gamma':np.arange(0,0.5,0.1),
#               'learning_rate':np.arange(0,0.1,0.01),
#              'min_child_weight':np.arange(1,10,1)
#              }


# gs=GridSearchCV(estimator=XGBClassifier(),scoring='roc_auc',param_grid=param_test,cv=3)
# gs.fit(X_train,np.ravel(y_train)) #used ravel to convert to 1-D array
# print("best parameters are: {0} for the best score of {1}".format(gs.best_params_,gs.best_score_))


# hyper_tuning="hyperopt"
hyper_tuning="grid_search"


# #you can try cv using below method
# params_lgb={'boosting_type':'gbdt',
#            'objective': 'binary',
#            'random_state':42}
# k_fold=10
# train_data=lgb.Dataset(X_train,label=y_train)
# validation_data=lgb.Dataset(X_test,label=y_test)
# time_to_train=time()
# lgbmc=lgb.cv(params_lgb,train_data,num_boost_round=10000,nfold=k_fold,metrics='auc',
#              verbose_eval=True, early_stopping_rounds=500)
# print("Training is completed!")
# print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - time_to_train))))
# print('-',30)
# print(lgbmc.best_score_)
# print(lgbmc.best_params)



#look carefully the default parameter; we will circle around the default values to find the optimum ones
LGBMClassifier()


# if hyper_tuning=="hyperopt":
    
#     MAX_EVALS = 500
#     N_FOLDS=5
#     train_set=lgb.Dataset(X,label=y)
#     #Objective function: part-1

#     def objective(params, n_folds = N_FOLDS):
#         """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""

#         # Keep track of evals
#         global ITERATION

#         ITERATION += 1

#         # Retrieve the subsample if present otherwise set to 1.0
#         subsample = params['boosting_type'].get('subsample', 1.0)

#         # Extract the boosting type
#         params['boosting_type'] = params['boosting_type']['boosting_type']
#         params['subsample'] = subsample

#         # Make sure parameters that need to be integers are integers
#         for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
#             params[parameter_name] = int(params[parameter_name])

#         start = timer()

#         # Perform n_folds cross validation
#         cv_results = lgb.cv(params, train_set=train_set, num_boost_round = 10000, nfold = n_folds, 
#                             early_stopping_rounds = 100, metrics = 'auc', seed = 50)

#         run_time = timer() - start

#         # Extract the best score
#         best_score = np.max(cv_results['auc-mean'])

#         # Loss must be minimized
#         loss = 1 - best_score

#         # Boosting rounds that returned the highest cv score
#         n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

#         # Write to the csv file ('a' means append)
#         of_connection = open(out_file, 'a')
#         writer = csv.writer(of_connection)
#         writer.writerow([loss, params, ITERATION, n_estimators, run_time])

#         # Dictionary with information for evaluation
#         return {'loss': loss, 'params': params, 'iteration': ITERATION,
#                 'estimators': n_estimators, 
#                 'train_time': run_time, 'status': STATUS_OK}
    
#     #defining search space: part-2
#     space = {
#         'class_weight': hp.choice('class_weight', [None, 'balanced']),
#         'boosting_type': hp.choice('boosting_type', 
#                                    [{'boosting_type': 'gbdt', 
#                                         'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
#                                      {'boosting_type': 'dart', 
#                                          'subsample': hp.uniform('dart_subsample', 0.5, 1)},
#                                      {'boosting_type': 'goss'}]),
#         'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
#         'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
#         'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
#         'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
#         'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
#         'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
#         'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
#     }
    
#     # boosting type domain 
#     boosting_type = {'boosting_type': hp.choice('boosting_type', 
#                                                 [{'boosting_type': 'gbdt', 'subsample': hp.uniform('subsample', 0.5, 1)}, 
#                                                  {'boosting_type': 'dart', 'subsample': hp.uniform('subsample', 0.5, 1)},
#                                                  {'boosting_type': 'goss', 'subsample': 1.0}])}

#     # Draw a sample
#     params = sample(boosting_type)
    
#     # Retrieve the subsample if present otherwise set to 1.0
#     subsample = params['boosting_type'].get('subsample', 1.0)

#     # Extract the boosting type
#     params['boosting_type'] = params['boosting_type']['boosting_type']
#     params['subsample'] = subsample
    
    
#     # File to save first results
#     out_file = 'gbm_trials.csv'
#     of_connection = open(out_file, 'w')
#     writer = csv.writer(of_connection)

#     # Write the headers to the file
#     writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
#     of_connection.close()



#     # Algorithm: part-3
#     tpe_algorithm = tpe.suggest
#     # Trials object to track progress
#     bayes_trials = Trials()
    



# %%capture
# #capture results: part-4
# # Global variable
# global  ITERATION

# ITERATION = 0

# # Run optimization
# best = fmin(fn = objective, space = space, algo = tpe.suggest, 
#             max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(50),
#             verbose=50,show_progressbar=True)

# results = pd.read_csv('gbm_trials.csv')

# # Sort with best scores on top and reset index for slicing
# results.sort_values('loss', ascending = True, inplace = True)
# results.reset_index(inplace = True, drop = True)
# results.head()



# # Convert from a string to a dictionary
# ast.literal_eval(results.loc[0, 'params'])

# # Extract the ideal number of estimators and hyperparameters
# best_bayes_estimators = int(results.loc[0, 'estimators'])
# best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()
# print("Best Estimator is:{}".format(best_bayes_estimators))
# print("-"* 30)
# print("Best Parameters are {}".format(best_bayes_params))


k_fold=5
kf=StratifiedKFold(n_splits=k_fold,shuffle=True, random_state=42)
if hyper_tuning=="grid_search":
    
    params_lgb_grid={
    'boosting_type': ['gbdt', 'dart'],'num_leaves': list(range(30, 150)),
    'learning_rate': list(np.logspace(np.log(0.005), np.log(0.2), base = np.exp(1), num = 1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),'reg_lambda': list(np.linspace(0, 1)),'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'n_estimators':list(np.arange(100,2000,50))
    }
    
    lgb_estimator=LGBMClassifier(objective='binary',random_state=42,max_depth=-1,num_boost_rounds=1000)
    rs=RandomizedSearchCV(estimator=lgb_estimator,scoring='roc_auc',param_distributions=params_lgb_grid,cv=kf,verbose=100, n_iter=20,n_jobs=2)
    rs.fit(X,np.ravel(y)) #used ravel to convert to 1-D array
    bs_score=rs.best_score_
    bs_params=rs.best_params_
    bs_est=rs.best_estimator_
    print("best parameters are: {0} for the best score of {1}".format(rs.best_params_,rs.best_score_))
    print("-"*30)
    print("best estimator is:{}".format(rs.best_estimator_))


# k_fold=5
# kf=StratifiedKFold(n_splits=k_fold,shuffle=True, random_state=42)
# training_start_time = time()
# aucs=[]
# for fold, (trn_idx,val_idx) in enumerate(kf.split(X,y)):
#     start_time = time()
#     print('Training on fold {}'.format(fold + 1))
#     trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
#     val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])
#     clf = lgb.train(params_lgb, trn_data, num_boost_round=10000, valid_sets = [trn_data, val_data], 
#                     verbose_eval=200, early_stopping_rounds=100)
#     aucs.append(clf.best_score['valid_1']['auc'])
#     print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
# print('-' * 30)
# print('Training is completed!.')
# print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
# print(clf.best_params_)
# print('-' * 30)



# def testing_model(my_model):
#     my_model.fit(X_train,np.ravel(y_train))
#     my_model_pred=my_model.predict(X_test)
#     print("accuracy score is {0}% and roc 
#     print(classification_report(y_test,my_model_pred))
#     print(my_model)


# xgbc_test=XGBClassifier(gamma=0.5,learning_rate=0.03,max_depth=9,n_estimators=200,min_child_weight=3, reg_alpha=0.005)
# testing_model(xgbc_test)


print(df_tran_tr.shape)
print(df_tran_ts.shape)


# #saving the model parameters
# params=clf.best_params_
# best_iter=clf.best_iteration


# tuned_lgb=LGBMClassifier(boosting_type='gbdt',class_weight='balanced', colsample_bytree=0.8970523178797932,
#                learning_rate=0.8970523178797932, min_child_samples=45,n_estimators=3323, n_jobs=-1, 
#                 num_leaves=138, objective='binary',random_state=42, reg_alpha=0.5005020213127344, 
#                 reg_lambda= 0.45121616279208887, silent=True,
#                subsample=0.6614743075688195, subsample_for_bin=260000)
tuned_lgb=LGBMClassifier(objective='binary',random_state=42,**bs_params)


def model_output(your_model):
    #training the model with inp and out df
    your_model.fit(df_tran_tr.drop(['isFraud'],axis=1),np.ravel(df_tran_tr[['isFraud']]))
    your_model_pred= your_model.predict_proba(df_tran_ts)[:,1]
    your_model_df= pd.DataFrame({'TransactionID':df_tran_ts['TransactionID'],'isFraud': your_model_pred.astype(float)})
    your_model_df.to_csv('submission_fraud.csv',index=False)

rdf_model=RandomForestClassifier(warm_start=True)
xgb_model=XGBClassifier()
nbg_model=GaussianNB()
mplc_model=MLPClassifier()
adb_model= AdaBoostClassifier()
gbb_model=GradientBoostingClassifier()
svc_model= SVC()
knn_model=KNeighborsClassifier()
sgd_model=SGDClassifier()
lgbmc_model= LGBMClassifier()
catb_model=CatBoostClassifier(eval_metric = 'AUC')
model_output(tuned_lgb)




