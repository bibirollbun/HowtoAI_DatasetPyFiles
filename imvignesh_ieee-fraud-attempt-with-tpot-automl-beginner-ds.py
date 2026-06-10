import os
import gc
import itertools

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score



%%time
train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print(train.shape)
print(test.shape)


def derive_hour_feature(df,tname):
    """
    Creates an hour of the day feature, encoded as 0-23. 
    Parameters: 
        df : pd.DataFrame
            df to manipulate.
        tname : str
            Name of the time column in df.
    """
    hours = df[tname] / (3600)        
    encoded_hours = np.floor(hours) % 24
    return encoded_hours

train['hours'] = derive_hour_feature(train,'TransactionDT')
test['hours'] = derive_hour_feature(test,'TransactionDT')



del train_transaction, train_identity, test_transaction, test_identity


X_train = train.drop(['TransactionDT'], axis=1)
X_test = test.drop(['TransactionDT'], axis=1)


del train, test


total = X_train.isnull().sum().sort_values(ascending=False)
percent = (X_train.isnull().sum()/X_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


# notuseful_features = missing_data[missing_data['Percent']>0.80]


num_fs = X_train.dtypes[X_train.dtypes != "object"].index
print("Number of Numerical features: ", len(num_fs))

cat_fs = X_train.dtypes[X_train.dtypes == "object"].index
print("Number of Categorical features: ", len(cat_fs))


# n = X_train.select_dtypes(include=object)
# for col in n.columns:
#     print(col, ':  ', X_train[col].unique())


## Let's see the distribuition of the categories: 
# for cat in list(cat_fs):
#     print('Distribuition of feature:', cat)
#     print(X_train[cat].value_counts(normalize=True))
#     print('#'*50)


# Seaborn visualization library
# import seaborn as sns
# Create the default pairplot
# sns.pairplot(X_train, hue = 'isFraud')


y_train = X_train['isFraud']
X_train.drop(['isFraud'], axis=1, inplace = True)
y_pred = sample_submission


# X_train = X_train.fillna(-999)
# X_test = X_test.fillna(-999)


X_train.columns, X_test.columns, y_train.shape


num_feats=len(cat_fs)+len(num_fs)-1
print(num_feats)


def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature
#cor_support, cor_feature = cor_selector(X_train, y_train,num_feats)
#print(str(len(cor_feature)), 'selected features')


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
X_norm = MinMaxScaler().fit_transform(X_train)
chi_selector = SelectKBest(chi2, k=num_feats)
chi_selector.fit(X_norm, y_train)
chi_support = chi_selector.get_support()
chi_feature = X_train.loc[:,chi_support].columns.tolist()
print(str(len(chi_feature)), 'selected features')


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
rfe_selector.fit(X_norm, y_train)
rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1"), max_features=num_feats)
embeded_lr_selector.fit(X_norm, y_train)

embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)), 'selected features')


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
embeded_rf_selector.fit(X_train, y_train)

embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features')


# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,
                                    'Random Forest':embeded_rf_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)


# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))   


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


#%%time
#X_train = reduce_mem_usage(X_train)


#X_test = reduce_mem_usage(X_test)#


X_train.shape, X_test.shape, y_train.shape, y_pred.shape


%%time
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train,train_size=0.90, test_size=0.10)


%%time
from tpot import TPOTClassifier
tpot = TPOTClassifier(generations=5, population_size=5, verbosity=2,cv=5, scoring='roc_auc', warm_start=True, early_stop=5 )
tpot.fit(X_tr, y_tr)


%%time
print("ROC_AUC is {}%".format(tpot.score(X_te, y_te)*100))


%%time
preds = tpot.predict(X_test)


%%time
preds_probab = tpot.predict_proba(X_test)


sample_submission['isFraud'] = '0'
sample_submission['isFraud'] = preds
sample_submission.to_csv('TPOT_automl_submission_pred_3.csv', index=True)


sample_submission['isFraud'] = '0'
sample_submission['isFraud'] = 1.000000 - preds_probab
sample_submission.to_csv('TPOT_automl_submission_probab_4.csv', index=True)

