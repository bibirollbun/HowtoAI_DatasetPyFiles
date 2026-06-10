# data analysis libraries:
import numpy as np
import pandas as pd

# data visualization libraries:
import matplotlib.pyplot as plt
import seaborn as sns

# to ignore warnings:
import warnings
warnings.filterwarnings('ignore')

# to display all columns:
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import numpy as np, pandas as pd, os, gc
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


# Read train and test data with pd.read_csv():
train_id= pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
test_id = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")
train_tr = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
test_tr = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")


train_id.head()


test_id.head()


train_tr.head()


test_tr.head()


train_id.info()


test_id.info()


train_tr.info()


train_tr.info()


test_tr.info()


train=pd.merge(train_tr, train_id, on = "TransactionID",how='left',left_index=True, right_index=True)
train.head()


test=pd.merge(test_tr, test_id, on = "TransactionID",how="left",left_index=True, right_index=True)
test.head()


del train_id, train_tr, test_id, test_tr


train.head()


test.columns=train.columns.drop("isFraud")


for i in range(1,16):
    if i in [1,2,3,5,9]: 
            train['D'+str(i)] =  train['D'+str(i)] - train.TransactionDT/np.float32(24*60*60)
            test['D'+str(i)] = test['D'+str(i)] - test.TransactionDT/np.float32(24*60*60)


for col in train.columns: 
    if sum(train[col].isnull())/float(len(train.index)) > 0.9: 
        del train[col], test[col]


%%time
# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
# WARNING! THIS CAN DAMAGE THE DATA 
def reduce_mem_usage2(df):
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


#train=reduce_mem_usage2(train)
#test=reduce_mem_usage2(test)


train.shape, test.shape


#train.iloc[:,2:21],test.iloc[:,1:20]=agg_func(train.iloc[:,2:21],test.iloc[:,1:20])


test.columns=train.columns.drop("isFraud")


train.shape, test.shape



train.head()


#Ktrain_transaction.loc[:,'V1':'V339'] = pd.DataFrame(EM().complete(np.array(Ktrain_transaction.loc[:,'V1':'V339'])), columns = var_names)



for col in train.loc[:,'V1':'V339']: 
    if sum(train[col].isnull())/float(len(train.index)) > 0.7: 
        del train[col], test[col]


train.head(50)



for i in train.loc[:,'V1':'V321']:   
     train[i]=train[i].fillna(train[i].min()-1)
for i in test.loc[:,'V1':'V321']:
    test[i]=test[i].fillna(train[i].min()-1)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



train.loc[:,'V1':'V100'] = StandardScaler().fit_transform(train.loc[:,'V1':'V100'])



train.loc[:,'V100':'V321'] = StandardScaler().fit_transform(train.loc[:,'V100':'V321'])



#train.loc[:,'V201':'V339'] = StandardScaler().fit_transform(train.loc[:,'V201':'V339'])


test.loc[:,'V1':'V100'] = StandardScaler().fit_transform(test.loc[:,'V1':'V100'])


test.loc[:,'V100':'V321'] = StandardScaler().fit_transform(test.loc[:,'V100':'V321'])


#test.loc[:,'V200':'V339'] = StandardScaler().fit_transform(test.loc[:,'V200':'V339'])


train.head(50)


#optimum bilesen sayisi
pca = PCA().fit(train.loc[:,'V1':'V321'])
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı");


#final
pca = PCA(n_components = 50)
n_v=["V_1","V_2","V_3","V_4","V_5","V_6","V_7","V_8","V_9","V_10","V_11"
     ,"V_12","V_13","V_14","V_15","V_16","V_17","V_18","V_19","V_20","V_21","V_22",
     "V_23","V_24","V_25","V_26","V_27","V_28","V_29","V_30","V_31","V_32","V_33",
     "V_34","V_35","V_36","V_37","V_38","V_39","V_40","V_41","V_42","V_43","V_44",
     "V_45","V_46","V_47","V_48","V_49","V_50",]
pca_fit_train_v = pca.fit_transform(train.loc[:,'V1':'V321'])
pca_fit_test_v = pca.fit_transform(test.loc[:,'V1':'V321'])
train[n_v] = pd.DataFrame(data = pca_fit_train_v)
test[n_v] = pd.DataFrame(data = pca_fit_test_v)
pca.explained_variance_ratio_


del pca_fit_train_v, pca_fit_test_v


train.head()


pca.explained_variance_ratio_.sum()


train=train.drop(train.loc[:,'V1':'V321'],axis=1)
test=test.drop(test.loc[:,'V1':'V321'],axis=1)


train.head()


num_id_colmns=train.loc[:,'id_01':'id_32']._get_numeric_data().columns

for i in num_id_colmns:   
         train[i]=train[i].fillna(train[i].min()-2)
         test[i]=test[i].fillna(train[i].min()-2)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
train[num_id_colmns] = StandardScaler().fit_transform(train[num_id_colmns])
test[num_id_colmns] = StandardScaler().fit_transform(test[num_id_colmns])
#optimum bilesen sayisi
pca = PCA().fit(train[num_id_colmns])
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı");


#final
pca = PCA(n_components = 8)
n_id=['id01', 'id02', 'id03', 'id04', 'id05', 'id06', 'id07', 'id08']
pca_fit_train_v = pca.fit_transform(train[num_id_colmns])
pca_fit_test_v = pca.fit_transform(test[num_id_colmns])
train[n_id] = pd.DataFrame(data = pca_fit_train_v)
test[n_id] = pd.DataFrame(data = pca_fit_test_v)
del pca_fit_train_v, pca_fit_test_v
print(pca.explained_variance_ratio_.sum())
train=train.drop(train[num_id_colmns],axis=1)
test=test.drop(test[num_id_colmns],axis=1)


test.head()


num_id_colmns=train.loc[:,'C1':'C14']._get_numeric_data().columns
for i in num_id_colmns:   
         train[i]=train[i].fillna(train[i].min()-2)
         test[i]=test[i].fillna(train[i].min()-2)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
train[num_id_colmns] = StandardScaler().fit_transform(train[num_id_colmns])
test[num_id_colmns] = StandardScaler().fit_transform(test[num_id_colmns])


#optimum bilesen sayisi
pca = PCA().fit(train[num_id_colmns])
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı");


#final
pca = PCA(n_components = 3)
n_c=["C_1","C_2","C_3"]
pca_fit_train_v = pca.fit_transform(train[num_id_colmns])
pca_fit_test_v = pca.fit_transform(test[num_id_colmns])
train[n_c] = pd.DataFrame(data = pca_fit_train_v)
test[n_c] = pd.DataFrame(data = pca_fit_test_v)
del pca_fit_train_v, pca_fit_test_v
pca.explained_variance_ratio_.sum()


train=train.drop(train[num_id_colmns],axis=1)
test=test.drop(test[num_id_colmns],axis=1)


train.head()


    emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 
          'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',
          'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',     
          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 
          'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',
          'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other',
          'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 
          'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 
          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',
          'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft',
          'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 
          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 
          'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 
          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 
          'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 
          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 
          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',
          'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']

# https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest-579654
for c in ['P_emaildomain',"P_emaildomain"]:
        train[c + '_bin'] = train[c].map(emails)
        test[c + '_bin'] = test[c].map(emails)
    
        train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
        test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])
    
        train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
        test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')


train_cat = train.select_dtypes(include=['object'])
train_cat_columns=train_cat.columns
train_cat_columns
train_columns=train.columns
test_cat = test.select_dtypes(include=['object'])
test_cat_columns=test_cat.columns
del test_cat, train_cat


train=reduce_mem_usage2(train)
test=reduce_mem_usage2(test)


train.shape


# FREQUENCY ENCODE TOGETHER
def encode_FE(df1, df2, cols):
    for col in cols:
        df = pd.concat([df1[col],df2[col]])
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col+'_FE'
        df1[nm] = df1[col].map(vc)
        df1[nm] = df1[nm].astype('float32')
        df2[nm] = df2[col].map(vc)
        df2[nm] = df2[nm].astype('float32')
        print(nm,', ',end='')
        
# LABEL ENCODE
def encode_LE(col,train=train,test=test,verbose=True):
    df_comb = pd.concat([train[col],test[col]],axis=0)
    df_comb,_ = df_comb.factorize(sort=True)
    nm = col
    if df_comb.max()>32000: 
        train[nm] = df_comb[:len(train)].astype('int32')
        test[nm] = df_comb[len(train):].astype('int32')
    else:
        train[nm] = df_comb[:len(train)].astype('int16')
        test[nm] = df_comb[len(train):].astype('int16')
    del df_comb; x=gc.collect()
    if verbose: print(nm,', ',end='')
        
# GROUP AGGREGATION MEAN AND STD
# https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda
def encode_AG(main_columns, uids, aggregations=['mean'], train_df=train, test_df=test, 
              fillna=True, usena=False):
    # AGGREGATION OF MAIN WITH UID FOR GIVEN STATISTICS
    for main_column in main_columns:  
        for col in uids:
            for agg_type in aggregations:
                new_col_name = main_column+'_'+col+'_'+agg_type
                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col,main_column]]])
                if usena: temp_df.loc[temp_df[main_column]==-1,main_column] = np.nan
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()   

                train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')
                test_df[new_col_name]  = test_df[col].map(temp_df).astype('float32')
                
                if fillna:
                    train_df[new_col_name].fillna(-1,inplace=True)
                    test_df[new_col_name].fillna(-1,inplace=True)
                
                print("'"+new_col_name+"'",', ',end='')
                
# COMBINE FEATURES
def encode_CB(col1,col2,df1=train,df2=test):
    nm = col1+'_'+col2
    df1[nm] = df1[col1].astype(str)+'_'+df1[col2].astype(str)
    df2[nm] = df2[col1].astype(str)+'_'+df2[col2].astype(str) 
    encode_LE(nm,verbose=False)
# GROUP AGGREGATION NUNIQUE
def encode_AG2(main_columns, uids, train_df=train, test_df=test):
    for main_column in main_columns:  
        for col in uids:
            comb = pd.concat([train_df[[col]+[main_column]],test_df[[col]+[main_column]]],axis=0)
            mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()
            train_df[col+'_'+main_column+'_ct'] = train_df[col].map(mp).astype('float32')
            test_df[col+'_'+main_column+'_ct'] = test_df[col].map(mp).astype('float32')
            print(col+'_'+main_column+'_ct, ',end='')


for i in range(1,16):
    if i in [1,2,3,5,9]:
        train['D'+str(i)] =  train['D'+str(i)] - train.TransactionDT/np.float32(24*60*60)
        test['D'+str(i)] = test['D'+str(i)] - test.TransactionDT/np.float32(24*60*60) 



encode_CB('card1','addr1')


train['day'] = train.TransactionDT / (24*60*60)
train['uid'] = train.card1_addr1.astype(str)+'_'+np.floor(train.day-train.D1).astype(str)

test['day'] = test.TransactionDT / (24*60*60)
test['uid'] = test.card1_addr1.astype(str)+'_'+np.floor(test.day-test.D1).astype(str)


numeric_columns=test._get_numeric_data().columns.drop('TransactionID','TransactionDT')
numeric_columns=numeric_columns.drop(['TransactionDT',"card1_addr1",'day'])
numeric_columns


categorical_columns=test.columns.drop(test._get_numeric_data().columns)
categorical_columns=categorical_columns.drop('uid')
categorical_columns


encode_FE(train,test,['uid'])
encode_AG(numeric_columns, ['uid'],['mean',"std"], train, test, fillna=True, usena=False)
encode_AG2(categorical_columns, ['uid'], train, test)



# FREQUENCY ENCODE: ADDR1, CARD1, CARD2, CARD3, P_EMAILDOMAIN
encode_FE(train,test,['addr1','card1','card2','card3','P_emaildomain'])
# COMBINE COLUMNS CARD1+ADDR1+P_EMAILDOMAIN
encode_CB('card1_addr1','P_emaildomain')
# FREQUENCY ENOCDE
encode_FE(train,test,['card1_addr1','card1_addr1_P_emaildomain'])
# GROUP AGGREGATE
encode_AG(['TransactionAmt','D9','D11','C_1', 'C_2','C_3', 'id01','id02',"V_1","V_2","V_3"],['card1','card1_addr1','card1_addr1_P_emaildomain'],['mean','std'],usena=True)


# TRANSACTION AMT CENTS
train['cents'] = (train['TransactionAmt'] - np.floor(train['TransactionAmt'])).astype('float32')
test['cents'] = (test['TransactionAmt'] - np.floor(test['TransactionAmt'])).astype('float32')
print('cents, ', end='')


del train['uid'], test['uid']



train.shape, test.shape


train.head()


from sklearn import preprocessing
for i in categorical_columns: 
    lbe=preprocessing.LabelEncoder()
    train[i]=lbe.fit_transform(train[i].astype(str))
    train[i] = train[i].astype('category')
    test[i]=lbe.fit_transform(test[i].astype(str))
    test[i] = test[i].astype('category')


for i in categorical_columns:
    if (test[i].max()== train[i].max())&(train[i].max()<10):
            test = pd.get_dummies(test, columns = [i])
            train=pd.get_dummies(train, columns = [i])



train_TransactionID= train["TransactionID"]
test_TransactionID=test["TransactionID"]
X= train.drop([ 'TransactionDT', 'TransactionID'], axis=1)
y = train.sort_values('TransactionDT')['isFraud']
test = test.drop(['TransactionDT', 'TransactionID'], axis=1)
del train


X=X.drop("isFraud", axis=1)


X.head()


X.shape, test.shape


params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47
         }


import seaborn as sns
import lightgbm as lgb
import gc
from time import time
import datetime
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
warnings.simplefilter('ignore')
sns.set()
%matplotlib inline


folds = TimeSeriesSplit(n_splits=5)

aucs = list()
feature_importances = pd.DataFrame()
feature_importances['feature'] = X.columns

training_start_time = time()
for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):
    start_time = time()
    print('Training on fold {}'.format(fold + 1))
    
    trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])
    clf = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)
    
    feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()
    aucs.append(clf.best_score['valid_1']['auc'])
    
    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
print('-' * 30)
print('Training has finished.')
print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
print('Mean AUC:', np.mean(aucs))
print('-' * 30)


feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(folds.n_splits)]].mean(axis=1)
feature_importances.to_csv('feature_importances.csv')

plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));


# clf right now is the last model, trained with 80% of data and validated with 20%
best_iter = clf.best_iteration


clf = lgb.LGBMClassifier(**params, num_boost_round=best_iter)
clf.fit(X, y)


#set the output as a dataframe and convert to csv file named submission.csv
predictions = clf.predict_proba(test)[:, 1]
output = pd.DataFrame({ "TransactionID" : test_TransactionID, "isFraud": predictions })
output.to_csv('submission_lgbm.csv', index=False)












