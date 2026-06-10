import numpy as np
import pandas as pd
import os
import math
from scipy import stats
import time

import matplotlib.pyplot as plt
%matplotlib inline
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import metrics
from sklearn import linear_model
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import eli5
import shap
from IPython.display import HTML
import json
import altair as alt

import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline


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

import pandas as pd
test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')


def check_dataframe(df):
    
    # excution time
    start_time = time.time()

    dict_check_value = {}
    list_check_value = []
    total_rows = len(df)
    for col in df.columns:
    #     print(col)
    
        value_count_base = df[col].value_counts()
        value_count_index = list(value_count_base.index)
        value_count_value = list(value_count_base.values)
        null_value = df[col].isnull().sum()
        unique_value = len(value_count_index)
        unique_value_exam = value_count_index[:5]
#         unique_value_exam = df[col][~df[col].isnull()].unique()
#         unique_value = len(unique_value_exam)
#         unique_value_exam = unique_value_exam[:5]
        value_type = df[col].dtype
    
        # include null value
        in_null_entropy = round(stats.entropy(df[col].value_counts()/total_rows), 4)
        
        # except null value
        except_null_entropy = round(stats.entropy(df[col].value_counts()/total_rows-null_value), 4)


        list_check_value = [unique_value, null_value, unique_value_exam, value_type, in_null_entropy, except_null_entropy]


        dict_check_value[col] = list_check_value
    
    new_df = pd.DataFrame.from_dict(dict_check_value, orient='index', columns=['uniques', 'missing', 'values_exam_top5', 'dtypes', 'total_entropy', 'exp_null_entropy'])
    
    print(f'The Excution Time(minutes) is {(time.time()-start_time)/60}')
    return new_df


# temp_df = train_transaction[:1000].copy(deep=True)



train_transaction_chk = check_dataframe(train_transaction)
train_transaction_chk


'....'.join([str(k)+" Data Types: "+str(v)+'' for (k, v) in train_transaction_chk.groupby('dtypes').size().to_dict().items()])



# Object Data Types


# basic_df 
# cross_check_df
# columns
# target columns


def check_obj_col(df, column):
    """
    # From: https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt
    parameter : df, column
    df(dataframe): Reference Dataframe 
    column(string): column to be based on
    """
    total = len(df)
    df[col] = df[column].fillna("Miss")
    cross_df = pd.crosstab(df[column], df['isFraud'], normalize='index')*100
    cross_df = cross_df.reset_index()
    cross_df.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    plt.figure(figsize=(14,10))
    subtile_str = '{} Distributions'.format(column)
    plt.suptitle(subtile_str, fontsize=22)
    plt.subplot(221)
    g = sns.countplot(x=column, data=df)
    
    g.set_title(subtile_str, fontsize=19)
    g.set_xlabel("{} Name".format(column), fontsize=17)
    g.set_ylabel("Count", fontsize=17)
    g.set_ylim(0,500000)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center", fontsize=14)
        
    plt.subplot(222)
    g1 = sns.countplot(x=column, hue='isFraud', data=df)
    plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])
    gt = g1.twinx()
    
    order_xcol = [t.get_text()  for t in g.get_xticklabels()]
    
    gt = sns.pointplot(x=column, y='Fraud', data=cross_df, color='black', order=order_xcol, legend=False)
    gt.set_ylabel("% of Fraud Transactions", fontsize=16)

    g1.set_title("{} by Target(isFraud)".format(column), fontsize=19)
    g1.set_xlabel("{} Name".format(column), fontsize=17)
    g1.set_ylabel("Count", fontsize=17)

    plt.subplot(212)
    g3 = sns.boxenplot(x=column, y='TransactionAmt', hue='isFraud', 
                  data=df[df['TransactionAmt'] <= 2000] )
    g3.set_title("Transaction Amount Distribuition by {} and Target".format(column), fontsize=20)
    g3.set_xlabel("{} Name".format(column), fontsize=17)
    g3.set_ylabel("Transaction Values", fontsize=17)
    
    plt.subplots_adjust(hspace = 0.6, top = 0.85)
    
    return plt.show()


object_col = [col for col in list(train_transaction_chk[train_transaction_chk['dtypes']=='object'].index) if col not in ['P_emaildomain', 'R_emaildomain']]
# print(object_col) # unique 개수가 너무 많은 컬럼 제외

# check_obj_col(train_transaction, column='ProductCD')

# From: https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt
for col in object_col:
    check_obj_col(train_transaction, column=col)
    
    
print(object_col)


train_transaction_chk[(train_transaction_chk.index == 'P_emaildomain') | (train_transaction_chk.index == 'R_emaildomain')]
# num_col_chk = [col for col in train_transaction_chk[train_transaction_chk['dtypes']!='object'].index if col not in ['TransactionID', 'isFraud', 'TransactionDT']]


train_transaction[['P_emaildomain', 'R_emaildomain']].head()
# NaN, NaN
# value, NaN
# value, value
# NaN, value


total = len(train_transaction)
print(f'p_emaildomain은 {(total-94456)/total}%로 값이 존재')
print(f'r_emaildomain은 {(total-453249)/total}%로 값이 존재')




# P is Null, R is Not Null
train_transaction[(train_transaction['P_emaildomain'].isnull()) & (train_transaction['R_emaildomain'].notnull())].groupby('isFraud').size()
p_null_r_notnull = np.array(train_transaction[(train_transaction['P_emaildomain'].isnull()) & (train_transaction['R_emaildomain'].notnull())].groupby('isFraud').size())

train_transaction[(train_transaction['P_emaildomain'].notnull()) & (train_transaction['R_emaildomain'].isnull())].groupby('isFraud').size()
p_notnull_r_null = np.array(train_transaction[(train_transaction['P_emaildomain'].notnull()) & (train_transaction['R_emaildomain'].isnull())].groupby('isFraud').size())

train_transaction[(train_transaction['P_emaildomain'].isnull()) & (train_transaction['R_emaildomain'].isnull())].groupby('isFraud').size()
p_null_r_null = np.array(train_transaction[(train_transaction['P_emaildomain'].isnull()) & (train_transaction['R_emaildomain'].isnull())].groupby('isFraud').size())

train_transaction[(train_transaction['P_emaildomain'].notnull()) & (train_transaction['R_emaildomain'].notnull())].groupby('isFraud').size()
p_notnull_r_notnull = np.array(train_transaction[(train_transaction['P_emaildomain'].notnull()) & (train_transaction['R_emaildomain'].notnull())].groupby('isFraud').size())

emaildomain_df = pd.DataFrame(np.stack((p_null_r_notnull, p_notnull_r_null, p_null_r_null, p_notnull_r_notnull)), columns=['Normal', 'isFraud'], index=['p_null_r_notnull', 'p_notnull_r_null', 'p_null_r_null', 'p_notnull_r_notnull'])

emaildomain_df['Normal_ratio'], emaildomain_df['isFraud_ratio'] = list(zip(*emaildomain_df.apply(lambda x: (x['Normal']/total, x['isFraud']/total), axis=1)))
emaildomain_df['isFraud_ratio'] = emaildomain_df.apply(lambda x: x['isFraud']/(x['Normal']+x['isFraud']), axis=1)

fraud_total = len(train_transaction[train_transaction['isFraud']==1])
emaildomain_df['fraudRatio_per_Fraud'] = emaildomain_df.apply(lambda x: x['isFraud']/fraud_total, axis=1)
emaildomain_df


# from https://www.kaggle.com/jesucristo/fraud-complete-eda
fig, ax = plt.subplots(1, 3, figsize=(32,10))

sns.countplot(y="P_emaildomain", ax=ax[0], data=train_transaction)
ax[0].set_title('P_emaildomain', fontsize=14)
sns.countplot(y="P_emaildomain", ax=ax[1], data=train_transaction.loc[train_transaction['isFraud'] == 1])
ax[1].set_title('P_emaildomain isFraud = 1', fontsize=14)
sns.countplot(y="P_emaildomain", ax=ax[2], data=train_transaction.loc[train_transaction['isFraud'] == 0])
ax[2].set_title('P_emaildomain isFraud = 0', fontsize=14)
plt.show()


# from https://www.kaggle.com/jesucristo/fraud-complete-eda
fig, ax = plt.subplots(1, 3, figsize=(32,10))

sns.countplot(y="R_emaildomain", ax=ax[0], data=train_transaction)
ax[0].set_title('R_emaildomain', fontsize=14)
sns.countplot(y="R_emaildomain", ax=ax[1], data=train_transaction.loc[train_transaction['isFraud'] == 1])
ax[1].set_title('R_emaildomain isFraud = 1', fontsize=14)
sns.countplot(y="R_emaildomain", ax=ax[2], data=train_transaction.loc[train_transaction['isFraud'] == 0])
ax[2].set_title('R_emaildomain isFraud = 0', fontsize=14)
plt.show()


train_transaction[(train_transaction['P_emaildomain'].notnull()) & (train_transaction['R_emaildomain'].notnull()) & (train_transaction['isFraud']==1)].apply(lambda x: (x['P_emaildomain']+'-'+x['R_emaildomain']), axis=1).value_counts()


train_transaction[(train_transaction['P_emaildomain'].notnull()) & (train_transaction['R_emaildomain'].notnull()) & (train_transaction['isFraud']==0)].apply(lambda x: (x['P_emaildomain']+'-'+x['R_emaildomain']), axis=1).value_counts()


# https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt 
def ploting_cnt_amt(df, col, lim=2000):
    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
    
    plt.figure(figsize=(16,14))    
    plt.suptitle(f'{col} Distributions ', fontsize=24)
    
    plt.subplot(211)
    g = sns.countplot( x=col,  data=df, order=list(tmp[col].values))
    gt = g.twinx()
    gt = sns.pointplot(x=col, y='Fraud', data=tmp, order=list(tmp[col].values),
                       color='black', legend=False, )
    gt.set_ylim(0,tmp['Fraud'].max()*1.1)
    gt.set_ylabel("%Fraud Transactions", fontsize=16)
    g.set_title(f"Most Frequent {col} values and % Fraud Transactions", fontsize=20)
    g.set_xlabel(f"{col} Category Names", fontsize=16)
    g.set_ylabel("Count", fontsize=17)
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    sizes = []
    for p in g.patches:
        height = p.get_height()
        sizes.append(height)
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center",fontsize=12) 
        
    g.set_ylim(0,max(sizes)*1.15)
    
    #########################################################################
    perc_amt = (df.groupby(['isFraud',col])['TransactionAmt'].sum() \
                / df.groupby([col])['TransactionAmt'].sum() * 100).unstack('isFraud')
    perc_amt = perc_amt.reset_index()
    perc_amt.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
    amt = df.groupby([col])['TransactionAmt'].sum().reset_index()
    perc_amt = perc_amt.fillna(0)
    plt.subplot(212)
    g1 = sns.barplot(x=col, y='TransactionAmt', 
                       data=amt, 
                       order=list(tmp[col].values))
    g1t = g1.twinx()
    g1t = sns.pointplot(x=col, y='Fraud', data=perc_amt, 
                        order=list(tmp[col].values),
                       color='black', legend=False, )
    g1t.set_ylim(0,perc_amt['Fraud'].max()*1.1)
    g1t.set_ylabel("%Fraud Total Amount", fontsize=16)
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    g1.set_title(f"{col} by Transactions Total + %of total and %Fraud Transactions", fontsize=20)
    g1.set_xlabel(f"{col} Category Names", fontsize=16)
    g1.set_ylabel("Transaction Total Amount(U$)", fontsize=16)
    g1.set_xticklabels(g.get_xticklabels(),rotation=45)    
    
    for p in g1.patches:
        height = p.get_height()
        g1.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total_amt*100),
                ha="center",fontsize=12) 
        
    plt.subplots_adjust(hspace=.4, top = 0.9)
    plt.show()


# https://www.kaggle.com/c/ieee-fraud-detection/discussion/100400#latest-579480
import datetime

START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
train_transaction["Date"] = train_transaction['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

train_transaction['_Hours'] = train_transaction['Date'].dt.hour

train_transaction['_Weekdays'] = train_transaction['Date'].dt.dayofweek


# total_amt = train_transaction.groupby(['isFraud'])['TransactionAmt'].sum().sum()
total_amt = train_transaction['TransactionAmt'].sum()
ploting_cnt_amt(train_transaction, '_Hours')


ploting_cnt_amt(train_transaction, '_Weekdays')


train_transaction.drop(['_Weekdays', 'Date'], axis=1, inplace=True)


# print(train_transaction_chk[train_transaction_chk['dtypes']=!'object'])
print('Numeric Columns {} in train_transaction'.format(len(train_transaction.select_dtypes(['float', 'int']).columns)))


list(train_transaction.select_dtypes(['float', 'int']).columns)
# TransactionID, isFraud, TransactionDT, TransactionAmt,_Hours 제외


eda_col_list = [col for col in list(train_transaction.select_dtypes(['float', 'int']).columns) if col not in ['TransactionID', 'isFraud', 'TransactionDT', '_Hours', 'TransactionAmt']]
# sns.pairplot(train_transaction[eda_col_list], kind="scatter", hue='type', corner=True)


train_transaction_chk.loc[eda_col_list, :] # dfObj.loc[ 'b' , : ]


train_transaction_chk.loc[eda_col_list, :].apply(lambda x: (total-x['missing'])/total, axis=1)


# This code from https://www.kaggle.com/nroman/eda-for-cis-fraud-detection
# def plot_numerical(feature):
#     """
#     Plot some information about a numerical feature for both train and test set.
#     Args:
#         feature (str): name of the column in DataFrame
#     """
#     fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 18))
#     sns.kdeplot(train[feature], ax=axes[0][0], label='Train');
#     sns.kdeplot(test[feature], ax=axes[0][0], label='Test');

#     sns.kdeplot(train[train['isFraud']==0][feature], ax=axes[0][1], label='isFraud 0')
#     sns.kdeplot(train[train['isFraud']==1][feature], ax=axes[0][1], label='isFraud 1')

#     test[feature].index += len(train)
#     axes[1][0].plot(train[feature], '.', label='Train');
#     axes[1][0].plot(test[feature], '.', label='Test');
#     axes[1][0].set_xlabel('row index');
#     axes[1][0].legend()
#     test[feature].index -= len(train)

#     axes[1][1].plot(train[train['isFraud']==0][feature], '.', label='isFraud 0');
#     axes[1][1].plot(train[train['isFraud']==1][feature], '.', label='isFraud 1');
#     axes[1][1].set_xlabel('row index');
#     axes[1][1].legend()

#     pd.DataFrame({'train': [train[feature].isnull().sum()], 'test': [test[feature].isnull().sum()]}).plot(kind='bar', rot=0, ax=axes[2][0]);
#     pd.DataFrame({'isFraud 0': [train[(train['isFraud']==0) & (train[feature].isnull())][feature].shape[0]],
#                   'isFraud 1': [train[(train['isFraud']==1) & (train[feature].isnull())][feature].shape[0]]}).plot(kind='bar', rot=0, ax=axes[2][1]);

#     fig.suptitle(feature, fontsize=18);
#     axes[0][0].set_title('Train/Test KDE distribution');
#     axes[0][1].set_title('Target value KDE distribution');
#     axes[1][0].set_title('Index versus value: Train/Test distribution');
#     axes[1][1].set_title('Index versus value: Target distribution');
#     axes[2][0].set_title('Number of NaNs');
#     axes[2][1].set_title('Target value distribution among NaN values');
    

def plot_numerical(train, test, feature):
    """
    Fix a little code from https://www.kaggle.com/nroman/eda-for-cis-fraud-detection
    Plot some information about a numerical feature for both train and test set.
    Args:
        train : train DataFrame
        test : test Dataframe
        feature (str): name of the column in DataFrame
    """
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 18))
    sns.kdeplot(train[feature], ax=axes[0][0], label='Train');
    sns.kdeplot(test[feature], ax=axes[0][0], label='Test');

    sns.kdeplot(train[train['isFraud']==0][feature], ax=axes[0][1], label='isFraud 0')
    sns.kdeplot(train[train['isFraud']==1][feature], ax=axes[0][1], label='isFraud 1')

    test[feature].index += len(train)
    axes[1][0].plot(train[feature], '.', label='Train');
    axes[1][0].plot(test[feature], '.', label='Test');
    axes[1][0].set_xlabel('row index');
    axes[1][0].legend()
    test[feature].index -= len(train)

    axes[1][1].plot(train[train['isFraud']==0][feature], '.', label='isFraud 0');
    axes[1][1].plot(train[train['isFraud']==1][feature], '.', label='isFraud 1');
    axes[1][1].set_xlabel('row index');
    axes[1][1].legend()

    pd.DataFrame({'train': [train[feature].isnull().sum()], 'test': [test[feature].isnull().sum()]}).plot(kind='bar', rot=0, ax=axes[2][0]);
    pd.DataFrame({'isFraud 0': [train[(train['isFraud']==0) & (train[feature].isnull())][feature].shape[0]],
                  'isFraud 1': [train[(train['isFraud']==1) & (train[feature].isnull())][feature].shape[0]]}).plot(kind='bar', rot=0, ax=axes[2][1]);

    fig.suptitle(feature, fontsize=18);
    axes[0][0].set_title('Train/Test KDE distribution');
    axes[0][1].set_title('Target value KDE distribution');
    axes[1][0].set_title('Index versus value: Train/Test distribution');
    axes[1][1].set_title('Index versus value: Target distribution');
    axes[2][0].set_title('Number of NaNs');
    axes[2][1].set_title('Target value distribution among NaN values');
    


# https://stackoverflow.com/questions/11350770/select-by-partial-string-from-a-pandas-dataframe
# Error가 나므로 이후 수정 필요
for col in [col for col in train_transaction.columns if ('card' in col) & (col not in ['card4', 'card6'])]:
    print(col)
    try:
        plot_numerical(train_transaction, test_transaction, col)
    except ValueError as e:
        print(e, col)
        continue


plot_numerical(train_transaction, test_transaction, 'card1')
# https://www.kaggle.com/c/ieee-fraud-detection/discussion/100340#latest-578626


plot_numerical(train_transaction, test_transaction, 'card2')


plot_numerical(train_transaction, test_transaction, 'card5')


# Target value에 따른, Feature의 isnull 값의 비율

# temp_cols = [col for col in train_transaction.columns if ('card' in col) & (col not in ['card4', 'card6'])]

temp_arr = np.empty([0, 3])

# np.array(pd.crosstab(train_transaction['card5'].fillna('missing'), train_transaction['isFraud'].fillna('missing'), margins=True).loc['missing', :])
for col in train_transaction.columns:
    print(col)
    try:
        new_arr = np.array(pd.crosstab(train_transaction[col].fillna('missing'), train_transaction['isFraud'].fillna('missing'), margins=True).loc['missing', :])
    except KeyError:
        new_arr = np.array([0, 0, 0])
#     print(pd.crosstab(train_transaction[col].fillna('missing'), train_transaction['isFraud'].fillna('missing'), margins=True))
#     print(np.array(pd.crosstab(train_transaction[col].fillna('missing'), train_transaction['isFraud'].fillna('missing'), margins=True).loc['missing', :]))
    temp_arr = np.vstack((temp_arr, new_arr))

null_df = pd.DataFrame(temp_arr, columns=['isFraud0', 'isFraud1', 'Null_Total'], index=train_transaction.columns)


import math
def fraudNull_perTotalNull(a, b):
    val = a/b*100
    if math.isnan(val):
        val = 0
    return val
# null_df.apply(lambda x: x['isFraud1']/x['Null_Total']*100, axis=1)
null_df['fraudNull_perTotalNull'] = null_df.apply(lambda x: fraudNull_perTotalNull(x['isFraud1'], x['Null_Total']), axis=1)


# null값은 의미가 있을까?
# Null값중 Fraud가 차지하는 비율
null_df['fraudNull_perTotalFraud'] = null_df.apply(lambda x: np.ceil(x['isFraud1']/fraud_total*100), axis=1)


train_transaction_chk = pd.concat([null_df[['isFraud1', 'fraudNull_perTotalNull', 'fraudNull_perTotalFraud']], train_transaction_chk], axis=1)


train_transaction_chk.head()


train_transaction_chk


[col for col in train_transaction.columns if ('C' in col) & ('ProductCD' not in col)]


plot_numerical(train_transaction, test_transaction, 'C1')


plot_numerical(train_transaction, test_transaction, 'C2')


plot_numerical(train_transaction, test_transaction, 'C6')


plot_numerical(train_transaction, test_transaction, 'C9')


plot_numerical(train_transaction, test_transaction, 'C11')


plot_numerical(train_transaction, test_transaction, 'C13')


plot_numerical(train_transaction, test_transaction, 'C14')


for col in [col for col in train_transaction.columns if ('D' in col) & (col not in ['TransactionID', 'TransactionDT', 'ProductCD', 'Date'])]:
    try:
        plot_numerical(train_transaction, test_transaction, col)
    except ValueError:
        print("{} Could not convert data to an integer.".format(col))
        continue
    except Exception as e:
        print("error", e)
        continue















































# Set columns
eda_col_list = [col for col in list(train.columns) if col not in ['id', 'fiberID', 'type']]

# plot
fig, ax = plt.subplots(figsize=(12, 15), sharex=True)
sns.despine(left=True)

num_fig = len(eda_col_list)
ncols_fig = 3
nrows_fig = math.ceil(num_fig/ncols_fig)

gs = gridspec.GridSpec(nrows_fig, ncols_fig)

fig.legend([plt.plot([], [], c=c)[0] for c in new_colors], unique_labels, loc='upper right', bbox_to_anchor=(1.2, 0.5))


for i, n in enumerate(range(num_fig)):
  # if i < 5:
    ax = fig.add_subplot(gs[n])

  # Plot a kernel density estimate and rug plot
    sns.distplot(df[eda_col_list[n]], hist=False, rug=True)

    ax.set_title(eda_col_list[n])
  # else:
    # break
plt.tight_layout()
plt.show()


# Set columns
eda_col_list = [col for col in list(train.columns) if col not in ['type']]

# Box Plot
num_fig = len(eda_col_list)
ncols_fig = 4
nrows_fig = math.ceil(num_fig/4)
gs = gridspec.GridSpec(nrows_fig, ncols_fig)
fig, axs = plt.subplots(figsize=(15,20))
red_square = dict(markerfacecolor='r', marker='s')

for n in range(num_fig):
    ax = fig.add_subplot(gs[n])
    ax.boxplot(df[eda_col_list[n]], flierprops=red_square, notch='True',patch_artist=True)
    ax.set_title(eda_col_list[n])


import itertools
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
import numpy as np

# ! pip3 install colorspacious
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict


eda_col_list = [col for col in train.columns if col not in ['fiberID', 'type', 'id']]

print(eda_col_list)

unique_labels = list(train['type'].unique())

# for Color
# https://stackoverflow.com/questions/53283813/legend-in-separate-subplot-and-grid
label2idx = {val: i for i, val in enumerate(unique_labels)}
new_colors = ['C'+str(label2idx[label]) for label in unique_labels]

# plot
fig, ax = plt.subplots(figsize=(12, 15))

num_fig = len(eda_col_list)
ncols_fig = 3
nrows_fig = math.ceil(num_fig/ncols_fig)

# gs = gridspec.GridSpec(rows, cols)
gs = gridspec.GridSpec(nrows_fig, ncols_fig)
# figure = plt.figure()
# plt.clf()

fig.legend([plt.plot([], [], c=c)[0] for c in new_colors], unique_labels, loc='upper right')


for i, n in enumerate(range(num_fig)):
  # if i < 5:
  ax = fig.add_subplot(gs[n])
      ax.scatter(train.id, train[eda_col_list[n]], c=['C'+str(label2idx[label]) for label in train.type.values], cmap=plt.cm.RdYlGn)
  # ax.text(train.id+.03, train[eda_col_list[n]]+.03, train['type'])
  ax.set_title(eda_col_list[n])
  # else:
  #   break
plt.show()


colormap = plt.cm.viridis
plt.figure(figsize=(25,25))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


len(train_transaction.columns)


# len(train_transaction_chk[train_transaction_chk['dtypes']!='object'].index)


len(train_transaction.select_dtypes('object').columns)


len(train_transaction.select_dtypes(['float', 'int']).columns)








train_transaction_chk[train_transaction_chk['dtypes']=='int64']


# max_val로 numeric value 체크
train_transaction[[col for col in train_transaction_chk[train_transaction_chk['dtypes']!='object'].index if col not in ['TransactionID', 'isFraud', 'TransactionDT']]].max()


num_col_chk = [col for col in train_transaction_chk[train_transaction_chk['dtypes']!='object'].index if col not in ['TransactionID', 'isFraud', 'TransactionDT']]


plt.hist(train_transaction[num_col_chk].max().values)


# avg - max
# avg - std
# median - iqr


np.percentile(train_transaction[num_col_chk], q = 75)


train_transaction[num_col_chk].agg([np.percentile(75)])


train_transaction[num_col_chk].describe(percentiles =[0.75, 0.25])


def percentile(n):
    def percentile_(x):
        return np.quantile(n)
    percentile_.__name__ = 'percentile_{:2.0f}'.format(n*100)
    return percentile_


train_transaction[num_col_chk].quantile([.25, .75])


train_transaction[num_col_chk].quantile([.75-.25])


train_transaction[num_col_chk].agg(['quantile'])


train_transaction[num_col_chk].quantile([.25, .75]).T.apply(lambda x: x[0.75]-x[0.25], axis=1)


np.quantile()


scatter_df = train_transaction[num_col_chk].agg(['max', 'mean']).T
g = sns.scatterplot(x='mean', y='max', data=scatter_df)



for i in range(len(scatter_df)):
    
    g.text(scatter_df['mean'][i], scatter_df['max'][i], scatter_df.index[i], rotation=45)



# (idea) 각 fraud와 non-fraud에서 갖는 값들을 비교.. 일정이상 차이가 나는 것과 안나는 것과 다른 labeling
## 참고 : https://seaborn.pydata.org/generated/seaborn.scatterplot.html


train_transaction_chk[train_transaction_chk.index=='V160']


train_transaction['V160'].value_counts()


train_transaction['card1'].hist()



for i in range(len(scatter_df)):
#     if i > 10:
#         break
#     print(i)
#     print(scatter_df['max'][i])
#     print(scatter_df['mean'][i])
#     print(scatter_df.index[i])
    
    g.text(scatter_df['mean'][i], scatter_df['max'][i], scatter_df.index[i])

    
    
# for p in g.patches:
#     height = p.get_height()
#     g.text(p.get_x()+p.get_width()/2., height + 3, '{:1.2f}%'.format(height/total*100),
#           ha="center", fontsize=10)



train_transaction[num_col_chk].agg(['max', 'mean']).T.plot.scatter(x='mean', y='max',c='DarkBlue')


train_transaction[num_col_chk].agg(['std', 'mean']).T.plot.scatter(x='std', y='mean',c='DarkBlue')


train_transaction[num_col_chk].std().max()


train_transaction[num_col_chk].std().min()


train_transaction[num_col_chk].max().max()


# max_val로 numeric value 체크
sns.distplot(train_transaction[[col for col in train_transaction_chk[train_transaction_chk['dtypes']!='object'].index if col not in ['TransactionID', 'isFraud', 'TransactionDT']]].max().values, hist=True)


train_transaction_chk[train_transaction_chk['dtypes']=='int64'].index











# Using From this kernael : https://www.kaggle.com/artgor/eda-and-models
import os
import time
import datetime
import json
import gc
from numba import jit

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics

from itertools import product

import altair as alt
from altair.vega import v5
from IPython.display import HTML

# using ideas from this kernel: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey

# using ideas from this kernel: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey
def prepare_altair():
    """
    Helper function to prepare altair for working.
    """

    vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v5.SCHEMA_VERSION
    vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'
    vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION
    vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'
    noext = "?noext"
    
    paths = {
        'vega': vega_url + noext,
        'vega-lib': vega_lib_url + noext,
        'vega-lite': vega_lite_url + noext,
        'vega-embed': vega_embed_url + noext
    }
    
    workaround = f"""    requirejs.config({{
        baseUrl: 'https://cdn.jsdelivr.net/npm/',
        paths: {paths}
    }});
    """
    
    return workaround





def add_autoincrement(render_func):
    # Keep track of unique <div/> IDs
    cache = {}
    def wrapped(chart, id="vega-chart", autoincrement=True):
        if autoincrement:
            if id in cache:
                counter = 1 + cache[id]
                cache[id] = counter
            else:
                cache[id] = 0
            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])
        else:
            if id not in cache:
                cache[id] = 0
            actual_id = id
        return render_func(chart, id=actual_id)
    # Cache will stay outside and 
    return wrapped




@add_autoincrement
def render(chart, id="vega-chart"):
    """
    Helper function to plot altair visualizations.
    """
    chart_str = """
    <div id="{id}"></div><script>
    require(["vega-embed"], function(vg_embed) {{
        const spec = {chart};     
        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);
        console.log("anything?");
    }});
    console.log("really...anything?");
    </script>
    """
    return HTML(
        chart_str.format(
            id=id,
            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)
        )
    )
    
    
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


@jit
def fast_auc(y_true, y_prob):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def eval_auc(y_true, y_pred):
    """
    Fast auc eval function for lgb.
    """
    return 'auc', fast_auc(y_true, y_pred), True


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()



def train_model_regression(X, X_test, y, params, folds=None, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200, n_estimators=50000, splits=None, n_folds=3):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.
    
    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type
    
    """
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]
    splits = folds.split(X) if splits is None else splits
    n_splits = folds.n_splits if splits is None else n_folds
    
    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                        'catboost_metric_name': 'MAE',
                        'sklearn_scoring_function': metrics.mean_absolute_error},
                    'group_mae': {'lgb_metric_name': 'mae',
                        'catboost_metric_name': 'MAE',
                        'scoring_function': group_mean_log_mae},
                    'mse': {'lgb_metric_name': 'mse',
                        'catboost_metric_name': 'MSE',
                        'sklearn_scoring_function': metrics.mean_squared_error}
                    }

    
    result_dict = {}
    
    # out-of-fold predictions on train data
    oof = np.zeros(len(X))
    
    # averaged predictions on train data
    prediction = np.zeros(len(X_test))
    
    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()
    
    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(splits):
        if verbose:
            print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = n_estimators, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')
            
            y_pred = model.predict(X_test).reshape(-1,)
        
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))

        prediction += y_pred    
        
        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_splits
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores
    
    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
            
            result_dict['feature_importance'] = feature_importance
        
    return result_dict





def train_model_classification(X, X_test, y, params, folds, model_type='lgb', eval_metric='auc', columns=None, plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200, n_estimators=50000, splits=None, n_folds=3, averaging='usual', n_jobs=-1):
    """
    A function to train a variety of classification models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.
    
    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type
    
    """
    columns = X.columns if columns is None else columns
    n_splits = folds.n_splits if splits is None else n_folds
    X_test = X_test[columns]
    
    # to set up scoring parameters
    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,
                        'catboost_metric_name': 'AUC',
                        'sklearn_scoring_function': metrics.roc_auc_score},
                    }
    
    result_dict = {}
    if averaging == 'usual':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))
        
    elif averaging == 'rank':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))

    
    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()
    
    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
        if model_type == 'lgb':
            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs = n_jobs)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)
            
            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')
            
            y_pred = model.predict_proba(X_test)
        
        if model_type == 'cat':
            model = CatBoostClassifier(iterations=n_estimators, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
                                      loss_function=Logloss)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
        
        if averaging == 'usual':
            
            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
            
            prediction += y_pred.reshape(-1, 1)

        elif averaging == 'rank':
                                  
            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
                                  
            prediction += pd.Series(y_pred).rank().values.reshape(-1, 1)        
        
        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_splits
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores
    
    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
            
            result_dict['feature_importance'] = feature_importance
            result_dict['top_columns'] = cols
        
    return result_dict



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

import pandas as pd
test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')


check_dataframe(train_transaction)


train_transaction['V335'].isnull().sum()


# unique()는 nan을 포함한다
len(train_transaction['V335'].unique())


# value_counts()는 nan을 포함 하지 않는다. 
len(train_transaction['V335'].value_counts().index)


def resumetable(df):
    
    ## Take This Function From https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    
    
    ## Add new
    for name in df.columns:
        


print('{} Columns ---> {}\n'.format('train_identity', len(test_identity.columns)))
# print(train_identity.info())
# train_identity.head()

print('{} Columns ---> {}\n'.format('train_transaction', len(test_transaction.columns)))
# print(train_transaction.info())
# train_transaction.head()

print("<Compare TransactinId>")
# TransactionID로 매핑이 된다고 했지만.. identity 정보가 훨씬 적음
print('train_transaction table -> Unique Of transactionId {} '.format(train_transaction['TransactionID'].nunique()))
print('train_identity table -> Unique Of transactionId {}'.format(train_identity['TransactionID'].nunique()))
print(str(math.floor(train_identity['TransactionID'].nunique()/train_transaction['TransactionID'].nunique()*100))+'%')


# missing value count
def missing_values_count(df):
    missing_values_count = df.isnull().sum()
    total_cells = np.product(df.shape) # ((590540, 394) -> (590540 * 394))
    total_missing = missing_values_count.sum()
    return "% of missing data = ",(total_missing/total_cells) * 100

print(missing_values_count(train_transaction))
print(missing_values_count(train_identity))




summary = pd.DataFrame(train_transaction.dtypes, columns=['dtypes'])


summary


summary = summary.reset_index()
summary


summary['Name'] = summary['index']
summary


summary = summary[['Name', 'dtypes']]
summary


summary.isnull().sum().values


summary = summary[['Name', 'dtypes']]

summary['Missing'] = train_transaction.isnull().sum().values
summary['Uniques'] = train_transaction.nunique().values
summary['First Value'] = train_transaction.loc[0].values
summary['Second Value'] = train_transaction.loc[1].values
summary['Third Value'] = train_transaction.loc[2].values


summary


for name in summary['Name'].value_counts().index:
    print(name)


# stats.entropy 
# scipy.stats.entropy(pk, qk=None, base=None, axis=0)[source]
# Calculate the entropy of a distribution for given probability values.
# About Entropy [entropy](https://datascienceschool.net/view-notebook/d3ecf5cc7027441c8509c0cae7fea088/)

import scipy.stats as stats

for name in summary['Name']:
    print(name)
#     summary.loc[summary['Name'] == name, 'Entropy']
    print(round(stats.entropy(train_transaction[name].value_counts(normalize=True), base=2), 2))


# from scipy.stats import entropy
import scipy








# TransactionID로 두 테이블을 조인
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

# del train_identity, train_transaction, test_transaction, test_identity

test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')


print(f'There are {train.isnull().any().sum()} columns in train dataset with missing values')


# Value check each columns

one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]
print(one_value_cols == one_value_cols_test)
print(one_value_cols)
print(one_value_cols_test)
# ['a', 'b'] ==  ['b', 'a'] 도 False가 나오므로 다른 방법 비교 필요


pd.merge(train_transaction, train_identity, on='TransactionID', how='inner')['TransactionID'].nunique()


print(f'{train_identity.shape}')
print(f'{train_transaction.shape}')


# 각 값들의 unique한 값과 null값을 얼마나 포함하고 있는지 확인
# 이들을 그래프로 그려보자
print(train_identity.columns)
print(train_transaction.columns)


def check_dataframe(df):

    dict_check_value = {}
    list_check_value = []
    for col in df.columns:
    #     print(col)
        null_value = df[col].isnull().sum()
        unique_value_exam = df[col][~df[col].isnull()].unique()
        unique_value = len(unique_value_exam)
        unique_value_exam = unique_value_exam[:5]
        value_type = df[col].dtype


        list_check_value = [unique_value, null_value, unique_value_exam, value_type]


        dict_check_value[col] = list_check_value
    
    new_df = pd.DataFrame.from_dict(dict_check_value, orient='index', columns=['unique_value', 'isNullcnt', 'value_exam', 'value_type'])
    return new_df


check_identity = check_dataframe(train_identity)
check_transaction = check_dataframe(train_transaction)


check_identity.head()


check_transaction.head()


# identity 부터 살펴보면 
check_identity['value_type'].value_counts().to_frame()
check_identity['value_type'].value_counts().keys()
check_identity['value_type'].value_counts().tolist() # https://stackoverflow.com/questions/35523635/extract-values-in-pandas-value-counts


# 각 테이블은 어떠한 데이터타입의 데이터로 이루어져 있는지
def split_ValueCounts(a, b):
    return [a, b]

def make_String(stat_df):
    """
    Using Statistics DataFrame
    """
    total = len(stat_df.index)
    stat_list = stat_df['value_type'].value_counts().to_frame().reset_index().apply(lambda x: split_ValueCounts(x['index'], x['value_type']), axis=1)
    stat_str = ','.join([(str(x[0])+"는 "+str(x[1])+'개('+str(math.floor(x[1]/total*100))+'%)') for x in stat_list])
    return f'{total}개 컬럼 중 {stat_str}이다.'

print(make_String(check_identity))
print(make_String(check_transaction))
# aa= check_identity['value_type'].value_counts().to_frame().reset_index().apply(lambda x: split_value(x['index'], x['value_type']), axis=1)

print(make_String(check_dataframe(train)))


import matplotlib.pyplot as plt
def check_DataEDA_withlineChart(df, columns): # figsize -> default (30, 20), 넣을 수 있도록.. 
    """
    df = Stats DataFrame
    columns = df.columns.to_list(), type=[]
    """
    
    print(len(columns))
    # Working With Columns that they are Selected
    new_df = df[df.index.isin(columns)].copy(deep=True)
#     new_df = df[columns].copy(deep=True)
    
    
    # Check Ordering and Ordered Number .. 
    col_2_idx_dict = {}
    for i, column in enumerate(columns):
        col_2_idx_dict[column] = i
    
    # inverted_dict 
    idx_2_col_dict = {val: key for key, val in col_2_idx_dict.items()}
    
    cols_idx = sorted(idx_2_col_dict.keys())
    
#     print(cols_idx)
#     print(len(cols_idx))
    # Draw Line Chart
    fig = plt.figure(figsize=(30, 20))
    
    
    x1 = new_df['unique_value']
    x2 = new_df['isNullcnt']
    
    
    plt.plot(cols_idx, new_df['unique_value'][idx_2_col_dict], label='unique_value', linestyle='-', marker='x')
    plt.plot(cols_idx, new_df['isNullcnt'][idx_2_col_dict], label='isNullcnt', linestyle='--', marker='o')
    
    plt.legend()
    plt.xlabel("Number Of Columns")
    plt.ylabel("Values")
    plt.title("CheckData with LineChart that NullVal and UniqVal")
    
    # Annotate Infomation
    print(idx_2_col_dict)
    for i in sorted(idx_2_col_dict.keys()):
#         print(i)
#         print(idx_2_col_dict[i])

        # Annotate DataType
        width = i
        height = new_df['unique_value'][idx_2_col_dict[i]]
        text = str(new_df['value_type'][idx_2_col_dict[i]])
        plt.annotate(text, xy=(width, height), xytext=(width+0.05, height+0.3), rotation=70)
        
        # Annotate Uniq Values Number
        text = str(height)+'(n)' # str(check_identity['value_type'][i])
        plt.annotate(text, xy=(width, height), xytext=(1, +50), textcoords="offset points", va="top", color='b', rotation=0)
        
        # Annotate Null Values 
#         total_num = max([new_df['unique_value'].max(), new_df['isNullcnt'].max()])
        
        # Stat Dataframe에 사용된 기본 데이터프레임
        total_num = len(train_identity)
    
        text = str(math.floor(new_df['isNullcnt'][idx_2_col_dict[i]]/total_num*100))+'%'
        plt.annotate(text, xy=(width, height), xytext=(1, +70), textcoords="offset points", va="top", color='orange', rotation=0)
        
        
    fx1, fx2, fy1, fy2 = plt.axis() # fig.axis()
    plt.annotate("The Red Text: Num of Unique Values", xy=(fx2, fy2), xytext=(1, 1), textcoords="offset points", ha="right", va="bottom", color='b')
    plt.annotate("The Black Text: type Of Data", xy=(fx2, fy2), xytext=(1, 10), textcoords="offset points", ha="right", va="bottom")
    plt.annotate("The Orange Text: Percentage Of Null value", xy=(fx2, fy2), xytext=(1, 20), textcoords="offset points", ha="right", va="bottom", color='orange')
    plt.xticks(cols_idx, list([val for key, val in sorted(idx_2_col_dict.items())]), rotation=90)
    
    return fig
    
    
    


columns = check_identity.index
check_DataEDA_withlineChart(check_identity, columns)


import matplotlib.pyplot as plt
# NULL값을 많이 포함하는 경우
check_identity['isNullper'] = check_identity['isNullcnt'].apply(lambda x: x/train_identity.shape[0])

# 모든 컬럼 조회
# select_col = check_identity.index

# Null값이 많은 컬럼 체크
select_col = check_identity[check_identity['isNullper']>0.9].index
print(f'{len(select_col)}과 {select_col}')


plt.hist(check_identity[check_identity.index.isin(select_col)]['isNullper'])


check_identity[check_identity.index.isin(select_col)]


# 사기거래 중 97%값은 비어있다. 
[train[train['isFraud']==1][col].isnull().sum()/train[train['isFraud']==1].shape[0] for col in select_col]


# 사기거래가 아닌 것 중 99%가 비어있다. .. 이 차이는 유의미한가?
[train[train['isFraud']==0][col].isnull().sum()/train[train['isFraud']==0].shape[0] for col in select_col]


remove_cols_identity = ['id_07', 'id_08', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26','id_27']


# Null value가 많은 값 제외
# Null값이 많은 컬럼을 제외한 칼럼들을 저장
select_col = check_identity[~check_identity.index.isin(select_col)].index
print(f'{len(select_col)}과 {select_col}')


plt.hist(check_identity[check_identity.index.isin(select_col)]['isNullper'])


remian_identity_col = ['TransactionID', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06',
       'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16',
       'id_17', 'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 'id_31',
       'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
       'DeviceType', 'DeviceInfo']


columns = check_identity[check_identity.index.isin(select_col)].index
check_DataEDA_withlineChart(check_identity, columns)


check_identity[(check_identity['value_type']=='object') & (check_identity.index.isin(select_col))]


# 너무 많은 값들을 가진 경우는 나중에 처리

tooMany_col_identity = check_identity[(check_identity['value_type']=='object') & (check_identity.index.isin(select_col))&(check_identity['unique_value']>=5)].index
print(tooMany_col_identity)


tooMany_col_identity = tooMany_col_identity.to_list()
remove_cols_identity.extend(tooMany_col_identity)


remove_cols_identity = ['id_07', 'id_08', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_30', 'id_31', 'id_33', 'DeviceInfo']


del check_identity


# NULL값을 많이 포함하는 경우
check_transaction['isNullper'] = check_transaction['isNullcnt'].apply(lambda x: x/train_transaction.shape[0])

# 모든 컬럼 조회
# select_col = check_identity.index

# Null값이 많은 컬럼 체크
select_col = check_transaction[check_transaction['isNullper']>0.7].index
print(f'{len(select_col)}과 {select_col}')

plt.hist(check_transaction['isNullper'])
plt.hist(check_transaction[check_transaction.index.isin(select_col)]['isNullper'])

print(len(select_col))
print(select_col)


# 우선 Null 값이 많은 칼럼을 제거하고, object중 값이 높은 것들도 제외... 
remove_cols_transacton = select_col.copy()

check_transaction[(~check_transaction.index.isin(remove_cols_transacton))&(check_transaction['value_type']=='object')]


remove_cols_transacton = remove_cols_transacton.to_list() # .append("P_emaildomain")

remove_cols_transacton.append("P_emaildomain")

remove_cols_transacton

# remove_cols_transacton = ['dist2','R_emaildomain','D6','D7','D8','D9','D12','D13','D14','V138','V139','V140','V141','V142','V143','V144','V145','V146','V147','V148','V149','V150','V151','V152','V153','V154','V155','V156','V157','V158','V159','V160','V161','V162','V163','V164','V165','V166','V167','V168','V169','V170','V171','V172','V173','V174','V175','V176','V177','V178','V179','V180','V181','V182','V183','V184','V185','V186','V187','V188','V189','V190','V191','V192','V193','V194','V195','V196','V197','V198','V199','V200','V201','V202','V203','V204','V205','V206','V207','V208','V209','V210','V211','V212','V213','V214','V215','V216','V217','V218','V219','V220','V221','V222','V223','V224','V225','V226','V227','V228','V229','V230','V231','V232','V233','V234','V235','V236','V237','V238','V239','V240','V241','V242','V243','V244','V245','V246','V247','V248','V249','V250','V251','V252','V253','V254','V255','V256','V257','V258','V259','V260','V261','V262','V263','V264','V265','V266','V267','V268','V269','V270','V271','V272','V273','V274','V275','V276','V277','V278','V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333','V334','V335','V336','V337','V338','V339','P_emaildomain']


remain_cols = [col for col in train.columns if col not in remove_cols_identity+remove_cols_transacton ]


print(remain_cols)
# remian_cols = ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D10', 'D11', 'D15', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_32', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType']


# Columns 지우기 전에 train 칼럼 보관
# _train = train.copy(deep=True)
train = train[remain_cols]


train.shape  # 칼럼 개수를 434개에서 252개로 .. 


test.columns = [val.replace('-', '_') if 'id' in val else val for val in list(test.columns)]


# _test = test.copy(deep=True)
remain_test_cols = [col for col in remain_cols if col not in ['isFraud']]
test = test[remain_test_cols]


train.info()
new_train = check_dataframe(train)
new_train.head()


print(len(train.select_dtypes('object').columns))
print(len(new_train[new_train['value_type']=='object']))
cat_cols = new_train[new_train['value_type']=='object'].index.to_list()


del new_train


from sklearn import preprocessing
for col in cat_cols:
    if col in train.columns:
        le = preprocessing.LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))


X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)



y = train.sort_values('TransactionDT')['isFraud']

X_test = test.drop(['TransactionDT', 'TransactionID'], axis=1)

test = test[['TransactionDT', 'TransactionID']]


# null 처리
def clean_inf_nan(df):
    return df.replace([np.inf, -np.inf], np.nan)

# Cleaning infinite values to NaN
X = clean_inf_nan(X)
X_test = clean_inf_nan(X_test)


import gc
gc.collect()
print("Garbage collection thresholds: {}".format(gc.get_threshold()))

# 출처: https://weicomes.tistory.com/277 [25%]


from sklearn.model_selection import train_test_split, cross_val_predict, TimeSeriesSplit, KFold, cross_val_score
n_fold = 5
folds = TimeSeriesSplit(n_splits=n_fold)
folds = KFold(n_splits=5)


categorical_features_indices = np.where(X.dtypes != np.float)[0]

clf = CatBoostClassifier(random_seed=rnd_state)

clf.fit(X_train, y_train, cat_features=categorical_features_indices)
clf.score(X_val, y_val)


params = {'num_leaves': 256,
          'min_child_samples': 79,
          'objective': 'binary',
          'max_depth': 13,
          'learning_rate': 0.03,
          "boosting_type": "gbdt",
          "subsample_freq": 3,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3,
          'reg_lambda': 0.3,
          'colsample_bytree': 0.9,
          #'categorical_feature': cat_cols
         }
result_dict_lgb = train_model_classification(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='lgb', eval_metric='auc', plot_feature_importance=True,
                                                      verbose=500, early_stopping_rounds=200, n_estimators=5000, averaging='usual', n_jobs=-1)


sample_submission['isFraud']


sample_submission['isFraud'] = result_dict_lgb['prediction']


sample_submission.to_csv('submission.csv', index=False)


sample_submission.head()


pd.DataFrame(result_dict_lgb['oof']).to_csv('lgb_oof.csv', index=False)


! ls


def check_DataEDA_withlineChart(df, columns): # figsize -> default (30, 20), 넣을 수 있도록.. 
    """
    df = DataFrame
    columns = df.columns.to_list(), type=[]
    """
    
    print(len(columns))
    # Working With Columns that they are Selected
    new_df = df[df.index.isin(columns)].copy(deep=True)
#     new_df = df[columns].copy(deep=True)
    
    
    # Check Ordering and Ordered Number .. 
    col_2_idx_dict = {}
    for i, column in enumerate(columns):
        col_2_idx_dict[column] = i
    
    # inverted_dict 
    idx_2_col_dict = {val: key for key, val in col_2_idx_dict.items()}
    
    cols_idx = sorted(idx_2_col_dict.keys())
    
    print(cols_idx)
    print(len(cols_idx))
    # Draw Line Chart
    fig = plt.figure(figsize=(30, 20))
    
    
    x1 = new_df['unique_value']
    x2 = new_df['isNullcnt']
    
    
    plt.plot(cols_idx, new_df['unique_value'][idx_2_col_dict], label='unique_value', linestyle='-', marker='x')
    plt.plot(cols_idx, new_df['isNullcnt'][idx_2_col_dict], label='isNullcnt', linestyle='--', marker='o')
    
    plt.legend()
    plt.xlabel("Number Of Columns")
    plt.ylabel("Values")
    plt.title("CheckData with LineChart that NullVal and UniqVal")
    
    # Annotate Infomation
    print(idx_2_col_dict)
    for i in sorted(idx_2_col_dict.keys()):
        print(i)
        print(idx_2_col_dict[i])

        
        # Annotate DataType
        width = i
        height = new_df['unique_value'][idx_2_col_dict[i]]
        text = str(new_df['value_type'][idx_2_col_dict[i]])
        plt.annotate(text, xy=(width, height), xytext=(width+0.05, height+0.3), rotation=70)
        
        # Annotate Uniq Values Number
        text = str(height)+'(n)' # str(check_identity['value_type'][i])
        plt.annotate(text, xy=(width, height), xytext=(1, +50), textcoords="offset points", va="top", color='b', rotation=0)
        
        # Annotate Null Values 
#         total_num = max([new_df['unique_value'].max(), new_df['isNullcnt'].max()])
        total_num = len(train)
        text = str(math.floor(new_df['isNullcnt'][idx_2_col_dict[i]]/total_num*100))+'%'
        plt.annotate(text, xy=(width, height), xytext=(1, +70), textcoords="offset points", va="top", color='orange', rotation=0)
        
        
    fx1, fx2, fy1, fy2 = plt.axis() # fig.axis()
    plt.annotate("The Red Text: Num of Unique Values", xy=(fx2, fy2), xytext=(1, 1), textcoords="offset points", ha="right", va="bottom", color='b')
    plt.annotate("The Black Text: type Of Data", xy=(fx2, fy2), xytext=(1, 10), textcoords="offset points", ha="right", va="bottom")
    plt.annotate("The Orange Text: Percentage Of Null value", xy=(fx2, fy2), xytext=(1, 20), textcoords="offset points", ha="right", va="bottom", color='orange')
    plt.xticks(cols_idx, list([val for key, val in sorted(idx_2_col_dict.items())]), rotation=90)
    
    return fig
    
    
    


select_col = check_train[(check_train['value_type']!='object') & (check_train['isNullper']<0.6) & (check_train['isNullper']>0.4)].index
check_DataEDA_withlineChart(check_train, select_col)


check_train[check_train.index.isin(['dist1', 'D2', 'D3'])]


select_col = check_train[check_train['value_type']!='object'].index
print(len(select_col))
plt.hist(check_train[check_train.index.isin(select_col)]['isNullper'])
plt.title("Histogram Of Percetage Of Null Data")


select_col = check_train.index
print(len(select_col))
plt.hist(check_train[check_train.index.isin(select_col)]['isNullper'])
plt.title("Histogram Of Percetage Of Null Data")


check_train[(check_train['value_type']!='object') & (check_train['isNullper']<0.6) & (check_train['isNullper']>0.4)]


train['card1'].isnull().sum()


train['card1'].nunique()


print(len(check_train[check_train['value_type']!='object'].index))
print(len(check_train[(check_train['value_type']!='object') & ((check_train['isNullper']<0.05))].index))


check_train[check_train.index.isin(select_col)]



plt.hist(check_train['isNullper'])
plt.title("Histogram Of Percetage Of Null Data")
check_train[check_train['isNullper']<0.3].index


check_train[check_train.index=='card3']


select_col = list(check_train[(check_train['value_type']!='object')&(check_train['isNullper']<0.3)].index)
check_DataEDA_withlineChart(check_train, select_col)


idx2col= {val: key for key, val in column_number.items()}


cols_idx = list(column_number.values())

for i in idx2col.keys():
    print(i)
    print(check_identity['unique_value'][idx2col[i]])


plt.figure(figsize=(30, 20)) # width, height
# plt.plot(check_identity.index, check_identity['unique_value'], label='unique_value', linestyle='-', marker='x')
# plt.plot(check_identity.index, check_identity['isNullcnt'], label='isNullcnt', linestyle='--', marker='o')
plt.plot(list(column_number.values()), check_identity['unique_value'], label='unique_value', linestyle='-', marker='x')
plt.plot(list(column_number.values()), check_identity['isNullcnt'], label='isNullcnt', linestyle='--', marker='o')
# plt.annotate
plt.legend()
plt.xlabel("Columns")
plt.ylabel("Values Each Cols")
for i in column_number.values():
    width = list(column_number.values())[i]
    height = check_identity['unique_value'][i]
#     print(width, height)
    plt.annotate(str(check_identity['value_type'][i]), xy=(width, height), xytext=(width+0.05, height+0.3), rotation=70)
    
for i in column_number.values():
    width = list(column_number.values())[i]
    height = check_identity['unique_value'][i]
#     print(width, height)
    plt.annotate(str(check_identity['unique_value'][i]), xy=(width, height), xytext=(width+0.05, height+5000), color='b', rotation=0)
# plt.annotate(check_identity['value_type'], xy=(list(column_number.values()), check_identity['isNullcnt']+2))

# null value의 percentage
for i in column_number.values():
    total_num = int(check_identity[check_identity.index=='TransactionID']['unique_value'])
    width = list(column_number.values())[i]
    height = check_identity['unique_value'][i]
#     print(width, height)
    plt.annotate(str(math.floor(check_identity['isNullcnt'][i]/total_num*100))+'%', xy=(width, height), xytext=(1, +70), textcoords="offset points", va="top", color='orange', rotation=0)


x1, x2, y1, y2 = plt.axis()
plt.annotate("The Red Text: Num of Unique Values", xy=(x2, y2), xytext=(1, 1), textcoords="offset points", ha="right", va="bottom", color='b')
plt.annotate("The Black Text: type Of Data", xy=(x2, y2), xytext=(1, 10), textcoords="offset points", ha="right", va="bottom")
plt.annotate("The Orange Text: Percentage Of Null value", xy=(x2, y2), xytext=(1, 20), textcoords="offset points", ha="right", va="bottom", color='orange')
plt.xticks(list(column_number.values()), list(column_number.keys()), rotation=90)
# plt.xticks(rotation=90, fontsize=20)
# plt.xticks(list(column_number.keys()))

plt.title("Check unique and null values")


check_train[check_train['value_type'] == 'object']


check_train['value_type'].value_counts()





check_identity


a = np.arange(5)
b = np.arange(5, 10)
plt.plot(a, b)
bbox_args = dict(boxstyle="round", fc="0.8")
for i in range(len(a)):
    plt.annotate('a', xy=(a[i], b[i]), xytext=(a[i]+0.1, b[i]+0))
    plt.annotate('b', xy=(a[i], b[i]), xytext=(a[i], b[i]), color='b')

x1, x2, y1, y2 = plt.axis()
plt.annotate('test', xy=(x2, y2), textcoords="offset points", ha="left", va="bottom")


plt.axis()


locs, labels = plt.xticks()


locs


labels


labels = ('Tom', 'Dick', 'Harry', 'Sally', 'Sue')
plt.xticks(np.arange(5), labels)


plt.plot([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4], 
         [1, 2, 3, 4], [2, 2, 2, 2])


import matplotlib.pyplot as plt
plt.plot('unique_value', 'isNullcnt')
plt.show()
# plt.plot(y = 'isNullcnt',kind='line')


check_identity.index


list(check_identity.index)




