import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import cufflinks as cf
cf.go_offline()
import os
import gc


application_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
POS_CASH_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv')
bureau_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')
previous_application = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')
installments_payments = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')
credit_card_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')
bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
application_test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')


application_train.head()


application_train.shape


POS_CASH_balance.head()


bureau_balance.head()


previous_application.head()


installments_payments.head()


credit_card_balance.head()


bureau.head()


application_train.isnull().mean().sort_values(ascending = False)


POS_CASH_balance.isnull().mean().sort_values(ascending = False)


bureau_balance.isnull().mean().sort_values(ascending = False)


previous_application.isnull().mean().sort_values(ascending = False)


installments_payments.isnull().mean().sort_values(ascending = False)


credit_card_balance.isnull().mean().sort_values(ascending = False)


bureau.isnull().mean().sort_values(ascending = False)


plt.figure(figsize=(10,5))
plt.title("Distribution of AMT_CREDIT")
sns.distplot(application_train["AMT_CREDIT"])
plt.show()


plt.figure(figsize=(10,5))
plt.title("Distribution of AMT_INCOME_TOTAL")
sns.distplot(application_train["AMT_INCOME_TOTAL"].dropna())
plt.show()


plt.figure(figsize=(10,5))
plt.title("Distribution of AMT_GOODS_PRICE")
sns.distplot(application_train["AMT_GOODS_PRICE"].dropna())
plt.show()


temp = pd.DataFrame({'labels': application_train["TARGET"].value_counts().index,
                   'values': application_train["TARGET"].value_counts().values})
temp.iplot(kind='bar',labels='labels',values='values', title='Loan Repayed or not')


temp = application_train["OCCUPATION_TYPE"].value_counts()
temp.iplot(kind='bar', xTitle = 'Occupation', yTitle = "Count", title = 'Occupation of Applicants', color = 'red')


temp = application_train["ORGANIZATION_TYPE"].value_counts()
temp.iplot(kind='bar', xTitle = 'Organization Name', yTitle = "Count", title = 'Types of Organizations who applied for loan ', color = 'green')


temp = previous_application["NAME_CONTRACT_TYPE"].value_counts()
fig = { "data": [{"values": temp.values,"labels": temp.index,"domain": {"x": [0, .48]},
      "hole": .6, "type": "pie"},], 
       "layout": {"title":"Contract product type",
        "annotations": [{"font": {"size": 20},"showarrow": False,"text": "Contract product type","x": 0.12,"y": 0.5}]}}
iplot(fig)


temp = pd.DataFrame({'labels': previous_application["NAME_CLIENT_TYPE"].value_counts().index,
                   'values': previous_application["NAME_CLIENT_TYPE"].value_counts().values})
temp.iplot(kind='pie',labels='labels',values='values', title='New or Old Client?', hole = 0.5)


temp = pd.DataFrame({'labels': previous_application["NFLAG_INSURED_ON_APPROVAL"].value_counts().index,
                   'values': previous_application["NFLAG_INSURED_ON_APPROVAL"].value_counts().values
                  })
temp.iplot(kind='bar',labels='labels',values='values', title='Did the client requested insurance? (YES : 1, NO : 0)')


cor = application_train.drop('SK_ID_CURR',axis=1).corr()
sns.heatmap(cor)
plt.show()


from sklearn.preprocessing import LabelEncoder
for c in application_train.columns:
    if (c!='SK_ID_CURR') & (application_train[c].dtypes==object):
        LE = LabelEncoder()
        LE.fit(list(application_train[c].values.astype('str')) + list(application_test[c].values.astype('str')))
        application_train[c] = LE.transform(list(application_train[c].values.astype('str')))
        application_test[c] = LE.transform(list(application_test[c].values.astype('str')))
application_train.head()


application_train.fillna(-1, inplace = True)


from xgboost import XGBClassifier
from xgboost import plot_importance
X = application_train.drop(['SK_ID_CURR', 'TARGET'],axis=1)
Y = application_train.TARGET
xgb = XGBClassifier(n_estimators=500, max_depth=8, random_state=2018)
xgb.fit(X, Y)


plot_importance(xgb)
plt.rcParams["figure.figsize"] = (10, 30)
plt.show()


df = application_train.append(application_test).reset_index()
# df = df[df['CODE_GENDER'] != 'XNA']

df['DAYS_EMPLOYED'].replace(365243, -1, inplace= True)
df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

previous_application['DAYS_FIRST_DRAWING'].replace(365243, -1, inplace= True)
previous_application['DAYS_FIRST_DUE'].replace(365243, -1, inplace= True)
previous_application['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, -1, inplace= True)
previous_application['DAYS_LAST_DUE'].replace(365243, -1, inplace= True)
previous_application['DAYS_TERMINATION'].replace(365243, -1, inplace= True)
previous_application['APP_CREDIT_PERC'] = previous_application['AMT_APPLICATION'] / previous_application['AMT_CREDIT']

installments_payments['PAYMENT_PERC'] = installments_payments['AMT_PAYMENT'] / installments_payments['AMT_INSTALMENT']
installments_payments['PAYMENT_DIFF'] = installments_payments['AMT_INSTALMENT'] - installments_payments['AMT_PAYMENT']
installments_payments['DPD'] = installments_payments['DAYS_ENTRY_PAYMENT'] - installments_payments['DAYS_INSTALMENT']
installments_payments['DBD'] = installments_payments['DAYS_INSTALMENT'] - installments_payments['DAYS_ENTRY_PAYMENT']
installments_payments['DPD'] = installments_payments['DPD'].apply(lambda x: x if x > 0 else 0)
installments_payments['DBD'] = installments_payments['DBD'].apply(lambda x: x if x > 0 else 0)


bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
# for c in bureau_balance.columns:
#     if bureau_balance[c].dtypes==object:
#         bb_aggregations[c] = ['mean']
bb_agg = bureau_balance.groupby('SK_ID_BUREAU').agg(bb_aggregations)
bb_agg.columns = pd.Index([col[0] + "_" + col[1].upper() for col in bb_agg.columns.tolist()])
bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
del bb_agg
gc.collect()

aggregations = {
    'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
    'DAYS_CREDIT_ENDDATE': ['mean','var'],
    'DAYS_CREDIT_UPDATE': ['mean'],
    'CREDIT_DAY_OVERDUE': ['mean','var'],
    'AMT_CREDIT_MAX_OVERDUE': ['mean'],
    'AMT_CREDIT_SUM': ['max', 'mean', 'var'],
    'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'var'],
    'AMT_CREDIT_SUM_OVERDUE': ['mean'],
    'AMT_CREDIT_SUM_LIMIT': ['mean', 'var'],
    'AMT_ANNUITY': ['max', 'mean'],
    'CNT_CREDIT_PROLONG': ['sum'],
    'MONTHS_BALANCE_MIN': ['min'],
    'MONTHS_BALANCE_MAX': ['max'],
    'MONTHS_BALANCE_SIZE': ['mean', 'var']
}

bureau_agg = bureau.groupby('SK_ID_CURR').agg({**aggregations})
bureau_agg.columns = pd.Index(['BURO_' + col[0] + "_" + col[1].upper() for col in bureau_agg.columns.tolist()])

del bureau, bureau_balance
gc.collect()
df = pd.merge(df, bureau_agg, how='left', on='SK_ID_CURR')
del bureau_agg
gc.collect()


for c in previous_application.columns:
    if previous_application[c].dtypes==object:
        LE = LabelEncoder()
        previous_application[c] = LE.fit_transform(list(previous_application[c].values.astype('str')))
aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'var'],
    }
prev_agg = previous_application.groupby('SK_ID_CURR').agg({**aggregations})
prev_agg.columns = pd.Index(['PREV_' + col[0] + "_" + col[1].upper() for col in prev_agg.columns.tolist()])

del previous_application
gc.collect()
df = pd.merge(df,prev_agg, how='left', on='SK_ID_CURR')
# df = df.join(bureau, how='left', on='SK_ID_CURR')

del prev_agg
gc.collect()


for c in POS_CASH_balance.columns:
    if POS_CASH_balance[c].dtypes==object:
        LE = LabelEncoder()
        POS_CASH_balance[c] = LE.fit_transform(list(POS_CASH_balance[c].values.astype('str')))
aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean', 'var'],
        'SK_DPD_DEF': ['max', 'mean', 'var']
    }
pos_agg = POS_CASH_balance.groupby('SK_ID_CURR').agg(aggregations)
pos_agg.columns = pd.Index(['POS_' + col[0] + "_" + col[1].upper() for col in pos_agg.columns.tolist()])
pos_agg['POS_COUNT'] = POS_CASH_balance.groupby('SK_ID_CURR').size()
del POS_CASH_balance
gc.collect()
df = pd.merge(df,pos_agg, how='left', on='SK_ID_CURR')
del pos_agg
gc.collect()


for c in installments_payments.columns:
    if installments_payments[c].dtypes==object:
        LE = LabelEncoder()
        installments_payments[c] = LE.fit_transform(list(installments_payments[c].values.astype('str')))

aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
ins_agg = installments_payments.groupby('SK_ID_CURR').agg(aggregations)
ins_agg.columns = pd.Index(['INSTAL_' + col[0] + "_" + col[1].upper() for col in ins_agg.columns.tolist()])

ins_agg['INSTAL_COUNT'] = installments_payments.groupby('SK_ID_CURR').size()
del installments_payments
gc.collect()        
df = pd.merge(df,ins_agg, how='left', on='SK_ID_CURR')
del ins_agg
gc.collect()


for c in credit_card_balance.columns:
    if credit_card_balance[c].dtypes==object:
        LE = LabelEncoder()
        credit_card_balance[c] = LE.fit_transform(list(credit_card_balance[c].values.astype('str')))
credit_card_balance.drop(['SK_ID_PREV'], axis= 1, inplace = True)
cc_agg = credit_card_balance.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
cc_agg.columns = pd.Index(['CC_' + col[0] + "_" + col[1].upper() for col in cc_agg.columns.tolist()])
cc_agg['CC_COUNT'] = credit_card_balance.groupby('SK_ID_CURR').size()

df = pd.merge(df,cc_agg, how='left', on='SK_ID_CURR')
del credit_card_balance, cc_agg
gc.collect()


df_train, df_test = df.iloc[:len(application_train)], df.iloc[len(application_train):]
del application_train, application_test, df
gc.collect()





from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
folds = StratifiedKFold(n_splits= 10, shuffle=True, random_state=2020)
sub_preds = np.zeros(df_test.shape[0])
feats = [f for f in df_train.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train[feats], df_train['TARGET'])):
    train_x, train_y = df_train[feats].iloc[train_idx], df_train['TARGET'].iloc[train_idx]
    valid_x, valid_y = df_train[feats].iloc[valid_idx], df_train['TARGET'].iloc[valid_idx]

    lgb = LGBMClassifier(nthread=4, n_estimators=12000, learning_rate=0.02, num_leaves=31,
        colsample_bytree=0.85,subsample=0.9, max_depth=8, reg_alpha=0.0415, reg_lambda=0.073,
        min_split_gain=0.022, min_child_weight=39.32, silent=-1, verbose=-1)

    lgb.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
        eval_metric= 'auc', verbose= 200, early_stopping_rounds= 100)

    sub_preds += lgb.predict_proba(df_test[feats], num_iteration=lgb.best_iteration_)[:, 1] / folds.n_splits

    del lgb, train_x, train_y, valid_x, valid_y
    gc.collect()

df_test['TARGET'] = sub_preds


df_test[['SK_ID_CURR', 'TARGET']].to_csv('submission', index= False)




