import pandas as pd
import numpy as np


pos_cash_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv')
train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')
bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
bureau_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')
cc_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')
ins_payment = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')
prev_app = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')


pos_cash_balance.head()


train.head()


bureau.head()


bureau_balance.head()


cc_balance.head()


ins_payment.head()


prev_app.head()


train.shape, test.shape, bureau.shape, bureau_balance.shape, cc_balance.shape, ins_payment.shape, prev_app.shape, pos_cash_balance.shape


operation_mean = ['mean']
operation_sum = ['sum']
operation_count = ['nunique']

bb_op = {'MONTHS_BALANCE': operation_mean, 'STATUS': operation_count}

bb_grouping = bureau_balance.groupby(['SK_ID_BUREAU'])
bb_groupby = bb_grouping.agg(bb_op)
bb_groupby.head()


bb_groupby.columns = ['BB_' + '_'.join(col).strip() for col in bb_groupby.columns.values]
bb_groupby.reset_index(inplace=True)
bb_groupby.head()


bureau = pd.merge(bureau, bb_groupby)
bureau.head()


bureau.isnull().sum() / bureau.shape[0]


bureau.describe()


bureau = bureau.drop(['AMT_CREDIT_MAX_OVERDUE'], axis=1)
bureau.fillna(0, inplace=True)


bureau.head()


bureau.columns


bureau_op = {'CREDIT_ACTIVE': operation_count, 'CREDIT_CURRENCY': operation_count, 
             'DAYS_CREDIT': operation_mean, 'CREDIT_DAY_OVERDUE': operation_mean, 
             'DAYS_CREDIT_ENDDATE': operation_mean, 'DAYS_ENDDATE_FACT': operation_mean,
             'CNT_CREDIT_PROLONG': operation_mean, 'AMT_CREDIT_SUM': operation_mean,
             'AMT_CREDIT_SUM_DEBT': operation_mean, 'AMT_CREDIT_SUM_LIMIT': operation_mean, 
             'AMT_CREDIT_SUM_OVERDUE': operation_mean, 'CREDIT_TYPE': operation_count,
             'DAYS_CREDIT_UPDATE': operation_mean, 'AMT_ANNUITY': operation_mean,
             'BB_MONTHS_BALANCE_mean': operation_mean, 'BB_STATUS_nunique': operation_sum}

bureau_grouping = bureau.groupby(['SK_ID_CURR'])
bureau_groupby = bureau_grouping.agg(bureau_op)
bureau_groupby.head()


bureau_groupby.columns = ['B_' + '_'.join(col).strip() for col in bureau_groupby.columns.values]
bureau_groupby.reset_index(inplace=True)
bureau_groupby.head()


pos_cash_balance.head()


pos_cash_balance_op = {'NAME_CONTRACT_STATUS': operation_count, 'MONTHS_BALANCE': operation_mean, 
                       'CNT_INSTALMENT': operation_mean, 'CNT_INSTALMENT_FUTURE': operation_mean, 
                       'SK_DPD': operation_mean, 'SK_DPD_DEF': operation_mean}

pos_cash_balance_grouping = pos_cash_balance.groupby(['SK_ID_PREV', 'SK_ID_CURR'])
pos_cash_balance_groupby = pos_cash_balance_grouping.agg(pos_cash_balance_op)
pos_cash_balance_groupby.head()


pos_cash_balance_groupby.columns = ['POS_' + '_'.join(col).strip() for col in pos_cash_balance_groupby.columns.values]
pos_cash_balance_groupby.reset_index(inplace=True)
pos_cash_balance_groupby.head()


cc_balance.isnull().sum() / cc_balance.shape[0]


cc_balance.head()


cc_balance_op = {'NAME_CONTRACT_STATUS': operation_count, 'MONTHS_BALANCE': operation_mean, 
                 'AMT_TOTAL_RECEIVABLE': operation_mean, 'CNT_DRAWINGS_CURRENT': operation_mean, 
                 'AMT_PAYMENT_TOTAL_CURRENT': operation_mean, 'AMT_CREDIT_LIMIT_ACTUAL': operation_mean,
                 'AMT_BALANCE': operation_mean, 'SK_DPD': operation_mean, 'SK_DPD_DEF': operation_mean}

cc_balance_grouping = cc_balance.groupby(['SK_ID_PREV', 'SK_ID_CURR'])
cc_balance_groupby = cc_balance_grouping.agg(cc_balance_op)
cc_balance_groupby.head()


cc_balance_groupby.columns = ['CC_' + '_'.join(col).strip() for col in cc_balance_groupby.columns.values]
cc_balance_groupby.reset_index(inplace=True)
cc_balance_groupby.head()


ins_payment.head()


ins_payment_op = {'NUM_INSTALMENT_VERSION': operation_mean, 
                 'NUM_INSTALMENT_NUMBER': operation_mean, 'DAYS_INSTALMENT': operation_mean, 
                 'DAYS_ENTRY_PAYMENT': operation_mean, 'AMT_INSTALMENT': operation_mean,
                 'AMT_PAYMENT': operation_mean}

ins_payment_grouping = ins_payment.groupby(['SK_ID_PREV', 'SK_ID_CURR'])
ins_payment_groupby = ins_payment_grouping.agg(ins_payment_op)
ins_payment_groupby.head()


ins_payment_groupby.columns = ['IP_' + '_'.join(col).strip() for col in ins_payment_groupby.columns.values]
ins_payment_groupby.reset_index(inplace=True)
ins_payment_groupby.head()


prev_app.shape


prev_app = pd.merge(prev_app, ins_payment_groupby, on=['SK_ID_CURR', 'SK_ID_PREV'], how='left')
prev_app = pd.merge(prev_app, cc_balance_groupby, on=['SK_ID_CURR', 'SK_ID_PREV'], how='left')
prev_app = pd.merge(prev_app, pos_cash_balance_groupby, on=['SK_ID_CURR', 'SK_ID_PREV'], how='left')
print(prev_app.shape)


prev_app.head()


prev_app.dtypes


pos_in_prevapp = prev_app[prev_app.columns[pd.Series(prev_app.columns).str.startswith('POS_')]]
floats = pos_in_prevapp.select_dtypes('float64').columns
prev_app[floats] = prev_app[floats].fillna(0)


ip_in_prevapp = prev_app[prev_app.columns[pd.Series(prev_app.columns).str.startswith('IP_')]]
floats = ip_in_prevapp.select_dtypes('float64').columns
prev_app[floats] = prev_app[floats].fillna(0)


cc_in_prevapp = prev_app[prev_app.columns[pd.Series(prev_app.columns).str.startswith('CC_')]]
floats = cc_in_prevapp.select_dtypes('float64').columns
prev_app[floats] = prev_app[floats].fillna(0)


objects = prev_app.select_dtypes('object').columns
prev_app[objects] = prev_app[objects].fillna('Unknown')
floats = prev_app.select_dtypes('float64').columns
prev_app[floats] = prev_app[floats].fillna(0)
ints = prev_app.select_dtypes('int64').columns
prev_app[ints] = prev_app[ints].fillna(0)


prev_app.head()


prev_app_op = {'AMT_ANNUITY': operation_mean,
               'AMT_APPLICATION': operation_mean,'AMT_CREDIT': operation_mean,
               'AMT_DOWN_PAYMENT': operation_mean,'AMT_GOODS_PRICE': operation_mean,
               'HOUR_APPR_PROCESS_START': operation_mean, 'NFLAG_LAST_APPL_IN_DAY': operation_mean,
               'RATE_DOWN_PAYMENT': operation_mean,'RATE_INTEREST_PRIMARY': operation_mean,
               'RATE_INTEREST_PRIVILEGED': operation_mean,
               'DAYS_DECISION': operation_mean,'SELLERPLACE_AREA': operation_mean,
               'CNT_PAYMENT': operation_mean,
               'DAYS_FIRST_DRAWING': operation_mean,'DAYS_FIRST_DUE': operation_mean,
               'DAYS_LAST_DUE_1ST_VERSION': operation_mean,'DAYS_LAST_DUE': operation_mean,
               'DAYS_TERMINATION': operation_mean,'NFLAG_INSURED_ON_APPROVAL': operation_mean,
               'IP_NUM_INSTALMENT_VERSION_mean': operation_mean,'IP_NUM_INSTALMENT_NUMBER_mean': operation_mean,
               'IP_DAYS_INSTALMENT_mean': operation_mean,'IP_DAYS_ENTRY_PAYMENT_mean': operation_mean,
               'IP_AMT_INSTALMENT_mean': operation_mean,'IP_AMT_PAYMENT_mean': operation_mean,'CC_MONTHS_BALANCE_mean': operation_mean,
               'CC_AMT_TOTAL_RECEIVABLE_mean': operation_mean,'CC_CNT_DRAWINGS_CURRENT_mean': operation_mean,
               'CC_AMT_PAYMENT_TOTAL_CURRENT_mean': operation_mean,'CC_AMT_CREDIT_LIMIT_ACTUAL_mean': operation_mean,
               'CC_AMT_BALANCE_mean': operation_mean,'CC_SK_DPD_mean': operation_mean,
               'CC_SK_DPD_DEF_mean': operation_mean,
               'POS_MONTHS_BALANCE_mean': operation_mean,'POS_CNT_INSTALMENT_mean': operation_mean,
               'POS_CNT_INSTALMENT_FUTURE_mean': operation_mean,'POS_SK_DPD_mean': operation_mean,
               'POS_SK_DPD_DEF_mean': operation_mean, 'NAME_CONTRACT_TYPE': operation_count,
               'WEEKDAY_APPR_PROCESS_START': operation_count,
               'FLAG_LAST_APPL_PER_CONTRACT': operation_count,'NAME_CASH_LOAN_PURPOSE': operation_count,
               'NAME_CONTRACT_STATUS': operation_count,'NAME_PAYMENT_TYPE': operation_count,
               'NAME_PAYMENT_TYPE': operation_count,
               'CODE_REJECT_REASON': operation_count,'NAME_TYPE_SUITE': operation_count,
               'NAME_CLIENT_TYPE': operation_count,'NAME_GOODS_CATEGORY': operation_count,
               'NAME_PORTFOLIO': operation_count,'NAME_PRODUCT_TYPE': operation_count,
               'CHANNEL_TYPE': operation_count,
               'NAME_SELLER_INDUSTRY': operation_count,
               'NAME_YIELD_GROUP': operation_count,
               'CC_NAME_CONTRACT_STATUS_nunique': operation_sum, 'POS_NAME_CONTRACT_STATUS_nunique': operation_sum}

prev_app_grouping = prev_app.groupby(['SK_ID_CURR'])
prev_app_groupby = prev_app_grouping.agg(prev_app_op)
prev_app_groupby.head()


prev_app_groupby.columns = ['PA_' + '_'.join(col).strip() for col in prev_app_groupby.columns.values]
prev_app_groupby.reset_index(inplace=True)
prev_app_groupby.head()


count_var = prev_app_groupby[prev_app_groupby.columns[pd.Series(prev_app_groupby.columns).str.contains('_nunique')]].columns
prev_app_groupby[count_var] = prev_app_groupby[count_var].astype('int64')


prev_app_groupby.head()


prev_app_groupby.isnull().sum()


bureau_groupby.isnull().sum()


combine = train.append(test)
combine.shape, train.shape, test.shape


combine = pd.merge(combine, bureau_groupby, how='left', on=['SK_ID_CURR'])
combine = pd.merge(combine, prev_app_groupby, how='left', on=['SK_ID_CURR'])
combine.shape, prev_app_groupby.shape, bureau_groupby.shape


print(len(combine.select_dtypes('object').columns))
objects = combine.select_dtypes('object').columns
combine[objects] = combine[objects].fillna('Unknown')
combine.select_dtypes('object').columns


print(len(combine.select_dtypes('float64').columns))
float64 = combine.select_dtypes('float64').columns[1:]
combine[float64] = combine[float64].fillna(combine[float64].mean())
combine.select_dtypes('float64').columns


print(len(combine.select_dtypes('int64').columns))
int64 = combine.select_dtypes('int64').columns
combine[int64] = combine[int64].fillna(combine[int64].mean())
combine.select_dtypes('int64').columns


count_var = combine[combine.columns[pd.Series(combine.columns).str.contains('_nunique')]].columns
combine[count_var] = combine[count_var].astype('int64')


combine.isnull().sum().sum(), combine.shape


combine = pd.get_dummies(combine)
combine.shape


X = combine[combine['TARGET'].isnull()!=True].drop(['TARGET', 'SK_ID_CURR'], axis=1)
y = combine[combine['TARGET'].isnull()!=True]['TARGET'].reset_index(drop=True)

X_test = combine[combine['TARGET'].isnull()==True].drop(['TARGET','SK_ID_CURR'], axis=1)

X.shape, y.shape, X_test.shape


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, roc_auc_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


err_as = []
err_rs = []
err_ps = []
err_roc = []
err_ll = []

y_pred_tot_lgm = []
features = X.columns
feature_importance_df = pd.DataFrame()

fold = StratifiedKFold(n_splits=5)
i = 1
for train_index, test_index in fold.split(X, y):
    x_train, x_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y[train_index], y[test_index]
    m = LGBMClassifier(max_depth=5,
                       learning_rate=0.05,
                       n_estimators=5000,
                       min_child_weight=0.01,
                       colsample_bytree=0.5,
                       random_state=1994)
    m.fit(x_train, y_train,
          eval_set=[(x_train,y_train),(x_val, y_val)],
          early_stopping_rounds=200,
          eval_metric='auc',
          verbose=200)
    pred_y = m.predict(x_val)
    prob_pred = m.predict_proba(x_val)[:,1]
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = m.feature_importances_
    fold_importance_df["fold"] = i + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    print("Fold ",i, " Accuracy: ",(accuracy_score(pred_y, y_val)))
    print("Fold ",i, " Recall: ",(recall_score(pred_y, y_val)))
    print("Fold ",i, " Precision: ",(precision_score(pred_y, y_val)))
    print("Fold ",i, " ROC AUC: ",(roc_auc_score(y_val, prob_pred)))
    print("Fold ",i, " Logloss: ",(log_loss(y_val, prob_pred)))
    print(confusion_matrix(pred_y, y_val))

    err_as.append(accuracy_score(pred_y, y_val))
    err_rs.append(recall_score(pred_y, y_val))
    err_ps.append(precision_score(pred_y, y_val))
    err_roc.append(roc_auc_score(y_val, prob_pred))
    err_ll.append(log_loss(y_val, prob_pred))

    pred_test = m.predict_proba(X_test)[:,1]
    i = i + 1
    y_pred_tot_lgm.append(pred_test)


print('Mean Accuracy Score on CV-5: ', np.mean(err_as, 0))
print('Mean Precision Score on CV-5: ', np.mean(err_ps, 0))
print('Mean Recall Score on CV-5: ', np.mean(err_rs, 0))
print('Mean ROC AUC Score on CV-5: ', np.mean(err_roc, 0))
print('Mean Logloss Score on CV-5: ', np.mean(err_ll, 0))


all_feat = feature_importance_df[["Feature",
                                  "importance"]].groupby("Feature").mean().sort_values(by="importance", 
                                                                                           ascending=False)
all_feat.reset_index(inplace=True)
important_feat = list(all_feat['Feature'])
all_feat.head(20)


df = X[important_feat]
corr_matrix = df.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

high_cor = [column for column in upper.columns if any(upper[column] > 0.98)]
print(len(high_cor))
print(high_cor)


features = [i for i in important_feat if i not in high_cor]
print(len(features))
print(features)


X1 = X[features]
X_test1 = X_test[features]


err_as = []
err_rs = []
err_ps = []
err_roc = []
err_ll = []

y_pred_tot_lgm_1 = []

fold = StratifiedKFold(n_splits=5)
i = 1
for train_index, test_index in fold.split(X1, y):
    x_train, x_val = X1.iloc[train_index], X1.iloc[test_index]
    y_train, y_val = y[train_index], y[test_index]
    m = LGBMClassifier(max_depth=5,
                       learning_rate=0.05,
                       n_estimators=5000,
                       min_child_weight=0.01,
                       colsample_bytree=0.5,
                       random_state=1994)
    m.fit(x_train, y_train,
          eval_set=[(x_train,y_train),(x_val, y_val)],
          early_stopping_rounds=200,
          eval_metric='auc',
          verbose=200)
    pred_y = m.predict(x_val)
    prob_pred = m.predict_proba(x_val)[:,1]
    
    print("Fold ",i, " Accuracy: ",(accuracy_score(pred_y, y_val)))
    print("Fold ",i, " Recall: ",(recall_score(pred_y, y_val)))
    print("Fold ",i, " Precision: ",(precision_score(pred_y, y_val)))
    print("Fold ",i, " ROC AUC: ",(roc_auc_score(y_val, prob_pred)))
    print("Fold ",i, " Logloss: ",(log_loss(y_val, prob_pred)))
    print(confusion_matrix(pred_y, y_val))

    err_as.append(accuracy_score(pred_y, y_val))
    err_rs.append(recall_score(pred_y, y_val))
    err_ps.append(precision_score(pred_y, y_val))
    err_roc.append(roc_auc_score(y_val, prob_pred))
    err_ll.append(log_loss(y_val, prob_pred))

    pred_test = m.predict_proba(X_test1)[:,1]
    i = i + 1
    y_pred_tot_lgm_1.append(pred_test)


print('Mean Accuracy Score on CV-5: ', np.mean(err_as, 0))
print('Mean Precision Score on CV-5: ', np.mean(err_ps, 0))
print('Mean Recall Score on CV-5: ', np.mean(err_rs, 0))
print('Mean ROC AUC Score on CV-5: ', np.mean(err_roc, 0))
print('Mean Logloss Score on CV-5: ', np.mean(err_ll, 0))


submission = pd.DataFrame()
submission['SK_ID_CURR'] = test['SK_ID_CURR']
submission['TARGET'] = np.mean(y_pred_tot_lgm, 0)
submission.head()


submission.to_csv('submission.csv', index=False)


! kaggle competitions submit -c home-credit-default-risk -f submission.csv -m "Home Credit"

