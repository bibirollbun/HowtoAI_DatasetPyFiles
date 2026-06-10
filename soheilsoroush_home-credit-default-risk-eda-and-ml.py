# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("../input/home-credit-default-risk/application_train.csv", index_col=0)
test = pd.read_csv("../input/home-credit-default-risk/application_test.csv", index_col=0)


display(train.head())
display(train.tail())


display(test.head())
display(test.tail())


columns = ["TARGET", "CODE_GENDER", "NAME_CONTRACT_TYPE", "FLAG_OWN_CAR",
           "FLAG_OWN_REALTY", "CNT_CHILDREN","NAME_TYPE_SUITE",
           "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "EMERGENCYSTATE_MODE","FLAG_DOCUMENT_2",
           "FLAG_DOCUMENT_3","FLAG_DOCUMENT_4","FLAG_DOCUMENT_5","FLAG_DOCUMENT_6",
           "FLAG_DOCUMENT_7","FLAG_DOCUMENT_8","FLAG_DOCUMENT_9","FLAG_DOCUMENT_10",
           "FLAG_DOCUMENT_11","FLAG_DOCUMENT_12","FLAG_DOCUMENT_13",
           "FLAG_DOCUMENT_14","FLAG_DOCUMENT_15","FLAG_DOCUMENT_16",
           "FLAG_DOCUMENT_17","FLAG_DOCUMENT_18","FLAG_DOCUMENT_19",
           "FLAG_DOCUMENT_20","FLAG_DOCUMENT_21"]
for column in columns:
  df = pd.DataFrame(train[column].value_counts())
  plt.figure(figsize=(12,12))
  plt.pie(df[column], labels=df.index, autopct='%2.1f%%')
  plt.title(f" {column}")
  plt.show()


 male_zero = train.loc[(train["CODE_GENDER"]=="M")&(train["TARGET"]==0), :].count()[0]
 male_one =  train.loc[(train["CODE_GENDER"]=="M")&(train["TARGET"]==1), :].count()[0]
 female_zero =  train.loc[(train["CODE_GENDER"]=="F")&(train["TARGET"]==0), :].count()[0]
 female_one =  train.loc[(train["CODE_GENDER"]=="F")&(train["TARGET"]==1), :].count()[0]

plt.figure(figsize=(10,5))
sns.countplot(data=train,x="CODE_GENDER", hue="TARGET", )
plt.text(-0.3,100000,male_zero)
plt.text(0.08,15000,male_one)
plt.text(0.7,190000,female_zero)
plt.text(1.1,20000,female_one)
plt.show()


plt.figure(figsize=(10,5))
sns.countplot(data=train,x="FLAG_OWN_CAR", hue="TARGET")
plt.show()


plt.figure(figsize=(10,4))
sns.countplot(data=train,x=pd.cut(train.DAYS_BIRTH/-365.25, bins=3,precision=0, right=True,retbins=False), hue="TARGET")
plt.xlabel("YEARS_BIRTH")
plt.show()


plt.figure(figsize=(10,4))
(train["DAYS_EMPLOYED"]/-365.25).plot.hist(bins=10)
plt.xlabel("YEARS_EMPLOYED")
plt.show()


plt.figure(figsize=(14,10))
sns.countplot(data=train,x=pd.cut(train.DAYS_EMPLOYED/-365.25, bins=5,precision=0, right=True,retbins=False), hue="TARGET")
plt.xlabel("YEARS_EMPLOYED")
plt.show()


plt.figure(figsize=(15,10))
sns.countplot(data=train, x="NAME_INCOME_TYPE", hue="TARGET")
plt.show()


plt.figure(figsize=(15,10))
sns.countplot(data=train,x="NAME_EDUCATION_TYPE", hue="TARGET", )
plt.show()



plt.figure(figsize=(15,10))
sns.countplot(data=train,x="NAME_FAMILY_STATUS", hue="TARGET", )
plt.show()



plt.figure(figsize=(15,8))
sns.countplot(data=train,x=pd.cut(train.DAYS_LAST_PHONE_CHANGE/-365.25, bins=12,precision=0, right=True,retbins=False), hue="TARGET")
plt.xlabel("YEARS_LAST_PHONE_CHANGE")
plt.show()


name_income_type=train.NAME_INCOME_TYPE.unique()
for name in name_income_type:
  plt.figure(figsize=(7,5))
  data=train.loc[(train.NAME_INCOME_TYPE==name), :]
  sns.countplot(data = data, x= "NAME_INCOME_TYPE", hue=train["TARGET"])
  plt.show()



pd.DataFrame(train.isnull().sum(), columns=["MISSING_VALUES"]).sort_values(by="MISSING_VALUES", ascending=False).head(30)


#categorical columns
pd.DataFrame(train.select_dtypes('object').apply(pd.Series.nunique, axis = 0),
             columns=["Categ_Data"]).sort_values(by="Categ_Data", ascending=False)


#categorical data in test data set
pd.DataFrame(train.select_dtypes('object').apply(pd.Series.nunique, axis = 0),
             columns=["Categ_Data"]).sort_values(by="Categ_Data", ascending=False)


colomns = ["HOUR_APPR_PROCESS_START", "EXT_SOURCE_1", "APARTMENTS_MODE",
           "YEARS_BEGINEXPLUATATION_MODE","DAYS_LAST_PHONE_CHANGE",
           "REGION_POPULATION_RELATIVE", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
           "DAYS_EMPLOYED", "AMT_ANNUITY", "AMT_CREDIT", "AMT_INCOME_TOTAL"
           ]

for colomn in colomns:
  plt.figure(figsize=(12,6))
  plt.subplot(1,2,1)
  train[colomn].plot(kind = "box")
  plt.subplot(1,2,2)
  train[colomn].plot(kind = "hist")
  plt.show()


train.AMT_INCOME_TOTAL.sort_values(ascending=False)


useless_columns = ["HOUSETYPE_MODE", "WALLSMATERIAL_MODE","FLAG_DOCUMENT_2",
                     "FLAG_DOCUMENT_4","FLAG_DOCUMENT_5","FLAG_DOCUMENT_7",
                     "FLAG_DOCUMENT_9","FLAG_DOCUMENT_10","FLAG_DOCUMENT_12",
                     "FLAG_DOCUMENT_15","FLAG_DOCUMENT_17","FLAG_DOCUMENT_19",
                   "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21","AMT_REQ_CREDIT_BUREAU_HOUR",
                   "YEARS_BUILD_MODE","ELEVATORS_MODE","ENTRANCES_MODE", "FLOORSMAX_MODE",
                     "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI",
                   "FLOORSMAX_MEDI", "FLOORSMIN_MEDI","ELEVATORS_AVG","ENTRANCES_AVG",
                   "FLOORSMAX_AVG","FLOORSMIN_AVG","LIVINGAPARTMENTS_AVG",
                   "NONLIVINGAPARTMENTS_AVG","ORGANIZATION_TYPE"]


train = train.drop(columns = useless_columns)
test = test.drop(columns=useless_columns)
train.shape, test.shape


#Fill the missing categorical data with the most frequent velue
train_categorical = train.select_dtypes(include ='object') 
train_categorical = train_categorical.apply(lambda x: x.fillna(x.value_counts().index[0]))

#Fill the missing numerical data with the median 
train_float = train.select_dtypes(include ='float64') 
train_float = train_float.fillna(train_float.median())

train_int = train.select_dtypes(include ='int64') 
train_int = train_int.fillna(train_int.median())

train_final = pd.concat([train_categorical,train_float,train_int], axis=1)
train_final["TARGET"] = train["TARGET"]
train=train_final


#fill missing categorical values in test data set with most frequent values in 
#training data set (don't use any info from test data set)

categorical_columns = test.select_dtypes(include ='object').columns
test_categorical = test.select_dtypes(include ='object') 

for column in categorical_columns:
  test_categorical[column] = test_categorical[column].fillna(train[column].value_counts().index[0])

#for numerical missing values in test data set, fill them with median value in 
#the training data set (don't use any info from test data set)
test_float = test.select_dtypes(include ='float64') 
test_float = test_float.fillna(train_float.median())

test_int = test.select_dtypes(include ='int64') 
test_int = test_int.fillna(train_int.median())

test_final = pd.concat([test_categorical,test_float,test_int], axis=1)
test=test_final


train.shape,test.shape


train.loc[train.DAYS_EMPLOYED==train.DAYS_EMPLOYED.max(), :].NAME_INCOME_TYPE.unique()


train.DAYS_EMPLOYED.max()


#replace anomalies with the extreme value in range of normal data:
#for years of employment, the maximum number should be 50 years for pensioners,
#and 0 for unemployed clients
train_dataframe=train.copy()
test_dataframe = test.copy()
def unemployed_anomaly(x):
  if x==365243:
    x= 0
  return x

def pensioner_anomaly(x):
  if x==365243:
    x=-18263
  return x

train_unemployed = train_dataframe.loc[train_dataframe.NAME_INCOME_TYPE=="Unemployed",:]
train_pensioner = train_dataframe.loc[train_dataframe.NAME_INCOME_TYPE=="Pensioner",:]
train_other = train_dataframe.loc[(train_dataframe.NAME_INCOME_TYPE!="Unemployed")&(train_dataframe.NAME_INCOME_TYPE!="Pensioner"),:]
train_unemployed.DAYS_EMPLOYED = train_unemployed.DAYS_EMPLOYED.apply(unemployed_anomaly)
train_pensioner.DAYS_EMPLOYED=train_pensioner.DAYS_EMPLOYED.apply(pensioner_anomaly)
train_dataframe=pd.concat([train_unemployed,train_pensioner,train_other],axis=0)

test_unemployed = test_dataframe.loc[test_dataframe.NAME_INCOME_TYPE=="Unemployed",:]
test_pensioner = test_dataframe.loc[test_dataframe.NAME_INCOME_TYPE=="Pensioner",:]
test_other = test_dataframe.loc[(test_dataframe.NAME_INCOME_TYPE!="Unemployed")&(test_dataframe.NAME_INCOME_TYPE!="Pensioner"),:]
test_unemployed.DAYS_EMPLOYED=test_unemployed.DAYS_EMPLOYED.apply(unemployed_anomaly)
test_pensioner.DAYS_EMPLOYED=test_pensioner.DAYS_EMPLOYED.apply(pensioner_anomaly)
test_dataframe=pd.concat([test_unemployed,test_pensioner,test_other],axis=0)

display(test_dataframe.head())
display(train_dataframe.head())


train_dataframe.DAYS_EMPLOYED.min()/365.25


train=train_dataframe
test=test_dataframe


#RATIOS:
#1- amount of annual payment to annual income:
train_df = train.copy()
train_df["PAYMENT_RATIO"] = train_df["AMT_ANNUITY"]/train_df["AMT_INCOME_TOTAL"]
train_df = train_df.drop(columns=["AMT_ANNUITY"])
test_df = test.copy()
test_df["PAYMENT_RATIO"] = test_df["AMT_ANNUITY"]/test_df["AMT_INCOME_TOTAL"]
test_df = test_df.drop(columns=["AMT_ANNUITY"])



train_df.head()


#Years Instead of Days!
def year_convertor(dataframe):
  df = dataframe.copy()
  df["AGE"] = df["DAYS_BIRTH"].apply(lambda x: np.int(-x/365))
  df["YEARS_EMPLOYED"] = df["DAYS_EMPLOYED"].apply(lambda x: np.int(-x/365))
  df["YEARS_REGISTERED"] = df["DAYS_REGISTRATION"].apply(lambda x: np.int(-x/365))
  df["YEARS_ID_PUBLISHED"] = df["DAYS_ID_PUBLISH"].apply(lambda x: np.int(-x/365))
  df["YEARS_LAST_PHONE_CHANGE"]=df["DAYS_LAST_PHONE_CHANGE"].apply(lambda x: np.int(-x/365))
  df = df.drop(columns = ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH","DAYS_LAST_PHONE_CHANGE"])
  return df


 training_df = year_convertor(train_df)
 testing_df = year_convertor(test_df)
training_df.head()


train=training_df
test=testing_df


train_1 = train.copy()
test_1 = test.copy()

#Binary Categorical Data (Y/N,F/M):
train_1['FLAG_OWN_CAR'] = train_1['FLAG_OWN_CAR'].replace(to_replace = ['Y', 'N'], value = [1,0] )
train_1['FLAG_OWN_REALTY'] = train_1['FLAG_OWN_REALTY'].replace(['Y', 'N'], [1, 0])
train_1['EMERGENCYSTATE_MODE'] = train_1['EMERGENCYSTATE_MODE'].replace(['Yes', 'No'], [1, 0])
train_1['CODE_GENDER'] = train_1['CODE_GENDER'].replace(['M', 'F'], [1, 0])

test_1['FLAG_OWN_CAR'] = test_1['FLAG_OWN_CAR'].replace(['Y', 'N'], [1, 0])
test_1['FLAG_OWN_REALTY'] = test_1['FLAG_OWN_REALTY'].replace(['Y', 'N'], [1, 0])
test_1['EMERGENCYSTATE_MODE'] = test_1['EMERGENCYSTATE_MODE'].replace(['Yes', 'No'], [1, 0])
test_1['CODE_GENDER'] = test_1['CODE_GENDER'].replace(['M', 'F'], [1, 0])
train_1.head()


train=train_1
test=test_1


#one-hot coding on categorical columns:
train = pd.get_dummies(train)
test = pd.get_dummies(test)
train.head()


train.shape,test.shape


#saving the targets firs:
train_labels = train['TARGET']

# Align the training and testing data
train, test = train.align(test, join = 'inner', axis = 1)

# Add the target
train['TARGET'] = train_labels


train.shape,test.shape


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


X_train, X_test, y_train, y_test = train_test_split(train.drop(columns="TARGET"),
                                                    train["TARGET"], test_size=0.10,
                                                    random_state=101)


X_train.shape,X_test.shape


reg_algorithm = LogisticRegression()
reg_algorithm.fit(X_train,y_train)
predictions = reg_algorithm.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


dt_model=DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
dt_pred = dt_model.predict(X_test)
print(confusion_matrix(y_test,dt_pred))
print(classification_report(y_test,dt_pred))


rf_model= RandomForestClassifier(n_estimators=300)
rf_model.fit(X_train,y_train)
rf_pre=rf_model.predict(X_test)
print(confusion_matrix(y_test,rf_pre))
print(classification_report(y_test,rf_pre))


xgb_model = XGBClassifier(n_estimators=300)
xgb_model.fit(X_train,y_train)
xg_pred = xgb_model.predict(X_test)
print(confusion_matrix(y_test,xg_pred))
print(classification_report(y_test,xg_pred))


X=train.drop(columns="TARGET")
y=train.TARGET
tuned_parameters = [{'n_estimators': [200,350], 'max_depth' : [5,10]}]
clf = GridSearchCV(XGBClassifier(), tuned_parameters, cv=5, scoring='roc_auc', return_train_score=True)
clf.fit(X, y)
print(clf.classes_)
print ("best score = ", clf.best_score_)
print("Best parameters set found on development set: ", clf.best_params_)
maxDepth=clf.best_params_["max_depth"]
nEstimator = clf.best_params_["n_estimators"]

xgb_model=XGBClassifier(max_depth=maxDepth,n_estimators=nEstimator)
X = train.drop(columns="TARGET")
y = train.TARGET
xgb_model=xgb_model.fit(X,y)
test_labels = xgb_model.predict_proba(test)



submission = pd.DataFrame({
    "SK_ID_CURR": test.index,
    "TARGET" : test_labels[:,1]
})
submission.to_csv('submission.csv', index=False)


