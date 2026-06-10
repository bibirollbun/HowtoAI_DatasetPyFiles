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


pd.options.display.max_columns=300
pd.options.display.max_rows=300


train=pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
test=pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")
display(train.head(),test.head())


train.isnull().sum()


train.dtypes


display(train.shape,
test.shape)


train.head(5)


import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(20,12))
train.boxplot()


plt.figure(figsize=(20,12))
train.iloc[:,0:10].boxplot()


train.boxplot(column="AMT_INCOME_TOTAL",return_type = 'axes')


train["AMT_INCOME_TOTAL"].describe()


train = train[(train.AMT_INCOME_TOTAL < 5000000)]


plt.figure(figsize=(20,12))
train.boxplot()


train.shape


train.describe()


(train["DAYS_BIRTH"]/-365).describe()


train["DAYS_EMPLOYED"].describe()


train.boxplot(column="DAYS_EMPLOYED",return_type = 'axes')


plt.figure(figsize =(20,12))
train['DAYS_EMPLOYED'] = (train['DAYS_EMPLOYED'].values)
plt.hist(train['DAYS_EMPLOYED'].values, bins=100)
plt.xlabel("DAYS_EMPLOYED")
plt.ylabel("count")
plt.show()


outlier = train[train['DAYS_EMPLOYED']==365243]
normal = train[train['DAYS_EMPLOYED']!=365243]


outlier["TARGET"].mean()*100


normal["TARGET"].mean()*100


train["DAYS_EMPLOYED_OUTLIER"] = train["DAYS_EMPLOYED"] == 365243


train["DAYS_EMPLOYED"].replace({365243: np.nan},inplace=True)


test['DAYS_EMPLOYED_OUTLIER'] = test["DAYS_EMPLOYED"] == 365243
test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)


train.head()


train.select_dtypes('object').apply(pd.Series.nunique,axis=0)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le_count=0

for col in train:
    if train[col].dtype == 'object':
        if len(list(train[col].unique())) == 2:
            le.fit(train[col])
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])
            le_count+=1
print(le_count)


train = pd.get_dummies(train)
test = pd.get_dummies(test)


y_train = train["TARGET"]
train , test = train.align(test,join='inner',axis=1)
train["TARGET"] = y_train

display(train.shape,test.shape)


display(train.shape,test.shape)


display(train, test)


from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

y_train = train["TARGET"]
x_train = train.drop(["TARGET"],axis=1)
feature = list(x_train.columns)

imputer = SimpleImputer(strategy = 'median')
scaler = MinMaxScaler(feature_range = (0,1))

imputer.fit(x_train)

x_train = imputer.transform(x_train)
test = imputer.transform(test)

scaler.fit(x_train)
x_train = scaler.transform(x_train)
test = scaler.transform(test)

display(x_train.shape,test.shape)


%%time
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, n_jobs=4, verbose=1,random_state=77)

model.fit(x_train,y_train)


result = model.predict_proba(test)[:,1]
result


sub = pd.read_csv("/kaggle/input/home-credit-default-risk/sample_submission.csv")


sub


sub["TARGET"] = result
sub.head(10)


sub.to_csv("home_credit_default_risk.csv",index=False)

