# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


data1 = pd.read_csv('../input/application_test.csv')
data1.head()
data2 = pd.read_csv('../input/application_train.csv')


data1 = data1.replace('NaN',np.nan)
data2 = data2.replace('NaN',np.nan)


s1 = data1.shape
s1


s2 = data2.shape
s2


g = []


for col in data2:
    if (data2[col].isnull().sum()/s2[0] > .6 ):
        print(col, data2[col].isnull().sum()/s2[0])
        g.append(col)


g1 = []


for col in data1:
    if (data1[col].isnull().sum()/s1[0] > .6 ):
        print(col, data1[col].isnull().sum()/s1[0])
        g1.append(col)


print(g)
print(g1)


test = data1.drop(g1, axis = 1)
print(test.columns)
train = data2.drop(g, axis = 1)
print(train.columns)


c = train.describe(include='number').columns
d = train.describe(include = 'object').columns


a = test.describe(include='number').columns
b = test.describe(include = 'object').columns


numdata_test = test[a]
objdata_test = test[b]


numdata_train = train[c]
objdata_train = train[d]


objdata_train.isnull().sum()


# numdata_train.isnull().sum()


pd.value_counts(objdata_train.columns).sum()


pd.value_counts(objdata_test.columns).sum()


objdata_test.isnull().sum()


? pd.get_dummies


objdata_train.shape


# for col in objdata_train.iloc[0:15]:
#     i = objdata_train.columns.get_loc(col)
#     #print(i,col)
# #     print((pd.get_dummies(objdata_train[col],drop_first=True)).head())
# #     objdata_train = pd.concat([objdata_train.iloc[:,:i],pd.get_dummies(objdata_train[col],drop_first=True),objdata_train.iloc[:,(i+1):]],axis = 1)


for col in objdata_train:
    objdata_train[col] = objdata_train[col].astype('category',copy=False)
#     objdata_train[col] = objdata_train[col].cat.codes
#     objdata_train[col] = objdata_train[col].astype('category',copy=False)


objdata_train.dtypes


objdata_train.head()


for col in objdata_test:
    objdata_test[col] = objdata_test[col].astype('category')
#     objdata_test[col] = objdata_test[col].cat.codes
#     objdata_test[col] = objdata_test[col].astype('category')





objdata_test.describe()


? pd.DataFrame.astype()


pd.value_counts(objdata_train.columns).sum()


pd.value_counts(objdata_test.columns).sum()


# for col in objdata1:
#      objdata1[col] = objdata1[col].astype('category',copy=False)
#      objdata1[col] = objdata1[col].cat.codes


# for col in objdata:
#      objdata[col] = objdata[col].astype('category',copy=False)
#      objdata[col] = objdata[col].cat.codes


# numdata = data1[a]
# objdata = data1[b]
# print(pd.value_counts(b).sum() + pd.value_counts(a).sum())


from sklearn.preprocessing import Imputer


imp_test = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp2_test = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)


imp_train = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp2_train = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)


imp_test.fit(numdata_test)


imp2_test.fit(objdata_test)


imp_train.fit(numdata_train)


imp2_train.fit(objdata_train)


impnumdata_test = pd.DataFrame(imp_test.transform(numdata_test))
# impnumdata.isnull().sum()
# print(impnumdata.shape)
# print(objdata.shape)


impobjdata_test = pd.DataFrame(imp2_test.transform(objdata_test))


impnumdata_train = pd.DataFrame(imp_train.transform(numdata_train))


impobjdata_train = pd.DataFrame(imp2_train.transform(objdata_train))


# impobjdata1 = pd.DataFrame(imp3.transform(objdata_train))


# impcatdata = pd.DataFrame(imp2.transform(objdata))
# impcatdata.isnull().sum()


impnumdata_test.columns = a
impobjdata_test.columns = b
impnumdata_train.columns = c
impobjdata_train.columns = d



for col in impobjdata_train:
    impobjdata_train[col] = impobjdata_train[col].astype('category')


impobjdata_train.dtypes


for col in impobjdata_test:
    impobjdata_test[col] = impobjdata_test[col].astype('category')


final_data_train = pd.concat([impnumdata_train,impobjdata_train],axis = 1)


# final_data_train.columns


final_data_test = pd.concat([impnumdata_test,impobjdata_test],axis = 1)


# final_data_test.dtypes


# final_data_test.isnull().sum()


from sklearn.utils import shuffle


train_data= final_data_train
train_data.columns
X_1 =train_data[ train_data["TARGET"]==1 ]
X_0=train_data[train_data["TARGET"]==0]
X_0=shuffle(X_0,random_state=42).reset_index(drop=True)
X_1=shuffle(X_1,random_state=42).reset_index(drop=True)

ALPHA=1.2

X_0=X_0.iloc[:round(len(X_1)*ALPHA),:]
final_data_train1=pd.concat([X_1, X_0])



d = pd.value_counts(final_data_train1['TARGET'])
d
c1 = d[0]/(d[0]+d[1] )
c2  = d[1]/(d[0]+d[1])
sizes = [c1,c2]
plot = plt.pie(sizes, labels = ['no','yes'],autopct='%1.1f%%',
        shadow=True, startangle=45 )
plt.axis('equal') 
plt.title("Balanced Data")
plt.show()


y = final_data_train1['TARGET']


x = final_data_train1.drop(['TARGET','SK_ID_CURR'],axis = 1)


X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2)


num = pd.value_counts(y)
num


model = LogisticRegression()


model.fit(x,y)


y_scores = model.predict(X_test)
model.score(X_test,y_test)


roc_auc_score(y_test, y_scores)


sample = pd.read_csv('../input/sample_submission.csv')


pd.value_counts(sample.TARGET)


test = final_data_test.drop('SK_ID_CURR',axis = 1)


y_pred1 = model.predict(test)


sample['TARGET'] = y_pred1


sample.to_csv('model1.csv')


model2 = GradientBoostingClassifier()


model2.fit(x, y)


y_pred = model2.predict(test)


sample['TARGET'] = y_pred


sample.to_csv('model2.csv')










