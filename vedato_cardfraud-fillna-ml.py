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


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, metrics, model_selection, svm
import missingno as msno

import numpy as np
import pandas as pd 
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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from warnings import filterwarnings
filterwarnings('ignore')

pd.set_option('display.max_columns', None)


!pip install ycimpute


from ycimpute.imputer import iterforest,EM
from fancyimpute import KNN
from sklearn.preprocessing import OrdinalEncoder


encoder=OrdinalEncoder()
imputer=KNN()

def encode(data):
    '''function to encode non-null data and replace it in the original data'''
    #retains only non-null values
    nonulls = np.array(data.dropna())
    #reshapes the data for encoding
    impute_reshape = nonulls.reshape(-1,1)
    #encode date
    impute_ordinal = encoder.fit_transform(impute_reshape)
    #Assign back encoded values to non-null values
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    return data


#Ktest_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
Ktest_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")
#Ktrain_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
Ktrain_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")


df = Ktrain_transaction.copy()
df2=Ktest_transaction.copy()
del Ktrain_transaction
del Ktest_transaction


"""
# nulls smaller than 60%
print(
    pd.DataFrame(df.isnull().sum()/len(df)>.7)
    .loc[pd.DataFrame(df.isnull().sum()/len(df)>.7)[0]==True].count())

print(
    pd.DataFrame(df2.isnull().sum()/len(df2)>.7)
    .loc[pd.DataFrame(df2.isnull().sum()/len(df2)>.7)[0]==True].count())
"""


"""
df_to_revome_cols=(
    pd.DataFrame(df.isnull().sum()/len(df)>.7)
    .loc[pd.DataFrame(df.isnull().sum()/len(df)>.7)[0]==True].T).columns

df= df.drop(df_to_revome_cols, axis=1)
df.shape
"""


print( df.loc[:,"addr1":"dist2"].isnull().sum()/len(df.loc[:,"addr1":"dist2"]))
print('---------------------')
print(df2.loc[:,"addr1":"dist2"].isnull().sum()/len(df2.loc[:,"addr1":"dist2"]))


df=df.drop(['dist2'], axis=1)
df2=df2.drop(['dist2'], axis=1)


addr= df.loc[:,"addr1":"dist1"]
addr= pd.DataFrame(EM().complete(np.array(addr)), columns=addr.columns)
addr2=df2.loc[:,"addr1":"dist1"]
addr2= pd.DataFrame(EM().complete(np.array(addr2)), columns=addr2.columns)

# no nulls for C series in df
c_series2= df2.loc[:,'C1':'C14']
c_series2= pd.DataFrame(EM().complete(np.array(c_series2)),columns= c_series2.columns)


# This is to check missing data greater than 60%

x= pd.DataFrame(df.loc[:,"D1":"D15"].isnull().sum()/len(df.loc[:,"D1":"D15"])>0.6)
x=x.loc[x[0]==True]
x_col=x.T.columns
print('---------------------')
x2= pd.DataFrame(df2.loc[:,"D1":"D15"].isnull().sum()/len(df2.loc[:,"D1":"D15"])>0.6)
x2= x2.loc[x2[0]==True]
x2_col=x.T.columns

print(x_col)
print(x2_col)


df= df.drop(x_col, axis=1)
df2=df2.drop(x2_col, axis=1)


d_series= df.loc[:,'D1':'D15']
d_series= pd.DataFrame(EM().complete(np.array(d_series)),columns= d_series.columns)

d_series2= df2.loc[:,'D1':'D15']
d_series2= pd.DataFrame(EM().complete(np.array(d_series2)),columns= d_series2.columns)


# cards have some are categorical data
card1card6= df.loc[:,'card1':'card6']
for i in card1card6:
    encode(card1card6[i])
    
card1card6_2= df2.loc[:,'card1':'card6']
for i in card1card6_2:
    encode(card1card6_2[i])
    
card1card6= pd.DataFrame(EM().complete(np.array(card1card6)), columns=card1card6.columns) 
card1card6_2= pd.DataFrame(EM().complete(np.array(card1card6_2)), columns=card1card6_2.columns)


# This is to check missing data greater than 60%

x= pd.DataFrame(df.loc[:,'TransactionID':'R_emaildomain'].isnull().sum()/len(df.loc[:,'TransactionID':'R_emaildomain'])>0.6)
x=x.loc[x[0]==True]
x_col=x.T.columns
print('---------------------')
x2= pd.DataFrame(df2.loc[:,'TransactionID':'R_emaildomain'].isnull().sum()/len(df2.loc[:,'TransactionID':'R_emaildomain'])>0.6)
x2= x2.loc[x2[0]==True]
x2_col=x.T.columns

print(x_col)
print(x2_col)


# we will delete R_emaildomain after filled b/c encode defdoesn't work in series

domain= df.loc[:,'P_emaildomain':'R_emaildomain']
domain2=df2.loc[:,'P_emaildomain':'R_emaildomain']

for i in domain:
    encode(domain[i])
    
for i in domain2:
    encode(domain2[i])
    
domain= pd.DataFrame(EM().complete(np.array(domain)), columns=domain.columns)
domain2= pd.DataFrame(EM().complete(np.array(domain2)), columns=domain2.columns)


df= df.drop(x_col, axis=1)
df2=df2.drop(x2_col, axis=1)


# m_ series
# This is to check missing data greater than 60%

x= pd.DataFrame(df.loc[:,'M1':'M9'].isnull().sum()/len(df.loc[:,'M1':'M9'])>0.6)
x=x.loc[x[0]==True]
x_col=x.T.columns
print('---------------------')
x2= pd.DataFrame(df2.loc[:,'M1':'M9'].isnull().sum()/len(df2.loc[:,'M1':'M9'])>0.6)
x2= x2.loc[x2[0]==True]
x2_col=x.T.columns

print(x_col)
print(x2_col)


# m series
m_series = df.loc[:,'M1':'M9']

for i in m_series:
    encode(m_series[i])
    
m_series= pd.DataFrame(EM().complete(np.array(m_series)), columns= m_series.columns)
# ----------

m_series2 = df2.loc[:,'M1':'M9']
for i in m_series2:
    encode(m_series2[i])
    
m_series2= pd.DataFrame(EM().complete(np.array(m_series2)), columns= m_series2.columns)



# This is to check missing data greater than 60%

x= pd.DataFrame(df.loc[:,'V1':'V11'].isnull().sum()/len(df.loc[:,'V1':'V11'])>0.6)
x=x.loc[x[0]==True]
x_col=x.T.columns
print('---------------------')
x2= pd.DataFrame(df2.loc[:,'V1':'V11'].isnull().sum()/len(df2.loc[:,'V1':'V11'])>0.6)
x2= x2.loc[x2[0]==True]
x2_col=x.T.columns

print(x_col)
print(x2_col)


# v series
v1v11= df.loc[:,'V1':'V11']
v1v11= pd.DataFrame(EM().complete(np.array(v1v11)), columns= v1v11.columns)

v12v34= df.loc[:,'V12':'V34']
v12v34= pd.DataFrame(EM().complete(np.array(v12v34)), columns= v12v34.columns)

v35v52= df.loc[:,"V35":"V52"]
v35v52= pd.DataFrame(EM().complete(np.array(v35v52)), columns=v35v52.columns)

v53v74= df.loc[:,'V53':'V74']
v53v74= pd.DataFrame(EM().complete(np.array(v53v74)), columns=v53v74.columns)


v1v11_2= df2.loc[:,'V1':'V11']
v1v11_2= pd.DataFrame(EM().complete(np.array(v1v11_2)), columns= v1v11_2.columns)

v12v34_2= df2.loc[:,'V12':'V34']
v12v34_2= pd.DataFrame(EM().complete(np.array(v12v34_2)), columns= v12v34_2.columns)
v35v52_2= df2.loc[:,"V35":"V52"]
v35v52_2= pd.DataFrame(EM().complete(np.array(v35v52_2)), columns=v35v52_2.columns)
v53v74_2= df2.loc[:,'V53':'V74']
v53v74_2= pd.DataFrame(EM().complete(np.array(v53v74_2)), columns=v53v74_2.columns)


# This is to check missing data greater than 60%

x= pd.DataFrame(df.loc[:,"V138":"V166"].isnull().sum()/len(df.loc[:,"V138":"V166"])>0.6)
x=x.loc[x[0]==True]
x_col=x.T.columns

#'---------------------')

x2= pd.DataFrame(df2.loc[:,"V138":"V166"].isnull().sum()/len(df2.loc[:,"V138":"V166"])>0.6)
x2= x2.loc[x2[0]==True]
x2_col=x.T.columns

print(x_col)
print('---------------------')
print(x2_col)


v75v94= df.loc[:,"V75":"V94"]
v75v94= pd.DataFrame(EM().complete(np.array(v75v94)), columns=v75v94.columns)
v95v137= df.loc[:,"V95":"V137"]
v95v137= pd.DataFrame(EM().complete(np.array(v95v137)), columns=v95v137.columns)
#v138-v166 dropped
#v167-v278 dropped
v279v321 = df.loc[:,"V279":"V321"]
v279v321= pd.DataFrame(EM().complete(np.array(v279v321)), columns=v279v321.columns)


v75v94_2= df2.loc[:,"V75":"V94"]
v75v94_2= pd.DataFrame(EM().complete(np.array(v75v94_2)), columns=v75v94_2.columns)
v95v137_2= df.loc[:,"V95":"V137"]
v95v137_2= pd.DataFrame(EM().complete(np.array(v95v137_2)), columns=v95v137_2.columns)
#v138-v166 dropped
#v167-v278 dropped
v279v321_2 = df.loc[:,"V279":"V321"]
v279v321_2= pd.DataFrame(EM().complete(np.array(v279v321_2)), columns=v279v321_2.columns)


dms= pd.get_dummies(df['ProductCD'])
dms.columns = ['ProductCD_C', 'ProductCD_H', 'ProductCD_R', 'ProductCD_S','ProductCD_W|']

dms2= pd.get_dummies(df2['ProductCD'])
dms2.columns = ['ProductCD_C', 'ProductCD_H', 'ProductCD_R', 'ProductCD_S','ProductCD_W|']
dms2.head()


domain=domain.drop(["R_emaildomain"], axis=1)
domain2=domain2.drop(["R_emaildomain"], axis=1)


df_ = df.loc[:,"TransactionID":"TransactionAmt"]
df2_= df2.loc[:,"TransactionID":"TransactionAmt"]


print(df_.shape)
print(df2_.shape)


'''
vars_to_removed=(card1card6.columns,
                 addr.columns,
                 d_series.columns,
                 domain.columns,
                 m_series.columns,
                 v1v11.columns,
                 v12v34.columns,
                 v35v52.columns,
                 v53v74.columns,
                 v75v94.columns,
                 v95v137.columns,
                 v279v321.columns)

for i in vars_to_removed:
    df_=df.drop(i, axis=1)

df_=df.drop(["ProductCD"], axis=1)
df_.shape
'''


'''
vars_to_removed=(card1card6_2.columns,
                 addr2.columns,
                 d_series2.columns,
                 domain2.columns,
                 m_series2.columns,
                 v1v11_2.columns,
                 v12v34_2.columns,
                 v35v52_2.columns,
                 v53v74_2.columns,
                 v75v94_2.columns,
                 v95v137_2.columns,
                 v279v321_2.columns)

for i in vars_to_removed:
    df2=df2.drop(i, axis=1)

df2=df2.drop(["ProductCD"], axis=1)
df.shape
'''



vars_to_reload=(card1card6,
                 addr,
                 d_series,
                 domain,
                 m_series,
                 v1v11,
                 v12v34,
                 v35v52,
                 v53v74,
                 v75v94,
                 v95v137,
                 v279v321,
                 dms)

for i in vars_to_reload:
    df_=pd.concat([df_,i], axis=1)
df_.shape


df_.head()


vars_to_repload=(card1card6_2,
                 addr2,
                 d_series2,
                 domain2,
                 m_series2,
                 v1v11_2,
                 v12v34_2,
                 v35v52_2,
                 v53v74_2,
                 v75v94_2,
                 v95v137_2,
                 v279v321_2,
                 dms2)

for i in vars_to_reload:
    df2_=pd.concat([df2_,i], axis=1)
df2_.head()


print(df_.shape)
print(df2_.shape)


import gc
delete = (card1card6,
          addr,
          d_series,
          domain,
          m_series,
          v1v11,
          v12v34,
          v35v52,
          v53v74,
          v75v94,
          v95v137,
          v279v321,
          card1card6_2,
          addr2,
          d_series2,
          domain2,
          m_series2,
          v1v11_2,
          v12v34_2,
          v35v52_2,
          v53v74_2,
          v75v94_2,
          v95v137_2,
          v279v321_2)
for i in delete:
    del i
    gc.collect()


del delete
gc.collect()


Ktrain_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
df1= Ktrain_identity

df1_col= pd.DataFrame(df1.isnull().sum()/len(df1)>.6).loc[(df1.isnull().sum()/len(df1)>.6)==True].T.columns
df1=df1.drop(df1_col, axis=1)
print(df1.shape)
df1.info()


df1_cat = df1.select_dtypes(include='object')
df1_cat.nunique()


df1_cat.isnull().sum()/len(df1_cat)


# crowded cat data
df1_cat1=df1_cat.drop(["id_30","id_31","id_33","DeviceInfo"], axis=1)


df1_cat1.shape


for i in df1_cat1:
    encode(df1_cat1[i])
df1_cat1.head()


df1_cat1= pd.DataFrame(EM().complete(np.array(df1_cat1)), columns=df1_cat1.columns)
df1_cat1.isnull().sum().any()


df1_cat1


df1_cat2=pd.concat([df1['id_30'],df1['id_31'],df1['id_33'],df1['DeviceInfo']], axis=1)
df1_cat2.head()


df1_cat2["id_30"].value_counts(dropna=False).count()


df1_cat2['id_30a']=df1_cat2['id_30'].str.slice(0,3)
df1_cat2['id_30a']=df1_cat2['id_30a'].str.lower()
df1_cat2['id_30a'].value_counts(dropna=False).count()


df1_cat2['id_31'].nunique()


df1_cat2['id_31a']=df1_cat2['id_31'].str.lower()
df1_cat2['id_31a']=df1_cat2['id_31a'].str.slice(0,3)
df1_cat2['id_31a'].nunique()


df1_cat2['DeviceInfo'].nunique()


df1_cat2['DeviceInfo_a']= df1['DeviceInfo'].str.slice(0,2)
df1_cat2['DeviceInfo_a']=df1_cat2['DeviceInfo_a'].str.lower()
df1_cat2['DeviceInfo_a'].nunique()


df1_cat2.head()


df1_cat2= df1_cat2.drop(['id_30','id_31','DeviceInfo'], axis=1)


df1_cat2


for i in df1_cat2:
    encode(df1_cat2[i])
df1_cat2.head()

df1_cat2= pd.DataFrame(EM().complete(np.array(df1_cat2)), columns=df1_cat2.columns)


df1_cat2


#df1_=df1.drop(df1_cat1.columns, axis=1)
#df1_=df1_.drop(['id_30','id_31','id_33','DeviceInfo'],axis=1)
df1_cat= pd.concat([df1_cat1,df1_cat2], axis=1)


df1_cat


df1.info()


df1_cont = df1.select_dtypes(include='float64')
df1_cont.nunique()


df1_cont.isnull().sum()/len(df1_cont)


df1_cont=pd.DataFrame(EM().complete(np.array(df1_cont)), columns=df1_cont.columns)
df1_cont.isnull().sum().any()


df1_=pd.concat([df1["TransactionID"],df1_cont,df1_cat], axis=1 )


df1_


del Ktrain_identity
del df1_cat
del df1_cat1
del df1_cat2
del df1_cont
gc.collect()


Ktest_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
df3= Ktest_identity
del Ktest_identity
df3_col= pd.DataFrame(df3.isnull().sum()/len(df3)>.6).loc[(df3.isnull().sum()/len(df3)>.6)==True].T.columns
df3=df3.drop(df3_col, axis=1)
df3_cat = df3.select_dtypes(include='object')
print(df3.shape)
df3_cat.nunique()


df3_cat.isnull().sum()/len(df3_cat)


# crowded cat data
df3_cat1=df3_cat.drop(["id-30","id-31","id-33","DeviceInfo"], axis=1)

for i in df3_cat1:
    encode(df3_cat1[i])
df3_cat1.head()

df3_cat1= pd.DataFrame(EM().complete(np.array(df3_cat1)), columns=df3_cat1.columns)
df3_cat1


df3_cat2=pd.concat([df3['id-30'],df3['id-31'],df3['id-33'],df3['DeviceInfo']], axis=1)
df3_cat2.shape


# make these cat. data larger groups

df3_cat2['id-30a']=df3_cat2['id-30'].str.slice(0,3)
df3_cat2['id-30a']=df3_cat2['id-30a'].str.lower()

df3_cat2['id-31a']=df3_cat2['id-31'].str.lower()
df3_cat2['id-31a']=df3_cat2['id-31a'].str.slice(0,3)

df3_cat2['DeviceInfo_a']= df3['DeviceInfo'].str.slice(0,2)
df3_cat2['DeviceInfo_a']=df3_cat2['DeviceInfo_a'].str.lower()

df3_cat2=df3_cat2.drop(["id-30", "id-31", "DeviceInfo"], axis=1)

df3_cat2.head()


# covert to numbers / fill nulls
for i in df3_cat2:
    encode(df3_cat2[i])

df3_cat2= pd.DataFrame(EM().complete(np.array(df3_cat2)), columns=df3_cat2.columns)


df3_cat2


df3_cat = pd.concat([df3_cat1, df3_cat2], axis=1)
df3_cat


df3_cont=df3.select_dtypes(include='float64')
df3_cont=pd.DataFrame(EM().complete(np.array(df3_cont)), columns=df3_cont.columns)
df3_=pd.concat([df3["TransactionID"],df3_cont,df3_cat], axis=1 )
print(df3_.shape)
df3_


print(df1_.shape, df3_.shape)



print(df_.shape, df1.shape)
print(df2_.shape, df3.shape)


Ktrain= pd.merge(df_,df1_, on='TransactionID', how='left')
Ktest= pd.merge(df2_,df3_, on='TransactionID', how='left')


print(Ktrain.shape)
print(Ktest.shape)


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


%%time
Ktest = reduce_mem_usage2(Ktest)
Ktrain = reduce_mem_usage2(Ktrain)


print(Ktrain.shape)
Ktrain.head()


print(Ktest.shape)
Ktest.head()


id=Ktrain['TransactionID']
y= Ktrain['isFraud']
X=Ktrain.drop(['isFraud','TransactionID'], axis=1).astype('float64')
X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.25, random_state=40)


lgb_model=LGBMClassifier().fit(X_train,y_train)
y_pred= lgb_model.predict(X_test)
np.sqrt(accuracy_score(y_test, y_pred))


Ktest_id=(Ktest["TransactionID"])
# Ktest_id= Ktest_id.astype("int64")
X_Ktest= Ktest.drop(['TransactionID'], axis=1).astype("float64")
Ktest_pred= lgb_model.predict(X_Ktest)


predictions=lgb_model.predict_proba(X_Ktest)[:,1]
output=pd.DataFrame({'TransactionID':Ktest_id, 'isFraud':predictions })
output=output.loc[pd.DataFrame(output["TransactionID"].isnull())["TransactionID"]==False]
output["TransactionID"]= output["TransactionID"].astype('int64')
output.to_csv('submission_lgbm.csv', index=False)




