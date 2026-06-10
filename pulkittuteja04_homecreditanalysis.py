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


import numpy as np 
import pandas as pd 
bureau_balance=pd.read_csv('../input/bureau_balance.csv')
bureau = pd.read_csv("../input/bureau.csv")
poshcash=pd.read_csv('../input/POS_CASH_balance.csv')
installement=pd.read_csv('../input/installments_payments.csv')
creditcard=pd.read_csv('../input/credit_card_balance.csv')


bureau_balance=bureau_balance.loc[0:400000,:]
for y in bureau_balance.columns :
    if(bureau_balance[y].isna().any() == True) :
        if( bureau_balance[y].dtype == np.float64) :
            meanvalue = bureau_balance[y].mean()
            bureau_balance[y] = bureau_balance[y].replace(np.nan, meanvalue)
        else :
            z=bureau_balance[y].value_counts().idxmax()
            bureau_balance[y] = bureau_balance[y].replace(np.nan, z)


bureau=bureau.loc[0:400000,:]
for y in bureau.columns :
    if(bureau[y].isna().any() == True) :
        if( bureau[y].dtype == np.float64) :
            meanvalue = bureau[y].mean()
            bureau[y] = bureau[y].replace(np.nan, meanvalue)
        else :
            z=bureau[y].value_counts().idxmax()
            bureau[y] = bureau[y].replace(np.nan, z)


merge_bureau = pd.merge(bureau_balance,bureau, on='SK_ID_BUREAU')


merge_bureau=merge_bureau.drop_duplicates(subset=['SK_ID_BUREAU'])
merge_bureau=merge_bureau.drop_duplicates(subset=['SK_ID_CURR'])


poshcash=poshcash.loc[0:400000,:]


poshcash=poshcash.loc[0:400000,:]


installement=installement.loc[0:400000,:]


creditcard=creditcard.loc[0:400000,]


for y in creditcard.columns :
    if(creditcard[y].isna().any() == True) :
        if( creditcard[y].dtype == np.float64) :
            meanvalue = creditcard[y].mean()
            creditcard[y] = creditcard[y].replace(np.nan, meanvalue)
        else :
            z=creditcard[y].value_counts().idxmax()
            creditcard[y] = creditcard[y].replace(np.nan, z)


creditcard.head()


creditcard.shape


for y in poshcash.columns :
    if(poshcash[y].isna().any() == True) :
        if( poshcash[y].dtype == np.float64) :
            meanvalue = poshcash[y].mean()
            poshcash[y] = poshcash[y].replace(np.nan, meanvalue)
        else :
            z=poshcash[y].value_counts().idxmax()
            poshcash[y] = poshcash[y].replace(np.nan, z)


poshcash.head()


poshcash.shape


for y in installement.columns :
    if(installement[y].isna().any() == True) :
        if( installement[y].dtype == np.float64) :
            meanvalue = installement[y].mean()
            installement[y] = installement[y].replace(np.nan, meanvalue)
        else :
            z=installement[y].value_counts().idxmax()
            installement[y] = installement[y].replace(np.nan, z)


 installement.head()


 installement.shape


merge_1=pd.merge(installement,creditcard, on='SK_ID_CURR')


merge_1.head()


merge_2=pd.merge(merge_1,poshcash, on='SK_ID_CURR')


merge_2.head()


previous_application = pd.read_csv("../input/previous_application.csv")



previous_application=previous_application.iloc[0:400000,]
p_bureau=previous_application.columns[previous_application.isna().any()].tolist()
for col in p_bureau:
        if (previous_application[col].dtype == 'object'): # for object type
            a_bureau=previous_application[col].value_counts().idxmax()
            previous_application[col]=previous_application[col].replace(np.nan,a_bureau)
        elif (previous_application[col].dtype == 'int64'):
            previous_application[col]=previous_application[col].fillna(previous_application[col].mean())
        else:
            previous_application[col]=previous_application[col].fillna(previous_application[col].mean())


previous_application.shape


previous_application.head()


merge_3=pd.merge(merge_2,previous_application, on='SK_ID_CURR')


merge_3.head()


merge_3=merge_3.drop_duplicates(subset=['SK_ID_CURR'])


merge_semi=pd.merge(merge_3,merge_bureau, on='SK_ID_CURR')


merge_semi=merge_semi.drop_duplicates(subset=['SK_ID_CURR'])


application_train = pd.read_csv("../input/application_train.csv")
application_test = pd.read_csv("../input/application_test.csv")
train = application_train.loc[:,['SK_ID_CURR','TARGET','CNT_CHILDREN','AMT_CREDIT','NAME_TYPE_SUITE','NAME_INCOME_TYPE','AMT_INCOME_TOTAL','CNT_FAM_MEMBERS','ORGANIZATION_TYPE','OCCUPATION_TYPE']]
test =  application_test.loc[:,['SK_ID_CURR','CNT_CHILDREN','AMT_CREDIT','NAME_TYPE_SUITE','NAME_INCOME_TYPE','AMT_INCOME_TOTAL','CNT_FAM_MEMBERS','ORGANIZATION_TYPE','OCCUPATION_TYPE']]



train.head()


test.head()


merge_final_train=pd.merge(merge_semi,train, on='SK_ID_CURR')


merge_final_train.shape


merge_final_test=pd.merge(merge_semi,test, on='SK_ID_CURR')


merge_final_test=merge_final_test.drop_duplicates(subset=['SK_ID_CURR'])
merge_final_test=merge_final_test.drop('SK_ID_PREV_x', axis=1 )
merge_final_test=merge_final_test.drop('SK_ID_PREV_y', axis=1 )


merge_final_test.shape


df1=pd.DataFrame(merge_final_test)


merge_final_train=merge_final_train.drop_duplicates(subset=['SK_ID_CURR'])
merge_final_train=merge_final_train.drop('SK_ID_PREV_x', axis=1 )
merge_final_train=merge_final_train.drop('SK_ID_PREV_y', axis=1 )



df=pd.DataFrame(merge_final_train)



from sklearn import preprocessing
temp=0
p=[]
df.drop(['SK_ID_CURR'],axis =1,inplace= True)
df.drop(['SK_ID_BUREAU'],axis =1,inplace= True)
for x in df.columns :
    if(x != 'TARGET') :
        if(df[x].dtype == 'int64' or df[x].dtype == 'uint8' or df[x].dtype == 'float64' ) :
            std_scale = preprocessing.StandardScaler().fit(df[[x]])
            df_std = std_scale.transform(df[[x]])
            minmax_scale = preprocessing.MinMaxScaler().fit(df[[x]])
            df_minmax = minmax_scale.transform(df[[x]])
            i=pd.DataFrame(df_minmax, columns=[[x]])
            p.append(i)
            df[x]=p[temp]
            temp=temp+1
        else :
            df= pd.get_dummies(df , columns=[x])


df.shape


df.head()


from sklearn import preprocessing
temp=0
p=[]
df1.drop(['SK_ID_CURR'],axis =1,inplace= True)
df1.drop(['SK_ID_BUREAU'],axis =1,inplace= True)
for x in df1.columns :
    
    
    #if(x != 'TARGET') :
    if(df1[x].dtype == 'int64' or df1[x].dtype == 'uint8' or df1[x].dtype == 'float64' ) :
        std_scale = preprocessing.StandardScaler().fit(df1[[x]])
        df1_std = std_scale.transform(df1[[x]])
        minmax_scale = preprocessing.MinMaxScaler().fit(df1[[x]])
        df1_minmax = minmax_scale.transform(df1[[x]])
        i=pd.DataFrame(df1_minmax, columns=[[x]])
        p.append(i)
        df1[x]=p[temp]
        temp=temp+1
    else :
        df1= pd.get_dummies(df1 , columns=[x])


df1.shape


df1.head()


missing_cols = (set(df.columns)).symmetric_difference(set(df1.columns))
for c in missing_cols:
    df1[c] = 0


missing_cols = (set(df1.columns)).symmetric_difference(set(df.columns))
for c in missing_cols:
    df[c] = 0


df.shape


df1.shape


df1.drop(['TARGET'],axis =1,inplace= True)


df1.shape


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
X = df.loc[:,df.columns != 'TARGET']
Y = df['TARGET']
#Z = df1
model = Sequential()
model.add(Dense(52, input_dim=213, activation='relu'))
model.add(Dense(25, activation='relu'))
#model.add(Dropout(10.0))
model.add(Dense(12, activation='relu'))
#model.add(Dropout(15.0))
model.add(Dense(1, activation='sigmoid'))
#model.add(Dropout(20.0))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history=model.fit(X, Y, epochs=30,validation_split=0.4, batch_size=400)
#predictions = model.predict(X)
# round predictions
#rounded = [round(x[0]) for x in predictions]
#print(rounded)


predictions = model.predict(df1)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)


from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




