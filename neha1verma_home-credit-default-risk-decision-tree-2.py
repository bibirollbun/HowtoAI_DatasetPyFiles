import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import gc
import warnings
import time
warnings.filterwarnings("ignore")

%matplotlib inline


#app_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
#app_test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')
df=pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')


df.describe()


def find_missing(data):
    ## Number of missing values
    missing_cnt = data.isnull().sum().values
    ## Total
    total = data.shape[0]
    ##Percentage of Missing values
    percentage = missing_cnt/total * 100
    missing_df = pd.DataFrame(data={'Total': total, 'Missing Count' : missing_cnt,'Percentage' : percentage}, 
                              index=data.columns.values)
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    return missing_df


find_missing(df)


df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


cat_df = df.select_dtypes(include=['object']).copy()
cat_df['TARGET'] = df['TARGET']
cat_df.shape


cat_df.drop(['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'], axis = 1,inplace =True) 
cat_df.head()


find_missing(df_cat)


cat_df.drop(['FONDKAPREMONT_MODE', 'WALLSMATERIAL_MODE','HOUSETYPE_MODE','EMERGENCYSTATE_MODE'], axis = 1,inplace =True) 



cat_df.shape


cat_df.columns


plt.figure(figsize=(11,7))
sns.countplot(x='NAME_TYPE_SUITE',data=cat_df,hue='TARGET')


df['NAME_TYPE_SUITE']=df['NAME_TYPE_SUITE'].fillna('Unaccompanied')
cat_df[cat_df['NAME_TYPE_SUITE']=='Other_A']=cat_df[cat_df['NAME_TYPE_SUITE']=='Children']
cat_df[cat_df['NAME_TYPE_SUITE']=='Other_B']=cat_df[cat_df['NAME_TYPE_SUITE']=='Children']
cat_df[cat_df['NAME_TYPE_SUITE']=='Group of people']=cat_df[cat_df['NAME_TYPE_SUITE']=='Children']
sns.countplot(x='NAME_TYPE_SUITE',data=cat_df,hue='TARGET')


plt.figure(figsize=(11,7))
sns.countplot(x='NAME_INCOME_TYPE',data=cat_df,hue='TARGET')


cat_df['NAME_INCOME_TYPE'].value_counts()



cat_df[cat_df['NAME_INCOME_TYPE']=='Student']=cat_df[cat_df['NAME_INCOME_TYPE']=='Unemployed']
cat_df[cat_df['NAME_INCOME_TYPE']=='Businessman']=cat_df[cat_df['NAME_INCOME_TYPE']=='Unemployed']
cat_df[cat_df['NAME_INCOME_TYPE']=='Maternity leave']=cat_df[cat_df['NAME_INCOME_TYPE']=='Unemployed']
sns.countplot(x='NAME_INCOME_TYPE',data=cat_df,hue='TARGET')


 def plot_stats(df_plot,feature,label_rotation=False,horizontal_layout=True):
    temp = df_plot[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Frequency': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = df_plot[[feature, 'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x = feature, y="Frequency",data=df1)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    s = sns.barplot(ax=ax2, x = feature, y='TARGET', order=cat_perc[feature], data=cat_perc)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    plt.show()


plot_stats(cat_df,'CODE_GENDER')


plot_stats(cat_df,'NAME_TYPE_SUITE')


plot_stats(cat_df,'NAME_INCOME_TYPE')


cat_df['NAME_EDUCATION_TYPE'].value_counts()


plot_stats(cat_df,'NAME_EDUCATION_TYPE')


plot_stats(cat_df,'NAME_FAMILY_STATUS')


plot_stats(cat_df,'WEEKDAY_APPR_PROCESS_START')


plot_stats(cat_df,'NAME_HOUSING_TYPE')


plot_stats(cat_df,'FLAG_OWN_CAR')



plot_stats(cat_df,'FLAG_OWN_REALTY')


plot_stats(cat_df,'NAME_CONTRACT_TYPE')


cat_df.columns


cat_df.drop(['WEEKDAY_APPR_PROCESS_START', 'NAME_TYPE_SUITE'], axis = 1,inplace =True) 


cat_df.apply(pd.Series.nunique, axis = 0)


num_df = df.select_dtypes(exclude=['object']).copy()
num_df.describe()



num_df.columns


find_missing(num_df)


# Find correlations with the target and sort
correlations = num_df.corr()['TARGET'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))


app_num_basic_col = [
'SK_ID_CURR',
'TARGET',
'CNT_CHILDREN',
'AMT_INCOME_TOTAL',
'AMT_CREDIT',
'AMT_ANNUITY',
'AMT_GOODS_PRICE',
'REGION_POPULATION_RELATIVE',
'DAYS_BIRTH',
'DAYS_EMPLOYED',
'DAYS_REGISTRATION',
'DAYS_ID_PUBLISH',
'CNT_FAM_MEMBERS',
'REGION_RATING_CLIENT',
'REGION_RATING_CLIENT_W_CITY',
'REG_REGION_NOT_LIVE_REGION',
'REG_REGION_NOT_WORK_REGION',
'LIVE_REGION_NOT_WORK_REGION',
'REG_CITY_NOT_LIVE_CITY',
'REG_CITY_NOT_WORK_CITY',
'LIVE_CITY_NOT_WORK_CITY',
'EXT_SOURCE_2',
'EXT_SOURCE_3'
]


app_cat_basic_col = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                     'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                    'NAME_HOUSING_TYPE',]


len(app_num_basic_col)


len(app_cat_basic_col)


df = df[app_num_basic_col + app_cat_basic_col]


df.shape


df['TARGET'].value_counts()


find_missing(df[app_num_basic_col])


df['AMT_GOODS_PRICE']=df['AMT_GOODS_PRICE'].fillna(df['AMT_GOODS_PRICE'].median())
df['AMT_ANNUITY']=df['AMT_ANNUITY'].fillna(df['AMT_ANNUITY'].median())
df['CNT_FAM_MEMBERS']=df['CNT_FAM_MEMBERS'].fillna(df['CNT_FAM_MEMBERS'].median())
df['EXT_SOURCE_3']=df['EXT_SOURCE_3'].fillna(df['EXT_SOURCE_3'].median())
df['EXT_SOURCE_2']=df['EXT_SOURCE_2'].fillna(df['EXT_SOURCE_2'].median())


find_missing(df[app_num_basic_col])


find_missing(df[app_cat_basic_col])


basic_features = app_num_basic_col + app_cat_basic_col 


len(basic_features)


find_missing(df[basic_features])


sns.boxplot(data=df['DAYS_EMPLOYED'])


df['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');


# Create an anomalous flag column
#df['DAYS_EMPLOYED_ANOM'] = df["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

df['DAYS_EMPLOYED']=df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].median())

(-1*df['DAYS_EMPLOYED']).plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');


df[df['DAYS_EMPLOYED'] / -365 > 8]['DAYS_EMPLOYED'].count()


(df['DAYS_BIRTH'] / 365).describe()


df['DAYS_BIRTH']


df['DAYS_BIRTH']=abs(df['DAYS_BIRTH'])
df['DAYS_EMPLOYED']=abs(df['DAYS_EMPLOYED'])


df(df['DAYS_BIRTH']/365).describe()


(df['DAYS_EMPLOYED']).describe()


df[df['CODE_GENDER'] == 'XNA']


#df = df[df['CODE_GENDER'] != 'XNA']
df['CODE_GENDER'].value_counts()


cat_df.apply(pd.Series.nunique, axis = 0)


one_hot_encode_col = app_cat_basic_col
one_hot_encode_col


len(one_hot_encode_col)


dummy_df = pd.get_dummies(df[app_cat_basic_col], dummy_na=False, drop_first=True)
dummy_df.head()


len(dummy_df.columns)


df.shape


len(basic_features)


df.drop(one_hot_encode_col, axis=1,inplace=True)


for f in one_hot_encode_col:
    basic_features.remove(f)


len(basic_features)


df.shape


df = pd.concat([df[basic_features], dummy_df], axis=1)


del dummy_df
gc.collect()


df.shape



df.head()



df.describe()


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix


X = df.drop('TARGET',axis=1)
y = df['TARGET']
print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=27)
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)



dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
print(classification_report(y_test,predictions))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=27)
print(X_train.shape[1])
print(X_train.shape[0])


dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
print(classification_report(y_test,predictions))



rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))


pd.value_counts(y_train).plot.bar()
plt.title('histogram')
plt.xlabel('TARGET')
plt.ylabel('Frequency')
df['TARGET'].value_counts()


from imblearn.over_sampling import SMOTE



sm = SMOTE(random_state=27)
X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)
pd.value_counts(y_train_sm).plot.bar()
plt.title('histogram')
plt.xlabel('TARGET')
plt.ylabel('Frequency')

print("Number transactions X_train dataset: ", X_train_sm.shape)
print("Number transactions y_train dataset: ", y_train_sm.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


dtree.fit(X_train_sm,y_train_sm)
predictions = dtree.predict(X_test)
print(classification_report(y_test,predictions))


rfc.fit(X_train_sm, y_train_sm)

rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))


#Standardization
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.transform(X_test)
df_x = pd.DataFrame(X_test_std)
df_x.head()


sm = SMOTE(random_state=27)
X_train_sm, y_train_sm = sm.fit_sample(X_train_std, y_train)
pd.value_counts(y_train_sm).plot.bar()
plt.title('histogram')
plt.xlabel('TARGET')
plt.ylabel('Frequency')


dtree.fit(X_train_sm,y_train_sm)
predictions = dtree.predict(X_test_std)
print(classification_report(y_test,predictions))


rfc.fit(X_train_sm, y_train_sm)

rfc_pred = rfc.predict(X_test_std)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))







