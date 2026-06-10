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


import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


app_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
app_test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')


app_train['data_type'] = 'train'
app_test['data_type'] = 'test'


app = pd.concat([app_train,app_test])


app.dtypes.value_counts()


# Katagrik değişkenlerin sınıfları
app_train.select_dtypes(include = [object]).apply(pd.Series.nunique, axis = 0)


mis_value = app_train.isnull().sum()
mis_value_percent = 100*app_train.isnull().sum()/len(app_train)
mis_value_table = pd.concat([mis_value,mis_value_percent], axis = 1)
mis_value_table.columns=['count', 'percent']
mis_value_table = mis_value_table.sort_values('percent',ascending=False)
mis_value_table = mis_value_table[mis_value_table['percent']>0]


# tüm değişkenleri görünütleyebilmek için
pd.set_option('display.max_rows', None)
mis_value_table


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,6))
ax = fig.add_axes([0,0,1,1])
ax.bar(mis_value_table.index,mis_value_table.percent, color = 'purple')
plt.xticks(rotation =90,fontsize =10)
plt.title('Missing Data')
plt.xlabel('Feature')
plt.ylabel('% Percentage')
plt.show()


app_train['TARGET'].value_counts()


plt.hist(app_train.TARGET)
plt.show()


# TARGET değişkeni ile diğer değişkenler arasındaki correlationlar
correlations = app.corr()['TARGET'].sort_values()

# Pozitif ve negatif yönlü en güçlü correlationlar
print('Negatif correlations:\n',correlations.head(20))
print('Pozitif correlations:\n',correlations.tail(20))


app.describe().T


app['CNT_CHILDREN'].value_counts()


sns.kdeplot(app['AMT_INCOME_TOTAL']/1000)


app.head()


IQR = app.describe().T
IQR['lower'] = IQR['25%']-1.5*(IQR['75%']-IQR['25%'])
IQR['upper'] = IQR['75%']+1.5*(IQR['75%']-IQR['25%'])
IQR.T


# Bu kısım çalışıyor ama sonuçlanmıyor
#out = app[app>IQR.upper]
#out


ıqr=202500.000000+1.5*(202500.000000+112500.000000)
ıqr


len(app[app['AMT_INCOME_TOTAL']>ıqr])


app['CONTACT'] = app.loc[:,'FLAG_MOBIL':'FLAG_EMAIL'].sum(axis = 1)


app.drop(app[['FLAG_MOBIL','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL']],axis=1,inplace=True)


app['CONTACT'].corr(app['TARGET'])


app.head()


# EŞLEŞMEYEN ADRES değişkenlerini toplamından WRONG_DRESS değişkeni oluşturma
app['WRONG_ADRESS'] = app.loc[:,'REG_REGION_NOT_LIVE_REGION':'LIVE_CITY_NOT_WORK_CITY'].sum(axis = 1)


app['WRONG_ADRESS'].corr(app['TARGET'])


# drop işlemi
app.drop(app.loc[:,'REG_REGION_NOT_LIVE_REGION':'LIVE_CITY_NOT_WORK_CITY'],axis=1,inplace=True)


list(app.columns)


app.head()


# # FLAG_DOCUMENT değişkenlerini toplamından DOCUMENT_SUM değişkeni oluşturma
app['DOCUMENT_SUM'] = (app.loc[:,'FLAG_DOCUMENT_2':'FLAG_DOCUMENT_21'].sum(axis = 1))**(1/2)


app['DOCUMENT_SUM'].corr(app['TARGET'])


# drop işlemi
# FLAG_DOCUMENT_3 nisbeten yüksek corr'a sahip olduğundan drop edilmedi
app.drop(app.loc[:,'FLAG_DOCUMENT_4':'FLAG_DOCUMENT_21'],axis=1,inplace=True)



app.drop('FLAG_DOCUMENT_2',axis=1,inplace=True)


app['REGION_RATING_CLIENT_W_CITY'].corr(app['REGION_RATING_CLIENT'])


app['REGION_RATING_CLIENT_W_CITY'].corr(app['TARGET'])


app['TARGET'].corr(app['REGION_RATING_CLIENT'])


# REGION_RATING_CLIENT, TARGET ile daha düşük bir corr sahip olduğundan drop ediliyor
app.drop('REGION_RATING_CLIENT',axis=1,inplace=True)


app.loc[:,'APARTMENTS_AVG':'EMERGENCYSTATE_MODE'].sample(10)


# APARTMENTS_AVG-NONLIVINGAREA_MEDI veriler toplanıp TARGET değişkeni ile corr hesaplanıyor
app['TARGET'].corr(app.loc[:,'APARTMENTS_AVG':'NONLIVINGAREA_MEDI'].sum(axis = 1))


app['TARGET'].corr((app.loc[:,'APARTMENTS_AVG':'NONLIVINGAREA_MEDI'].sum(axis = 1))*(4-app['REGION_RATING_CLIENT_W_CITY']))



df = app.dropna()


df['TARGET'].corr((df.loc[:,'APARTMENTS_AVG':'NONLIVINGAREA_AVG'].sum(axis = 1))*(4-df['REGION_RATING_CLIENT_W_CITY']))


# Null veriler Median ile dolduruluyor. 
for col in app.loc[:,'APARTMENTS_AVG':'NONLIVINGAREA_MEDI'].columns:
    app[col] = app[col].fillna(app[col].median())


# Null verilerin manüplasyonu correlationu etkilemedi
df['TARGET'].corr((df.loc[:,'APARTMENTS_AVG':'NONLIVINGAREA_AVG'].sum(axis = 1))*(4-df['REGION_RATING_CLIENT_W_CITY']))


# LIVINGAREA*REGION değişkeni veriye ekleniyor
app['LIVINGAREA*REGION'] = (app.loc[:,'APARTMENTS_AVG':'NONLIVINGAREA_AVG'].sum(axis = 1))*(4-app['REGION_RATING_CLIENT_W_CITY'])


app['LIVINGAREA*REGION'].head()


# bu bölümdeki değişkenler drop ediliyor
app.drop(app.loc[:,'APARTMENTS_AVG':'NONLIVINGAREA_MEDI'],axis=1,inplace=True)


list(app.columns)


app['CODE_GENDER'].value_counts()


app.drop(app[app['CODE_GENDER']=='XNA'].index, inplace=True)



app['CODE_GENDER'].value_counts()


app['DAYS_EMPLOYED'].describe()


len(app[app['DAYS_EMPLOYED'] == 365243])


app['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)



app_corr = app.corr()


?sns.heatmap


plt.figure(figsize = (28, 26))

# Heatmap of correlations
sns.heatmap(app_corr, cmap = plt.cm.RdYlBu_r, vmin = -0.25,fmt='.0g', annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');


# Eksik değerler. En az bir eksi deger bulunan gözlem(axis=1)
# app[app.isnull().any(axis=1)]


# yukardaki koodun tam olarak değili
# app[app.notnull().all(axis=1)]


# eksik verileri doldurma
# app['CODE_GENDER'].fillna(df['CODE_GENDER'].mean())


# tüm eksik verileri doldurmak için
# 1.yol
# app.apply(lambda x: x.fillna(x.mean()),axis=0)
# 2.yol
# app.fillna(app.mean()[:])
# bazı değişkenlere uygulamak için
# app.fillna(app.mean()['v1':'v2'])
# 3.yol
# app.where(pd.notna(app), app.mean(), axis='columns')


# Tüm değişkenleri NaN olan gözlemlerin silinmesi
# app.dropna(how='all')


# Değişken bazlı silme
# app.dropna(axis=1)
# Tüm gözlemleri eksik olan değişkenin silinmesi
# app.dropna(axis=1, how='all')


#!pip install missingno


import missingno as msno


msno.bar(app);


msno.matrix(app.sample(200));


msno.heatmap(app);


app1 = app[['EXT_SOURCE_3','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY',
            'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
            'AMT_REQ_CREDIT_BUREAU_YEAR']]


app1_corr = app1.corr()


plt.figure(figsize = (8, 6))

# Heatmap of correlations
sns.heatmap(app1_corr, cmap = plt.cm.RdYlBu_r, vmin = -0.25,fmt='.0g', annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');


len(app1[app1.notnull().all(axis=1)])


len(app1[app1.isnull().any(axis=1)])


app1.sample(100)


app2 = app[['HOUSETYPE_MODE','TOTALAREA_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE']]


app2.sample(100)


app2['HOUSETYPE_MODE'].value_counts()


app2['WALLSMATERIAL_MODE'].value_counts()


app2['EMERGENCYSTATE_MODE'].value_counts()


app3 = app[['AMT_GOODS_PRICE', 'NAME_TYPE_SUITE']]


app3.sample(100)


app3.describe()/1000


len(app3[app3['AMT_GOODS_PRICE']>1.336500e+06])


# app.loc[app['AMT_GOODS_PRICE'] > 1.336500e+06,'AMT_GOODS_PRICE']=np.nan
# veya
# app['AMT_GOODS_PRICE'] = np.where(app['AMT_GOODS_PRICE'] > 1.336500e+06, np.nan, app['AMT_GOODS_PRICE'])


len(app[app['AMT_GOODS_PRICE']>1.336500e+06])


len(app[app['AMT_GOODS_PRICE'].isnull()])


app_cat = app.select_dtypes(include = [object])
del app_cat['data_type']


#app.select_dtypes(include = [object]).apply(pd.Series.nunique, axis = 0)


app_cat.select_dtypes(include = [object]).apply(pd.Series.nunique, axis = 0)


for col in list(app_cat.columns):
    app_cat[col].fillna(method = "ffill")


list(app_cat.columns)


app_cat.isnull().sum()


app_cat.head()


print('app_cat shape:',app_cat.shape)


app_cat[['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','EMERGENCYSTATE_MODE']].head()


from sklearn.preprocessing import LabelEncoder
# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in app_cat:
    if app_cat[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(app_cat[col].unique())) <= 2:
            # Train on the training data
            le.fit(app_cat[col])
            # Transform both training and testing data
            app_cat[col] = le.transform(app_cat[col])
                        
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)


app_cat.head()


app_cat = pd.get_dummies(app_cat)


print('app_cat shape:',app_cat.shape)


app_cat.head()


import seaborn as sns
import missingno as msno


app_num = app.select_dtypes(include = ['float64','int64'])
app_num.drop(['SK_ID_CURR','TARGET'],axis=1,inplace=True)


app_num.isnull().sum()


!pip install ycimpute


from ycimpute.imputer import EM


var_names = list(app_num) 


var_names


np_app_num =np.array(app_num)


dff = EM().complete(np_app_num)


dff = pd.DataFrame(dff, columns = var_names)


dff.isnull().sum()


dff.head()


dff_corr = dff.corr()
plt.figure(figsize = (28, 26))

# Heatmap of correlations
sns.heatmap(dff_corr, cmap = plt.cm.RdYlBu_r, vmin = -0.25,fmt='.0g', annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');


from sklearn.neighbors import LocalOutlierFactor


# n_neighbors = 10 komşuluk sayısı, contamination = 0.1 saydamlık
clf = LocalOutlierFactor(n_neighbors = 10, contamination = 0.1)


clf.fit_predict(dff)


# negatif skorlar 
dff_scores = clf.negative_outlier_factor_


np.sort(dff_scores)[0:1000]


esik_deger = np.sort(dff_scores)[7]
esik_deger


len(dff[dff_scores<esik_deger])


dff[dff_scores==esik_deger]


# eşik skora sahip gözlem baskılama verisi olarak belirleniyor
baskılama_deg = dff[dff_scores==esik_deger]


# esik skordan daha küçük skora sahip gözlemler için True-False şeklinde ARRAY oluşturulyor
outlier_array = dff_scores<esik_deger
outlier_array


# outlier_array'ın döndürdüğü True-False değerler ile filtreleme yapılarak Outlier gözlemler ile DATAFRAME oluşturuluyor
outlier_df = dff[outlier_array]
len(outlier_df)


outlier_df 


# outlier_df indexlerinden arındırılarak ARRAY'a dönüştürülüyor.
outlier_df.to_records(index=False)


# Bu array res olarak tutuluyor.
res = outlier_df.to_records(index=False)


# res'deki tüm veriler yerine baskılama dergerleri atanıyor
res[:] = baskılama_deg.to_records(index=False)


res


dff[outlier_array]


# Bir array olan res aykırı gözlemlerin indexleri kullanılarak DATAFRAME dönüştürülüyor ve dff deki aykırı gözlemlerin yerine atanyor
dff[outlier_array] = pd.DataFrame(res, index = dff[outlier_array].index)


dff[outlier_array]


df_app = app[['data_type','SK_ID_CURR','TARGET']]


print(df_app.shape,dff.shape,app_cat.shape)


list(df_app.columns)


list(dff.columns)


list(app_cat.columns)


dff.head()


app_cat.head()


app.head()


dff.insert(0,'data_type',app['data_type'].values)
dff.insert(1,'SK_ID_CURR',app['SK_ID_CURR'].values)
dff.insert(2,'TARGET',app['TARGET'].values)
app_cat.insert(0,'SK_ID_CURR',app['SK_ID_CURR'].values)


dff.head()


app_cat.head()


df = pd.merge( dff, app_cat, on='SK_ID_CURR')


df.head()


df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']


correlations = df.corr()['TARGET'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))


df.head()


import re
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))


df_train = df[df['data_type']=='train']
del df_train['data_type']
df_train.shape


df_test = df[df['data_type']=='test']
del df_test['data_type']
df_test.shape


del df_test['TARGET']


df_test.head()


plt.figure(figsize = (10, 8))

# KDE plot of loans that were repaid on time
sns.kdeplot(df_train.loc[df_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(df_train.loc[df_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')

# Labeling of plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');


plt.figure(figsize = (10, 8))

# KDE plot of loans that were repaid on time
sns.kdeplot(df_train.loc[df_train['TARGET'] == 0, 'CNT_CHILDREN'] , label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(df_train.loc[df_train['TARGET'] == 1, 'CNT_CHILDREN'] , label = 'target == 1')

# Labeling of plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');



plt.figure(figsize = (10, 8))

# KDE plot of loans that were repaid on time
sns.kdeplot(df_train.loc[df_train['TARGET'] == 0, 'EXT_SOURCE_2'] , label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(df_train.loc[df_train['TARGET'] == 1, 'EXT_SOURCE_2'] , label = 'target == 1')

# Labeling of plot
plt.xlabel('EXT_SOURCE_2'); plt.ylabel('Density'); plt.title('Distribution of EXT_SOURCE_2');


len(df_train[(df_train['EXT_SOURCE_2']>0.4)|df_train['TARGET']==1])/len(df_train[(df_train['EXT_SOURCE_2']>0.4)|df_train['TARGET']==0])


len(df_train[(df_train['EXT_SOURCE_2']<0.4)|df_train['TARGET']==0])/len(df_train[(df_train['EXT_SOURCE_2']<0.4)|df_train['TARGET']==1])



plt.figure(figsize = (10, 8))

# KDE plot of loans that were repaid on time
sns.kdeplot(df_train.loc[df_train['TARGET'] == 0, 'EXT_SOURCE_3'] , label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(df_train.loc[df_train['TARGET'] == 1, 'EXT_SOURCE_3'] , label = 'target == 1')

# Labeling of plot
plt.xlabel('EXT_SOURCE_3'); plt.ylabel('Density'); plt.title('Distribution of EXT_SOURCE_3');



plt.figure(figsize = (10, 8))

# KDE plot of loans that were repaid on time
sns.kdeplot(df_train.loc[df_train['TARGET'] == 0, 'DAYS_EMPLOYED_PERC'] , label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(df_train.loc[df_train['TARGET'] == 1, 'DAYS_EMPLOYED_PERC'] , label = 'target == 1')

# Labeling of plot
plt.xlabel('DAYS_EMPLOYED_PERC'); plt.ylabel('Density'); plt.title('Distribution of DAYS_EMPLOYED_PERC');


df_train.isnull().sum()


del df_train['SK_ID_CURR']


df_y = df_train['TARGET']
df_X = df_train.copy()


df_X.head()


from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(df_X, 
                                test_size=0.20, 
                                stratify = df_train['TARGET'],
                                random_state = 12)


y_train, y_test = train_test_split(df_y, 
                                test_size=0.20, 
                                stratify = df_train['TARGET'],
                                random_state = 13)


from lightgbm import LGBMClassifier


y_train.isnull().sum()


lgbm_model = LGBMClassifier().fit(X_train, y_train)



from sklearn.metrics import accuracy_score 
y_predc = lgbm_model.predict(X_test)
accuracy_score(y_test, y_predc)


?lgbm_model


lgbm_params = {
    'n_estimators': [100,500, 1000, 2000],
    'subsample': [0.6, 0.8, 1],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.1, 0.01, 0.02, 0.05],
    'min_child_samples': [5,10,20]}


from sklearn.model_selection import GridSearchCV
lgbm = LGBMClassifier()

lgbm_cv_model = GridSearchCV(lgbm, lgbm_params,
                            cv = 5,
                            n_jobs = -1,
                            verbose = 2)


lgbm_cv_model.fit(X_train, y_train)

