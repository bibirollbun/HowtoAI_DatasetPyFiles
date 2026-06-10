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


# analisis de datos
import pandas as pd
import numpy as np
import random as rnd
from pandas import read_csv

# visualización
import seaborn as sns
from scipy.stats import norm, skew
from scipy import stats
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

## scikit modeling libraries
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                             GradientBoostingClassifier, ExtraTreesClassifier,
                             VotingClassifier)

from sklearn.model_selection import (GridSearchCV, cross_val_score, cross_val_predict,
                                     StratifiedKFold, learning_curve)

## Predictive modeling
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import RFE

#Principal components & otros
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB



original = '../input/'
url1 = "/kaggle/input/ieee-fraud-detection/train_identity.csv"
url2 = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"

train_identity = read_csv(url1)
train_transaction = read_csv(url2)



#exploración inicial del fichero transaction
print('El tamaño del fichero train_transaction es: ', train_transaction.shape)


train_transaction.columns.values # Nombre de las variables


train_transaction.info()


#exploración inicial del fichero identity
print('El tamaño del fichero train_identity es: ', train_identity.shape)
train_identity.columns.values # Nombre de las variables


train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')


#Por ahorrar memoria, ahora proceso a eliminar las base de datos _train transaction_ y _train identity_
del train_identity, train_transaction


#exploración inicial del fichero con todos los datos _train_
print('El tamaño del fichero train es: ', train.shape)


train.columns.values # Nombre de las variables


#Explorando los datos vacíos para posible eliminación
train.isnull().sum()


total = train.isnull().sum().sort_values(ascending = False) #sumando valores nulos por columna
porcentaje = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False) #y % del total
missing_train_data  = pd.concat([total, porcentaje], axis=1, keys=['Total', 'Porcentaje']) #df para explorar
missing_train_data.head(50)


missing_train_data.head(235)


#creo nuevo df con las columnas con más del 70% de las observaciones
Train_new = train.drop(train.loc[:,list((100*(train.isnull().sum()/len(train.index))>30))].columns, 1)
#compruebo presencia de NAs
total1 = Train_new.isnull().sum().sort_values(ascending = False) #sumando valores nulos por columna
porcentaje1 = (Train_new.isnull().sum()/Train_new.isnull().count()*100).sort_values(ascending = False) #y % del total
missing_train_data1  = pd.concat([total1, porcentaje1], axis=1, keys=['Total', 'Porcentaje']) #df para explorar
missing_train_data1.head(50)


print('El tamaño del fichero train es: ', Train_new.shape)
Train_new.columns.values # Nombre de las variables


#creo una nueva version del dataset sin valores nulos
Train_new2 = Train_new.dropna()
print('El tamaño del fichero train es: ', Train_new2.shape)


#compruebo presencia de valores missing y nulos
total1 = Train_new2.isna().sum().sort_values(ascending = False) #sumando valores nulos por columna
porcentaje1 = (Train_new2.isna().sum()/Train_new2.isna().count()*100).sort_values(ascending = False) #y % del total
missing_train_data1  = pd.concat([total1, porcentaje1], axis=1, keys=['Total', 'Porcentaje']) #df para explorar
missing_train_data1.head(50)


interrogante = Train_new2.apply(lambda x: True if "?" in list(x) else False, axis=1)
numOfRows = len(interrogante[interrogante == True].index)
 
print('El número de observaciones con caracter ? es ', numOfRows)


missing = Train_new2.apply(lambda x: True if "missing" in list(x) else False, axis=1)
numOfRows = len(interrogante[interrogante == True].index)
 
print('El número de observaciones con caracter missing es ', numOfRows)


# Extrayendo Muestra
Train_sample = Train_new2.sample(frac =.1524, random_state = 2)
print('El tamaño de la muestra de train es: ', Train_sample.shape)


Train_sample.head()


print("La variable objetivo _Is Fraud_ tiene {0} obervaciones y {1} son valores únicos.".format(Train_sample['isFraud'].count(),Train_sample['isFraud'].nunique()))


print(Train_sample['isFraud'].describe())


Train_sample.groupby('isFraud') \
    .count()['TransactionID'] \
    .plot(kind='barh',
          title='Distribución de IsFraud',
          figsize=(15, 3))
plt.show()


print(Train_sample['isFraud'].value_counts())


#TransactionDT
print(Train_sample['TransactionDT'].describe())


#Histograma de columna en log, para suavizar escala
Train_sample['TransactionDT'] \
    .apply(np.log) \
    .plot(kind='hist',
          bins=100,
          figsize=(15, 5),
          title='Distribución del Log de TransactionDT')
plt.show()


#relación con columna objetivo
Train_sample[['TransactionDT', 'isFraud']].groupby(['isFraud'], as_index=False).mean()


print('La media de TransactionDT con IsFraud igual a 1 es: {:.4f}'.format(Train_sample.loc[Train_sample['isFraud'] == 1]['TransactionDT'].mean()))
print('La media de TransactionDT con IsFraud igual a 0 es: {:.4f}'.format(Train_sample.loc[Train_sample['isFraud'] == 0]['TransactionDT'].mean()))


#TransactionAmt
print(Train_sample['TransactionAmt'].describe())


#Histograma de columna en log, para suavizar escala
Train_sample['TransactionAmt'] \
    .apply(np.log) \
    .plot(kind='hist',
          bins=100,
          figsize=(15, 5),
          title='Distribución del Log de TransactionAmt')
plt.show()


#relación con columna objetivo
Train_sample[['TransactionAmt', 'isFraud']].groupby(['isFraud'], as_index=False).mean()


print('La media de TransactionAmt con IsFraud igual a 1 es: {:.4f}'.format(Train_sample.loc[Train_sample['isFraud'] == 1]['TransactionAmt'].mean()))
print('La media de TransactionAmt con IsFraud igual a 0 es: {:.4f}'.format(Train_sample.loc[Train_sample['isFraud'] == 0]['TransactionAmt'].mean()))


#Histogramas de TransactionAmt vs. IsFraud
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 6))
Train_sample.loc[Train_sample['isFraud'] == 1] \
    ['TransactionAmt'].apply(np.log) \
    .plot(kind='hist',
          bins=100,
          title='Log TransactionAmt vs IsFraud = 1',
          xlim=(-3, 10),
         ax= ax1)
Train_sample.loc[Train_sample['isFraud'] == 0] \
    ['TransactionAmt'].apply(np.log) \
    .plot(kind='hist',
          bins=100,
          title='Log TransactionAmt vs IsFraud = 0',
          xlim=(-3, 10),
         ax=ax2)
Train_sample.loc[Train_sample['isFraud'] == 1] \
    ['TransactionAmt'] \
    .plot(kind='hist',
          bins=100,
          title='TransactionAmt vs IsFraud = 1',
         ax= ax3)
Train_sample.loc[Train_sample['isFraud'] == 0] \
    ['TransactionAmt'] \
    .plot(kind='hist',
          bins=100,
          title='TransactionAmt vs IsFraud = 0',
         ax=ax4)
plt.show()


print(Train_sample['ProductCD'].value_counts())


Train_sample.groupby('ProductCD') \
    .count()['TransactionID'] \
    .plot(kind='barh',
          title='Distribución de ProductCD',
          figsize=(15, 3))
plt.show()


print(Train_new2['ProductCD'].value_counts())


#Eliminando la columna _ProductCD_ de las base de datos
Train_sample = Train_sample.drop(['ProductCD'], axis=1)
Train_new2 = Train_new2.drop(['ProductCD'], axis=1)


#creando vector con nombre de las columnas para facilitar exploración
card_cols = ["card1", "card2", "card3", "card4", "card5", "card6"]
Train_sample[card_cols].head(50)


Train_sample[card_cols].tail(50)


#Distribución de _Card1_
Train_sample['card1'] \
    .plot(kind='hist',
          bins=100,
          figsize=(7.5, 2.5),
          title='Distribución de card1')
plt.show()


#Distribución de _Card1_ relativo a IsFraud
Train_sample[['card1', 'isFraud']].groupby(['isFraud'], as_index=False).mean()


#Distribución de _Card2_
Train_sample['card2'] \
    .plot(kind='hist',
          bins=100,
          figsize=(7.5, 2.5),
          title='Distribución de card2')
plt.show()


#Distribución de _Card2_ relativo a IsFraud
Train_sample[['card2', 'isFraud']].groupby(['isFraud'], as_index=False).mean()


#Distribución de _Card5_
Train_sample['card5'] \
    .plot(kind='hist',
          bins=100,
          figsize=(7.5, 2.5),
          title='Distribución de card5')
plt.show()


#Distribución de _Card5_ relativo a IsFraud
Train_sample[['card5', 'isFraud']].groupby(['isFraud'], as_index=False).mean()


#Distribución de _Card3_
Train_sample['card3'] \
    .plot(kind='hist',
          bins=100,
          figsize=(7.5, 2.5),
          title='Distribución de card3')
plt.show()


#Valores únicos de _Card3_ en la base poblacional 
print(Train_new2['card3'].value_counts())


#Eliminando la columna _card3_ de las base de datos
Train_sample = Train_sample.drop(['card3'], axis=1)
Train_new2 = Train_new2.drop(['card3'], axis=1)


#Distribución de variable categórica _Card4_ y _Card6_ referente a tipo de tarjeta
#_Card4_
print(Train_sample['card4'].value_counts())


Train_sample.groupby('card4') \
    .count()['TransactionID'] \
    .plot(kind='barh',
          title='Distribución de ProductCD',
          figsize=(15, 3))
plt.show()


#_Card6_
print(Train_sample['card6'].value_counts())


Train_sample.groupby('card6') \
    .count()['TransactionID'] \
    .plot(kind='barh',
          title='Distribución de ProductCD',
          figsize=(15, 3))
plt.show()


#Ahora evaluamos la relación de estas con la variable objetivo _IsFraud_
Train_sample_fr1 = Train_sample.loc[Train_sample['isFraud'] == 1]
Train_sample_fr0 = Train_sample.loc[Train_sample['isFraud'] == 0]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
Train_sample_fr1.groupby('card4')['card4'].count().plot(kind='barh', ax=ax1, title='card4 con IsFraud = 1')
Train_sample_fr0.groupby('card4')['card4'].count().plot(kind='barh', ax=ax2, title='card4 con IsFraud = 0')
Train_sample_fr1.groupby('card6')['card6'].count().plot(kind='barh', ax=ax3, title='card6 con IsFraud = 1')
Train_sample_fr0.groupby('card6')['card6'].count().plot(kind='barh', ax=ax4, title='card6 con IsFraud = 0')
plt.show()


#Card_4
Train_sample['visa'] = np.where(Train_sample['card4'] == 'visa', 1, 0)
print(Train_sample['visa'].value_counts())
Train_sample['mastercard'] = np.where(Train_sample['card4'] == 'mastercard', 1, 0)
print(Train_sample['mastercard'].value_counts())
Train_new2['visa'] = np.where(Train_new2['card4'] == 'visa', 1, 0)
Train_new2['mastercard'] = np.where(Train_new2['card4'] == 'mastercard', 1, 0)

#Card_6
Train_sample['debit'] = np.where(Train_sample['card6'] == 'debit', 1, 0)
print(Train_sample['debit'].value_counts())
Train_sample['credit'] = np.where(Train_sample['card6'] == 'credit', 1, 0)
print(Train_sample['credit'].value_counts())
Train_new2['debit'] = np.where(Train_new2['card6'] == 'debit', 1, 0)
Train_new2['credit'] = np.where(Train_new2['card6'] == 'credit', 1, 0)

#Elimando ambas columnas de las dataframes muestral y poblacional
Train_sample = Train_sample.drop(['card4'], axis=1)
Train_new2 = Train_new2.drop(['card4'], axis=1)
Train_sample = Train_sample.drop(['card6'], axis=1)
Train_new2 = Train_new2.drop(['card6'], axis=1)



#Distribución de _Addr1_
Train_sample['addr1'] \
    .plot(kind='hist',
          bins=100,
          figsize=(7.5, 2.5),
          title='Distribución de addr1')
plt.show()


#Distribución de _Addr1_ relativo a IsFraud
Train_sample[['addr1', 'isFraud']].groupby(['isFraud'], as_index=False).mean()


#Distribución de _Addr2_
Train_sample['addr2'] \
    .plot(kind='hist',
          bins=100,
          figsize=(7.5, 2.5),
          title='Distribución de addr2')
plt.show()


#Valores únicos de _addr2_ en la base poblacional 
print(Train_new2['addr2'].value_counts())


#Eliminando la columna _addr2_ de las base de datos
Train_sample = Train_sample.drop(['addr2'], axis=1)
Train_new2 = Train_new2.drop(['addr2'], axis=1)


print(Train_sample['P_emaildomain'].value_counts())


Train_sample.groupby('P_emaildomain') \
    .count()['TransactionID'] \
    .plot(kind='barh',
          title='Distribución de P_emaildomain',
          figsize=(15, 15))
plt.show()


#Ahora evaluamos la relación de estas con la variable objetivo _IsFraud_
#fraude
Train_sample_fr1 = Train_sample.loc[Train_sample['isFraud'] == 1]
print(Train_sample_fr1['P_emaildomain'].value_counts())


#No fraude
Train_sample_fr0 = Train_sample.loc[Train_sample['isFraud'] == 0]
print(Train_sample_fr0['P_emaildomain'].value_counts())


#Eliminando la columna _P_emaildomain_ de las base de datos
Train_sample = Train_sample.drop(['P_emaildomain'], axis=1)
Train_new2 = Train_new2.drop(['P_emaildomain'], axis=1)


#Creando una dataframes con el conjunto de columnas _C_ para su exploración
c_cols = [c for c in Train_sample if c[0] == 'C']
c_df = Train_sample[c_cols]
c_df.head(50)
c_df.shape


#PCA en columnas _C 01-14_

#1) Normalizar ó Estandalizar los datos
scaler=StandardScaler()#instantiate
scaler.fit(c_df) # calcula la media y estandar para cada dimension
X_scaled=scaler.transform(c_df)# transforma los datos a su nueva escala

#2) Aplicando PCA
pca=PCA(n_components=14)
pca.fit(X_scaled) # buscar los componentes principales
X_pca=pca.transform(X_scaled) 
#revisemos la forma del array
print("shape of X_pca", X_pca.shape)

#3) Comprobando variabilidad del PCA
expl = pca.explained_variance_ratio_
print(expl)
print('suma:',sum(expl[0:2]))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


#Incluyendo las dos nuevas componentes en columnas _C PCA1_ y _C PCA2_
Train_sample['C_PCA1'] = X_pca[:,0]
Train_sample['C_PCA2'] = X_pca[:,1]

#Elimando las columnas C
Train_sample = Train_sample.drop([c for c in c_cols], axis=1)

Train_sample.shape


#Replicando pasos anteriores en la base poblacional

#PCA en columnas _C 01-14_ en df _Train_new2_

c_df = Train_new2[c_cols]

#1) Normalizar ó Estandalizar los datos
scalerp=StandardScaler()#instantiate
scalerp.fit(c_df) # calcula la media y estandar para cada dimension
X_scaled=scalerp.transform(c_df)# transforma los datos a su nueva escala

#2) Aplicando PCA
pca=PCA(n_components=14)
pca.fit(X_scaled) # buscar los componentes principales
X_pca=pca.transform(X_scaled) 
#revisemos la forma del array
print("shape of X_pca", X_pca.shape)

#3) Comprobando variabilidad del PCA
expl = pca.explained_variance_ratio_
print(expl)
print('suma:',sum(expl[0:2]))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


#Incluyendo las dos nuevas componentes en columnas _C PCA1_ y _C PCA2_ en df _Train_new2_
Train_new2['C_PCA1'] = X_pca[:,0]
Train_new2['C_PCA2'] = X_pca[:,1]

#Elimando las columnas C
Train_new2 = Train_new2.drop([c for c in c_cols], axis=1)

Train_new2.shape


#Creando una dataframes con el conjunto de columnas _V_ para su exploración
v_cols = [v for v in Train_sample if v[0] == 'V']
v_df = Train_sample[v_cols]
v_df.head(50)
v_df.shape


#PCA en columnas _V 01-339_

#1) Normalizar ó Estandalizar los datos
scaler=StandardScaler()#instantiate
scaler.fit(v_df) # calcula la media y estandar para cada dimension
X_scaled=scaler.transform(v_df)# transforma los datos a su nueva escala

#2) Aplicando PCA
pca=PCA(n_components=20)
pca.fit(X_scaled) # buscar los componentes principales
X_pca=pca.transform(X_scaled) 
#revisemos la forma del array
print("shape of X_pca", X_pca.shape)

#3) Comprobando variabilidad del PCA
expl = pca.explained_variance_ratio_
print(expl)
print('suma:',sum(expl[0:20]))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


#Incluyendo las 20 nuevas componentes en columnas _V PCA1_ al _V PCA20_ en df _Train_new2_
Train_sample['V_PCA1'] = X_pca[:,0]
Train_sample['V_PCA2'] = X_pca[:,1]
Train_sample['V_PCA3'] = X_pca[:,2]
Train_sample['V_PCA4'] = X_pca[:,3]
Train_sample['V_PCA5'] = X_pca[:,4]
Train_sample['V_PCA6'] = X_pca[:,5]
Train_sample['V_PCA7'] = X_pca[:,6]
Train_sample['V_PCA8'] = X_pca[:,7]
Train_sample['V_PCA9'] = X_pca[:,8]
Train_sample['V_PCA10'] = X_pca[:,9]
Train_sample['V_PCA11'] = X_pca[:,10]
Train_sample['V_PCA12'] = X_pca[:,11]
Train_sample['V_PCA13'] = X_pca[:,12]
Train_sample['V_PCA14'] = X_pca[:,13]
Train_sample['V_PCA15'] = X_pca[:,14]
Train_sample['V_PCA16'] = X_pca[:,15]
Train_sample['V_PCA17'] = X_pca[:,16]
Train_sample['V_PCA18'] = X_pca[:,17]
Train_sample['V_PCA19'] = X_pca[:,18]
Train_sample['V_PCA20'] = X_pca[:,19]

#Elimando las columnas V
Train_sample = Train_sample.drop([v for v in v_cols], axis=1)

Train_sample.shape


#Replicando pasos anteriores en la base poblacional

#PCA en columnas _V 01-_339 en df _Train_new2_

v_df = Train_new2[v_cols]

#1) Normalizar ó Estandalizar los datos
scalerp=StandardScaler()#instantiate
scalerp.fit(v_df) # calcula la media y estandar para cada dimension
X_scaled=scalerp.transform(v_df)# transforma los datos a su nueva escala

#2) Aplicando PCA
pca=PCA(n_components=20)
pca.fit(X_scaled) # buscar los componentes principales
X_pca=pca.transform(X_scaled) 
#revisemos la forma del array
print("shape of X_pca", X_pca.shape)

#3) Comprobando variabilidad del PCA
expl = pca.explained_variance_ratio_
print(expl)
print('suma:',sum(expl[0:20]))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


#Incluyendo las 20 nuevas componentes en columnas _V PCA1_ al _V PCA20_ en df _Train_new2_
Train_new2['V_PCA1'] = X_pca[:,0]
Train_new2['V_PCA2'] = X_pca[:,1]
Train_new2['V_PCA3'] = X_pca[:,2]
Train_new2['V_PCA4'] = X_pca[:,3]
Train_new2['V_PCA5'] = X_pca[:,4]
Train_new2['V_PCA6'] = X_pca[:,5]
Train_new2['V_PCA7'] = X_pca[:,6]
Train_new2['V_PCA8'] = X_pca[:,7]
Train_new2['V_PCA9'] = X_pca[:,8]
Train_new2['V_PCA10'] = X_pca[:,9]
Train_new2['V_PCA11'] = X_pca[:,10]
Train_new2['V_PCA12'] = X_pca[:,11]
Train_new2['V_PCA13'] = X_pca[:,12]
Train_new2['V_PCA14'] = X_pca[:,13]
Train_new2['V_PCA15'] = X_pca[:,14]
Train_new2['V_PCA16'] = X_pca[:,15]
Train_new2['V_PCA17'] = X_pca[:,16]
Train_new2['V_PCA18'] = X_pca[:,17]
Train_new2['V_PCA19'] = X_pca[:,18]
Train_new2['V_PCA20'] = X_pca[:,19]

#Elimando las columnas V
Train_new2 = Train_new2.drop([v for v in v_cols], axis=1)

Train_new2.shape



#D1
#TransactionDT
print(Train_sample['D1'].describe())


#Histograma de columna D1
Train_sample['D1'] \
    .plot(kind='hist',
          bins=100,
          figsize=(15, 5),
          title='Distribución de D1 TransactionDT')
plt.show()


#relación con columna objetivo
Train_sample[['D1', 'isFraud']].groupby(['isFraud'], as_index=False).mean()


print('La media de D1 con IsFraud igual a 1 es: {:.4f}'.format(Train_sample.loc[Train_sample['isFraud'] == 1]['D1'].mean()))
print('La media de D1 con IsFraud igual a 0 es: {:.4f}'.format(Train_sample.loc[Train_sample['isFraud'] == 0]['D1'].mean()))


#D4
#TransactionDT
print(Train_sample['D4'].describe())


#Histograma de columna D4
Train_sample['D4'] \
    .plot(kind='hist',
          bins=100,
          figsize=(15, 5),
          title='Distribución de D4')
plt.show()


#relación con columna objetivo
Train_sample[['D4', 'isFraud']].groupby(['isFraud'], as_index=False).mean()


print('La media de D4 con IsFraud igual a 1 es: {:.4f}'.format(Train_sample.loc[Train_sample['isFraud'] == 1]['D4'].mean()))
print('La media de D4 con IsFraud igual a 0 es: {:.4f}'.format(Train_sample.loc[Train_sample['isFraud'] == 0]['D4'].mean()))


#D10
#TransactionDT
print(Train_sample['D10'].describe())


#Histograma de columna D10
Train_sample['D10'] \
    .plot(kind='hist',
          bins=100,
          figsize=(15, 5),
          title='Distribución de D10')
plt.show()


#relación con columna objetivo
Train_sample[['D10', 'isFraud']].groupby(['isFraud'], as_index=False).mean()


print('La media de D10 con IsFraud igual a 1 es: {:.4f}'.format(Train_sample.loc[Train_sample['isFraud'] == 1]['D10'].mean()))
print('La media de D10 con IsFraud igual a 0 es: {:.4f}'.format(Train_sample.loc[Train_sample['isFraud'] == 0]['D10'].mean()))


#D15
#TransactionDT
print(Train_sample['D15'].describe())


#Histograma de columna D15
Train_sample['D15'] \
    .plot(kind='hist',
          bins=100,
          figsize=(15, 5),
          title='Distribución de D15')
plt.show()


#relación con columna objetivo
Train_sample[['D15', 'isFraud']].groupby(['isFraud'], as_index=False).mean()


print('La media de D15 con IsFraud igual a 1 es: {:.4f}'.format(Train_sample.loc[Train_sample['isFraud'] == 1]['D15'].mean()))
print('La media de D15 con IsFraud igual a 0 es: {:.4f}'.format(Train_sample.loc[Train_sample['isFraud'] == 0]['D15'].mean()))


print(Train_sample['M6'].value_counts())


#Convirtiendo columna _M6_ en binaria, T == 1
Train_sample['M6'] = np.where(Train_sample['M6'] == 'T', 1, 0)
print(Train_sample['M6'].value_counts())

#Replicando para la df poblacional
Train_new2['M6'] = np.where(Train_new2['M6'] == 'T', 1, 0)
print(Train_new2['M6'].value_counts())


Train_sample.groupby('M6') \
    .count()['TransactionID'] \
    .plot(kind='barh',
          title='Distribución de M6',
          figsize=(15, 3))
plt.show()


#relación con columna objetivo
#fraude
Train_sample_fr1 = Train_sample.loc[Train_sample['isFraud'] == 1]
Train_sample_fr1.groupby('M6') \
    .count()['TransactionID'] \
    .plot(kind='barh',
          title='Distribución de M6 con isFraud = 1',
          figsize=(15, 3))
plt.show()


#No fraude
Train_sample_fr0 = Train_sample.loc[Train_sample['isFraud'] == 0]
Train_sample_fr0.groupby('M6') \
    .count()['TransactionID'] \
    .plot(kind='barh',
          title='Distribución de M6 con isFraud = 0',
          figsize=(15, 3))
plt.show()


#Creo un backup de las bases de datos por resguardo antes de eliminar nuevas columnas.
Train_sample_bk = Train_sample
Train_new2_bk = Train_new2


Train_new2_bk.info()


correlation_matrix = Train_sample.corr()
correlation_matrix

plt.figure(figsize=(23.0,23.0))
plt.title('Matriz de Correlación de Pearson')
sns.heatmap(correlation_matrix, annot=True)


corrmat = Train_sample.corr()
cols_corrmat = corrmat['isFraud'].abs()
cols_corrmat = cols_corrmat.sort_values(ascending=False)
cols_corrmat.head(39)


#Guardando columnas finales en array para proceso de modelado
cols = cols_corrmat.index
cols = cols[0:14]
cols
df_model = Train_new2[cols]
df_model.shape


Train_new2.to_csv('Train_new2.csv',index=False)
Train_sample.to_csv('Train_sample.csv',index=False)
df_model.to_csv('df_model.csv',index=False)




