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


import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv',index_col='TransactionID')
test_identity  = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv',index_col='TransactionID')
train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv',index_col='TransactionID')
test_transaction  = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv',index_col='TransactionID')
train_identity.head()


print("Shape of Train_identity:: ", train_identity.shape)
print("Shape of Test_identity:: ", test_identity.shape)
print("Shape of Train_transaction:: ", train_transaction.shape)
print("Shape of Test_transaction:: ", test_transaction.shape)


train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)


train_transaction.head()


train_transaction.info()


train_transaction.describe()


train_transaction.reset_index()['TransactionID'].isin(train_identity.reset_index()['TransactionID']).value_counts()


train_identity.reset_index()['TransactionID'].isin(train_transaction.reset_index()['TransactionID']).value_counts()


X = pd.merge(train_transaction,
             train_identity,
             on='TransactionID',
             how='left')
print("train_transaction dimensions: {} ".format(train_transaction.shape))
print("train_identity dimensions:    {} ".format(train_identity.shape))
print("Merged X dimensions:          {} ".format(X.shape))


Y = X.isFraud
X = X.reset_index().drop('isFraud', axis=1)


X.head()


print("Fraud Transaction Count is {}".format(Y[Y==1].count()))
print("non-Fraud Transaction Count is {}".format(Y[Y==0].count()))
print("% Fraud Transaction Count is {}".format((Y[Y==1].count()/Y.count())*100))


Y.dropna().plot(kind='hist',bins=10)
plt.xlabel('ISFraud')
plt.title('ISFRAUD Histogram Plot')
plt.xlim(0,1)


#Splitting bthe data to train and validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.20, random_state=42)


print("X_train dimensions:  {}".format(X_train.shape))
print("y_train dimensions:  {}".format(y_train.shape))
print("X_valid dimensions:  {}".format(X_valid.shape))
print("y_valid dimensions:  {}".format(y_valid.shape))


numeric_columns = list(X_train.select_dtypes(exclude='object').columns)
categorical_columns = list(X_train.select_dtypes(include='object').columns)


categorical_columns


columns = ['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'ProductCD',
 'card4',
 'card6',
 'P_emaildomain',
 'R_emaildomain',
 'M1',
 'M2',
 'M3',
 'M4',
 'M5',
 'M6',
 'M7',
 'M8',
 'M9',
 'id_12',
 'id_15',
 'id_16',
 'id_23',
 'id_27',
 'id_28',
 'id_29',
 'id_30',
 'id_31',
 'id_33',
 'id_34',
 'id_35',
 'id_36',
 'id_37',
 'id_38',
 'DeviceType',
 'DeviceInfo']

for col in columns:
    X_train[col] = X_train[col].astype('category')
    X_valid[col] = X_valid[col].astype('category')


numeric_columns = list(X_train.select_dtypes(exclude='category').columns)
categorical_columns = list(X_train.select_dtypes(include='category').columns)


numeric_columns


sns.set(style="white")

df_corr= X_train[['C1','C2', 'C3', 'C4','C5','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14']]

# Compute the correlation matrix
corr = df_corr.dropna().corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(30, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, square=True, annot=True,linewidths=.5, ax=ax)


X_train.columns[54:393]


X_train.iloc[:,54:393].corr()


X_train.loc[:,["V319","V320"]].corr()


X_train.loc[:,["V109","V110"]].corr()


X_train.loc[:,["V329","V330"]].corr()


X_train.loc[:,["V316","V331"]].corr()


train["diff_V319_V320"] = np.zeros(train.shape[0])

train.loc[train["V319"]!=train["V320"],"diff_V319_V320"] = 1

test["diff_V319_V320"] = np.zeros(test.shape[0])

test.loc[test["V319"]!=test["V320"],"diff_V319_V320"] = 1

train["diff_V109_V110"] = np.zeros(train.shape[0])

train.loc[train["V109"]!=train["V110"],"diff_V109_V110"] = 1

test["diff_V109_V110"] = np.zeros(test.shape[0])

test.loc[test["V109"]!=test["V110"],"diff_V109_V110"] = 1

train["diff_V329_V330"] = np.zeros(train.shape[0])

train.loc[train["V329"]!=train["V330"],"diff_V329_V330"] = 1

test["diff_V329_V330"] = np.zeros(test.shape[0])

test.loc[test["V329"]!=test["V330"],"diff_V329_V330"] = 1


train["diff_V316_V331"] = np.zeros(train.shape[0])

train.loc[train["V331"]!=train["V316"],"diff_V316_V331"] = 1

test["diff_V316_V331"] = np.zeros(test.shape[0])

test.loc[test["V316"]!=test["V331"],"diff_V316_V331"] = 1


plt.bar(train.groupby("diff_V319_V320").mean().isFraud.index,train.groupby("diff_V319_V320").mean().isFraud.values)


plt.bar(train.groupby("diff_V109_V110").mean().isFraud.index,train.groupby("diff_V109_V110").mean().isFraud.values)


plt.bar(train.groupby("diff_V329_V330").mean().isFraud.index,train.groupby("diff_V329_V330").mean().isFraud.values)


plt.bar(train.groupby("diff_V316_V331").mean().isFraud.index,train.groupby("diff_V316_V331").mean().isFraud.values)


train = train.drop("diff_V109_V110",axis=1)
test = test.drop("diff_V109_V110",axis=1)

train = train.drop("diff_V329_V330",axis=1)
test = test.drop("diff_V329_V330",axis=1)


#Missing Value Check
X_train[['D1','D2', 'D3', 'D4','D5','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15']].isnull().sum()/X_train.shape[0]


X_train.shape


X_train.columns


X_train['ProductCD'].dropna().plot(kind='hist',bins=10)
plt.xlabel('ProductCD')
plt.title('ProductCD Histogram Plot')
plt.xlim(0,1)


X_train[0:60000].corr()





sns.set(style="white")
# Compute the correlation matrix
corr = train_transaction.reset_index()[numeric_columns].dropna().corr()

corr


categorical_columns


print("Numeric Missing Numbers: {}".format(X_train[numeric_columns].isnull().sum()[X_train[numeric_columns].isnull().sum() > 0]))


print("Categorical Missing Numbers: {}".format(X_train[categorical_columns].isnull().sum()[X_train[categorical_columns].isnull().sum() > 0]))


X_train[categorical_columns].nunique()


good_category_columns  = [col for col in categorical_columns if set(X_train[col]) == set(X_valid[col])]
bad_category_columns  = list(set(categorical_columns) - set(good_category_columns))
bad_category_columns


#Seperating Low cardinal categorical features.
categorical_cols_low = [cname for cname in categorical_columns if
                    X_train[cname].nunique() < 10 and 
                    X_train[cname].dtype == "category"]
categorical_cols_high = list(set(categorical_columns) - set(categorical_cols_low))

print("Low cardinal categorical features:  {}".format(categorical_cols_low))
print("High cardinal categorical features:  {}".format(categorical_cols_high))


# Keep selected columns only
my_cols = categorical_cols_low + numeric_columns
X_train_1 = X_train[my_cols].copy()
X_valid_1 = X_valid[my_cols].copy()
#X_test = X_test[my_cols].copy()


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_cols_low)
    ])

# Define model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Preprocessing of training data, fit model 
clf.fit(X_train_1, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid_1)

print('R2 Test:',  r2_score(y_valid, preds))
print('R2 Train:', r2_score(y_train, clf.predict(X_train)))


del train_transaction, train_identity, test_transaction, test_identity


print(train.shape)
print(test.shape)

y_train = train['isFraud'].copy()

# Drop target, fill in NaNs
X_train = train.drop('isFraud', axis=1)
X_test = test.copy()
X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)


del train, test


from sklearn import preprocessing
# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values)) 


import xgboost as xgb
from xgboost import XGBClassifier
clf = xgb.XGBClassifier(n_estimators=500,
                        n_jobs=4,
                        max_depth=9,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        missing=-999)

clf.fit(X_train, y_train)


# Predicting the Test set results
%time y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# Explained variance score: 1 is perfect prediction
%time print('Variance score: %.2f' % classifier.score(X_test, y_test))


sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]
sample_submission.to_csv('simple_xgboost.csv')







