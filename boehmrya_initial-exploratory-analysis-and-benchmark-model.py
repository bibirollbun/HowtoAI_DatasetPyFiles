# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

# other libraries and functions
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


df = pd.read_csv("../input/application_train.csv")


train, test = train_test_split(df, test_size=0.2)


train.shape


test.shape


train.head()


test.head()


# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# Missing values statistics
missing_train = missing_values_table(train)
missing_train.head(10)


# Number of each type of column
train.dtypes.value_counts()


# Number of unique classes in each object column
train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


train.describe()


(train['DAYS_BIRTH'] / -365).describe()


train['DAYS_EMPLOYED'].describe()


train['CNT_CHILDREN'].describe()


train['AMT_INCOME_TOTAL'].describe()


train['DAYS_REGISTRATION'].describe()


# TARGET value 0 means loan is repayed, value 1 means loan is not repayed.
plt.figure(figsize=(15,5))
sns.countplot(train.TARGET)
plt.xlabel('Target (0 = repaid, 1 = not repaid)'); plt.ylabel('C'); plt.title('Distribution of Loan Repayment');


plt.figure(figsize=(15,5))
sns.countplot(train.NAME_CONTRACT_TYPE.values,data=train)
plt.xlabel('Contract Type'); plt.ylabel('Count'); plt.title('Distribution of Contract Types');


plt.figure(figsize=(15,5))
sns.countplot(train.CODE_GENDER.values,data=train)
plt.xlabel('Gender'); plt.ylabel('Number of Clients'); plt.title('Distribution of Gender');


plt.figure(figsize=(15,5))
sns.countplot(train.NAME_EDUCATION_TYPE.values,data=train)
plt.xlabel('Education Type/Level'); plt.ylabel('Number of Clients'); plt.title('Distribution of Education Type/Level');


plt.figure(figsize=(15,5))
sns.countplot(train.FLAG_OWN_CAR.values,data=train)
plt.xlabel('Car Ownership (Y = Yes, N = No)'); plt.ylabel('Number of Clients'); plt.title('Distribution of Car Ownership');


plt.figure(figsize=(15,5))
sns.countplot(train.FLAG_OWN_REALTY.values,data=train)
plt.xlabel('Home Ownership (Y = Yes, N = No)'); plt.ylabel('Number of Clients'); plt.title('Distribution of Home Ownership');


plt.figure(figsize=(15,5))
sns.countplot(train.CNT_CHILDREN.values,data=train)
plt.xlabel('Number of Children'); plt.ylabel('Number of Clients'); plt.title('Distribution of Children Per Client');


plt.figure(figsize=(15,5))
sns.countplot(train.NAME_FAMILY_STATUS.values,data=train)
plt.xlabel('Family Status'); plt.ylabel('Number of Clients'); plt.title('Family Status Distribution');


plt.figure(figsize=(15,5))
sns.countplot(train.NAME_HOUSING_TYPE.values,data=train)
plt.xlabel('Housing Type'); plt.ylabel('Number of Clients'); plt.title('Housing Type Distribution');


train['DAYS_BIRTH'] = abs(train['DAYS_BIRTH'])

plt.figure(figsize=(15,5))
sns.distplot(train['DAYS_BIRTH'] / 365,bins=5)
plt.xlabel('Age (Years)'); plt.ylabel('Density'); plt.title('Age Distribution');


# Age information into a separate dataframe
age_data = train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

# Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age_data.head(10)


# Group by the bin and calculate averages
age_groups  = age_data.groupby('YEARS_BINNED').mean()
age_groups


plt.figure(figsize = (8, 8))

# Graph the age bins and the average of the target as a bar plot
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');


# Find correlations with the target and sort
correlations = train.corr()['TARGET'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))


# Copy data into a different dataframe to preserve the original
bench_train = train.copy()
bench_test = test.copy()

# one-hot encoding of categorical variables
bench_train = pd.get_dummies(bench_train)
bench_test = pd.get_dummies(bench_test)

# capture the labels
bench_train_labels = bench_train['TARGET']
bench_test_labels = bench_test['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
bench_train, bench_test = bench_train.align(bench_test, join = 'inner', axis = 1)

# Drop the target from the training and testing data
bench_train = bench_train.drop(columns = ['TARGET'])
bench_test = bench_test.drop(columns = ['TARGET'])

# Median imputation of missing values
imputer = Imputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(bench_train)
imputer.fit(bench_test)

# Transform both training and testing data
bench_train = imputer.transform(bench_train)
bench_test = imputer.transform(bench_test)

# Repeat with the scaler
scaler.fit(bench_train)
scaler.fit(bench_test)
bench_train = scaler.transform(bench_train)
bench_test = scaler.transform(bench_test)

print('Training data shape: ', bench_train.shape)
print('Testing data shape: ', bench_test.shape)


# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)

shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(log_reg, bench_train, bench_train_labels, cv=shuffle, scoring='roc_auc')
print(scores)


print(train.shape)
print(test.shape)


# Copy data into a different dataframe to preserve the original
main_train = train.copy()
main_test = test.copy()


main_train['DAYS_BIRTH'] = abs(main_train['DAYS_BIRTH'])
main_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)


# one-hot encoding of categorical variables
main_train = pd.get_dummies(main_train)
main_test = pd.get_dummies(main_test)

# Align the training and testing data, keep only columns present in both dataframes
main_train, main_test = main_train.align(main_test, join = 'inner', axis = 1)


print(main_train.shape)
print(main_test.shape)


# Threshold for removing correlated variables
threshold = 0.8

# Absolute value correlation matrix
corr_matrix = main_train.corr().abs()
corr_matrix.head()


upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head()


# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove.' % (len(to_drop)))


main_train = main_train.drop(columns = to_drop)
main_test = main_test.drop(columns = to_drop)

print('Training shape: ', main_train.shape)
print('Testing shape: ', main_test.shape)


# Train missing values (in percent)
train_missing = (main_train.isnull().sum() / len(main_train)).sort_values(ascending = False)
train_missing.head()


# Test missing values (in percent)
test_missing = (main_test.isnull().sum() / len(main_test)).sort_values(ascending = False)
test_missing.head()


# Identify missing values above threshold
train_missing = train_missing.index[train_missing > 0.50]
test_missing = test_missing.index[test_missing > 0.50]

all_missing = list(set(set(train_missing) | set(test_missing)))
print('There are %d columns with more than 50%% missing values' % len(all_missing))


# Need to save the labels because aligning will remove this column
main_train_labels = main_train["TARGET"]
main_train_ids = main_train['SK_ID_CURR']
main_test_ids = main_test['SK_ID_CURR']

main_train = pd.get_dummies(main_train.drop(columns = all_missing))
main_test = pd.get_dummies(main_test.drop(columns = all_missing))

main_train, main_test = main_train.align(main_test, join = 'inner', axis = 1)

print('Training set full shape: ', main_train.shape)
print('Testing set full shape: ' , main_test.shape)


main_train = main_train.drop(columns = ['SK_ID_CURR'])
main_test = main_test.drop(columns = ['SK_ID_CURR'])


print('Training set full shape: ', main_train.shape)
print('Testing set full shape: ' , main_test.shape)


# capture the labels
main_train_labels = main_train['TARGET']
main_test_labels = main_test['TARGET']

# Drop the target from the training and testing data
main_train = main_train.drop(columns = ['TARGET'])
main_test = main_test.drop(columns = ['TARGET'])

# impute median values
imputer = Imputer(strategy = 'median')
imputer.fit(main_train)
imputer.fit(main_test)
main_train = imputer.transform(main_train)
main_test = imputer.transform(main_test)

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))
scaler.fit(main_train)
scaler.fit(main_test)
main_train = scaler.transform(main_train)
main_test = scaler.transform(main_test)

print('Training data shape: ', main_train.shape)
print('Testing data shape: ', main_test.shape)


log_reg = LogisticRegression(C = 0.0001)

shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(log_reg, main_train, main_train_labels, cv=shuffle, scoring='roc_auc')


print("All Scores:")
print(scores)
print("Average Score:")
print(round((sum(scores) / len(scores)), 4)) 


knn = KNeighborsClassifier()

shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(knn, main_train, main_train_labels, cv=shuffle, scoring='roc_auc')


print("All Scores:")
print(scores)
print("Average Score:")
print(round((sum(scores) / len(scores)), 4)) 


nb = GaussianNB()

shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(nb, main_train, main_train_labels, cv=shuffle, scoring='roc_auc')


print("All Scores:")
print(scores)
print("Average Score:")
print(round((sum(scores) / len(scores)), 4)) 


svm = SVC()

shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(svm, main_train, main_train_labels, cv=shuffle, scoring='roc_auc')


print("All Scores:")
print(scores)
print("Average Score:")
print(round((sum(scores) / len(scores)), 4)) 


dtc = DecisionTreeClassifier(random_state=0)

shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(dtc, main_train, main_train_labels, cv=shuffle, scoring='roc_auc')


print("All Scores:")
print(scores)
print("Average Score:")
print(round((sum(scores) / len(scores)), 4)) 


rfc = RandomForestClassifier()

shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(rfc, main_train, main_train_labels, cv=shuffle, scoring='roc_auc')


print("All Scores:")
print(scores)
print("Average Score:")
print(round((sum(scores) / len(scores)), 4)) 


abc = AdaBoostClassifier(DecisionTreeClassifier())

shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(abc, main_train, main_train_labels, cv=shuffle, scoring='roc_auc')


print("All Scores:")
print(scores)
print("Average Score:")
print(round((sum(scores) / len(scores)), 4)) 


xgb = XGBClassifier()

shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(xgb, main_train, main_train_labels, cv=shuffle, scoring='roc_auc')


print("All Scores:")
print(scores)
print("Average Score:")
print(round((sum(scores) / len(scores)), 4)) 













