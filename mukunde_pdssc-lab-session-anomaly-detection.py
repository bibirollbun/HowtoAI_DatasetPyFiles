import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns





train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_transaction.head()


train_transaction.isFraud.head(10000)[train_transaction.isFraud == 0].count()


train_transaction.isFraud.head(10000).hist()


reduced_transaction_df = train_transaction.head(10000)


# head of dataset
reduced_transaction_df.head()


reduced_transaction_df.isnull().values.any


reduced_transaction_df.info()


reduced_transaction_df.describe()


print('number of non-fraudulent transactions is {}'.format(reduced_transaction_df.isFraud[reduced_transaction_df.isFraud == 0].count()))
print('number of fraudulent transactions is {}'.format(reduced_transaction_df.isFraud[reduced_transaction_df.isFraud == 1].count()))


print('Rate of fraudulent transaction is {} %'.format((reduced_transaction_df.isFraud[reduced_transaction_df.isFraud == 1].count() / 10000 * 100)))


print('Categorical colums are :')
list_non_cat = reduced_transaction_df.loc[:,reduced_transaction_df.dtypes == np.object].columns.tolist()
list_non_cat


print('Numeric colums are :')
list_num = reduced_transaction_df.loc[:,reduced_transaction_df.dtypes != np.object].columns.tolist()
list_num


reduced_transaction_df.loc[:,list_num].describe()


# rate of nan values per column
df_missing = (reduced_transaction_df.isna().sum() / 10000)
df_missing



# rate of zeros per column
'''df_missing = (reduced_transaction_df.isna().sum() / 10000)
df_missing'''


def show3D_transaction_data(transac_dataset, x_axis_name, y_axis_name, z_axis_name):
    zOffset = 0.02
    limit = len(transac_dataset)
    sns.reset_orig()
    fig = plt.figure(figsize=(10,12))
    ax = fig.add_subplot(111,projection='3d')
    

fig = plt.figure()
ax = plt.axes(projection="3d")


def show3D_transation_data(training_set, x_points, y_points, z_points):
    z_line = np.linspace(0, 15, 1000)
    x_line = np.sin(z_line)
    y_line = np.cos(z_line)
    ax.plot3D(x_line, y_line, z_line, 'black')
    ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv');
    
    
show3D_transation_data(training_set, training_set['TransactionAmt'], training_set['card1'], training_set['addr1'])


def show3D_transaction_data_fraud_only(transac_dataset, x_axis_name, y_axis_name,z_axis_name):
    


reduced_transaction_df.head()


one_hot_encoded_X = pd.get_dummies(reduced_transaction_df.copy())


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(missing_values=np.nan,strategy='median')
X_with_imputed_values = pd.DataFrame(my_imputer.fit_transform(one_hot_encoded_X))
X_with_imputed_values.columns = one_hot_encoded_X.columns


X_with_imputed_values.isnull().values.any


X_with_imputed_values.head()


np.array(X_with_imputed_values.isFraud.values)


from sklearn.ensemble import IsolationForest
from scipy import stats
outlier_ratio = 0.3
rng = np.random.RandomState(99)
labels = X_with_imputed_values.isFraud.values
to_model_columns=X_with_imputed_values.columns[2:]
x = X_with_imputed_values[to_model_columns]
clf = IsolationForest(max_samples='auto', contamination=outlier_ratio, \
                        random_state=rng, behaviour='new')




clf.fit(x)

y_pred = clf.predict(x)
num_errors = sum(y_pred != labels)
print('Number of errors = {}'.format(num_errors))



X_with_imputed_values['if_outliers'] = y_pred
X_with_imputed_values['if_outliers']


scores_pred = clf.decision_function(x)
threshold = stats.scoreatpercentile(scores_pred,100 * outlier_ratio)
xx, yy = np.meshgrid(np.linspace(-11,11,1000),np.linspace(-11,11,1000))
Z = clf.decision_function(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)

inlier_plot = plt.plot(x[:num_inliers,0],x[:num_inliers,1],'go',label='inliers')
outlier_plot = plt.plot(x[-num_inliers:,0],x[-num_inliers:,1],'ko',label='outliers')
plt.contour(xx,yy,Z,levels=[threshold],linewidths=5,colors='gray')
plt.contour(xx,yy,Z,levels=np.linspace(Z.min(),threshold,7),cmap=plt.cm.Greys_r)
plt.contour(xx,yy,Z,levels=[threshold,Z.max()],colors='gray')

plt.xlim(-11,11)
plt.ylim(-11,11)
plt.legend(numpoints=1)
plt.show()

