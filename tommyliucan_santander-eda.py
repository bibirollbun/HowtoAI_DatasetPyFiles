## Data manipulation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats

## Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
%matplotlib inline
plt.style.use('seaborn')

## Modeling
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA, LatentDirichletAllocation, NMF
from sklearn.linear_model import Ridge,ElasticNet, SGDRegressor, LogisticRegression
from sklearn.model_selection import train_test_split
from scipy import sparse
from scipy.stats import norm, skew

from sklearn.manifold import TSNE

## others
import copy
import os
import time
import warnings
import gc
import os
import pickle
from six.moves import urllib
import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
full_data = pd.concat([train_data, test_data])

print("train_data:", train_data.info())
print("test_data:", test_data.info())



train_data.head(10)


# Numerical features
num_vars = []
# Categorical features
cat_vars = []
for var, dtype in full_data.dtypes.items():
    if "float" in str(dtype) or "int" in str(dtype):
        num_vars.append(var)
    if "object" in str(dtype):
        cat_vars.append(var)

id_var = "ID_code" # this is just the order of data
cat_vars.remove(id_var)
target_var = "target"
num_vars.remove(target_var)
print("There are %d numerical features: %s" %(len(num_vars), num_vars))
print("There are %d numerical features: %s" %(len(cat_vars), cat_vars))



train_data.describe()


test_data.describe()


sns.countplot(train_data["target"])


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis= 1, keys = ['total', 'percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt["Type"] = types
    return np.transpose(tt)


missing_data(train_data)


missing_data(test_data)


train_unique_1= train_data[num_vars].nunique().reset_index()
train_unique_2 = train_unique_1.rename(columns = {"index":'feature', 0:'unique'}).sort_values('unique')
sns.barplot(x = 'feature', y = 'unique', color = 'b', data = train_unique_2)


test_unique_1= test_data[num_vars].nunique().reset_index()
test_unique_2 = test_unique_1.rename(columns = {"index":'feature', 0:'unique'}).sort_values('unique')
sns.barplot(x = 'feature', y = 'unique', color = 'b', data = test_unique_2)


std_scaler = StandardScaler()
# Notice we are using full datasets in order to capture more information
std_scaler.fit(full_data[num_vars].values) 
train_std_df = pd.DataFrame(std_scaler.transform(train_data[num_vars].values), columns=num_vars)
test_std_df = pd.DataFrame(std_scaler.transform(test_data[num_vars].values) , columns=num_vars)

train_std_df['target'] = train_data['target'].values
train_std_df[num_vars].describe()


corr_data = full_data[num_vars].corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True) # Use different colors by palette
# Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr_data)    , cbar_kws={"shrink": .5}
sns.heatmap(corr_data, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5)


train_norm_data = train_data[num_vars].apply(lambda x: stats.normaltest(x)[1])
print("There are %d features normally distributed." % ((train_norm_data<0.05).sum()))

print("Top 10 features with highest P value:")
train_norm_data.sort_values(ascending=False).head(10)


test_norm_data = test_data[num_vars].apply(lambda x: stats.normaltest(x)[1])
print("There are %d features normally distributed." % ((test_norm_data<0.05).sum()))

print("Top 10 features with highest P value:")
test_norm_data.sort_values(ascending=False).head(10)


## Plot var_146
sns.distplot(train_data['var_146'])


sns.pairplot(train_data[num_vars[:20] + ['target']][:10000], hue='target')
# 1: green; 0: blue


%%time
pca = PCA(n_components = 2)
pca2d = pca.fit_transform(train_data[num_vars][:10000].values)
pca2d_df = pd.DataFrame({'pca_0':pca2d[:,0], 'pca_1':pca2d[:,1], 'target': train_data['target'][:10000].values})
sns.lmplot(x='pca_0', y='pca_1', data=pca2d_df, hue='target', fit_reg=False)


%%time
## 2D
svd = TruncatedSVD(n_components=2)
svd2d = svd.fit_transform(train_data[num_vars][:10000].values)
svd2d_df = pd.DataFrame({'svd_0':svd2d[:,0],'svd_1':svd2d[:,1],'target':train_data['target'][:10000].values})
sns.lmplot(x='svd_0', y='svd_1', data=svd2d_df, hue='target', fit_reg=False)


%%time
# 1D
tsne = TSNE(n_components=1)
tsne1d = tsne.fit_transform(train_data[num_vars][:10000].values)
tsne1d_df = pd.DataFrame({'tsne_0':tsne1d.reshape(-1), 'target':train_data['target'][:10000].values})
sns.distplot(tsne1d_df.query('target==0')['tsne_0'], label='target:0')
sns.distplot(tsne1d_df.query('target==1')['tsne_0'], label='target:1')
plt.legend()


%%time
## 2D
tsne = TSNE(n_components=2, perplexity = 50, n_iter = 2000)
tsne2d = tsne.fit_transform(train_data[num_vars][:10000].values)
tsne2d_df = pd.DataFrame({'tsne_0':tsne2d[:,0],'tsne_1':tsne2d[:,1],'target':train_data['target'][:10000].values})
sns.lmplot(x='tsne_0', y='tsne_1', data=tsne2d_df, hue='target', fit_reg=False)
plt.legend()


%%time
train_x = train_std_df[num_vars].values
train_y = train_std_df['target'].values
test_x = test_std_df[num_vars].values

lr = LogisticRegression()
lr.fit(train_x, train_y)


lr_feature_importance = pd.DataFrame({'feature':num_vars, 'lr_importance':lr.coef_.reshape(-1), 
                                      'abs_lr_importance': abs(lr.coef_.reshape(-1))})
                            
lr_feature_importance.sort_values('abs_lr_importance', ascending=False).head(10)


%%time
lgb_clf = lgb.LGBMClassifier(n_jobs=-1)
lgb_clf.fit(train_x, train_y)


lgb_feature_importance = pd.DataFrame({'feature':num_vars, 
                                       'lgb_importance':lgb_clf.feature_importances_.reshape(-1)})
                                        
lgb_feature_importance.sort_values('lgb_importance', ascending=False).head()


feature_importance = pd.merge(lr_feature_importance, lgb_feature_importance, on='feature')
feature_importance.head(20)


from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

output_notebook()

TOOLTIPS = [
    ("Feature", "@feature"),
    ("LR importance", "@abs_lr_importance"),
    ("LGB importance", "@lgb_importance")
]
p = figure(plot_width=400, plot_height=400, tooltips=TOOLTIPS)
p.circle('abs_lr_importance', 
         'lgb_importance', source=ColumnDataSource(feature_importance), size=8)
show(p)


sns.distplot(train_std_df['var_53'])


# var_53 against target
sns.distplot(train_std_df.query('target==0')['var_53'], label='target:0')
sns.distplot(train_std_df.query('target==1')['var_53'], label='target:1')
plt.legend()


# var_81 against target
sns.distplot(train_std_df.query('target==0')['var_81'], label='target:0')
sns.distplot(train_std_df.query('target==1')['var_81'], label='target:1')
plt.legend()

