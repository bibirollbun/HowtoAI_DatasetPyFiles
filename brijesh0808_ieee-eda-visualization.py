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


path = "/kaggle/input/ieee-fraud-detection/"


pip install chart-studio


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Standard plotly imports
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import iplot, init_notebook_mode
import cufflinks
import cufflinks as cf
import plotly.figure_factory as ff

# Using plotly + cufflinks in offline mode
init_notebook_mode(connected=True)
cufflinks.go_offline(connected=True)

import os
print(os.listdir("../input"))


df_id = pd.read_csv(path+"train_identity.csv", nrows=50_000)
df_txn = pd.read_csv(path+"train_transaction.csv", nrows=50_000)


def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

def plot_distribution(df, var_select=None, title=None, bins=1.0): 
    # Calculate the correlation coefficient between the new variable and the target
    tmp_fraud = df[df['isFraud'] == 1]
    tmp_no_fraud = df[df['isFraud'] == 0]    
    corr = df['isFraud'].corr(df[var_select])
    corr = np.round(corr,3)
    tmp1 = tmp_fraud[var_select].dropna()
    tmp2 = tmp_no_fraud[var_select].dropna()
    hist_data = [tmp1, tmp2]
    
    group_labels = ['Fraud', 'No Fraud']
    colors = ['seagreen','indianred', ]

    fig = ff.create_distplot(hist_data,
                             group_labels,
                             colors = colors, 
                             show_hist = True,
                             curve_type='kde', 
                             bin_size = bins
                            )
    
    fig['layout'].update(title = title+' '+'(corr target ='+ str(corr)+')')

    iplot(fig, filename = 'Density plot')

def plot_dist_churn(df, col, binary=None):
    tmp_churn = df[df[binary] == 1]       #fraud 
    tmp_no_churn = df[df[binary] == 0]    #no Fraud
    tmp_attr = round(tmp_churn[col].value_counts().sort_index() / df[col].value_counts().sort_index(),2)*100
    print(f'Distribution of {col}: ')
    trace1 = go.Bar(
        x=tmp_churn[col].value_counts().sort_index().index,
        y=tmp_churn[col].value_counts().sort_index().values, 
        name='Fraud',opacity = 0.8, marker=dict(
            color='seagreen',
            line=dict(color='#000000',width=1)))

    trace2 = go.Bar(
        x=tmp_no_churn[col].value_counts().sort_index().index,
        y=tmp_no_churn[col].value_counts().sort_index().values,
        name='No Fraud', opacity = 0.8, 
        marker=dict(
            color='indianred',
            line=dict(color='#000000',
                      width=1)
        )
    )

    trace3 =  go.Scatter(   
        x=tmp_attr.sort_index().index,
        y=tmp_attr.sort_index().values,
        yaxis = 'y2', 
        name='% Fraud', opacity = 0.6, 
        marker=dict(
            color='black',
            line=dict(color='#000000',
                      width=2 )
        )
    )
    
    layout = dict(title =  f'Distribution of {str(col)} feature by %Fraud',
              xaxis=dict(type='category'), 
              yaxis=dict(title= 'Count'), 
              yaxis2=dict(range= [0, 15], 
                          overlaying= 'y', 
                          anchor= 'x', 
                          side= 'right',
                          zeroline=False,
                          showgrid= False, 
                          title= 'Percentual Fraud Transactions'
                         ))

    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    iplot(fig)

## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

#Fxns: resumetable, plot_distribution, plot_dist_churn, reduce_mem_usage
## REducing memory
df_txn = reduce_mem_usage(df_txn)
df_id = reduce_mem_usage(df_id)


df_id.shape


df_txn.shape


resumetable(df_id)



resumetable(df_txn)


df_txn.columns[:20]


df_id.columns


df_txn.head()


df_txn['isFraud'].value_counts(normalize=True)*100


df_train_trans = df_txn.copy()


print("Transactions % Fraud:")
print(round(df_train_trans[['isFraud', 'TransactionID']]['isFraud'].value_counts(normalize=True) * 100,2))

trace0 = go.Bar(
    x=df_train_trans[['isFraud', 'TransactionID']].groupby('isFraud')['TransactionID'].count().index,
    y=df_train_trans[['isFraud', 'TransactionID']].groupby('isFraud')['TransactionID'].count().values,
    marker=dict(
        color=['indianred', 'seagreen']),
)

data = [trace0] 
layout = go.Layout(
    title='Fraud (Target) Distribution <br>## 0: No Fraud | 1: Is Fraud ##', 
    xaxis=dict(
        title='Transaction is fraud', 
        type='category'),
    yaxis=dict(
        title='Count')
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ones=len(df_txn[df_txn.isFraud==1])
# zeros=50000-ones

# plt.bar([0,1],[zeros, ones], color=['r', 'g'], tick_label=[0,1])
# # plt.grid()
# plt.xlabel('Transaction is Fraud')
# plt.ylabel('Count')


import matplotlib.pyplot as plt


s = df_txn['isFraud'].value_counts()
s


plt.xticks(s.index, [0,1])
plt.bar(s.index, s.values, label='#Label', color=['red', 'blue'])
plt.xlabel('isFraud')
plt.ylabel('#Total')
plt.legend()
plt.title("0:No Fraud, 1:Fraud")


def print_trans(tmp, num_col='TransactionAmt'):
    print(f"The mininum value in Transaction Amount is {tmp[num_col].min()}, median is {round(tmp[num_col].median(),2)}, and the maximum is {df_train_trans[num_col].max()}")
    print(f"The mean Transaction Amount of Fraudulent Transactions is {round(tmp[tmp['isFraud'] == 1][num_col].median(),2)}\
          \nThe mean Transaction Amount of No-Fraudulent Transactions is {round(tmp[tmp['isFraud'] == 0][num_col].median(),2)}")
    
print_trans(df_train_trans[['isFraud', 'TransactionAmt']], 'TransactionAmt')


df_txn[df_txn.isFraud==1].TransactionAmt.astype('float').mean()


df_txn[df_txn.isFraud==0].TransactionAmt.astype('float64').mean()


print("Transaction Amount Quantiles: ")
print(df_train_trans['TransactionAmt'].quantile([0.01, .025, .1, .25, .5, .75, .975, .99]))


# # df_train_trans['TransactionAmt_log'] = df_train_trans['TransactionAmt'].apply(np.log)
# tmp = df_train_trans[['TransactionAmt', 'isFraud']]
# tmp['TransactionAmt_log'] = tmp['TransactionAmt'].apply(np.log)
# ## Calling the function
# plot_distribution(tmp[(tmp['TransactionAmt'] <= 800)], 'TransactionAmt', 'Transaction Amount Distribution', bins=10.0,)
# plot_distribution(tmp[(tmp['TransactionAmt'] <= 800)], 'TransactionAmt_log', 'Transaction Amount Log Distribution', bins=0.1)


import seaborn as sns


tmp = df_txn[['TransactionAmt', 'isFraud']]
tmp_log = tmp.copy()
tmp_log.TransactionAmt = tmp_log.TransactionAmt.apply(np.log)

sns.distplot(tmp[tmp.isFraud==1].TransactionAmt)
# sns.distplot(tmp[tmp.isFraud==0].TransactionAmt)


sns.distplot(tmp[tmp.isFraud==0].TransactionAmt)


sns.distplot(tmp_log[tmp_log.isFraud==1].TransactionAmt)


sns.distplot(tmp_log[tmp_log.isFraud==0].TransactionAmt)


plot_dist_churn(df_train_trans[['ProductCD', 'isFraud']], 'ProductCD', 'isFraud')


df_txn.ProductCD.unique()


pd.crosstab(df_txn.ProductCD, df_txn.isFraud, normalize=True)


df1 = pd.crosstab(df_txn.ProductCD, df_txn.isFraud)
df1['%Fraud'] = (df1.iloc[:,1]*100)/(df1.iloc[:,0]+df1.iloc[:,1])


df1


for col in ['card4', 'card6']:
    df_train_trans[col] = df_train_trans[col].fillna('NoInf')
    plot_dist_churn(df_train_trans, col, 'isFraud')


df_txn.card4.unique()


print("Card Features Quantiles: ")
print(df_train_trans[['card1', 'card2', 'card3', 'card5']].quantile([0.01, .025, .1, .25, .5, .75, .975, .99]))





for col in ['card1', 'card2', 'card3', 'card5']:
    df_train_trans[str(col)+'_log'] = np.log(df_train_trans[col])




df_train_trans.columns[:20]


# plot_dist_churn(df_train_trans[['ProductCD', 'isFraud']], 'ProductCD', 'isFraud')
binary = 'isFraud'
col= 'ProductCD'


df = df_train_trans[['ProductCD', 'isFraud']]



tmp_churn = df[df[binary] == 1]       #fraud 
tmp_no_churn = df[df[binary] == 0]    #no Fraud
tmp_attr = round(tmp_churn[col].value_counts().sort_index() / df[col].value_counts().sort_index(),2)*100
tmp_attr


tmp = df_txn[['ProductCD', 'isFraud']].copy()
Fraud = tmp[tmp.isFraud==1].ProductCD.value_counts().sort_index()
noFraud = tmp[tmp.isFraud==0].ProductCD.value_counts().sort_index()



noFraud


noFraud.index


perc = Fraud*100/(noFraud+Fraud)
perc


fig, ax1 = plt.subplots(figsize=(10,5))
ax2 = ax1.twinx()

xarr = np.arange(len(Fraud.index))
width=0.4

plt.xticks(xarr, Fraud.index)
c1 = ax1.bar(xarr-width, Fraud.values, width=0.4, label = 'Fraud', color='g')
c1 = ax1.bar(xarr, noFraud.values, width=0.4, label = 'noFraud', color='r')

c2 =ax2.plot(xarr-0.2, perc.values, label='%Fraud')


ax1.set_xlabel('Code')
ax1.set_ylabel('#no. of codes')
ax2.set_ylabel('% Fraud')

ax1.legend()
ax2.legend()
plt.show()




