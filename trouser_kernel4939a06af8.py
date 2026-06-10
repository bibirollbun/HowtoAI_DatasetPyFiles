import pandas as pd
sample_submission = pd.read_csv("../input/ieee-fraud-detection/sample_submission.csv")
test_identity = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")
test_transaction = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")
train_identity = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")
train_transaction = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")


train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')


print(f"Training data contains {train.shape[0]} rows and {train.shape[1]} columns.")
print(f"Test data contains {test.shape[0]} rows and {test.shape[1]} columns.")


train.head()


del test_identity, test_transaction, train_identity, train_transaction


print(f"There are {train.isnull().any().sum()} columns in training data with null values.")
print(f"There are {test.isnull().any().sum()} columns in test data with null values.")


[c for c in train.columns if train[c].nunique() <= 1]


[c for c in test.columns if test[c].nunique() <= 1]


import numpy as np
import matplotlib.pyplot as plt

color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]


train['TransactionDT'].plot(kind='hist',
                                        figsize=(15, 5),
                                        label='train',
                                        bins=50,
                                        title='Train vs Test TransactionDT distribution')
test['TransactionDT'].plot(kind='hist',
                                       label='test',
                                       bins=50)
plt.legend()
plt.show()


train['TransactionAmt'].plot(
    kind='hist',
    bins=1000,
    figsize=(15, 5),
    xlim=(0,10000),
    label='train',
    title="Distribution of TransactionAmt in linear scale"
)
test['TransactionAmt'].plot(
    kind='hist',
    bins=1000,
    figsize=(15, 5),
    xlim=(0,10000),
    label='test',
)
plt.legend()
plt.show()


train['TransactionAmt'].apply(np.log) \
    .plot(
        kind='hist',
        bins=100,
        figsize=(15, 5),
        title="Distribution of TransactionAmt in log scale",
        label='train',
    
)
test['TransactionAmt'].apply(np.log) \
    .plot(
        kind='hist',
        bins=100,
        figsize=(15, 5),
        label='test',
)
plt.legend()
plt.show()


train.groupby("ProductCD")["TransactionID"].count() \
    .plot(kind='barh', title="Number of transactions by ProductCD")


train.groupby("ProductCD")["isFraud"].mean() \
    .plot(kind='barh', title="Percentage of fraud by ProductCD")

plt.show()


card_features = [c for c in train.columns if 'card' in c]
card_cols = train[card_features]


card_cols.head()


card_cols.nunique()


color_idx = 0
for c in card_cols:
    if train[c].dtype in ['float64','int64']:
        train[c].plot(kind='hist',
              title=c,
              bins=50,
              figsize=(15, 2),
              color=color_pal[color_idx])
    color_idx += 1
    plt.show()


addr_features = [c for c in train.columns if 'addr' in c]
addr_cols = train[addr_features]


addr_cols.head()


addr_cols.nunique()




