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
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
import sklearn

import seaborn as sns

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (12, 8)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

print('Libraries imported.')


transactions = pd.read_csv('../input/train.csv')
print('train data imported.')
transactions.head()


print('number of unique ids {}'.format(len(transactions.ID_code.unique())))
print('number of rows {}'.format(len(transactions)))
print('number of columns {}'.format(len(transactions.columns)))
print('missing values: {}'.format(transactions.isna().any().any()))


transactions.describe()


neg, pos = np.bincount(transactions['target'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))


plt.figure(figsize=(6, 6))
sns.countplot(x=transactions.target)
plt.title('Class imbalance')
plt.show()


transactions.drop(['ID_code'], axis=1, inplace=True)
transactions.head()


nrows = 2
ncols = 2

fig = plt.gcf()
fig.set_size_inches(ncols*7, nrows*6)

for i, stat in enumerate(['mean', 'std', 'max', 'min']):
    ax = plt.subplot(nrows, ncols, i + 1)
    sns.histplot(transactions.drop(['target'], axis=1).describe().loc[stat], ax= ax)
    ax.set(xlabel=None)
    plt.title('Distribution of numeric features {} values'.format(stat.upper()))
plt.show()


from sklearn.model_selection import train_test_split

# Use a utility from sklearn to split and shuffle your dataset.
train_df, test_df = train_test_split(transactions, test_size=0.2, stratify=transactions.target)
train_df, valid_df = train_test_split(train_df, test_size=0.2, stratify=train_df.target)


X_train, y_train = train_df.drop(['target'], axis=1).values, train_df.target.values
X_valid, y_valid = valid_df.drop(['target'], axis=1).values, valid_df.target.values
X_test, y_test = test_df.drop(['target'], axis=1).values, test_df.target.values

bool_train_labels = y_train != 0

print('Train features shape: {}'.format(X_train.shape))
print('Valid features shape: {}'.format(X_valid.shape))
print('Test features shape: {}'.format(X_test.shape))

print('\nTrain labels shape: {}'.format(y_train.shape))
print('Valid labels shape: {}'.format(y_valid.shape))
print('Test labels shape: {}'.format(y_test.shape))


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


pos_df = pd.DataFrame(X_train[bool_train_labels], columns=train_df.drop(['target'], axis=1).columns)
neg_df = pd.DataFrame(X_train[~bool_train_labels], columns=train_df.drop(['target'], axis=1).columns)

sns.jointplot(x=pos_df['var_5'], y=pos_df['var_6'], kind='hex')
plt.suptitle("Positive distribution")

sns.jointplot(x=neg_df['var_5'], y=neg_df['var_6'], kind='hex')
_ = plt.suptitle("Negative distribution")


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def build_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    model = keras.models.Sequential([
        keras.layers.Dense(100, activation='relu', input_shape=[X_train.shape[-1]]),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=keras.losses.BinaryCrossentropy(),metrics=metrics)
    
    return model


EPOCHS = 100
BATCH_SIZE = 2048

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

# calculate initial bias
initial_bias = np.log([pos/neg])

baseline_model = build_model(output_bias=initial_bias)
baseline_model.summary()


baseline_history = baseline_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping], validation_data=(X_valid, y_valid))


def plot_metrics(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], color=colors[0], linestyle="--", label='Valid')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':      
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.legend()

plot_metrics(baseline_history)


train_predictions_baseline = baseline_model.predict(X_train, batch_size=BATCH_SIZE)
test_predictions_baseline = baseline_model.predict(X_test, batch_size=BATCH_SIZE)


from sklearn.metrics import confusion_matrix

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    print('Non-Target Customers Detected (True Negatives): ', cm[0][0])
    print('Non-Target Customers Incorrectly Detected (False Positives): ', cm[0][1])
    print('Target Customers Missed (False Negatives): ', cm[1][0])
    print('Target Customers Detected (True Positives): ', cm[1][1])
    print('Total Target Customers: ', np.sum(cm[1]))


baseline_results = baseline_model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)

for name, value in zip(baseline_model.metrics_names, baseline_results):
    print(name, ': ', value)
print()

plot_cm(y_test, test_predictions_baseline)


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    
plot_roc("Train Baseline", y_train, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", y_test, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right')
plt.show()


# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))


weighted_model = build_model(output_bias=initial_bias)

weighted_history = weighted_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping], 
                                      validation_data=(X_valid, y_valid),
                                      class_weight=class_weight)


plot_metrics(weighted_history)


train_predictions_weighted = weighted_model.predict(X_train, batch_size=BATCH_SIZE)
test_predictions_weighted = weighted_model.predict(X_test, batch_size=BATCH_SIZE)


weighted_results = weighted_model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(weighted_model.metrics_names, weighted_results):
    print(name, ': ', value)
print()
plot_cm(y_test, test_predictions_weighted)


plot_roc("Train Baseline", y_train, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", y_test, test_predictions_baseline, color=colors[0], linestyle='--')

plot_roc("Train Weighted", y_train, train_predictions_weighted, color=colors[1])
plot_roc("Test Weighted", y_test, test_predictions_weighted, color=colors[1], linestyle='--')

plt.legend(loc='lower right')
plt.show()


pos_features = X_train[bool_train_labels]
neg_features = X_train[~bool_train_labels]

pos_labels = y_train[bool_train_labels]
neg_labels = y_train[~bool_train_labels]

print('Positive features shape: {}'.format(pos_features.shape))
print('Negative features shape: {}'.format(neg_features.shape))

print('\nPositive labels shape: {}'.format(pos_labels.shape))
print('Negative labels shape: {}'.format(neg_labels.shape))


ids = np.arange(len(pos_features))
choices = np.random.choice(ids, len(neg_features))

res_pos_features = pos_features[choices]
res_pos_labels = pos_labels[choices]

print('Resampled Positive features shape: {}'.format(res_pos_features.shape))
print('Resampled Positive labels shape: {}'.format(res_pos_labels.shape))


resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

order = np.arange(len(resampled_labels))
np.random.shuffle(order)
resampled_features = resampled_features[order]
resampled_labels = resampled_labels[order]

resampled_features.shape


resampled_model = build_model(output_bias=[0])

resampled_history = resampled_model.fit(resampled_features, resampled_labels, batch_size=BATCH_SIZE, 
                                        epochs=EPOCHS, callbacks=[early_stopping], validation_data=(X_valid, y_valid))


plot_metrics(resampled_history)


train_predictions_resampled = resampled_model.predict(X_train, batch_size=BATCH_SIZE)
test_predictions_resampled = resampled_model.predict(X_test, batch_size=BATCH_SIZE)


resampled_results = resampled_model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(resampled_model.metrics_names, resampled_results):
    print(name, ': ', value)
print()

plot_cm(y_test, test_predictions_resampled)


plot_roc("Train Baseline", y_train, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", y_test, test_predictions_baseline, color=colors[0], linestyle='--')

plot_roc("Train Weighted", y_train, train_predictions_weighted, color=colors[1])
plot_roc("Test Weighted", y_test, test_predictions_weighted, color=colors[1], linestyle='--')

plot_roc("Train Resampled", y_train, train_predictions_resampled, color=colors[2])
plot_roc("Test Resampled", y_test, test_predictions_resampled, color=colors[2], linestyle='--')
plt.legend(loc='lower right')
plt.show()


test_df = pd.read_csv('../input/test.csv')
print('test data imported.')
print(test_df.shape)
test_df.head()


X_test = test_df.drop(['ID_code'], axis=1)
X_test = scaler.transform(X_test)
print(X_test.shape)


test_df['target'] = (weighted_model.predict(X_test, batch_size=BATCH_SIZE) > 0.5).astype(int)
test_df.head()


test_df[['ID_code', 'target']].to_csv('submission.csv', index=False)




