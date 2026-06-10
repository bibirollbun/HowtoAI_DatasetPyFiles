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


pd.options.display.max_columns = 500


train = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
test = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")


credit_card = pd.read_csv("/kaggle/input/home-credit-default-risk/credit_card_balance.csv")


display(train,test,credit_card)


credit_card.sort_values("SK_ID_CURR")


credit_num = credit_card.groupby("SK_ID_CURR").mean().reset_index()
credit_num


alldata = pd.concat([train,test])
alldata = pd.merge(alldata,credit_num,how="left",on="SK_ID_CURR")
alldata


alldata2 = alldata.drop("TARGET",axis=1)


alldata2.dtypes.value_counts()


alldata.corr()["TARGET"].sort_values()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in alldata2.columns[alldata2.dtypes==object]:
    alldata2[i] = le.fit_transform(list(alldata2[i]))


alldata2 = alldata2.fillna(-1)


train2 =alldata2[:len(train)]
test2 =alldata2[len(train):]


from catboost import CatBoostClassifier

cbc = CatBoostClassifier(task_type="GPU")

cbc.fit(train2,train["TARGET"])

result = cbc.predict_proba(test2)

sub = pd.read_csv("/kaggle/input/home-credit-default-risk/sample_submission.csv")

sub.head()


sub["TARGET"] = result[:,1]
sub.to_csv("sub.csv",index=False)

