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


import pandas as pd
import numpy as np
import statistics as sts
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as warning
warning.filterwarnings('ignore')
homecredit_dataset=pd.read_csv('../input/application_train.csv')


homecredit_dataset.shape


homecredit_dataset.describe().columns


len(homecredit_dataset.describe().columns)


def nullValueAnalysis(Base_DataSet):
# find the percentage of null values in the base dataset 
     null_values=(Base_DataSet.isna().sum()/Base_DataSet.shape[0])*100 
# make a dataframe for this null value percentage 
     null_values=pd.DataFrame(null_values)
# Add null value percentage column to existing data Frame
     null_values.columns=['Null_Value %']
# sorting the Null value percentage column by descending order
     null_values=null_values['Null_Value %'].sort_values(ascending=False)
# make a dataframe for this null value percentage column
     null_values=pd.DataFrame(null_values)
# take only less than 30 % of the null values to prepare the analytical data set 
     Base_DataSet.drop(null_values[null_values['Null_Value %'] > 30].index,axis= 1,inplace=True) 
     for i in Base_DataSet.describe().columns:
        Base_DataSet[i].fillna(Base_DataSet[i].median(),inplace=True)
        #for i in Base_DataSet.describe(include='object').columns:
        #    Base_DataSet[i].fillna(Base_DataSet[i].median(),inplace=True)
 
     return Base_DataSet


homecredit_dataset=nullValueAnalysis(homecredit_dataset)


for i in homecredit_dataset.isna().sum():
    print(i)


# Outlier formula Q1=0.25 % of base data set
#                 Q3=0.75 % of base data set
#                 IQR(Inter Quartile range)=Q3-Q1
#                 LTV(lower Threshold value)=Q1-(1.5*IQR)
#                 UTV(Upper Threshold value)=Q3+(1.5*IQR)
# if data sets below LTU or above UTV then take median for data sets
def Oulier_Analysis(Base_dataSet):
    for i in Base_dataSet.describe().columns:
        x=np.array(Base_dataSet[i])
        p=[]
        Q1=Base_dataSet[i].quantile(0.25)
        Q3=Base_dataSet[i].quantile(0.75)
        IQR=Q3-Q1
        LTV=Q1-(1.5 * IQR)
        UTV=Q3+(1.5 * IQR)
        for j in x:
            if j <=LTV or j >= UTV:
                p.append(sts.median(x))
            else :
                 p.append(j)
        Base_dataSet[i]=p         

        return Base_dataSet 


homecredit_dataset=Oulier_Analysis(homecredit_dataset)
homecredit_dataset


homecredit_dataset.shape


len(homecredit_dataset.describe().columns)






















