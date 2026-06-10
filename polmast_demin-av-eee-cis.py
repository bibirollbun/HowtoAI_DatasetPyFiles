#EEE-CIS обнаружение мошенничества
# Reboot. DS -75. Первый поток
# Выпускная работа Демина А.В


# Задача поставлена в соревновании:  https://www.kaggle.com/c/ieee-fraud-detection/overview
# За основу взяты нотебуки: 
#                                    https://www.kaggle.com/artgor/eda-and-models 
#                                    https://www.kaggle.com/jesucristo/fraud-complete-eda
#                                    https://www.kaggle.com/cybercat/naive-modeling-using-minimum-analysis  
#                                    https://www.kaggle.com/dejavu23/titanic-survival-seaborn-and-ensembles


# Лучший способ участия в соревновании — найти чужое ядро с хорошим результатом в таблице лидеров, скопировать его и попытаться улучшить результат. 
# (c) https://tproger.ru/translations/kaggle-competitions-introduction/


# Материалы оцениваются с помощью ROC AUC между прогнозируемой вероятностью и наблюдаемой целью.

# Хотя интуитивно кажется, что нужно использовать точность для задачи бинарной классификации, 
# это будет плохим решением, потому что мы имеем дело с проблемой несбалансированного класса. 
# Вместо точности, решения оцениваются с помощью ROC AUC (Receiver Operating Characteristic curve Area Under the Curve). 
# Чем выше результат, тем лучше. Чтобы вести подсчёты с помощью ROC AUC, нужно делать прогнозы в терминах вероятностей, а не бинарные — 0 или 1. 
# ROC показывает истинную положительную оценку по сравнению с ложно положительной оценкой, как функцию порога, согласно которому мы классифицируем экземпляр как положительный.

# Обычно нравится делать наивное базовое предсказание, но в этом случае мы уже знаем, что случайные догадки по задаче будут равны 0,5 по ROC AUC. 


# Файл представления
# Для каждого TransactionID в наборе тестов вы должны предсказать вероятность для переменной isFraud. Файл должен содержать заголовок и иметь следующий формат:
# TransactionID,isFraud
# 3663549,0.5
# 3663550,0.5
# 3663551,0.5
# и т.д.

# Categorical Features - Transaction 
# Функция TransactionDT представляет собой временную дельту из заданной контрольной даты и времени (не фактическая временная метка)


# Таблица транзакций (описание https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203)
# TransactionDT: timedelta из заданной контрольной даты и времени (не фактическая временная метка)
# TransactionAMT: сумма оплаты транзакции в долларах США
# ProductCD: код товара, товар для каждой транзакции
# card1 - card6: информация о платежной карте, такая как тип карты, категория карты, банк-эмитент, страна и т. д.
# addr: адрес
# dist: расстояние
# P_ и (R__) emaildomain: домен электронной почты покупателя и получателя
# C1-C14: подсчет, например, количество найденных адресов, связанных с платежной картой, и т. Д. Фактическое значение маскируется.
# D1-D15: интервал времени, например дни между предыдущими транзакциями и т. Д.
# M1-M9: совпадение, например имена на карте и адрес и т. Д.
# Vxxx: Vesta разработала богатые возможности, включая ранжирование, подсчет и другие отношения сущностей.

# Категориальные признаки:
# ProductCD
# card1 - card6
# addr1, addr2
# P emaildomain R emaildomain
# M1 - M9

# Идентификационная таблица *
# Переменные в этой таблице представляют собой информацию об идентичности - информацию о сетевом соединении (IP, ISP, Proxy и т. Д.) И цифровую подпись (UA / browser / os / version и т. Д.), Связанную с транзакциями.
# Они собраны системой защиты от мошенничества Vesta и партнерами по цифровой безопасности.
# (Имена полей замаскированы и парный словарь не будет предоставлен для защиты конфиденциальности и заключения договора)

# Категориальные признаки:
# DeviceType
# DeviceInfo
# ID 12 - ID 38


# Файлы
# train_{transaction, identity}.csv - the training set
# test_{transaction, identity}.csv - the test set (you must predict the isFraud value for these observations)
# sample_submission.csv - a sample submission file in the correct format


# загрузка библиотек
import pandas as pd
import numpy as np
import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt
%matplotlib inline

from tqdm import tqdm_notebook

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import lightgbm as lgb
import catboost

import time

pd.options.display.precision = 15
import seaborn as sns
%matplotlib inline
sns.set()

# plotly library
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import gc


# просмотр содержимого входной директории https://www.kaggle.com/c/ieee-fraud-detection/data
from subprocess import check_output
folder_path = "../input/ieee-fraud-detection/"
print(check_output(["ls", folder_path]).decode("utf8"))


# загрузка первичных датасетов
train_identity = pd.read_csv(f'{folder_path}train_identity.csv')
train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv')
test_identity = pd.read_csv(f'{folder_path}test_identity.csv')
test_transaction = pd.read_csv(f'{folder_path}test_transaction.csv')
sub = pd.read_csv(f'{folder_path}sample_submission.csv')   


# Наменование столбцов в train_identity

# TransactionID,
# id_01,id_02,id_03,id_04,id_05,id_06,id_07,id_08,id_09,id_10,id_11,id_12,id_13,id_14,id_15,id_16,id_17,id_18,id_19,id_20,id_21,id_22,id_23,id_24,id_25,id_26,id_27,id_28,id_29,id_30,id_31,id_32,id_33,id_34,id_35,id_36,id_37,id_38,
# DeviceType,DeviceInfo

# Первая строка: 2987004,0.0,70787.0,,,,,,,,,100.0,NotFound,,-480.0,New,NotFound,166.0,,542.0,144.0,,,,,,,,New,NotFound,Android 7.0,samsung browser 6.2,32.0,2220x1080,match_status:2,T,F,T,T,mobile,SAMSUNG SM-G892A Build/NRD90M

# Наменование столбцов в train_transaction

# TransactionID,isFraud,TransactionDT,TransactionAmt,ProductCD,card1,card2,card3,card4,card5,card6,addr1,addr2,dist1,dist2,P_emaildomain,R_emaildomain,

# C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,
# D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15,
# M1,M2,M3,M4,M5,M6,M7,M8,M9,
# V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,V29,V30,V31,V32,V33,V34,V35,V36,V37,V38,V39,V40,V41,V42,V43,V44,V45,V46,V47,V48,V49,V50,
# V51,V52,V53,V54,V55,V56,V57,V58,V59,V60,V61,V62,V63,V64,V65,V66,V67,V68,V69,V70,V71,V72,V73,V74,V75,V76,V77,V78,V79,V80,V81,V82,V83,V84,V85,V86,V87,V88,V89,V90,V91,V92,V93,V94,V95,V96,V97,V98,V99,V100,
# V101,V102,V103,V104,V105,V106,V107,V108,V109,V110,V111,V112,V113,V114,V115,V116,V117,V118,V119,V120,V121,V122,V123,V124,V125,V126,V127,V128,V129,V130,V131,V132,V133,V134,V135,V136,V137,V138,V139,V140,
# V141,V142,V143,V144,V145,V146,V147,V148,V149,V150,V151,V152,V153,V154,V155,V156,V157,V158,V159,V160,V161,V162,V163,V164,V165,V166,V167,V168,V169,V170,V171,V172,V173,V174,V175,V176,V177,V178,V179,V180,
# V181,V182,V183,V184,V185,V186,V187,V188,V189,V190,V191,V192,V193,V194,V195,V196,V197,V198,V199,V200,V201,V202,V203,V204,V205,V206,V207,V208,V209,V210,V211,V212,V213,V214,V215,V216,V217,V218,V219,V220,
# V221,V222,V223,V224,V225,V226,V227,V228,V229,V230,V231,V232,V233,V234,V235,V236,V237,V238,V239,V240,V241,V242,V243,V244,V245,V246,V247,V248,V249,V250,V251,V252,V253,V254,V255,V256,V257,V258,V259,V260,
# V261,V262,V263,V264,V265,V266,V267,V268,V269,V270,V271,V272,V273,V274,V275,V276,V277,V278,V279,V280,V281,V282,V283,V284,V285,V286,V287,V288,V289,V290,V291,V292,V293,V294,V295,V296,V297,V298,V299,V300,
# V301,V302,V303,V304,V305,V306,V307,V308,V309,V310,V311,V312,V313,V314,V315,V316,V317,V318,V319,V320,V321,V322,V323,V324,V325,V326,V327,V328,V329,V330,V331,V332,V333,V334,V335,V336,V337,V338,V339
# Первая строка:2987000,0,86400,68.5,W,13926,,150.0,discover,142.0,credit,315.0,87.0,19.0,,,,1.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,2.0,0.0,1.0,1.0,14.0,,13.0,,,,,,,13.0,13.0,,,,0.0,T,T,T,M2,F,T,,,,
# 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,,,,,,,,,,,,,,,,,,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,
# 0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
# 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,117.0,0.0,0.0,0.0,0.0,0.0,117.0,0.0,0.0,0.0,0.0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
# 0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,117.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,117.0,0.0,0.0,0.0,0.0,,,,,,,,,,,,,,,,,,



print(train_transaction.head(1))


print(train_identity.head(1))


# Комбинация данных с помощью функции merge(), которая соединяет два набора данных (аналог join в SQL)
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
print(f'Датасет Train содержит {train.shape[0]} строк and {train.shape[1]} столбцов')
print(f'Датасет Test содержит {test.shape[0]} строк and {test.shape[1]} столбцов')


# Итак, у нас есть два набора данных среднего размера с большим количеством столбцов. Train и test имеют примерно одинаковое количество строк


# Поскольку загруженные данные занимают много оперативной памяти, то удалим из памяти первично загруженные датасеты:
del train_identity, train_transaction, test_identity, test_transaction


# дополнительнопринудительно вызывем сборщик мусора, который почистит память (примерно 3 гб):
gc.collect()


# ✔ Исследование данных


print(train.head(5))


print (train.info())


print(test.head(5))


print (test.info())


print(f'Из всех {train.shape[1]} столбцов датасета Train {train.isnull().any().sum()} столбцов с хотя бы одним пустым значением')


#Подсчет различных наблюдений по каждому столбцу (по умолчанию игнорируем значения NaN, если нужно их учитывать, то nunique( self , axis = 0 , dropna = False)
one_value_cols = [col for col in train.columns if train[col].nunique() <= 1] 
one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]    
one_value_cols == one_value_cols_test


print(f'Всего {len(one_value_cols)} столбцов в train с одним уникальным значением.')
print(f'Всего {len(one_value_cols_test)} столбцов в test с одним уникальным значением')


missing_values_count = train.isnull().sum()
print (missing_values_count[0:10])
total_cells = np.product(train.shape)
total_missing = missing_values_count.sum()
print ("% пропущенных данных = ",(total_missing/total_cells) * 100)


del missing_values_count, total_cells, total_missing, one_value_cols, one_value_cols_test
gc.collect()


# Промежуточные выводы:
#   В большинстве столбцов отсутствуют данные 
#   Также есть столбцы с одним уникальным значением (или все отсутствуют). 
#   Есть много непрерывных переменных и некоторые категориальные. 


# Поиск категориальных и числовых признаков
# Справочно типы в pandas: https://pbpython.com/pandas_dtypes.html
numerical_feats = train.dtypes[ (train.dtypes != "object") & (train.dtypes != "category") ].index
numerical_feats_kol = len(numerical_feats)
print("Количество числовых фичей в датасете train: ", numerical_feats_kol)

categorical_feats = train.dtypes[ (train.dtypes == "object") | (train.dtypes == "category") ].index
categorical_feats_kol = len(categorical_feats)
print("Количество категореальных фичей в датасете train: ", categorical_feats_kol)


# установим формат, чтобы вывести на экран все названия фич (без ..)
# Более подробный формат https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
#pd.set_option('display.max_rows', 1000)
#pd.set_option('display.max_columns', 1000)

pd.set_option('display.width', 200)  # ширина выводимого экрана
pd.set_option('max_seq_items', 500)  # то из-чего перестало автоматически пропускать и ставить ..

print("Имена числовых фичей в датасете train:")
print(train[numerical_feats].columns)
print("*"*100)
print("Имена категореальных столбцов в датасете train:")
print(train[categorical_feats].columns)


#для всех числовых фичей рассчитаем describe(), фичей много, поэтому сделаем разбивку:
gr_kol = 50 # кол-во фичей за 1 проход
n1 = 0
for i in tqdm_notebook(range (1,numerical_feats_kol // gr_kol + 2)):
    n2 = i * gr_kol 
    if n2 > (numerical_feats_kol - 1):
        n2 =  numerical_feats_kol - 1
    cols = [numerical_feats[j] for j in range (n1,n2+1) ]
    #print (cols)
    
    plot_data = pd.DataFrame(train[cols])

    if 0: # для тестирования можно ограничить кол-во строк для экономии памяти и увеличения быстродействия
        max_rows =1000    
        plot_data = pd.DataFrame(plot_data[:max_rows])
    
    print(plot_data.describe())   
    n1 = n2 +1


#для всех числовых фичей построим гистограммы. Фичей много, но при разбивке сделанной выше графики подключивыют, поэтому разбивать не будем (просчитывает долго)
plot_data = pd.DataFrame(train[numerical_feats])
if 0: # для тестрования можно ограничить кол-во строк для экономии памяти и увеличения быстродействия
    max_rows =1000    
    plot_data = pd.DataFrame(plot_data[:max_rows])

_ = plot_data.hist(plot_data.columns, figsize=(60, 40))


# Анализ отдельных фичей


fig, ax = plt.subplots(1, 2, figsize=(18,4))

time_val = train['TransactionDT'].values

sns.distplot(time_val, ax=ax[0], color='r')
ax[0].set_title('Распределение TransactionDT', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

sns.distplot(np.log(time_val), ax=ax[1], color='b')
ax[1].set_title('Распеределение логарифма TransactionDT', fontsize=14)
ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])

plt.show()


fig, ax = plt.subplots(1, 2, figsize=(18,4))

time_val = train.loc[train['isFraud'] == 1]['TransactionDT'].values

sns.distplot(np.log(time_val), ax=ax[0], color='r')
ax[0].set_title('Распеределение логарифма TransactionDT, isFraud=1', fontsize=14)
ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])

time_val = train.loc[train['isFraud'] == 0]['TransactionDT'].values

sns.distplot(np.log(time_val), ax=ax[1], color='b')
ax[1].set_title('Распеределение логарифма TransactionDT, isFraud=0', fontsize=14)
ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])


plt.show()


train['TransactionDT'].plot(kind='hist',
                                        figsize=(15, 5),
                                        label='train',
                                        bins=50,
                                        title='Train против Test TransactionDT распределения')
test['TransactionDT'].plot(kind='hist',
                                       label='test',
                                       bins=50)
plt.legend()
plt.show()


plt.figure(figsize=(30, 10))
c_features = [col for col in train[numerical_feats].columns if (col[:1] == "c") | (col[:1] == "C")] 
uniques = [len(train[col].unique()) for col in c_features]
sns.set(font_scale=1.2)
ax = sns.barplot(c_features, uniques, log=True)
ax.set(xlabel='Признаки', ylabel='логарифм(кол-во уникальных)', title='Число уникальных значений фичей TRAIN')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 


print(train['id_01'].unique())
print(len(train['id_01'].unique())-1)


plt.hist(train['id_01'], bins=77); #bins - если задано целое число, бины + 1 ребро бина рассчитываются и возвращаются в соответствии с numpy.histogram
plt.title('Распределение значений признака id_01');


# id_01 имеет интересное распределение: у него 77 уникальных неположительных значений.


# Посмотрим список из количеств уникальных значений признка id_03
# полученный список будет в порядке убывания, так что первый элемент является наиболее часто встречающимся элементом. Исключает значения NA по умолчанию.
# normalize: логический, по умолчанию False, Если True, то возвращаемый объект будет содержать относительные частоты уникальных значений
train['id_03'].value_counts(dropna=False, normalize=True).head()


# d_03 содержит 88% пропущенных значений, а 98% значений либо отсутствуют, либо равны 0.


train['id_11'].value_counts(dropna=False, normalize=True).head()


# 22% значений в id_11 равны 100, а 76% отсутствуют. 


plt.hist(train['id_07']);
plt.title('Распределение значений признака id_07');


%%time
# за основу взята функция из ноутбука https://www.kaggle.com/jesucristo/fraud-complete-eda
# который в свою очередь ссылается на ноутбук https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
# ПРЕДУПРЕЖДЕНИЕ! Это может повредить данные
def reduce_mem_usage2(df):
    # переберите все столбцы кадра данных и измените тип данных чтобы уменьшить использование памяти.      
    start_mem = df.memory_usage().sum() / 1024**2
    print('Использование памяти датафрейма составляет {:.2f} MB'.format(start_mem))
    
    for col in tqdm_notebook(df.columns):
        col_type = df[col].dtype
        
        if col_type != object:
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Использование памяти после оптимизации составляет: {:.2f} MB'.format(end_mem))
    print('Экономия составила {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# для дальнейшего анализа обработаем оба датасета train и test для сокращения пямяти и создадим один объедененный датасет для преобразования фичей и создания новых


# Для чистоты удалим ранее созданные датасеты и загрузим из первичных файлов
del train, test
gc.collect()


# загрузка из первичных файлов

train_identity = pd.read_csv(f'{folder_path}train_identity.csv')
train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv')
test_identity = pd.read_csv(f'{folder_path}test_identity.csv')
test_transaction = pd.read_csv(f'{folder_path}test_transaction.csv')
  


%%time
# клеточная магия %% должна начинаться с первой строки по соглашению. Вот почему она называются клеточной магией 😊
# Выполнение пунктов ниже займет некоторое время но сократит использование памяти
gc.collect()
train_tr = reduce_mem_usage2(train_transaction)
train_id = reduce_mem_usage2(train_identity)


%%time
gc.collect()
test_tr = reduce_mem_usage2(test_transaction)
test_id = reduce_mem_usage2(test_identity)


del train_identity,train_transaction, test_identity, test_transaction
gc.collect()


# создадим три датасета: train, test и их объединение dataset
train = pd.merge(train_tr, train_id, on='TransactionID', how='left')
test = pd.merge(test_tr, test_id, on='TransactionID', how='left')
dataset = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)
train_len = len(train)


del train_tr, train_id, test_tr, test_id
gc.collect()


# Числовые фичи
num_cols = [col for col in dataset.columns if dataset[col].dtype in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]
dataset[num_cols].describe()


#Категориальные фичи
cat_cols = [col for col in dataset.columns if dataset[col].dtype not in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]
dataset[cat_cols].describe()


# по всем категориальным фичам сгруппируем значения и посчитаем среднее isFraud 
for col in cat_cols:
    print('-'*25+'['+col+']'+'-'*25)
    print(dataset[[col, 'isFraud']].groupby(col).mean()*100)


# Преобразования фичей и создания новых


# преобразуем TransactionDT в стандартную дату и создадим новую фичу Date
import datetime

genesis = datetime.datetime.strptime('2019-01-01', '%Y-%m-%d')
dataset['Date'] = dataset['TransactionDT'].apply(lambda x : genesis+datetime.timedelta(seconds=x))


# создадим новые фичи:
dataset['Weekdays'] = dataset['Date'].dt.dayofweek
dataset['Days'] = dataset['Date'].dt.day
dataset['Hours'] = dataset['Date'].dt.hour


fig, ax = plt.subplots(1, 3, figsize=(15, 5))

g = sns.barplot(dataset[dataset.index<train_len].Weekdays, train.isFraud, ax=ax[0])
ax[0].set_title('Fraud Charges by Weekdays')
plt.setp(g.get_xticklabels(), visible=False)

g = sns.barplot(dataset[dataset.index<train_len].Days, train.isFraud, ax=ax[1])
ax[1].set_title('Fraud Charges by Days')
plt.setp(g.get_xticklabels(), visible=False)

g = sns.barplot(dataset[dataset.index<train_len].Hours, train.isFraud, ax=ax[2])
ax[2].set_title('Fraud Charges by Hours')
plt.setp(g.get_xticklabels(), visible=False)

plt.show()


# можно увидидеть явную зависимость от часов и дней недели


dataset.drop('Date', axis=1, inplace=True)


# Обработка редких или пропущенных почтовых доменов


print(dataset['P_emaildomain'].value_counts().head())
print('Data type : {}'.format(dataset['P_emaildomain'].dtype))


# объеденим значения нескольких почтовых доменов в значение другие (etc)
dataset.loc[(dataset.P_emaildomain!='gmail.com')&(dataset.P_emaildomain!='yahoo.com')&(dataset.P_emaildomain!='hotmail.com')&(dataset.P_emaildomain!='anonymous.com')&(dataset.P_emaildomain!='aol.com'), 'P_emaildomain'] = 'etc'


sns.countplot(dataset['P_emaildomain'])
fig = plt.gcf()
fig.set_size_inches(10, 4)
plt.show()


print(dataset['R_emaildomain'].value_counts().head())
print('Data type : {}'.format(dataset['P_emaildomain'].dtype))


sns.countplot(dataset['R_emaildomain'])
fig = plt.gcf()
fig.set_size_inches(10, 4)
plt.show()


# Анализ операционок


top_os = dataset[['id_30', 'isFraud']].groupby(['id_30']).mean().sort_values(by=['isFraud'], ascending=False).head(10)
top_os.T


# Вывод: обвинения в мошенничестве в основном использовались мобильными устройствами или устройствами, работающими под управлением редких операционных систем (обозначенных другими)


top_os = list(top_os.index)


all_os = list(dataset['id_30'].unique())
safe_os = [os for os in all_os if os not in top_os]


dataset.id_30.replace(safe_os, 'etc', inplace=True)


dataset[['id_30', 'isFraud']].groupby(['id_30']).mean().T


# Анализ браузеров


top_browsers = dataset[['id_31', 'isFraud']].groupby(['id_31']).mean().sort_values(by=['isFraud'], ascending=False).head(10)
top_browsers.T


top_browsers = list(top_browsers.index)


all_browsers = list(dataset['id_31'].unique())
safe_browsers = [brw for brw in all_browsers if brw not in top_browsers]


dataset.id_31.replace(safe_browsers, 'etc', inplace=True)


dataset[['id_31', 'isFraud']].groupby('id_31').mean().sort_values(by='isFraud', ascending=False).T


# Анализ размер экрана
# размеры экрана могут быть факторами для отслеживания определенных типов устройств


top_scrsz = dataset[['id_33', 'isFraud']].groupby(['id_33']).mean().sort_values(by=['isFraud'], ascending=False).head(15)
top_scrsz.T


top_scrsz = list(top_scrsz.index)


all_scrsz = dataset['id_33'].unique()
safe_scrsz = [s for s in all_scrsz if s not in top_scrsz]


dataset.id_33.replace(safe_scrsz, 'etc', inplace=True)


dataset[['id_33', 'isFraud']].groupby('id_33').mean().sort_values(by='isFraud', ascending=False).T


# Анализ информации об устройстве


top_dev = dataset[['DeviceInfo', 'isFraud']].groupby(['DeviceInfo']).mean().sort_values(by='isFraud', ascending=False).head(10)
top_dev.T


top_dev = list(top_dev.loc[top_dev.isFraud>0.5].index)
top_dev


all_dev = dataset['DeviceInfo'].unique()
safe_dev = [dev for dev in all_dev if dev not in top_dev]


dataset.DeviceInfo.replace(safe_dev, 'etc', inplace=True)


dataset[['DeviceInfo', 'isFraud']].groupby('DeviceInfo').mean().sort_values(by=['isFraud'], ascending=False).T


# Векторизация: one-hot



dataset_num = dataset.select_dtypes(exclude=['object'])
dataset_num.head()


dataset_cat = dataset.select_dtypes(include=['object'])
dataset_cat.head()


print('Added Columns : {}'.format(sum(dataset_cat.nunique().values)-len(dataset_cat.columns)))


dataset_cat_new = pd.get_dummies(dataset_cat) #Преобразовать категориальные признаки
dataset = pd.concat([dataset_num, dataset_cat_new], axis=1) #сформируем новый датасет
dataset.shape


dataset.drop('TransactionID', axis=1, inplace=True) # удалим TransactionID
del dataset_num, dataset_cat
gc.collect()


# Моделирование


# Подготовка данных


dataset.head()


# из обработанного датасета сформируем train и test
train = dataset[:train_len]
test = dataset[train_len:]


# сформируем датасеты для моделирования
y = train.isFraud
X = train.drop('isFraud', axis=1)
test_y = test.isFraud
test_X = test.drop('isFraud', axis=1)


print(train.head(5))


for f in X.columns:
    if X[f].dtype=='object' or test_X[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X[f].values) + list(test_X[f].values))
        X[f] = lbl.transform(list(X[f].values))
        test_X[f] = lbl.transform(list(test_X[f].values))   


np.unique(y)


# Light Gradient Boosting Machine(LGBM) 


from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)


%%time
import lightgbm as lgbm

hyper = {
    'num_leaves' : 500,
    'min_child_weight': 0.03,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.4,
    'min_data_in_leaf': 100,
    'objective': 'binary',
    'max_depth': 6,
    'learning_rate': 0.05,
    'boosting_type': 'gbdt',
    'bagging_seed': 10,
    'metric': 'auc',
    'verbosity': 0,
    'reg_alpha': 0.4,
    'reg_lambda': 0.6,
    'random_state': 0
}

dtrain = lgbm.Dataset(train_X, label=train_y)
dvalid = lgbm.Dataset(val_X, label=val_y)
model = lgbm.train(hyper, dtrain, 10000, valid_sets=[dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500) #лучший резльтат дает 10000, для ускорения можно зададим 200
# model = lgbm.train(hyper, dtrain, 200, valid_sets=[dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500) #лучший резльтат дает 10000, для ускорения можно зададим 200


preds_lgb = model.predict(test_X)


del train_X, val_X, train_y, val_y
gc.collect()


# LGBM дает отличные результаты, применение других методов вряд ли их улучшат + они выполняются очень долго

Pr_Other = 0 # 0 - не применять другие методы, 1 - применять


Pr_Test = 0
if Pr_Test:
    del y, X, test_y, test_X 
    gc.collect()
    y = train.isFraud
    X = train.drop('isFraud', axis=1)
    test_y = test.isFraud
    test_X = test.drop('isFraud', axis=1)


if Pr_Other:
    # применение других методов требует более глубокую подготовку данных
    
    #дропнем оставшиеся категориальные фичи
    categorical_feats = X.dtypes[ (train.dtypes == "object") | (train.dtypes == "category") ].index
    categorical_feats_kol = len(categorical_feats)
    X.drop(categorical_feats, axis=1, inplace=True)
    test_X.drop(categorical_feats, axis=1, inplace=True)
    
    missing_values_count = X.isnull().sum()
    print (missing_values_count[0:10])
    total_cells = np.product(X.shape)
    total_missing = missing_values_count.sum()
    print ("% до fillna пропущенных данных = ",(total_missing/total_cells) * 100)
    
    X.fillna(-1,inplace=True)
    test_X.fillna(-1,inplace=True)
    
    missing_values_count = X.isnull().sum()
    print (missing_values_count[0:10])
    total_cells = np.product(X.shape)
    total_missing = missing_values_count.sum()
    print ("% после fillna пропущенных данных = ",(total_missing/total_cells) * 100)



if Pr_Other:
    # Нормулизуем D фичи
    for i in tqdm_notebook(range(1,16)):
        if i in [1,2,3,5,9]: continue
        X['D'+str(i)] =  X['D'+str(i)] - X.TransactionDT/np.float32(24*60*60)
        test_X['D'+str(i)] = test_X['D'+str(i)] - test_X.TransactionDT/np.float32(24*60*60) 


if 0 & Pr_Other:
    # для других методов дропнем часть информации для освобождения памяти
    print(f"До удаления, посмотрим топ фичей с пропущенными значениями:\n{X.isna().sum().sort_values(ascending = False).head(5)}\n")
    thresh = 0.80 #Из-за множества значений NA (%), то что больше 80% - это слишком много - шум
    X_less_nas = X.dropna(thresh=X.shape[0]*(1-thresh), axis='columns')
    cols_dropped  = list(set(X.columns)-set(X_less_nas.columns))
    test_X.drop(cols_dropped, axis=1, inplace=True)
    print(f"После удаления, топ фичей с пропущенными значениями:\n{X_less_nas.isna().sum().sort_values(ascending = False).head(5)}")
    print(f"\nКол-во удаленных фичей = {len(set(X.columns)-set(X_less_nas.columns))}, or {len(set(X.columns)-set(X_less_nas.columns))/len(X.columns)*100:.2f}% фичей")
    X = X_less_nas
    del X_less_nas
    gc.collect()


if 0 & Pr_Other:
    cols = ['ProductCD', 'card4', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_28', 'id_29', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType']
    for col in cols:
        print('-'*25+'['+col+']'+'-'*25)
        print(dataset[[col, 'isFraud']].groupby(col).mean()*100)
    X.drop(cols,axis=1, inplace=True)
    test_X.drop(cols, axis=1, inplace=True)


Pr_SVC = 0
if Pr_SVC & Pr_Other: # очень долго выполняется
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    param_grid = {'C': [0.1,10, 100, 1000,5000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
    svc_grid = GridSearchCV(SVC(), param_grid, cv=10, refit=True, verbose=1, scoring='roc_auc')
    svc_grid.fit(X,y)
    sc_svc = get_best_score(svc_grid)
    pred_all_svc = svc_grid.predict(test_X)



Pr_knn = 0 # очень долго выполняется, память переполняется
if Pr_knn & Pr_Other:
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    #leaf_range = list(range(3, 15, 1))
    #k_range = list(range(1, 15, 1))
    leaf_range = list(range(3, 4, 1))
    k_range = list(range(1, 2, 1))
    weight_options = ['uniform', 'distance']
    param_grid = dict(leaf_size=leaf_range, n_neighbors=k_range, weights=weight_options)
    print(param_grid)

    knn_grid = GridSearchCV(knn, param_grid, cv=10, verbose=1, scoring='roc_auc')
    knn_grid.fit(X, y)
    sc_knn = get_best_score(knn_grid)
    pred_all_knn = knn_grid.predict(test_X)
    print('KNN: ', roc_auc_score(test_y, pred_all_knn))    


Pr_Tree = 0 # очень долго выполняется, память не переполняется
if Pr_Tree & Pr_Other:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV

    dtree = DecisionTreeClassifier()
    param_grid = {'min_samples_split': [4,7,10,12]}
    dtree_grid = GridSearchCV(dtree, param_grid, cv=10, refit=True, verbose=1)
    dtree_grid.fit(X,y)
    pred_all_dtree = dtree_grid.predict(test_X)
    print('Tree: ', roc_auc_score(test_y, pred_all_dtree))    
    #print(dtree_grid.best_score_)
    #print(dtree_grid.best_params_)
    #print(dtree_grid.best_estimator_)
    


# Submission


submission = sub 
submission['isFraud'] = np.nan
submission.head()


submission['isFraud'] = preds_lgb
submission.head()


submission.to_csv('submission_demin_3.csv', index=False)


sub.loc[ sub['isFraud']>0.99 , 'isFraud'] = 1
b = plt.hist(sub['isFraud'], bins=50)


# del sub
gc.collect()

