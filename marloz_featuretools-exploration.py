# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# featuretools for automated feature engineering
# !pip install featuretools --upgrade
import featuretools as ft

# matplotlit and seaborn for visualizations
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (15, 8)
pylab.rcParams['font.size'] = 10
import seaborn as sns

# Suppress warnings from pandas
import warnings
warnings.filterwarnings('ignore')


# Utility to quickly inspect data
def inspect_df(df, target_col=None):
    print(f'df shape: {df.shape}')
    print('_____________________')
    print(f'datatypes: {df.dtypes.value_counts()}')
    print('_____________________')
    print(f'Num null vals: {df.isnull().sum().sum()}')
    print('_____________________')
    if target_col is not None:
        print(f'{target_col} classes: \n{df[target_col].value_counts()}')
    print('_____________________')
    return df.head()


app_df = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
inspect_df(app_df, target_col='TARGET')


# Downsample majority class to have same number of rows as minority class
def balance_target(df, target_col, positive_class=1, random_state=42):
    positive_idx = df[df[target_col] == positive_class].index
    negative_idx = (df.loc[~df.index.isin(positive_idx)]
                    .sample(len(positive_idx), replace=False, random_state=random_state)).index
    return df.loc[positive_idx.union(negative_idx)]


# Reduction from 300k rows to 50k should speed up exploration and since classes are balanced, shouldn't affect the accuracy too much.
app_df_sample = balance_target(app_df, target_col='TARGET')
app_df_sample.TARGET.value_counts()


# Splitter into features and target
def split_x_y(df, target_col):
    return df.drop(target_col, axis=1), df[target_col]


class DataPreparator:
    
    def __init__(self, id_cols, add_rand_cols=False):
        self.id_cols = id_cols
        self.add_rand_cols = add_rand_cols
        np.random.seed(42)
    
    def prepare_data(self,
                     X,
                     cat_fill_val='none', 
                     cat_trans_func=lambda x: pd.factorize(x)[0],
                     cat_rand_func=lambda x: np.random.choice([0, 1], x.shape[0]),
                     num_fill_val=0,
                     num_trans_func=lambda x: x.astype('float32'),
                     num_rand_func=lambda x: np.random.rand(x.shape[0])):
        ids, X = X[self.id_cols], X.drop(self.id_cols, axis=1)
        X_cat = self._preprocess(X, 'object', cat_fill_val, cat_trans_func, cat_rand_func)
        X_num = self._preprocess(X, 'number', num_fill_val, num_trans_func, num_rand_func)
        return pd.concat([ids, X_cat, X_num], axis=1)
        
    def _preprocess(self, X, dtypes, fill_val, trans_func, rand_func):
        X_proc = (X.select_dtypes(include=dtypes)
                    .fillna(fill_val)
                    .apply(trans_func))
        if X_proc.shape[0] > 0:
            return X_proc.assign(**{f'rand_{dtypes}': rand_func}) if self.add_rand_cols else X_proc


app_df_feat, y = split_x_y(app_df_sample, target_col='TARGET')
app_df_proc = DataPreparator(id_cols=['SK_ID_CURR'], add_rand_cols=True).prepare_data(app_df_feat)
inspect_df(app_df_proc)


from sklearn.ensemble import RandomForestClassifier

class FeatureSelector:
    
    def __init__(self, X, y, id_cols, rand_cols):
        self.X = X
        self.y = y
        self.id_cols = id_cols
        self.rand_cols = rand_cols

    def select_important_features(self):
        rf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)
        rf.fit(self.X, self.y)
        print(f'Model score with full feature set {rf.oob_score_}')
        important_cols = self.get_important_cols(rf, self.X.columns)
        rf.fit(self.X[important_cols], self.y)
        print(f'Model score with reduced feature set {rf.oob_score_}')
        return self.X[self.id_cols + important_cols]

    def get_important_cols(self, model, column_names):
        importances = pd.Series(model.feature_importances_, index=column_names)
        rand_importance = np.max(importances.loc[importances.index.isin(self.rand_cols)])
        important_cols = importances[importances > rand_importance].index.tolist()
        print(f'Number of features with greater than random column importance {len(important_cols)}')
        importances.sort_values().plot(title='feature importance')
        return important_cols


app_df_reduced = FeatureSelector(app_df_proc, y, id_cols=['SK_ID_CURR'], rand_cols=['rand_object', 'rand_number']).select_important_features()
inspect_df(app_df_reduced)


def sample_from_parent_df(parent_df, id_col, child_df):
    sample_ids = parent_df.set_index(id_col).index
    child_df = (child_df.set_index(id_col)
                .apply(lambda x: x.loc[x.index.isin(sample_ids)])
                .reset_index())
    print(f'Num ids in parent df: {len(sample_ids)}, '
          f'num ids in child df: {child_df[id_col].nunique()}')
    return child_df


bureau_df = sample_from_parent_df(parent_df=app_df_reduced, id_col='SK_ID_CURR', 
                                  child_df=pd.read_csv('../input/home-credit-default-risk/bureau.csv'))
inspect_df(bureau_df)


bureau_df_proc = DataPreparator(id_cols=['SK_ID_CURR', 'SK_ID_BUREAU']).prepare_data(bureau_df)
inspect_df(bureau_df_proc)


bureau_bal_df = sample_from_parent_df(parent_df=bureau_df_proc, id_col='SK_ID_BUREAU', 
                                      child_df=pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv'))
inspect_df(bureau_bal_df)


bureau_bal_df_proc = DataPreparator(id_cols=['SK_ID_BUREAU']).prepare_data(bureau_bal_df)
inspect_df(bureau_bal_df_proc)


es = ft.EntitySet(id='credit_data')
es = es.entity_from_dataframe(entity_id='applications',
                              dataframe=app_df_reduced,
                              index='SK_ID_CURR')
es = es.entity_from_dataframe(entity_id='bureau',
                              dataframe=bureau_df_proc,
                              index='SK_ID_BUREAU')
es = es.entity_from_dataframe(entity_id='bureau_balance',
                              dataframe=bureau_bal_df_proc,
                              index='SK_ID_BUREAU_BAL',
                              make_index=True)
es.plot()


rel_app_bureau = ft.Relationship(parent_variable=es['applications']['SK_ID_CURR'], 
                                 child_variable=es['bureau']['SK_ID_CURR'])
rel_bureau_bal = ft.Relationship(parent_variable=es['bureau']['SK_ID_BUREAU'], 
                                 child_variable=es['bureau_balance']['SK_ID_BUREAU'])
es = es.add_relationships([rel_app_bureau, rel_bureau_bal])
es.plot()


feat_mat, feat_def = ft.dfs(entityset=es, target_entity='applications', n_jobs=-1, max_depth=2)


inspect_df(feat_mat)


feat_mat.head().reset_index()


feat_mat_proc = DataPreparator(id_cols=['SK_ID_CURR'], add_rand_cols=True).prepare_data(feat_mat.reset_index())
inspect_df(feat_mat_proc)


feat_mat_imp = FeatureSelector(feat_mat_proc, y, id_cols=['SK_ID_CURR'], rand_cols=['rand_object', 'rand_number']).select_important_features()

