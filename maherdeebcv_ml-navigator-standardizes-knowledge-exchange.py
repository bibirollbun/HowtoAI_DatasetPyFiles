!git clone https://github.com/KI-labs/ML-Navigator.git ML_Navigator


!cp -fr ./ML_Navigator/* ./
!rm -r ./ML_Navigator
!rm -r ./docs


!cat requirements.txt | xargs -n 1 pip install


from flows.flows import Flows


flow = Flows(3)


path = '/kaggle/input/ieee-fraud-detection/'
files_list = ['train_transaction.csv','test_transaction.csv']


dataframe_dict, columns_set = flow.load_data(path, files_list, rows_amount=10000)


dataframe_dict["train_transaction"].head()


files_list_2 = ['train_identity.csv','test_identity.csv']
dataframe_dict_identity, columns_set_identity = flow.load_data(path, files_list_2, rows_amount=10000)


dataframe_dict_identity["train_identity"].head()


import pandas as pd
dataframe_train =  pd.merge(dataframe_dict["train_transaction"],
                            dataframe_dict_identity["train_identity"], how="left",
                            on='TransactionID') 
dataframe_test = pd.merge(dataframe_dict["test_transaction"],
                            dataframe_dict_identity["test_identity"], how="left",
                            on='TransactionID') 


print(dataframe_train.shape)
print(dataframe_test.shape)


dataframe_dict = {}
dataframe_dict["train"] = dataframe_train
dataframe_dict["test"] = dataframe_test


columns_set = flow.update_data_summary(dataframe_dict)


dataframe_dict, columns_set = flow.encode_categorical_feature(dataframe_dict, print_results=10)


ignore_columns = ['isFraud']


dataframe_dict, columns_set = flow.drop_columns_constant_values(dataframe_dict, ignore_columns)


dataframe_dict, columns_set = flow.drop_correlated_columns(dataframe_dict, ignore_columns)


ignore_columns = ["TransactionID", "isFraud"]


dataframe_dict, columns_set = flow.scale_data(dataframe_dict, ignore_columns)


flow.exploring_data(dataframe_dict, "train")


columns = dataframe_dict["train"].columns
total_columns = columns_set["train"]["continuous"] +\
columns_set["train"]["categorical_integer"]

train_dataframe = dataframe_dict["train"][
    [x for x in total_columns if x not in ignore_columns]]
test_dataframe = dataframe_dict["test"][
    [x for x in total_columns if x not in ignore_columns]]
train_target = dataframe_dict["train"]["isFraud"]


parameters_lightgbm = {
    "data": {
        "train": {"features": train_dataframe, "target": train_target.to_numpy()},
    },
    "split": {
        "method": "kfold",  # "method":"kfold"
        "fold_nr": 5,  # foldnr:5 , "split_ratios": 0.8 # "split_ratios":(0.7,0.2)
    },
    "model": {"type": "lightgbm",
              "hyperparameters": dict(objective='binary', metric='cross-entropy', num_leaves=5,
                                      boost_from_average=True,
                                      learning_rate=0.05, bagging_fraction=0.99, feature_fraction=0.99, max_depth=-1,
                                      num_rounds=10000, min_data_in_leaf=10, boosting='dart')
              },
    "metrics": ["accuracy_score", "roc_auc_score"],
    "predict": {
        "test": {"features": test_dataframe}
    }
}


model_index_list, save_models_dir, y_test = flow.training(parameters_lightgbm)




