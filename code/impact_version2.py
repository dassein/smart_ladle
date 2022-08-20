import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import joblib
import copy  # used for deep copy


def load_raw_data(path):
    df = pd.read_csv(path)
    x_raw, y_raw = df.values[:, :-2], df.values[:, -2:]
    return x_raw, y_raw


def get_impact(x_norm, mean_y, std_y,
               gbm_temp, gbm_slope,
               name_feature):
    dict_impact = {}
    y_norm_pred = np.transpose(np.concatenate([[gbm_temp.predict(x_norm, num_iteration=gbm_temp.best_iteration_)],
                                               [gbm_slope.predict(x_norm, num_iteration=gbm_slope.best_iteration_)]]))
    y_actual_pred = y_norm_pred * std_y + mean_y
    for ind, feature in enumerate(name_feature):
        x_feature = copy.deepcopy(x_norm)
        x_feature[:, ind] = 0
        x_norm_feature_pred = np.transpose(np.concatenate([[gbm_temp.predict(x_feature, num_iteration=gbm_temp.best_iteration_)],
                                                           [gbm_slope.predict(x_feature, num_iteration=gbm_slope.best_iteration_)]]))
        y_feature_pred = x_norm_feature_pred * std_y + mean_y
        impact_feature = y_actual_pred - y_feature_pred
        dict_impact[feature] = impact_feature
    return dict_impact


if __name__ == "__main__":
    x_train_raw, y_train_raw = load_raw_data("../data/train+excluded9heat_no.csv")
    x_val_raw, y_val_raw = load_raw_data("../data/val.csv")
    x_test_raw, y_test_raw = load_raw_data("../data/test.csv")
    mean_x, std_x = np.mean(x_train_raw, axis=0), np.std(x_train_raw, axis=0)
    mean_y, std_y = np.mean(y_train_raw, axis=0), np.std(y_train_raw, axis=0)
    x_train, y_train = (x_train_raw - mean_x) / std_x, (y_train_raw - mean_y) / std_y
    x_val, y_val = (x_val_raw - mean_x) / std_x, (y_val_raw - mean_y) / std_y
    x_test, y_test = (x_test_raw - mean_x) / std_x, (y_test_raw - mean_y) / std_y
    name_feature = list(pd.read_csv("../data/train+excluded9heat_no.csv").columns[: -2]) # name of input features
    '''load model'''
    gbm_temp = joblib.load('./lightGBM_temp+excluded9heat_no.pkl')
    gbm_slope = joblib.load('./lightGBM_slope+excluded9heat_no.pkl')
    ''' impact = predicted(0, ..., normalized param_actual, 0, ,..., 0) - predicted(o, ..., 0) '''
    impact_train = get_impact(x_train, mean_y, std_y, gbm_temp, gbm_slope, name_feature)
    impact_val   = get_impact(x_val,   mean_y, std_y, gbm_temp, gbm_slope, name_feature)
    impact_test  = get_impact(x_test,  mean_y, std_y, gbm_temp, gbm_slope, name_feature)
    print(impact_test)