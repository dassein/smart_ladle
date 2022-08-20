import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import joblib


def load_raw_data(path):
    df = pd.read_csv(path)
    df = df.drop(['heat_no', 'ladle_no', 'location_code', 'temp_liquidus'], axis=1)
    x_raw, y_raw = df.values[:, :-2], df.values[:, -2:]
    return x_raw, y_raw


def get_impact(x_norm, mean_y, std_y,
               gbm_temp, gbm_slope,
               name_feature):
    dict_impact = {}
    x_average = np.zeros(x_norm.shape)
    y_norm_pred = np.transpose(np.concatenate([[gbm_temp.predict(x_average, num_iteration=gbm_temp.best_iteration_)],
                                               [gbm_slope.predict(x_average, num_iteration=gbm_slope.best_iteration_)]]))
    y_average_pred = y_norm_pred * std_y + mean_y
    ############### actual
    y_norm_actual_pred = np.transpose(np.concatenate([[gbm_temp.predict(x_norm, num_iteration=gbm_temp.best_iteration_)],
                                                      [gbm_slope.predict(x_norm,num_iteration=gbm_slope.best_iteration_)]]))
    y_actual_pred = y_norm_actual_pred * std_y + mean_y
    impact_actual = y_actual_pred - y_average_pred
    ##################
    for ind, feature in enumerate(name_feature):
        x_feature = np.zeros(x_norm.shape)
        x_feature[:, ind] = x_norm[:, ind]
        y_norm_feature_pred = np.transpose(np.concatenate([[gbm_temp.predict(x_feature, num_iteration=gbm_temp.best_iteration_)],
                                                           [gbm_slope.predict(x_feature, num_iteration=gbm_slope.best_iteration_)]]))
        y_feature_pred = y_norm_feature_pred * std_y + mean_y
        impact_feature = y_feature_pred - y_average_pred
        dict_impact[feature] = impact_feature
    return dict_impact, impact_actual

def get_name_feature():
    name_feature = list(pd.read_csv("../data/train+excluded9heat_no.csv").columns[: -2])
    name_feature[-2:-1] = [] # no temp_liquidus
    name_feature[0:3] = [] # no heat_no, ladle_no, location_code
    return name_feature

if __name__ == "__main__":
    x_train_raw, y_train_raw = load_raw_data("../data/train+excluded9heat_no.csv")
    x_val_raw, y_val_raw = load_raw_data("../data/val.csv")
    x_test_raw, y_test_raw = load_raw_data("../data/test.csv")
    mean_x, std_x = np.mean(x_train_raw, axis=0), np.std(x_train_raw, axis=0)
    mean_y, std_y = np.mean(y_train_raw, axis=0), np.std(y_train_raw, axis=0)
    x_train, y_train = (x_train_raw - mean_x) / std_x, (y_train_raw - mean_y) / std_y
    x_val, y_val = (x_val_raw - mean_x) / std_x, (y_val_raw - mean_y) / std_y
    x_test, y_test = (x_test_raw - mean_x) / std_x, (y_test_raw - mean_y) / std_y
    name_feature = get_name_feature() # name of input features
    '''load model'''
    '''lightGBM'''
    hyper_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['l2', 'auc'],
        'learning_rate': 0.005,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.7,
        'bagging_freq': 10,
        'verbose': 0,
        "max_depth": 8,
        "num_leaves": 128,
        "max_bin": 512,
        "num_iterations": 100000,
        "n_estimators": 1000
    }
    gbm_temp = lgb.LGBMRegressor(**hyper_params)
    gbm_slope = lgb.LGBMRegressor(**hyper_params)
    gbm_temp.fit(x_train, y_train[:, 0].ravel(),
                 eval_set=[(x_val, y_val[:, 0].ravel())],
                 eval_metric='l1',
                 early_stopping_rounds=1000)
    gbm_slope.fit(x_train, y_train[:, 1].ravel(),
                  eval_set=[(x_val, y_val[:, 1].ravel())],
                  eval_metric='l1',
                  early_stopping_rounds=1000)
    y_train_pred = np.transpose(np.concatenate([[gbm_temp.predict(x_train, num_iteration=gbm_temp.best_iteration_)],
                                                [gbm_slope.predict(x_train, num_iteration=gbm_slope.best_iteration_)]]))
    y_train_pred = y_train_pred * std_y + mean_y
    y_val_pred = np.transpose(np.concatenate([[gbm_temp.predict(x_val, num_iteration=gbm_temp.best_iteration_)],
                                              [gbm_slope.predict(x_val, num_iteration=gbm_slope.best_iteration_)]]))
    y_val_pred = y_val_pred * std_y + mean_y
    y_test_pred = np.transpose(np.concatenate([[gbm_temp.predict(x_test, num_iteration=gbm_temp.best_iteration_)],
                                               [gbm_slope.predict(x_test, num_iteration=gbm_slope.best_iteration_)]]))
    y_test_pred = y_test_pred * std_y + mean_y
    '''evaluate and plot'''
    # gbm_temp = joblib.load('./lightGBM_temp+excluded9heat_no.pkl')
    # gbm_slope = joblib.load('./lightGBM_slope+excluded9heat_no.pkl')
    ''' impact = predicted(0, ..., normalized param_actual, 0, ,..., 0) - predicted(o, ..., 0) '''
    # impact_train, _ = get_impact(x_train, mean_y, std_y, gbm_temp, gbm_slope, name_feature)
    # impact_val, _   = get_impact(x_val,   mean_y, std_y, gbm_temp, gbm_slope, name_feature)
    # impact_test, _  = get_impact(x_test,  mean_y, std_y, gbm_temp, gbm_slope, name_feature)
    # print(impact_test)
    for name in name_feature:
        print(name)
    # heat, ladle, caster =  22022490,2,13 and no temp_liquidus
    x_example = np.asarray([[15,11,898,2843,8858,366,2277,8244,0,2983,10097,1227,2451,5034,129.1534269,2939,2095,3010,2895,2805,10.0]])
    y_example = np.asarray([[2815,0.001163499]])
    x_eg = (x_example - mean_x) / std_x
    impact_example, impact_actual = get_impact(x_eg, mean_y, std_y, gbm_temp, gbm_slope, name_feature)
    print(impact_example)
    list_impact = []
    impact_sum  = 0
    for key, val in impact_example.items():
        print(key)
        print("%.2f" % val[0][0])
        list_impact.append(round(val[0][0], 2))
        impact_sum += val[0][0]
    df =pd.DataFrame(list_impact)
    df.to_csv("./impact_example.csv", index=False)
    print("sum of impact:", round(impact_sum, 2))
    print("actual impact:", round(impact_actual[0][0], 2))
