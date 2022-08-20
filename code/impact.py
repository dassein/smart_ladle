import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import joblib


def load_raw_data(path):
    df = pd.read_csv(path)
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
        x_norm_feature_pred = np.transpose(np.concatenate([[gbm_temp.predict(x_feature, num_iteration=gbm_temp.best_iteration_)],
                                                           [gbm_slope.predict(x_feature, num_iteration=gbm_slope.best_iteration_)]]))
        y_feature_pred = x_norm_feature_pred * std_y + mean_y
        impact_feature = y_feature_pred - y_average_pred
        dict_impact[feature] = impact_feature
    return dict_impact, impact_actual


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
    # impact_train, _ = get_impact(x_train, mean_y, std_y, gbm_temp, gbm_slope, name_feature)
    # impact_val, _  = get_impact(x_val,   mean_y, std_y, gbm_temp, gbm_slope, name_feature)
    # impact_test, _  = get_impact(x_test,  mean_y, std_y, gbm_temp, gbm_slope, name_feature)
    # print(impact_test)
    for name in name_feature:
        print(name)
    x_example = np.asarray([[22022490,2,13,15,11,898,2843,8858,366,2277,8244,0,2983,10097,1227,2451,5034,129.1534269,2939,2095,3010,2895,2805,2783,10.0]])
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

    ####################
    # For outlier heat 32021030
    x_outlier = np.asarray([[32021030, 6, 13, 74, 18, 1317, 2842, 8918, 279, 3399, 8233, 1589, 1594, 9184, 2524, 2295, 5030, 129.24461200000002, 1760, 3270, 2977, 2831, 2796, 2780, 2.0]])
    y_outlier = np.asarray([[2794, -0.003170154]])
    x_norm_outlier = (x_outlier - mean_x) / std_x
    y_norm_outlier_pred = np.transpose(np.concatenate([[gbm_temp.predict(x_norm_outlier, num_iteration=gbm_temp.best_iteration_)],
                                               [gbm_slope.predict(x_norm_outlier,num_iteration=gbm_slope.best_iteration_)]]))
    y_outlier_pred = y_norm_outlier_pred * std_y + mean_y
    print(y_outlier_pred)





