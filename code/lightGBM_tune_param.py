import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform as sp_uniform
import joblib


def load_raw_data(path):
    df = pd.read_csv(path)
    x_raw, y_raw = df.values[:, 1:-2], df.values[:, -2:]
    return x_raw, y_raw


def evaluate(y_pred, y_raw, str_output="temp_mid"):
    rmse = np.sqrt(mean_squared_error(y_pred, y_raw))
    mae  = np.absolute(np.subtract(y_pred, y_raw)).mean()
    print(str_output + " MAE = ", mae)
    print(str_output + " RMSE = ", rmse)
    print(str_output + " min = ", min(y_pred - y_raw))
    print(str_output + " max = ", max(y_pred - y_raw))
    print("\n")


def plot_hist_error_temp(list_error_temp, str_fig="Train"):
    num_bins = 30
    n, bins, patches = plt.hist(list_error_temp, num_bins, facecolor='blue', alpha=0.5)
    str_title = r"$T_{mid}$ error (" + str_fig + ")"
    plt.title(str_title)
    plt.xlabel(r"error ($\circ F$)")
    plt.grid()
    plt.show()


def plot_hist_error_slope(list_error_slope, str_fig="Train"):
    num_bins = 30
    n, bins, patches = plt.hist(list_error_slope, num_bins, facecolor='blue', alpha=0.5)
    str_title = r"$\frac{dT}{dt}_{mid}$ error (" + str_fig + ")"
    plt.title(str_title)
    plt.xlabel(r"error ($\circ F/s$)")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    '''prepare dataset'''
    x_train_raw, y_train_raw = load_raw_data("../data/train.csv")
    x_val_raw  , y_val_raw   = load_raw_data("../data/val.csv")
    x_test_raw , y_test_raw  = load_raw_data("../data/test.csv")
    mean_x, std_x = np.mean(x_train_raw, axis=0), np.std(x_train_raw, axis=0)
    mean_y, std_y = np.mean(y_train_raw, axis=0), np.std(y_train_raw, axis=0)
    x_train, y_train = (x_train_raw - mean_x) / std_x, (y_train_raw - mean_y) / std_y
    x_val  , y_val   = (x_val_raw   - mean_x) / std_x, (y_val_raw   - mean_y) / std_y
    x_test , y_test  = (x_test_raw  - mean_x) / std_x, (y_test_raw  - mean_y) / std_y
    '''lightGBM'''
    hyper_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['l2', 'auc'],
        # 'learning_rate': 0.005,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.7,
        'bagging_freq': 10,
        'verbose': 0,
        "max_depth": 8,
        "num_leaves": 128,
        "max_bin": 512,
        "num_iterations": 100000,
        # "n_estimators": 1000
    }
    param_grid = {
        'learning_rate': [0.001, 0.005, 0.01],
        'num_leaves': [100, 128, 150],
        'n_estimators': [1000, 1500]
    }
    gbm_temp  = lgb.LGBMRegressor(**hyper_params)
    gbm_slope = lgb.LGBMRegressor(**hyper_params)
    gsearch_temp = GridSearchCV(
        estimator=gbm_temp,
        param_grid=param_grid,
        n_jobs=1, cv=3, verbose=1
    )
    model_temp_best = gsearch_temp.fit(x_train, y_train[:, 0].ravel())
    model_slope_best = gsearch_temp.fit(x_train, y_train[:, 0].ravel())
    # print(model_temp_best.grid_scores_)
    print(model_temp_best.best_params_)
    print(model_temp_best.best_score_)
    best_param_temp  = model_temp_best.best_params_
    best_param_slope = model_slope_best.best_params_
    gbm_temp_best  = lgb.LGBMRegressor(**best_param_temp)
    gbm_slope_best = lgb.LGBMRegressor(**best_param_slope)
    gbm_temp_best.set_params(**best_param_temp)
    gbm_slope_best.set_params(**best_param_slope)




    # gsearch = GridSearchCV(
    #     estimator=LGBMClassifier(boosting_type='gbdt', num_leaves=30, max_depth=5, learning_rate=0.1, n_estimators=50,
    #                              max_bin=225,
    #                              subsample_for_bin=0.8, objective=None, min_split_gain=0,
    #                              min_child_weight=5,
    #                              min_child_samples=10, subsample=1, subsample_freq=1,
    #                              colsample_bytree=1,
    #                              reg_alpha=1, reg_lambda=0, seed=410, nthread=7, silent=True),
    #     param_grid=param_set, scoring='roc_auc', n_jobs=7, iid=False, cv=10)
    # lgb_model2 = gsearch.fit(features_train, label_train)
    # lgb_model2.grid_scores_, lgb_model2.best_params_, lgb_model2.best_score_

    #######################################
    '''save and load models'''
    joblib.dump(gbm_temp_best, './lightGBM_temp_best.pkl')
    joblib.dump(gbm_slope_best, './lightGBM_slope_best.pkl')
    gbm_temp_load = joblib.load('./lightGBM_temp_best.pkl')
    gbm_slope_load = joblib.load('./lightGBM_slope_best.pkl')
    '''provide predictions'''
    y_train_pred = np.transpose(np.concatenate([[gbm_temp_best.predict(x_train, num_iteration=gbm_temp_best.best_iteration_)],
                                                [gbm_slope_best.predict(x_train, num_iteration=gbm_slope_best.best_iteration_)]]))
    y_train_pred = y_train_pred * std_y + mean_y
    y_val_pred = np.transpose(np.concatenate([[gbm_temp_best.predict(x_val, num_iteration=gbm_temp_best.best_iteration_)],
                                              [gbm_slope_best.predict(x_val, num_iteration=gbm_slope_best.best_iteration_)]]))
    y_val_pred = y_val_pred * std_y + mean_y
    y_test_pred = np.transpose(np.concatenate([[gbm_temp_best.predict(x_test, num_iteration=gbm_temp_best.best_iteration_)],
                                               [gbm_slope_best.predict(x_test, num_iteration=gbm_slope_best.best_iteration_)]]))
    y_test_pred = y_test_pred * std_y + mean_y

    '''evaluate and plot'''
    # temp_mid: train, val, test
    evaluate(y_train_pred[:, 0], y_train_raw[:, 0], str_output="temp_mid")
    evaluate(y_val_pred[:, 0], y_val_raw[:, 0], str_output="temp_mid")
    evaluate(y_test_pred[:, 0], y_test_raw[:, 0], str_output="temp_mid")
    plot_hist_error_temp(y_train_pred[:, 0] - y_train_raw[:, 0], str_fig="Train")
    plot_hist_error_temp(y_val_pred[:, 0] - y_val_raw[:, 0], str_fig="Val")
    plot_hist_error_temp(y_test_pred[:, 0] - y_test_raw[:, 0], str_fig="Test")
    # slope_mid: train, val, test
    evaluate(y_train_pred[:, 1], y_train_raw[:, 1], str_output="slope_mid")
    evaluate(y_val_pred[:, 1], y_val_raw[:, 1], str_output="slope_mid")
    evaluate(y_test_pred[:, 1], y_test_raw[:, 1], str_output="slope_mid")
    plot_hist_error_slope(y_train_pred[:, 1] - y_train_raw[:, 1], str_fig="Train")
    plot_hist_error_slope(y_val_pred[:, 1] - y_val_raw[:, 1], str_fig="Val")
    plot_hist_error_slope(y_test_pred[:, 1] - y_test_raw[:, 1], str_fig="Test")
