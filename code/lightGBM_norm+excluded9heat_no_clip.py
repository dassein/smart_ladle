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


def plot_importance(importance_feature, name_feature):
    feature_imp = pd.DataFrame({'Value': importance_feature, 'Feature': name_feature})
    plt.figure(figsize=(40, 20))
    sns.set(font_scale=5)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                        ascending=False))
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.savefig('./lightGBM_importance+excluded9heat_no_clip.png')
    plt.show()


if __name__ == "__main__":
    '''prepare dataset'''
    x_train_raw, y_train_raw = load_raw_data("../data/train+excluded9heat_no_clip.csv")
    x_val_raw  , y_val_raw   = load_raw_data("../data/val_clip.csv")
    x_test_raw , y_test_raw  = load_raw_data("../data/test_clip.csv")
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
    gbm_temp  = lgb.LGBMRegressor(**hyper_params)
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
    '''export heat no with worst performance in val, test dataset'''
    thresh_error_temp = 10  # degree
    df_val = pd.read_csv('../data/val_clip.csv')
    list_heat_no_val = df_val['heat_no'].values
    list_error_temp_val = (y_val_pred[:, 0] - y_val_raw[:, 0]).ravel()
    list_heat_no_val_worst = [heat_no for (heat_no, error) in list(zip(list_heat_no_val, list_error_temp_val))
                              if error > thresh_error_temp]
    print(list_heat_no_val_worst)
    df_test = pd.read_csv('../data/test_clip.csv')
    list_heat_no_test = df_test['heat_no'].values
    list_error_temp_test = (y_test_pred[:, 0] - y_test_raw[:, 0]).ravel()
    list_heat_no_test_worst = [heat_no for (heat_no, error) in list(zip(list_heat_no_test, list_error_temp_test))
                               if error > thresh_error_temp]
    print(list_heat_no_test_worst)
    '''save and load models'''
    joblib.dump(gbm_temp , './lightGBM_temp+excluded9heat_no_clip.pkl')
    joblib.dump(gbm_slope, './lightGBM_slope+excluded9heat_no_clip.pkl')
    gbm_temp_load  = joblib.load('./lightGBM_temp+excluded9heat_no_clip.pkl')
    gbm_slope_load = joblib.load('./lightGBM_slope+excluded9heat_no_clip.pkl')
    '''plot the importance of variables'''
    importance_feature = list(gbm_temp.feature_importances_)
    print(importance_feature)
    name_feature = list(pd.read_csv("../data/train+excluded9heat_no_clip.csv").columns[: -2])
    print(name_feature)
    plot_importance(importance_feature, name_feature)