import pandas as pd
import numpy as np
from math import exp

def sigmoid(x):
    return 1. / (1. + exp(-x))

def nonlinear(x, A):
    return 2 * A * (sigmoid(x / (3 * A)) - 0.5)

def clip_time_nonlinear(path_in, path_out):
    df = pd.read_csv(path_in)
    time_threshold1 = 12 * 60 * 60  # 12 hours for with no steel
    time_threshold2 =  8 * 60 * 60  #  8 hours for preheat
    df['time_interval_preheat_1'] = df['time_interval_preheat_1'].map(lambda x : nonlinear(x, time_threshold2))
    df['time_interval_preheat_2'] = df['time_interval_preheat_2'].map(lambda x : nonlinear(x, time_threshold2))
    df['time_interval_preheat_3'] = df['time_interval_preheat_3'].map(lambda x : nonlinear(x, time_threshold2))
    df['time_interval_no_steel_1'] = df['time_interval_no_steel_1'].map(lambda x : nonlinear(x, time_threshold1))
    df['time_interval_no_steel_2'] = df['time_interval_no_steel_2'].map(lambda x : nonlinear(x, time_threshold1))
    df['time_interval_no_steel_3'] = df['time_interval_no_steel_3'].map(lambda x : nonlinear(x, time_threshold1))
    df.to_csv(path_out, index=False)


if __name__ == "__main__":
    list_csv_in  = ["../data/val.csv", "../data/test.csv", "../data/train+excluded9heat_no.csv"]
    list_csv_out = ["../data/val_nonlinear.csv", "../data/test_nonlinear.csv",
                    "../data/train+excluded9heat_no_nonlinear.csv"]
    for path_in, path_out in list(zip(list_csv_in, list_csv_out)):
        clip_time_nonlinear(path_in,path_out)
