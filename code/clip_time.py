import pandas as pd
import numpy as np


def clip_time(path_in, path_out):
    df = pd.read_csv(path_in)
    time_threshold = 12 * 60 * 60  # 12 hours for preheat, with no steel
    time_threshold =  8 * 60 * 60  #  8 hours for preheat
    df['time_interval_preheat_1'] = df['time_interval_preheat_1'].map(lambda x : min(x, time_threshold))
    df['time_interval_preheat_2'] = df['time_interval_preheat_2'].map(lambda x : min(x, time_threshold))
    df['time_interval_preheat_3'] = df['time_interval_preheat_3'].map(lambda x : min(x, time_threshold))
    df['time_interval_no_steel_1'] = df['time_interval_no_steel_1'].map(lambda x : min(x, time_threshold))
    df['time_interval_no_steel_2'] = df['time_interval_no_steel_2'].map(lambda x : min(x, time_threshold))
    df['time_interval_no_steel_3'] = df['time_interval_no_steel_3'].map(lambda x : min(x, time_threshold))
    df.to_csv(path_out, index=False)


if __name__ == "__main__":
    list_csv_in  = ["../data/train.csv", "../data/val.csv", "../data/test.csv", "../data/train+excluded9heat_no.csv"]
    list_csv_out = ["../data/train_clip.csv", "../data/val_clip.csv", "../data/test_clip.csv",
                    "../data/train+excluded9heat_no_clip.csv"]
    for path_in, path_out in list(zip(list_csv_in, list_csv_out)):
        clip_time(path_in,path_out)
