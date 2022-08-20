import configparser
import pyodbc
import joblib
import numpy as np
from datetime import datetime, timedelta
import re
import argparse
from math import exp, ceil, floor
import matplotlib.pyplot as plt
import csv
import lightgbm
import sys
# root path for exe file and other resource file, output files
import os

root = str(os.path.dirname(__file__)) + "/"  # "./Assets/Resources/SmartLadle/"
root = "./SmartLadle_Data/Resources/SmartLadle/"

def str2time(time):
    if isinstance(time, datetime):
        return time
    return datetime.strptime(time, "%Y-%m-%d %H:%M:%S")

def time2str(time):
    if isinstance(time, datetime):
        return datetime.strftime(time, "%Y-%m-%d %H:%M:%S")
    return time

def delta_time(time_24hour, time_reference):
    if isinstance(time_24hour, datetime):
        datetime_24hour = time_24hour
    else:
        list_time = re.split(" |-|:|\.", time_24hour)
        if len(list_time) == 6:
            datetime_24hour = datetime.strptime(time_24hour, "%Y-%m-%d %H:%M:%S")   # no .%f second
        else:
            datetime_24hour = datetime.strptime(time_24hour, "%Y-%m-%d %H:%M:%S.%f")  # no .%f second
    if isinstance(time_reference, datetime):
        datetime_reference = time_reference
    else:
        list_ref = re.split(" |-|:|\.", time_reference)
        if len(list_ref) == 6:
            datetime_reference = datetime.strptime(time_reference, "%Y-%m-%d %H:%M:%S")
        else:
            datetime_reference = datetime.strptime(time_reference, "%Y-%m-%d %H:%M:%S.%f")
    dt = datetime_24hour - datetime_reference
    return int( dt.days * 86400 + dt.seconds )  # 1 day = 86400 seconds

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#########################################################################################################
## new function edited by Zhankun, March 23, 2021
# LMF1, LMF2, LMF3  8, 9, 10
# C1_LOAD, C2_LOAD  12, 14
# C1_CAST, C2_CAST  13, 15
# PREHEAT1, ~, PREHEAT6 27~32
'''
From 1-Ladle_Event, @return:
* heat_no
* 12 history intervals
* reline_flag
'''
def extract_interval_relineflag(heat_no, table, cursor):
    def datetime2sec(dt):
        return int(dt.days * 86400 + dt.seconds)
    sql_query_time_current = \
        "select t_stamp, ladle_no from %s \
        where heat_no = ? \
        order by t_stamp desc limit 0, 1;" % (table)  # Get latest time record of heat_no
    cursor.execute(sql_query_time_current, (heat_no))
    result = cursor.fetchall()
    if len(result) == 0:
        return -1, [-1] * 12, -1, -1
    time_current, ladle_no = result[0][0], round(result[0][1])
    sql_query = \
        "select distinct heat_no from %s \
        where t_stamp <= ? and heat_no is not null and heat_no != '        ' and ladle_no = ? \
        order by t_stamp desc limit 0, 4;" % (table)  # Get latest 4 heat_no
    cursor.execute(sql_query, (time_current, ladle_no))
    result = cursor.fetchall()
    list_heat_no = []
    for item in result:
        list_heat_no.append(item[0])
    # start time for latest 4 heat_no (3 previous + 1 current)
    list_time_steel_start = []
    sql_query_time_start = \
        "select t_stamp, ladle_no, location_code from %s \
        where heat_no = ? order by t_stamp limit 0, 1;" % (table)  # find 1st record
    for heat_no in list_heat_no:
        cursor.execute(sql_query_time_start, (heat_no))
        result_time_start = cursor.fetchall()
        list_time_steel_start.append(result_time_start[0][0])
    # Calc intervals for steel, empty, preheat, gap
    list_interval = []
    for i, time_steel_start in enumerate(list_time_steel_start):
        # extract records
        if i == 0:  # i represent previous i-th heat_no
            continue
        time_next = list_time_steel_start[i - 1]
        # sql_query_part = \
        #     "select t_stamp, location_str from %s \
        #     where t_stamp >= ? and t_stamp < ? and ladle_no = ? ;" % (table)
        sql_query_part = \
            "select t_stamp, location_code from %s \
            where t_stamp >= ? and t_stamp < ? and ladle_no = ? by t_stamp;" % (table) # edited by Zhankun, July 06, 2020
        cursor.execute(sql_query_part, (time_steel_start, time_next, ladle_no))
        result_part = cursor.fetchall()
        # calculate intervals
        time_steel_end, time_preheat_start, time_preheat_end = False, False, False
        count_preheat = 0
        for item in result_part:
            # if (item[1] == "C1_LOAD" or item[1] == "C2_LOAD"):
            # if str_value = "LADLE_SPIN_OUT"
            if (int(item[2]) == 18):  # edited by Zhankun, July 06, 2020
                time_steel_end = item[0]  # time_steel_end <= datetime
                continue
            # if item[1].startswith("PREHEAT"):
            if int(item[1]) >= 27 and int(item[1]) <= 32:  # edited by Zhankun, July 06, 2020
                count_preheat += 1
                if count_preheat % 2 == 1:
                    time_preheat_start = item[0]
                else:
                    time_preheat_end = item[0]
        if time_steel_end == False:
            print(list_heat_no[i] + "doesn't LADLE_SPIN_OUT (no event_code==18)" + "for ladle_no=" + ladle_no)
            list_interval += [-1, -1, -1, -1]
            del time_steel_start, time_steel_end, time_preheat_start, time_preheat_end, time_next
            del result_part
            continue
        if count_preheat == 0 or count_preheat % 2 == 1:
            time_interval_steel = datetime2sec(time_steel_end - time_steel_start)
            time_interval_empty = datetime2sec(time_next - time_steel_end)
            time_interval_preheat = 0
            time_interval_gap = 0
        else:
            time_interval_steel = datetime2sec(time_steel_end - time_steel_start)
            time_interval_empty = datetime2sec(time_preheat_start - time_steel_end)
            time_interval_preheat = datetime2sec(time_preheat_end - time_preheat_start)
            time_interval_gap = datetime2sec(time_next - time_preheat_end)
        del time_steel_start, time_steel_end, time_preheat_start, time_preheat_end, time_next
        # save intervals
        list_interval += [time_interval_steel, time_interval_empty, time_interval_preheat, time_interval_gap]
        del result_part
        del time_interval_steel, time_interval_empty, time_interval_preheat, time_interval_gap
    # Count ladle_flag
    sql_query_time_reline = \
        "select t_stamp from %s \
        where t_stamp < ? and ladle_no = ? and location_code = 25 \
        order by t_stamp desc limit 0, 1;" % (table)  # location_code = 25 represent TEAR_OUT (means reline)
    cursor.execute(sql_query_time_reline, (time_current, ladle_no))
    time_reline = cursor.fetchall()[0][0]
    sql_query_reline_flag = \
        "select count(event_code) from %s \
        where t_stamp < ? and t_stamp >= ? and ladle_no = ? \
        and event_code = 17 \
        order by t_stamp;" % (table)  # 17 => LADLE_SPIN_IN, revised by Zhankun, April 23, 2021
    cursor.execute(sql_query_reline_flag, (time_current, time_reline, ladle_no))
    reline_flag = cursor.fetchall()[0][0] + 1  # add 1 (current heat_no itself)
    ## ladle_no is in LMF or NOT ?
    heat_no = list_heat_no[0]
    flag_in_LMF = False
    sql_query_in_LMF = \
        "select location_code from %s \
        where t_stamp < ? and heat_no = ? \
        order by t_stamp desc limit 0, 1;" % (table)
    cursor.execute(sql_query_in_LMF, (time_current, heat_no))
    # if cursor.fetchall()[0][0].startswith("LMF"):
    location_code = int(cursor.fetchall()[0][0])
    if location_code == 8 \
        or location_code == 9 \
        or location_code == 10:  # edited by Zhankun, July 06, 2020
        flag_in_LMF = True
    return flag_in_LMF, list_interval, reline_flag, ladle_no

'''
From 3-Caster Event, @return:
* replace flag
'''
def extract_replaceflag(heat_no, table, cursor):
    # 13 => C1_CAST; 15 => C2_CAST \
    str_heat = ['C1_Weight_Next', 'C2_Weight_Next', 'START_OF_HEAT']
    sql_query_time_current = \
        "select t_stamp, location_code from %s \
        where heat_no = ? and str_value in ('%s')  \
        order by t_stamp desc limit 0, 1;" % (table, '\',\''.join(str for str in str_heat))
    cursor.execute(sql_query_time_current, (heat_no))
    time_current, location_code = cursor.fetchall()[0]
    if location_code == 13:
        str_seq_start = ['C1_Seq_Start', 'SEQUENCE_START']
        str_heat = ['C1_Weight_Next', 'START_OF_HEAT']
    elif location_code == 15:
        str_seq_start = ['C2_Seq_Start', 'SEQUENCE_START']
        str_heat = ['C2_Weight_Next', 'START_OF_HEAT']
    else:
        raise Exception("location_code wrong")
    sql_query_time_replace = \
        "select t_stamp from %s \
        where t_stamp < ? and location_code = ? and str_value in ('%s')  \
        order by t_stamp desc limit 0, 1;" % (table, '\',\''.join(str for str in str_seq_start) )
    cursor.execute(sql_query_time_replace, (time_current, location_code))
    time_replace = cursor.fetchall()[0][0]
    sql_query_replace_flag = \
        "select count(str_value) from %s \
        where t_stamp < ? and t_stamp >= ? and location_code = ? \
        and str_value in ('%s') \
        order by t_stamp;" % (table, '\',\''.join(str for str in str_heat) )
    cursor.execute(sql_query_replace_flag, (time_current, time_replace, location_code))
    replace_flag = cursor.fetchall()[0][0] + 1 + 1  # running tundish itself + next heat_no for tundish
    ## consider 1st heat_no after tundish
    if replace_flag != 2:
        sql_query_replace_ornot1 = \
            "select heat_no from %s \
            where t_stamp < ? and t_stamp >= ? and location_code = ? \
            and str_value = '%s' \
            order by t_stamp desc limit 0, 1;" % (table, str_heat[1]) # verify last start with empty heat_no ?
        cursor.execute(sql_query_replace_ornot1, (time_current, time_replace, location_code))
        result = cursor.fetchall()
        if (len( result ) != 0):
            heat_no = result[0][0]
            if heat_no == '        ':
                replace_flag = 1    # just replace tundish
    else: # replace_flag == 2:  # if C1_seq with no heat_no && no start => 1st
        sql_query_replace_ornot2 = \
            "select heat_no from %s \
            where t_stamp = ? and location_code = ? \
            and str_value = '%s' \
            order by t_stamp desc;" % (table, str_seq_start[0])  # verify last start with empty heat_no ?
        cursor.execute(sql_query_replace_ornot2, (time_replace, location_code))
        result = cursor.fetchall()
        if (len(result) != 0):
            heat_no = result[0][0]
            if heat_no == '        ':
                replace_flag = 1  # just replace tundish
    return replace_flag

def predict_replaceflag(location_code, table, cursor):
    # 13 => C1_CAST; 15 => C2_CAST \
    if location_code == 13:
        str_seq_start = ['C1_Seq_Start', 'SEQUENCE_START']
        str_heat = ['C1_Weight_Next', 'START_OF_HEAT']
    elif location_code == 15:
        str_seq_start = ['C2_Seq_Start', 'SEQUENCE_START']
        str_heat = ['C2_Weight_Next', 'START_OF_HEAT']
    else:
        raise Exception("location_code wrong")
    sql_query_time_replace = \
        "select t_stamp from %s \
        where location_code = ? and str_value in ('%s')  \
        order by t_stamp desc limit 0, 1;" % (table, '\',\''.join(str for str in str_seq_start) )
    cursor.execute(sql_query_time_replace, (location_code))
    result = cursor.fetchall()
    if len(result) == 0:
        return -1
    time_replace = result[0][0]
    sql_query_replace_flag = \
        "select count(str_value) from %s \
        where t_stamp >= ? and location_code = ? \
        and str_value in ('%s') \
        order by t_stamp;" % (table, '\',\''.join(str for str in str_heat) )
    cursor.execute(sql_query_replace_flag, (time_replace, location_code))
    replace_flag = cursor.fetchall()[0][0] + 1 + 1  # running tundish itself + next heat_no for tundish
    ## consider 1st heat_no after tundish
    if replace_flag != 2:
        sql_query_replace_ornot1 = \
            "select heat_no from %s \
            where t_stamp >= ? and location_code = ? \
            and str_value = '%s' \
            order by t_stamp desc limit 0, 1;" % (table, str_heat[1]) # verify last start with empty heat_no ?
        cursor.execute(sql_query_replace_ornot1, (time_replace, location_code))
        result = cursor.fetchall()
        if (len( result ) != 0):
            heat_no = result[0][0]
            if heat_no == '        ':
                replace_flag = 1    # just replace tundish
    else: # replace_flag == 2:  # if C1_seq with no heat_no && no start => 1st
        sql_query_replace_ornot2 = \
            "select heat_no from %s \
            where t_stamp = ? and location_code = ? \
            and str_value = '%s' \
            order by t_stamp desc;" % (table, str_seq_start[0])  # verify last start with empty heat_no ?
        cursor.execute(sql_query_replace_ornot2, (time_replace, location_code))
        result = cursor.fetchall()
        if (len(result) != 0):
            heat_no = result[0][0]
            if heat_no == '        ':
                replace_flag = 1  # just replace tundish
    return replace_flag



'''
From 4-EAF, @return:
* time, temp_EAF
* weight
'''
def extract_EAF(heat_no, table, cursor):
    sql_query_EAF = \
        "select tap_temp_ts, tap_temp, tap_weight from %s \
        where heat_no = ? order by tap_temp_ts desc limit 0, 1;" % (table)  # find 1st record
    cursor.execute(sql_query_EAF, (heat_no))
    time_EAF, temp_EAF, weight = cursor.fetchall()[0]
    return time_EAF, temp_EAF, weight

'''
From 2-LMF Event, @return:
* time, temp_LMF_start; time, temp_LMF_end
'''
def extract_LMF(heat_no, table, cursor):
    sql_query_LMF_end = \
        "select t_stamp, num_value from %s \
        where heat_no = ? and str_value = 'TEMPERATURE_SAMPLE'\
        order by t_stamp desc limit 0, 1;" % (table)  # find 1st record
    cursor.execute(sql_query_LMF_end, (heat_no))
    time_LMF_end, temp_LMF_end = cursor.fetchall()[0]
    sql_query_LMF_start = \
        "select t_stamp, num_value from %s \
        where heat_no = ? and str_value = 'TEMPERATURE_SAMPLE'\
        order by t_stamp limit 0, 1;" % (table)
    cursor.execute(sql_query_LMF_start, (heat_no))
    time_LMF_start, temp_LMF_start = cursor.fetchall()[0]
    return time_LMF_start, temp_LMF_start, time_LMF_end, temp_LMF_end

'''
From Caster Process NOV
* weight_speed ( NOW() )
* time_interval_NOW2CAST := weight( NOW() ) / weight_speed ( NOW() )
=> time_interval_LMF2CAST = time_interval_LMF2NOW + time_interval_NOW2CAST
* dT/dt_NOW least min square (positive): compute from last 10 min Temp
=> temp_tundish_start := temp( NOW() ) - dT/dt_NOW * time_interval_NOW2CAST
'''
def convert_time(time_in):
    if isinstance(time_in, datetime):
        return time_in
    list_time = re.split(" |-|:|\.", time_in)
    if len(list_time) == 6:
        datetime_out = datetime.strptime(time_in, "%Y-%m-%d %H:%M:%S")  # no .%f second
    else:
        datetime_out = datetime.strptime(time_in, "%Y-%m-%d %H:%M:%S.%f")  # no .%f second
    return datetime_out
def datetime2sec(dt):
    return int(dt.days * 86400 + dt.seconds)

def extract_CAST(heat_no, table, cursor):
    sql_query_current = \
        "select t_stamp from %s \
        where heat_no = ? \
        order by t_stamp desc limit 0, 1;" % (table) # find the latest record, make sure tundish start wthin 1 hour
    cursor.execute(sql_query_current, (heat_no))
    time_current = convert_time(cursor.fetchall()[0][0])
    time_limit = time_current - timedelta(hours=1)
    sql_query = \
        "select t_stamp, tundish_temp from %s \
        where heat_no = ? and t_stamp > ? \
        order by t_stamp limit 0, 1;" % (table)
    cursor.execute(sql_query, (heat_no, time_limit))
    time_tundish_start, temp_tundish_start = cursor.fetchall()[0]
    return time_tundish_start, temp_tundish_start

def predict_CAST(location_code, table, cursor):
    sql_query_time_current = \
        "select t_stamp from %s \
        where location_code = ? \
        order by t_stamp desc limit 0, 1;" % (table)
    cursor.execute(sql_query_time_current, (location_code))
    time_current = cursor.fetchall()[0][0]
    time_current = convert_time(time_current) # str to datetime
    sql_query_weight_speed = \
        "select heat_no, cast_weight, tundish_temp, cast_speed from %s \
        where t_stamp < ? and location_code = ? \
        order by t_stamp desc limit 0, 1;" % (table)
    cursor.execute(sql_query_weight_speed, (time_current, location_code))
    heat_no, weight_remain, temp_now, cast_speed = cursor.fetchall()[0]
    # if heat_no == '        ':  # NOW() still replacing tundish
    #     return -1, -1, -1  # time_tundish_start, temp_tundish_start, weight_speed = [-1] * 3
    # if cast_speed < 1:
    #     return -2, -2, -2  # now the cast location is empty
    # calc dT/dt at NOW()
    time_current_prev5min = time_current - timedelta(minutes=15)
    sql_query_dT_dt = \
        "select t_stamp, tundish_temp, cast_weight from %s \
        where t_stamp < ? and t_stamp >= ? and heat_no = ? \
        order by t_stamp desc limit 0, 200;" % (table)
    cursor.execute(sql_query_dT_dt, (time_current, time_current_prev5min, heat_no))
    result = cursor.fetchall()
    list_time_fit, list_temp_fit, list_weight_fit = zip(*result)
    list_dt_fit = [datetime2sec(time_fit - time_current) for time_fit in list_time_fit]
    # if len(list_dt_fit) <= (15 * 60 / 5) * 0.95:
    #     return -3, -3, -3  # cast location just start casting
    dT_dt = - (len(list_dt_fit) * sum(map(lambda t, T: t * T, list_dt_fit, list_temp_fit) ) - sum(list_dt_fit)*sum(list_temp_fit)) / \
            (len(list_dt_fit) * sum(map(lambda t: t * t, list_dt_fit) ) - sum(list_dt_fit) * sum(list_dt_fit) )
    weight_speed = - (len(list_dt_fit) * sum(map(lambda t, W: t * W, list_dt_fit, list_weight_fit) ) - sum(list_dt_fit)*sum(list_weight_fit)) / \
            (len(list_dt_fit) * sum(map(lambda t: t * t, list_dt_fit) ) - sum(list_dt_fit) * sum(list_dt_fit) )
    time_interval_NOW2CAST = weight_remain / weight_speed
    time_tundish_start = time_current + timedelta(seconds=time_interval_NOW2CAST)
    temp_tundish_start = temp_now - dT_dt * time_interval_NOW2CAST
    return time_tundish_start, temp_tundish_start, weight_speed


'''
From Caster Process, @return:
* temp_liquidus
'''
def extract_temp_liquidus(heat_no, table, cursor):
    sql_query = "select cast_liquidus from %s \
                where heat_no = ? order by t_stamp desc limit 0, 1;" % (table)
    cursor.execute(sql_query, (heat_no))
    result = cursor.fetchall()
    if len(result) == 0:
        return -1
    return result[0][0]


def predict_temp_liquidus(location_code, table, cursor):
    sql_query = "select cast_liquidus from %s \
                where location_code = ? order by t_stamp desc limit 0, 1;" % (table)
    cursor.execute(sql_query, (location_code))
    result = cursor.fetchall()
    if len(result) == 0:
        return -1
    return result[0][0]

'''
From Caster Process, @return:
* caster throughput
'''
def extract_throughput(heat_no, table, cursor):
    rho = 0.2828 # density lb/(in)^3
    depth = 2.56 # inch
    sql_query = "select cast_speed, ram_width_inches from %s \
                where heat_no = ? and cast_speed > 0 order by t_stamp;" % (table)
    cursor.execute(sql_query, (heat_no))
    result = cursor.fetchall()
    if len(result) == 0:
        return -1
    list_prod = []
    for row in result:
        list_prod.append(row[0] * row[1])
    mean_prod = sum(list_prod) / len(list_prod)
    throughput = rho * depth * mean_prod / 60 # rho * depth * (width*cast_speed) / 60
    return throughput

def predict_throughput(location_code, table, cursor):
    rho = 0.2828  # density lb/(in)^3
    depth = 2.56  # inch
    sql_query_current = \
        "select t_stamp from %s \
        where location_code = ? \
        order by t_stamp desc limit 0, 1;" % (table)  # find the latest record, make sure tundish start wthin 1 hour
    cursor.execute(sql_query_current, (location_code))
    time_current = convert_time(cursor.fetchall()[0][0])
    time_limit = time_current - timedelta(minutes=5)
    sql_query = "select cast_speed, ram_width_inches from %s \
                where location_code = ? and t_stamp > ? order by t_stamp;" % (table)
    cursor.execute(sql_query, (location_code, time_limit))
    result = cursor.fetchall()
    if len(result) == 0:
        return -1
    list_prod = []
    for row in result:
        list_prod.append(row[0] * row[1])
    mean_prod = sum(list_prod) / len(list_prod)
    throughput = rho * depth * mean_prod / 60  # rho * depth * (width*cast_speed) / 60
    return throughput


'''
From Caster Process, @return:
* previous deviation, cast_max_temp_speed, cast_min_temp_speed
'''
def extract_deviation(heat_no, table, cursor):
    sql_query_time = \
        "select t_stamp, location_code from %s \
        where heat_no = ? \
        order by t_stamp asc limit 0, 1;" % (table)
    cursor.execute(sql_query_time, (heat_no))
    result = cursor.fetchall()
    if len(result) == 0:
        return -1
    time_current, location_code = result[0][0], round(result[0][1])
    sql_query_heat_no = \
        "select heat_no from %s \
        where  t_stamp < ? and location_code = ? \
        order by t_stamp desc limit 0, 1;" % (table)
    cursor.execute(sql_query_heat_no, (time_current, location_code))
    result = cursor.fetchall()
    if len(result) == 0:
        return -1
    heat_no_prev = result[0][0]
    def repr_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False
    if not repr_int(heat_no_prev):
        return -1
    sql_query_weight_speed = \
        "select cast_weight, t_stamp from %s \
        where heat_no = ? and cast_weight > 1 \
        order by t_stamp asc limit 0, 1;" % (table)
    cursor.execute(sql_query_weight_speed, (heat_no_prev))
    result = cursor.fetchall()
    if len(result) == 0:
        return -1
    weight_overall, time_tundish_start = result[0]
    ## input weight_overall ( weight, weight_speed )
    ## output: temp_mid, cast_max_temp_speed, cast_min_temp_speed,
    sql_query_temp_mid = \
        "select tundish_temp, cast_max_temp_speed, cast_min_temp_speed from %s \
        where  heat_no = ? and t_stamp > ? and cast_weight <= ? \
        order by t_stamp asc limit 0, 1;" % (table)
    cursor.execute(sql_query_temp_mid, (heat_no_prev, time_tundish_start, 0.5 * weight_overall))
    result = cursor.fetchall()
    if len(result) == 0:
        return -1
    temp_mid, cast_max_temp_speed, cast_min_temp_speed = result[0]
    return temp_mid - 0.5 * (cast_max_temp_speed + cast_min_temp_speed)

def extract_temp_mix_min(heat_no, table, cursor):
    sql_query_temp = \
        "select cast_max_temp_speed, cast_min_temp_speed from %s \
        where heat_no = ?  \
        order by t_stamp desc limit 0, 1;" % (table)
    cursor.execute(sql_query_temp, (heat_no))
    result = cursor.fetchall()
    cast_max_temp_speed, cast_min_temp_speed = result[0]
    return cast_max_temp_speed, cast_min_temp_speed

def predict_deviation(location_code, table, cursor):
    sql_query_temp_mid = \
        "select tundish_temp, cast_max_temp_speed, cast_min_temp_speed from %s \
        where  location_code = ?  \
        order by t_stamp desc limit 0, 1;" % (table)
    cursor.execute(sql_query_temp_mid, (location_code))
    result = cursor.fetchall()
    if len(result) == 0:
        return -1
    temp_mid, cast_max_temp_speed, cast_min_temp_speed = result[0]
    return temp_mid - 0.5 * (cast_max_temp_speed + cast_min_temp_speed), cast_max_temp_speed, cast_min_temp_speed


'''
From Caster Process, @return:
* previous deviation
'''
def extract_time_mid(heat_no, table, cursor):
    sql_query_weight_speed = \
        "select cast_weight, t_stamp from %s \
        where heat_no = ? and cast_weight > 1 \
        order by t_stamp asc limit 0, 1;" % (table)
    cursor.execute(sql_query_weight_speed, (heat_no))
    result = cursor.fetchall()
    if len(result) == 0:
        return -1
    weight_overall, time_tundish_start = result[0]
    sql_query_temp_mid = \
        "select t_stamp from %s \
        where  heat_no = ? and t_stamp > ? and cast_weight <= ? \
        order by t_stamp asc limit 0, 1;" % (table)
    cursor.execute(sql_query_temp_mid, (heat_no, time_tundish_start, 0.5 * weight_overall))
    result = cursor.fetchall()
    if len(result) == 0: # lastest heat_no which is casting
        sql_query_weight_speed = \
            "select t_stamp, cast_weight from %s \
            where t_stamp >= ? and heat_no = ? \
            order by t_stamp desc limit 0, 1;" % (table)
        cursor.execute(sql_query_weight_speed, (time_tundish_start, heat_no))
        result = cursor.fetchall()
        list_time_fit, list_weight_fit = zip(*result)
        list_dt_fit = [delta_time(time_fit, time_tundish_start) for time_fit in list_time_fit]
        weight_speed = - (len(list_dt_fit) * sum(map(lambda t, W: t * W, list_dt_fit, list_weight_fit)) - sum(
            list_dt_fit) * sum(list_weight_fit)) / \
                       (len(list_dt_fit) * sum(map(lambda t: t * t, list_dt_fit)) - sum(list_dt_fit) * sum(list_dt_fit))
        # calc dt_mid := 0.5 * weight / weight_speed = time_mid - time_tundish_start
        dt_mid = round(0.5 * weight_overall / weight_speed)
        time_mid = time2str(str2time(time_tundish_start) + timedelta(seconds=dt_mid))
        return dt_mid, time_mid
    time_mid = result[0][0]
    dt_mid = delta_time(time_mid, time_tundish_start)
    return dt_mid, time_mid


'''
Grab all the data from current heat_no
'''
def grab_current_data(heat_no, location_code):
    flag_in_LMF, list_interval, reline_flag, ladle_no = extract_interval_relineflag(heat_no, table_ladle_event, cursor)
    replace_flag = predict_replaceflag(location_code, table_caster_event, cursor)
    time_EAF, temp_EAF, weight = extract_EAF(heat_no, table_EAF, cursor)
    time_LMF_start, temp_LMF_start, time_LMF_end, temp_LMF_end = extract_LMF(heat_no, table_LMF_event, cursor)
    time_tundish_start, temp_tundish_start, weight_speed = predict_CAST(location_code, table_caster_process, cursor)
    throughput = predict_throughput(location_code, table_caster_process, cursor)
    temp_liquidus = predict_temp_liquidus(location_code, table_caster_process, cursor)
    deviation, cast_max_temp_speed, cast_min_temp_speed = predict_deviation(location_code, table_caster_process, cursor)
    # organize data
    list_interval = merge_time(list_interval)
    time_interval_LMF2CAST = delta_time(time_tundish_start, time_LMF_end)
    time_interval_EAF2LMF = delta_time(time_LMF_start, time_EAF)
    time_interval_LMF = delta_time(time_LMF_end, time_LMF_start)
    time_interval_steel = time_interval_EAF2LMF + time_interval_LMF
    list_data = [heat_no, ladle_no, location_code] + [reline_flag, replace_flag] + \
        [time_interval_LMF2CAST, temp_LMF_end] + list_interval + [time_interval_steel, throughput] + \
        [time_interval_EAF2LMF, time_interval_LMF, temp_EAF, temp_LMF_start, temp_tundish_start] + \
        [temp_liquidus, deviation]
    # calc dt_mid := 0.5 * weight / weight_speed = time_mid - time_tundish_start
    dt_mid = round(0.5 * weight / weight_speed)
    time_mid = time2str(str2time(time_tundish_start) + timedelta(seconds=dt_mid))
    return list_data, dt_mid, time_mid, cast_max_temp_speed, cast_min_temp_speed


'''
Grab all the data from previous heat_no
'''
def grab_prev_data(heat_no, location_code):
    # 13 => C1_CAST; 15 => C2_CAST \
    str_heat_all = ['C1_Weight_Next', 'START_OF_HEAT', 'C2_Weight_Next', 'START_OF_HEAT']
    sql_query_time_current = \
        "select t_stamp, location_code from %s \
        where heat_no = ? and str_value in ('%s')  \
        order by t_stamp desc limit 0, 1;" % (table_caster_event, '\',\''.join(str for str in str_heat_all))
    cursor.execute(sql_query_time_current, (heat_no))
    result = cursor.fetchall()
    if len(result) == 0:  # when do prediction
        sql_query_time_current = \
            "select t_stamp from %s \
            where location_code = ? and str_value in ('%s')  \
            order by t_stamp desc limit 0, 1;" % (table_caster_event, '\',\''.join(str for str in str_heat_all))
        cursor.execute(sql_query_time_current, (location_code))
        result = cursor.fetchall()
    else:
        location_code = result[0][1]
    time_current = result[0][0]
    if location_code == 13: # after specify the location
        str_seq_start = ['C1_Seq_Start', 'SEQUENCE_START']
        str_heat = ['C1_Weight_Next', 'START_OF_HEAT']
    elif location_code == 15:
        str_seq_start = ['C2_Seq_Start', 'SEQUENCE_START']
        str_heat = ['C2_Weight_Next', 'START_OF_HEAT']
    else:
        raise Exception("location_code wrong")
    sql_query_time_replace = \
        "select t_stamp from %s \
        where location_code = ? and str_value in ('%s') and t_stamp < ? \
        order by t_stamp desc limit 0, 1;" % (table_caster_event, '\',\''.join(str for str in str_seq_start))
    cursor.execute(sql_query_time_replace, (location_code, time_current))
    time_replace = cursor.fetchall()[0][0]
    sql_query_time_prev = \
        "select heat_no, t_stamp from %s \
        where location_code = ? and t_stamp < ? \
        and str_value in ('%s') \
        order by t_stamp desc limit 0, 1;" % (table_caster_event, '\',\''.join(str for str in str_heat))
    cursor.execute(sql_query_time_prev, (location_code, time_current))
    heat_no, time_prev = cursor.fetchall()[0]  # previous heat_no, start time for previous heat
    if (convert_time(time_prev) < convert_time(time_replace)):   # do nothing, exit the program when heat_count <= 1
        exit()
    # grab data with previous heat_no
    flag_in_LMF, list_interval, reline_flag, ladle_no = extract_interval_relineflag(heat_no, table_ladle_event, cursor)
    replace_flag = extract_replaceflag(heat_no, table_caster_event, cursor)
    time_EAF, temp_EAF, weight = extract_EAF(heat_no, table_EAF, cursor)
    time_LMF_start, temp_LMF_start, time_LMF_end, temp_LMF_end = extract_LMF(heat_no, table_LMF_event, cursor)
    time_tundish_start, temp_tundish_start = extract_CAST(heat_no, table_caster_process, cursor)
    throughput = extract_throughput(heat_no, table_caster_process, cursor)
    temp_liquidus = extract_temp_liquidus(heat_no, table_caster_process, cursor)
    deviation = extract_deviation(heat_no, table_caster_process, cursor)
    # organize data
    list_interval = merge_time(list_interval)
    time_interval_LMF2CAST = delta_time(time_tundish_start, time_LMF_end)
    time_interval_EAF2LMF = delta_time(time_LMF_start, time_EAF)
    time_interval_LMF = delta_time(time_LMF_end, time_LMF_start)
    time_interval_steel = time_interval_EAF2LMF + time_interval_LMF
    list_data = [heat_no, ladle_no, location_code] + [reline_flag, replace_flag] + \
                [time_interval_LMF2CAST, temp_LMF_end] + list_interval + [time_interval_steel, throughput] + \
                [time_interval_EAF2LMF, time_interval_LMF, temp_EAF, temp_LMF_start, temp_tundish_start] + \
                [temp_liquidus, deviation]
    # extract dt_mid := 0.5 * weight / weight_speed = time_mid - time_tundish_start
    dt_mid, time_mid = extract_time_mid(heat_no, table_caster_process, cursor)
    cast_max_temp_speed, cast_min_temp_speed = extract_temp_mix_min(heat_no, table_caster_process, cursor)
    return list_data, dt_mid, time_mid, cast_max_temp_speed, cast_min_temp_speed

'''
Grab all the data from next heat_no
'''
def grab_next_data(heat_no, location_code):
    str_heat_all = ['C1_Weight_Next', 'START_OF_HEAT', 'C2_Weight_Next', 'START_OF_HEAT']
    sql_query_time_current = \
        "select t_stamp, location_code from %s \
        where heat_no = ? and str_value in ('%s')  \
        order by t_stamp desc limit 0, 1;" % (table_caster_event, '\',\''.join(str for str in str_heat_all))
    cursor.execute(sql_query_time_current, (heat_no))
    result = cursor.fetchall()
    if len(result) == 0:  # when do prediction
        exit()
    time_current, location_code = result[0][0], result[0][1]
    # 13 => C1_CAST; 15 => C2_CAST \
    if location_code == 13:
        str_seq_start = ['C1_Seq_Start', 'SEQUENCE_START']
        str_heat = ['C1_Weight_Next', 'START_OF_HEAT']
    elif location_code == 15:
        str_seq_start = ['C2_Seq_Start', 'SEQUENCE_START']
        str_heat = ['C2_Weight_Next', 'START_OF_HEAT']
    else:
        raise Exception("location_code wrong")
    sql_query_time_next = \
        "select heat_no, t_stamp from %s \
        where location_code = ? and t_stamp > ? \
        and str_value in ('%s') \
        order by t_stamp limit 0, 1;" % (table_caster_event, '\',\''.join(str for str in str_heat))
    cursor.execute(sql_query_time_next, (location_code, time_current))
    result = cursor.fetchall()
    if len(result) == 0:  # when heat has no next heats
        exit()
    heat_no, time_next = result[0]  # next heat_no, start time for next heat
    flag_in_LMF, list_interval, reline_flag, ladle_no = extract_interval_relineflag(heat_no, table_ladle_event, cursor)
    replace_flag = extract_replaceflag(heat_no, table_caster_event, cursor)
    time_EAF, temp_EAF, weight = extract_EAF(heat_no, table_EAF, cursor)
    time_LMF_start, temp_LMF_start, time_LMF_end, temp_LMF_end = extract_LMF(heat_no, table_LMF_event, cursor)
    time_tundish_start, temp_tundish_start = extract_CAST(heat_no, table_caster_process, cursor)
    throughput = extract_throughput(heat_no, table_caster_process, cursor)
    temp_liquidus = extract_temp_liquidus(heat_no, table_caster_process, cursor)
    deviation = extract_deviation(heat_no, table_caster_process, cursor)
    # organize data
    list_interval = merge_time(list_interval)
    time_interval_LMF2CAST = delta_time(time_tundish_start, time_LMF_end)
    time_interval_EAF2LMF = delta_time(time_LMF_start, time_EAF)
    time_interval_LMF = delta_time(time_LMF_end, time_LMF_start)
    time_interval_steel = time_interval_EAF2LMF + time_interval_LMF
    list_data = [heat_no, ladle_no, location_code] + [reline_flag, replace_flag] + \
                [time_interval_LMF2CAST, temp_LMF_end] + list_interval + [time_interval_steel, throughput] + \
                [time_interval_EAF2LMF, time_interval_LMF, temp_EAF, temp_LMF_start, temp_tundish_start] + \
                [temp_liquidus, deviation]
    # extract dt_mid := 0.5 * weight / weight_speed = time_mid - time_tundish_start
    dt_mid, time_mid = extract_time_mid(heat_no, table_caster_process, cursor)
    cast_max_temp_speed, cast_min_temp_speed = extract_temp_mix_min(heat_no, table_caster_process, cursor)
    return list_data, dt_mid, time_mid, cast_max_temp_speed, cast_min_temp_speed


'''
From Caster Process, @return:
* list_temp, list_dt
note: where dt := t_stamp - time_tundish_start
'''
def extract_actual(heat_no, table, cursor):
    sql_query_time_current = \
        "select t_stamp from %s \
        where heat_no = ? \
        order by t_stamp desc limit 0, 1;" % (table)
    cursor.execute(sql_query_time_current, (heat_no))
    time_current = cursor.fetchall()[0][0]
    time_current = convert_time(time_current)  # str to datetime
    time_limit = time_current - timedelta(hours=1)
    sql_query_weight_speed = \
        "select t_stamp, tundish_temp from %s \
        where t_stamp > ? and heat_no = ? \
        order by t_stamp;" % (table)
    cursor.execute(sql_query_weight_speed, (time_limit, heat_no))
    result = cursor.fetchall()
    list_time, list_temp = zip(*result)
    list_time_plot = [elem for (i, elem) in enumerate(list_time) if i % int(len(list_time) / 30) == 0]
    list_temp_plot = [elem for (i, elem) in enumerate(list_temp) if i % int(len(list_temp) / 30) == 0]
    list_dt_plot = [delta_time(time, list_time_plot[0]) for time in list_time_plot]
    return list_dt_plot, list_temp_plot
#########################################################################################################

#########################################################################################################
'''
nonlinear mapping
'''
def sigmoid(x):
    return 1./ (1. + exp(-x))

def nonlinear(x, A):
    return 2 * A * (sigmoid(x / (3 * A)) - 0.5)

def merge_time(list_interval):
    list_out = []
    for ind in range(round(len(list_interval) / 4)):
        list_out.append(list_interval[4*ind])  # steel time
        list_out.append(list_interval[4*ind+2])# preheat time
        list_out.append(list_interval[4*ind+1] + list_interval[4*ind+3]) # gap time + empty time
    return list_out

def get_prediction(x_norm, mean_y, std_y, gbm_temp, gbm_slope):
    y_norm_pred = np.transpose(np.concatenate([[gbm_temp.predict(x_norm, num_iteration=gbm_temp.best_iteration_)],
                                                [gbm_slope.predict(x_norm, num_iteration=gbm_slope.best_iteration_)]]))
    y_pred = y_norm_pred * std_y + mean_y
    return y_pred

def get_impact(x_norm, mean_y, std_y,
               gbm_temp, gbm_slope,
               name_feature):
    dict_impact = {}
    x_average = np.zeros(x_norm.shape)
    y_norm_pred = np.transpose(np.concatenate([[gbm_temp.predict(x_average, num_iteration=gbm_temp.best_iteration_)],
                                               [gbm_slope.predict(x_average, num_iteration=gbm_slope.best_iteration_)]]))
    y_average_pred = y_norm_pred * std_y + mean_y
    for ind, feature in enumerate(name_feature):
        x_feature = np.zeros(x_norm.shape)
        x_feature[:, ind] = x_norm[:, ind]
        y_norm_feature_pred = np.transpose(np.concatenate([[gbm_temp.predict(x_feature, num_iteration=gbm_temp.best_iteration_)],
                                                           [gbm_slope.predict(x_feature, num_iteration=gbm_slope.best_iteration_)]]))
        y_feature_pred = y_norm_feature_pred * std_y + mean_y
        impact_feature = y_feature_pred - y_average_pred
        dict_impact[feature] = round(impact_feature[0][0])
    return dict_impact

def plot_prediction(dict_data, y_pred, dt_mid, time_mid, cast_max_temp_speed, cast_min_temp_speed):
    heat_no = str(dict_data["heat_no"])
    ladle_no = round(dict_data["ladle_no"])
    location_code = round(dict_data["location_code"])
    dict_caster = {"13": 1, "15": 2}
    caster_no = dict_caster[str(location_code)]
    time_interval_mid = dt_mid
    temp_mid = y_pred[0][0]
    slope_mid = y_pred[0][1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1 / 1.5)
    ax.get_xaxis().set_ticks([])
    time_start = 0.5 * time_interval_mid
    time_end = 2 * time_interval_mid
    func_temp_pred = lambda t: - slope_mid * (t - time_interval_mid) + temp_mid
    list_time_pred = np.linspace(time_start, time_end, num=21)
    list_temp_pred = func_temp_pred(list_time_pred)
    plt.plot(time_interval_mid / 60, temp_mid, 'bo', markersize=10)
    plt.plot([elem / 60 for elem in list_time_pred], list_temp_pred, 'b-', label='Predicted')
    plt.xlabel('Time (min)')
    plt.ylabel('Temprature ($^\circ$F)', multialignment='center')
    plt.title(f"Ladle {ladle_no}, Caster {caster_no}")
    plt.margins(0, 0)
    plt.axhline(y=cast_max_temp_speed, color='g', linestyle='-')
    plt.axhline(y=cast_min_temp_speed, color='g', linestyle='-')
    plt.xlim([0, time_end / 60])
    temp_min, temp_max = max(2770, cast_min_temp_speed), min(2850, cast_max_temp_speed)
    ylim_min, ylim_max = floor(temp_min / 5) * 5 - 5, ceil(temp_max / 5) * 5 + 5
    ylim_min, ylim_max = min([ylim_min, ylim_max]), max([ylim_min, ylim_max])
    plt.ylim([ylim_min, ylim_max])
    plt.plot([0, time_interval_mid / 60], [temp_mid, temp_mid], 'k--', lw=0.7)
    plt.text(0, temp_mid, r'$T_{mid}=$' + str(round(temp_mid)) + r'$^\circ$ F', ha='left', va='bottom', color='r')
    plt.plot([time_interval_mid / 60, time_interval_mid / 60], [ylim_min, temp_mid], 'k--', lw=0.7)
    plt.text(time_interval_mid / 60, ylim_min, r'$\Delta t_{mid}=$' + str(round(time_interval_mid / 60)) + ' min  ' +
             r'$t_{mid}=$' + time2str(time_mid), ha='center',
             va='bottom', color='r')
    plt.grid(True)
    plt.legend()
    plt.savefig(root + "output" + str(caster_no) + ".png",
                bbox_inches='tight',
                pad_inches=0)
    # plt.show()

def plot_prediction_actual(dict_data, y_pred, dt_mid, time_mid, cast_max_temp_speed, cast_min_temp_speed, list_dt_plot, list_temp_plot):
    heat_no = str(dict_data["heat_no"])
    ladle_no = round(dict_data["ladle_no"])
    location_code = round(dict_data["location_code"])
    dict_caster = {"13": 1, "15": 2}
    caster_no = dict_caster[str(location_code)]
    time_interval_mid = dt_mid
    temp_mid = y_pred[0][0]
    slope_mid = y_pred[0][1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1 / 1.5)
    ax.get_xaxis().set_ticks([])
    plt.plot([elem / 60 for elem in list_dt_plot], list_temp_plot, 'rv', label='Actual')
    plt.fill_between([elem / 60 for elem in list_dt_plot], [elem + 5 for elem in list_temp_plot],
                     [elem - 5 for elem in list_temp_plot],
                     interpolate=True, color='mistyrose')
    time_start = 0.5 * time_interval_mid
    time_end = min(2 * time_interval_mid, list_dt_plot[-1])
    func_temp_pred = lambda t: - slope_mid * (t - time_interval_mid) + temp_mid
    list_time_pred = np.linspace(time_start, time_end, num=21)
    list_temp_pred = func_temp_pred(list_time_pred)
    plt.plot(time_interval_mid / 60, temp_mid, 'bo', markersize=10)
    plt.plot([elem / 60 for elem in list_time_pred], list_temp_pred, 'b-', label='Predicted')
    plt.xlabel('Time (min)')
    plt.ylabel('Temprature ($^\circ$F)', multialignment='center')
    plt.title(f"Ladle {ladle_no}, Caster {caster_no}")
    plt.margins(0, 0)
    plt.axhline(y=cast_max_temp_speed, color='g', linestyle='-')
    plt.axhline(y=cast_min_temp_speed, color='g', linestyle='-')
    plt.xlim([0, time_end / 60])
    temp_min, temp_max = max(2770, cast_min_temp_speed), min(2850, cast_max_temp_speed)
    ylim_min, ylim_max = floor(temp_min / 5) * 5 - 5, ceil(temp_max / 5) * 5 + 5
    ylim_min, ylim_max = min([ylim_min, ylim_max]), max([ylim_min, ylim_max])
    plt.ylim([ylim_min, ylim_max])
    plt.plot([0, time_interval_mid / 60], [temp_mid, temp_mid], 'k--', lw=0.7)
    plt.text(0, temp_mid, r'$T_{mid}=$' + str(round(temp_mid)) + r'$^\circ$ F', ha='left', va='bottom', color='r')
    plt.plot([time_interval_mid / 60, time_interval_mid / 60], [ylim_min, temp_mid], 'k--', lw=0.7)
    plt.text(time_interval_mid / 60, ylim_min, r'$\Delta t_{mid}=$' + str(round(time_interval_mid/60)) +' min  ' +
             r'$t_{mid}=$' + time2str(time_mid), ha='center',
             va='bottom', color='r')
    plt.grid(True)
    plt.legend()
    plt.savefig(root + "output1.png",
                bbox_inches='tight',
                pad_inches=0)
    plt.savefig(root + "output2.png",
                bbox_inches='tight',
                pad_inches=0)
    # plt.show()

def revise_impact_dict(dict_impact, name_impact_drop):
    for key in name_impact_drop:
        dict_impact.pop(key)
    return dict(("impact " + key, value) for key, value in dict_impact.items())

#########################################################################################################


if __name__ == "__main__":
    # # config
    # config = configparser.ConfigParser()
    # config.read(root + 'settings.ini')
    # # read values from section [SQL Database]
    # driver = config.get('SQL Database', 'driver')
    # server = config.get('SQL Database', 'server')
    # database = config.get('SQL Database', 'database')
    # username = config.get('SQL Database', 'username')
    # password = config.get('SQL Database', 'password')
    # has_port = False
    # if 'port' in list(dict(config['SQL Database'].items()).keys()):
    #     port = config.get('SQL Database', 'port')
    #     has_port = True
    # # read values from section [Table]
    # table_ladle_event = config.get('Table', 'table_ladle_event')
    # table_caster_event = config.get('Table', 'table_caster_event')
    # table_EAF = config.get('Table', 'table_EAF')
    # table_LMF_event = config.get('Table', 'table_LMF_event')
    # table_caster_process = config.get('Table', 'table_caster_process')
    # # connect to database
    # if has_port:
    #     cxnstring = "DRIVER={" + driver + "};SERVER=" + server + ";DATABASE=" + database + \
    #                 ";UID=" + username + ";PWD=" + password + ";PORT=" + port
    # else:
    #     cxnstring = "DRIVER={" + driver + "};SERVER=" + server + ";DATABASE=" + database + \
    #             ";UID=" + username + ";PWD=" + password

    # for the security, we include the database info of SDI in the codes instead if using "settings.ini"
    driver = 'MySQL ODBC 8.0 Unicode Driver'
    server = '127.0.0.1'
    database = 'database'
    username = 'root'  # ''
    password = ''  # ''
    port = '3306'
    cxnstring = "DRIVER={" + driver + "};SERVER=" + server + ";DATABASE=" + database + \
                ";UID=" + username + ";PWD=" + password + ";PORT=" + port
    # tables in database of SDI
    table_ladle_event = '`1-Ladle Event`'
    table_caster_event = '`3-Caster Event`'
    table_EAF = '`4-EAF`'
    table_LMF_event = '`2-LMF Event`'
    table_caster_process = '`Caster Process`'
    # connect to database of SDI
    cnxn = pyodbc.connect(cxnstring)
    cursor = cnxn.cursor()
    # print args of system
    print("args of system:")
    print(sys.argv)
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('heat_no',  help="for data lookup")
    parser.add_argument('caster_no', type=int, help="for history lookup, not used for prediction")
    parser.add_argument('prediction', type=str2bool,
                        help="true on 'run prediction', false on 'next/previous'")
    parser.add_argument('direction', type=int, help="for saying whether to pull next or previous heat")
    args = parser.parse_args()
    # name of features, outputs, impacts
    name_feature = ['heat_no', 'ladle_no', 'location_code', 'reline_flag', 'replace_flag', 'time_interval_LMF2CAST',
                    'temp_LMF_end',
                    'time_interval_steel_1', 'time_interval_preheat_1', 'time_interval_no_steel_1',
                    'time_interval_steel_2', 'time_interval_preheat_2', 'time_interval_no_steel_2',
                    'time_interval_steel_3', 'time_interval_preheat_3', 'time_interval_no_steel_3',
                    'time_interval_steel', 'throughput', 'time_interval_EAF2LMF', 'time_interval_LMF',
                    'temp_EAF', 'temp_LMF_start', 'temp_tundish_start', 'temp_liquidus', 'deviation']
    # name_output = ['temp_mid', 'slope_mid', 'time_mid']
    # name_impact = ['impact reline_flag', 'impact replace_flag', 'impact time_interval_LMF2CAST', 'impact temp_LMF_end',
    #                'impact time_interval_steel_1', 'impact time_interval_preheat_1', 'impact time_interval_no_steel_1',
    #                'impact time_interval_steel_2', 'impact time_interval_preheat_2', 'impact time_interval_no_steel_2',
    #                'impact time_interval_steel_3', 'impact time_interval_preheat_3', 'impact time_interval_no_steel_3',
    #                'impact time_interval_steel', 'impact throughput',
    #                'impact time_interval_EAF2LMF', 'impact time_interval_LMF',
    #                'impact temp_EAF', 'impact temp_LMF_start', 'impact temp_tundish_start', 'impact deviation']
    name_impact_drop = ['heat_no', 'ladle_no', 'location_code', 'temp_liquidus']
    # nonlinear map
    list_index_preheat = [ind for (ind, feature) in enumerate(name_feature) if
                          feature.startswith('time_interval_preheat')]
    list_index_no_steel = [ind for (ind, feature) in enumerate(name_feature) if
                           feature.startswith('time_interval_no_steel')]
    threshold_preheat, threshold_no_steel = 8 * 60 * 60, 12 * 60 * 60  # 8 hours for preheat; 12 hours for with no steel
    # normalize the nonlinear-mapped data, output with mean, std of nonlinear-mapped data, output (x, y）
    mean_x = np.asarray([2.68488258e+07, 7.50855992e+00, 1.39217858e+01, 4.76794226e+01, 2.25276939e+01, 1.21864485e+03,
                         2.84334911e+03,
                         8.90388687e+03, 3.33475679e+02, 4.58920025e+02,
                         9.04401746e+03, 4.78135359e+02, 8.15153510e+02,
                         8.97713998e+03, 3.30156874e+02, 4.62591935e+02,
                         5.16754985e+03, 1.24557015e+02, 1.81111749e+03, 3.35643236e+03,
                         3.00899496e+03, 2.91249413e+03, 2.80734710e+03, 2.77035582e+03, 1.07774421e+01])
    std_x = np.asarray([1.12939631e+07, 4.13755314e+00, 9.96936580e-01, 3.57642539e+01, 3.67477656e+01, 5.16047621e+02,
                        1.21011587e+01,
                        2.79491382e+03, 5.19926177e+02, 2.54947080e+02,
                        5.76725628e+03, 1.47150337e+03, 3.36538713e+03,
                        2.89892330e+03, 5.41523551e+02, 2.59678187e+02,
                        2.00029648e+03, 1.47457854e+01, 1.11552324e+03, 1.56964338e+03,
                        3.59705267e+01, 4.75506996e+01, 1.23990804e+01, 1.61491723e+02, 2.16096753e+02])
    mean_y = np.asarray([2.80887043e+03, 3.04070017e-03])
    std_y = np.asarray([1.11953568e+01, 2.80258664e-03])
    # load the lightGBM model
    gbm_temp = joblib.load(root + 'model_temp.pkl')
    gbm_slope = joblib.load(root + 'model_slope.pkl')
    # access the input data from SQL database
    dict_location_code = {"1": 13, "2": 15} # caster_no 1: location_code 13; caster_no 2: location_code 15
    if args.prediction:
        data_1, dt_mid_1, time_mid_1, cast_max_temp_speed_1, cast_min_temp_speed_1 = grab_current_data(args.heat_no, dict_location_code["1"])
        data_2, dt_mid_2, time_mid_2, cast_max_temp_speed_2, cast_min_temp_speed_2 = grab_current_data(args.heat_no, dict_location_code["2"])
        # 1. nonlinear mapping, 2. normalize, 3. fill the data with averages (set 0, after normalization)
        data_mapped_1 = [int(x) if ind == 0
                         else -1 if x < 0
                         else nonlinear(x, threshold_preheat) if ind in list_index_preheat
                         else nonlinear(x, threshold_no_steel) if ind in list_index_no_steel
                         else x for (ind, x) in enumerate(data_1)]
        data_mapped_2 = [int(x) if ind == 0
                         else -1 if x < 0
                         else nonlinear(x, threshold_preheat) if ind in list_index_preheat
                         else nonlinear(x, threshold_no_steel) if ind in list_index_no_steel
                         else x for (ind, x) in enumerate(data_2)]
        data_norm_1 = np.asarray([[0 if data < 0
                                  else (data - mean) / std
                                  for (data, mean, std) in list(zip(data_mapped_1, mean_x, std_x))]])
        data_norm_2 = np.asarray([[0 if data < 0
                                  else (data - mean) / std
                                  for (data, mean, std) in list(zip(data_mapped_2, mean_x, std_x))]])
        dict_data_1 = {key: val for (key, val) in list(zip(name_feature, data_1))}
        dict_data_2 = {key: val for (key, val) in list(zip(name_feature, data_2))}
        y_pred_1 = get_prediction(data_norm_1, mean_y, std_y, gbm_temp, gbm_slope)
        y_pred_2 = get_prediction(data_norm_2, mean_y, std_y, gbm_temp, gbm_slope)
        plot_prediction(dict_data_1, y_pred_1, dt_mid_1, time_mid_1, cast_max_temp_speed_1, cast_min_temp_speed_1)
        plot_prediction(dict_data_2, y_pred_2, dt_mid_2, time_mid_2, cast_max_temp_speed_2, cast_min_temp_speed_2)
        dict_impact_1 = get_impact(data_norm_1, mean_y, std_y, gbm_temp, gbm_slope, name_feature)
        dict_impact_2 = get_impact(data_norm_2, mean_y, std_y, gbm_temp, gbm_slope, name_feature)
        dict_impact_1 = revise_impact_dict(dict_impact_1, name_impact_drop)
        dict_impact_2 = revise_impact_dict(dict_impact_2, name_impact_drop)
        dict_input_1 = dict((key, value) if key == 'heat_no'
                            else (key, -9999) if value < 0
                            else (key, round(value / 60)) if key.startswith("time_interval")
                            else (key, round(value)) for key, value in dict_data_1.items())
        dict_input_2 = dict((key, value) if key == 'heat_no'
                            else (key, -9999) if value < 0
                            else (key, round(value / 60)) if key.startswith("time_interval")
                            else (key, round(value)) for key, value in dict_data_2.items())
        dict_output_1 = {'temp_mid': round(y_pred_1[0][0]), 'slope_mid': round(y_pred_1[0][1] * 60, 2), 'time_mid': time_mid_1}
        dict_output_2 = {'temp_mid': round(y_pred_2[0][0]), 'slope_mid': round(y_pred_2[0][1] * 60, 2), 'time_mid': time_mid_2}
        dict_row_1 = {**dict_input_1, **dict_output_1, **dict_impact_1}
        dict_row_2 = {**dict_input_2, **dict_output_2, **dict_impact_2}
        with open(root + "output.csv", 'w', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=list(dict_row_1.keys()))
            csvwriter.writeheader()
            csvwriter.writerow(dict_row_1)
            csvwriter.writerow(dict_row_2)
        with open(root + "output_log.csv", 'a+', newline='') as csvfile:  # save logging file
            csvwriter = csv.DictWriter(csvfile, fieldnames=list(dict_row_1.keys()))
            if not os.path.isfile(root + "output_log.csv"):
                csvwriter.writeheader()
            csvwriter.writerow(dict_row_1)
            csvwriter.writerow(dict_row_2)
    elif args.direction == -1:
        location_code = dict_location_code[str(args.caster_no)]
        data, dt_mid, time_mid, cast_max_temp_speed, cast_min_temp_speed = grab_prev_data(args.heat_no, location_code)
        list_dt_plot, list_temp_plot = extract_actual(data[0], table_caster_process, cursor)
        data_mapped = [int(x) if ind == 0
                       else -1 if x < 0
                       else nonlinear(x, threshold_preheat) if ind in list_index_preheat
                       else nonlinear(x, threshold_no_steel) if ind in list_index_no_steel
                       else x for (ind, x) in enumerate(data)]
        data_norm = np.asarray([[0 if data < 0
                                else (data - mean) / std
                                for (data, mean, std) in list(zip(data_mapped, mean_x, std_x))]])
        dict_data = {key: val for (key, val) in list(zip(name_feature, data))}
        y_pred = get_prediction(data_norm, mean_y, std_y, gbm_temp, gbm_slope)
        plot_prediction_actual(dict_data, y_pred, dt_mid, time_mid, cast_max_temp_speed, cast_min_temp_speed,
                              list_dt_plot, list_temp_plot)
        dict_impact = get_impact(data_norm, mean_y, std_y, gbm_temp, gbm_slope, name_feature)
        dict_impact = revise_impact_dict(dict_impact, name_impact_drop)
        dict_input = dict((key, value) if key == 'heat_no'
                          else (key, -9999) if value < 0
                          else (key, round(value / 60)) if key.startswith("time_interval")
                          else (key, round(value)) for key, value in dict_data.items())
        dict_output = {'temp_mid': round(y_pred[0][0]), 'slope_mid': round(y_pred[0][1] * 60, 2), 'time_mid': time_mid}
        dict_row = {**dict_input, **dict_output, **dict_impact}
        with open(root + "output.csv", 'w', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=list(dict_row.keys()))
            csvwriter.writeheader()
            csvwriter.writerow(dict_row)
            csvwriter.writerow(dict_row)
        with open(root + "output_log.csv", 'a+', newline='') as csvfile:  # save logging file
            csvwriter = csv.DictWriter(csvfile, fieldnames=list(dict_row.keys()))
            if not os.path.isfile(root + "output_log.csv"):
                csvwriter.writeheader()
            csvwriter.writerow(dict_row)
            csvwriter.writerow(dict_row)
    else: # args.direction == +1
        location_code = dict_location_code[str(args.caster_no)]
        data, dt_mid, time_mid, cast_max_temp_speed, cast_min_temp_speed = grab_next_data(args.heat_no, location_code)
        list_dt_plot, list_temp_plot = extract_actual(data[0], table_caster_process, cursor)
        data_mapped = [int(x) if ind == 0
                       else -1 if x < 0
                       else nonlinear(x, threshold_preheat) if ind in list_index_preheat
                       else nonlinear(x, threshold_no_steel) if ind in list_index_no_steel
                       else x for (ind, x) in enumerate(data)]
        data_norm = np.asarray([[0 if data < 0
                                else (data - mean) / std
                                for (data, mean, std) in list(zip(data_mapped, mean_x, std_x))]])
        dict_data = {key: val for (key, val) in list(zip(name_feature, data))}
        y_pred = get_prediction(data_norm, mean_y, std_y, gbm_temp, gbm_slope)
        plot_prediction_actual(dict_data, y_pred, dt_mid, time_mid, cast_max_temp_speed, cast_min_temp_speed,
                               list_dt_plot, list_temp_plot)
        dict_impact = get_impact(data_norm, mean_y, std_y, gbm_temp, gbm_slope, name_feature)
        dict_impact = revise_impact_dict(dict_impact, name_impact_drop)
        dict_input = dict((key, value) if key == 'heat_no'
                          else (key, -9999) if value < 0
                          else (key, round(value / 60)) if key.startswith("time_interval")
                          else (key, round(value)) for key, value in dict_data.items())
        dict_output = {'temp_mid': round(y_pred[0][0]), 'slope_mid': round(y_pred[0][1] * 60, 2), 'time_mid': time_mid}
        dict_row = {**dict_input, **dict_output, **dict_impact}
        with open(root + "output.csv", 'w', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=list(dict_row.keys()))
            csvwriter.writeheader()
            csvwriter.writerow(dict_row)
            csvwriter.writerow(dict_row)
        with open(root + "output_log.csv", 'a+', newline='') as csvfile:  # save logging file
            csvwriter = csv.DictWriter(csvfile, fieldnames=list(dict_row.keys()))
            if not os.path.isfile(root + "output_log.csv"):
                csvwriter.writeheader()
            csvwriter.writerow(dict_row)
            csvwriter.writerow(dict_row)
