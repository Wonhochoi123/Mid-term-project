#!/usr/bin/env python

# import modules
from data.keys import weather_keys
import calendar
import numpy as np
import pandas as pd
import requests
import os
import json
from IPython.display import JSON  # JSON(response.json())
from datetime import datetime
import sqlite3
from sqlite3 import Error
import pickle


# Function for sending GET requests
def request_weather(url, querys, keys, interval):
    """pull JSONs from hwww.worldweatheronline.com

    Args:
        url (str): endpoint URL
        querys (list of lists): params for GET request payload; [[location, startdate, enddate],[]...]
        keys (list): API keys ['key1', 'key2',...]
        interval (list/tuple): the start and end indices of 'querys'. Increments +1 from start until end is reached. 

    Returns:
        weather_jsons [list of lists]: 1 json for each querys[index].
    """
    import numpy as np
    import pandas as pd
    import json
    import requests

    weather_jsons = []  # list for collecting JSON files
    count = interval[0]  # start
    end = interval[1]
    key = 0

    while count < (end+1):
        try:
            payload = {'q': querys[count][0],
                       'date': querys[count][1],
                       'enddate': querys[count][2],
                       'tp': '1',
                       'format': 'json',
                       'key': keys[key]}
            r = requests.get(url=url, params=payload)

            if (r.status_code == 200):
                weather_jsons.append(r.json())
                count += 1
                print(count)
            else:
                try:
                    key += 1
                except Exception:
                    key = 0
        except Exception:
            pass

    # write list of jsons to file
    with open(f'data/weather_{count}-{end}.txt', 'w', encoding='utf-8') as outfile:
        try:
            json.dump(weather_jsons, outfile, ensure_ascii=False, indent=4)
        finally:
            outfile.close()

    return weather_jsons


def add_jsons_to_db(jsons):
    """Normalizes JSONs into pandas DF, then appends table to SQLite database

    Args:
        jsons (list of dicts): List of JSONs

    Returns:
        Dataframe: Final constructed dataframe
    """
    import numpy as np
    import pandas as pd
    import json
    from datetime import datetime

    # create db connection
    cnx = create_connection("data/weather.sqlite")

    # Set columns for table
    cols = ['city', 'date', 'time', 'tempC', 'windspeedKmph', 'winddirDegree', 'WindGustKmph', 'visibility',
            'visibilityMiles', 'cloudcover', 'pressure', 'humidity', 'precipMM', 'totalSnow_cm', 'sunrise', 'sunset']

    # instantiate DataFrame to build on
    df = pd.DataFrame(data=None, columns=cols)

    # get all unique city names
    origins = pd.read_csv('data/origin_cities.csv')
    origins = origins.origin_city_name.tolist()

    # set parameters for building df from JSONs
    path = ['data', 'weather', 'hourly']
    meta = [['data', 'weather', 'date'],
            ['data', 'weather', 'totalSnow_cm'],
            ['data', 'weather', 'astronomy'],
            ['data', 'request']]

    json_size = len(jsons)

    # create dataframe from JSONs
    for idx in range(json_size):
        try:
            if (list(jsons[idx]['data'].keys())[0]) == 'error':
                del jsons[idx]
        except Exception:
            pass
        finally:
            data = pd.json_normalize(
                data=jsons[idx], record_path=path, meta=meta, errors='ignore')
            data = data.rename(columns={'data.weather.astronomy': 'sun',
                                        'data.weather.totalSnow_cm': 'totalSnow_cm',
                                        'data.weather.date': 'date',
                                        'data.request': 'city'})
            data['city'] = data.city.apply(lambda x: x['query'].split(', ')[0])
            data['sunrise'] = data.sun.apply(lambda x: x['sunrise'])
            data['sunset'] = data.sun.apply(lambda x: x['sunset'])
            #data['sunrise'] = pd.to_datetime(data['sunrise'], format='%H:%M %p').dt.time
            #data['sunset'] = pd.to_datetime(data['sunrise'], format='%H:%M %p').dt.time
            data = data[cols]

            df = pd.concat([df, data], axis=0)
            del data
            print(f'{file} | progress: {idx}/{json_size}')

    # convert time to int64 (to match flights data)
    df['time'] = df['time'].astype('int64')

    # append data to database table
    df.to_sql('weather_data', cnx, if_exists='append')

    return df

# function for calculating intervals


def interval_calc(start, end, intervals, roundup=False):
    """returns numpy array of n 'intervals' between 'start' and 'end'

    Args:
        start (int): the start indices of 'querys'
        end (int): the end indices of 'querys'
        intervals (list/tuple): number of intervals to split 'querys' into
        roundup (boolean): when True, round up to nearest hundred. Default = False.
    """
    import numpy as np
    import math

    result = []

    # round 'end' up by nearest hundred
    if roundup:
        end = int(math.ceil(end/100)*100)

    # compute intervals
    interval_size = (end - start) / intervals
    end = start + interval_size
    while True:
        result.append([int(start), int(end)])
        start = end + 1
        end = end + interval_size
        if len(result) == intervals:
            break

    return np.array(result)


def create_connection(path):
    """create DB connection

    Args:
        path (string): path to SQLite database

    Returns:
        connection: connect variable
    """
    import sqlite3
    from sqlite3 import Error

    connection = None
    try:
        connection = sqlite3.connect(path)
        print("Connection to SQLite DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection


def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query executed successfully")
    except Error as e:
        print(f"The error '{e}' occurred")


def save_model(model, filepath):
    """Save a trained model to filepath (e.g. 'model/filename')

    Args:
        model (var): trained model (e.g. Linear Regression)
        filepath (str): path to save model (excluding file extension)

    Returns:
        msg (str): confirmation message
    """
    import pickle

    pickle.dump(model, open(f'{filepath}.sav', 'wb'))
    return f"model saved to: {filepath}.sav"


def load_model(filepath):
    """Load model into variable

    Args:
        filepath (string): path to model (e.g. models/LinRegression.sav)
    """
    import pickle

    loaded_model = pickle.load(open(filepath, 'rb'))
    return loaded_model


def sample_data(db_path, table, size):
    """Generate random sample from db table: of a particular size

    Args:
        db_path (str): path to database
        table (str): name of table in DB
        size (int): number of rows to sample from DB table  

    Returns:
        sample [pd.DataFrame]: DF of random rows, index same as DB table 
    """
    import numpy as np
    import pandas as pd

    cnx = create_connection(db_path)
    sample = pd.read_sql_query(
        f'SELECT * FROM flights_X ORDER BY RANDOM() LIMIT {size};', cnx, index_col='index')
    return sample


def send_data_to_db(data, table_name, db_path='data/training.sqlite', if_exist='fail', mode='interval'):
    """Send data to training.sqlite database

    Args:
        data (DataFrame): pd.DataFrame, for adding to DB as table_name
        table_name (str): name of table in database
        db_path (str): path to sqlite database. Default 'data/training.sqlite'
        if_exist (str): {‘fail’, ‘replace’, ‘append’}. Default ‘fail’
        mode (str): {'interval', 'None'}. Default 'interval'

    Returns:
        msg (str): confirmation message
    """
    import numpy as np
    import pandas as pd
    import time

    cnx = create_connection(db_path)

    if if_exist == 'replace':
        execute_query(cnx, f"DROP TABLE IF EXISTS {table_name}")
        if_exist = 'append'
        cnx = create_connection(db_path)

    if mode == 'interval':
        intervals = interval_calc(0, len(data)+1, 30)
        print(intervals)
        [data.iloc[inv[0]:inv[1], :].to_sql(
            table_name, cnx, if_exists=if_exist) for inv in intervals]

    else:
        data.to_sql(table_name, cnx, if_exists=if_exist)

    return f"Saved DF to table: {table_name}, in database: {db_path}"


def incr_train(model, filepath, db_path, invs, select):
    """train & save a model incrementally using partial_fit (and default parameters).
    https://scikit-learn.org/0.15/modules/scaling_strategies.html

    Args:
        model (function): incremental learners (e.g. SGDRegressor, PassiveAggressiveRegressor)
        filepath (str): model save path
        db_path (str): DB retrieval path
        invs (int): number of intervals to break data into
        select (str): select statement from db_path

    Returns:
        msg (str): confirmation message, saved to file.
    """
    import os
    import numpy as np
    import pandas as pd
    import pickle
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor

    # set DB connection
    cnx = create_connection(db_path)
    count = 0
    end = pd.read_sql_query(f'SELECT COUNT(*) FROM {table}', cnx)

    # setup intervals for training
    intervals = interval_calc(0, end+1, invs)
    print(intervals[-1])

    for inv in intervals:

        # get data inside interval
        data = pd.read_sql_query(
            f"SELECT * FROM flight_ready WHERE rowid BETWEEN {inv[0]} AND {inv[1]}", cnx)
        X = data.iloc[:, 2:]
        X.drop(columns='arr_delay', inplace=True)
        y = data['arr_delay']

        # scale data
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X).astype(float))

        # load model
        if os.path.exists(filepath):
            model = pickle.load(open(filepath, 'rb'))

        # partial fit
        model.partial_fit(X_scaled, y)

        # save model
        save_model(model, filepath)

        count += 1
        print(f'partial-fit progress: {count/invs*100}% or {count}/16')

    return f"model trained and saved to: {filepath}"


''' MODELING '''


def incr_train(model, filepath, db_path, table, invs, mode):
    """train & save a model incrementally using partial_fit (and default parameters).
    https://scikit-learn.org/0.15/modules/scaling_strategies.html

    Args:
        model (function): incremental learners (e.g. SGDRegressor, PassiveAggressiveRegressor)
        filepath (str): model save path
        db_path (str): DB retrieval path
        table (str): table to select data from
        invs (int): number of intervals to break data into
        select (str): select statement from db_path
        mode (str): {'A30','A20'}

    Returns:
        msg (str): confirmation message, saved to file.
    """
    import os
    import numpy as np
    import pandas as pd
    import pickle
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor

    # set DB connection
    cnx = create_connection(db_path)
    count = 0
    end = pd.read_sql_query(f'SELECT COUNT(*) FROM {table}', cnx).iat[0, 0]

    # setup intervals for training
    intervals = interval_calc(0, end+1, invs)
    print(intervals[-1])

    for inv in intervals:

        # get data inside interval
        cnx = create_connection('data/fl_final.sqlite')

        if mode == 'A20':
            data = pd.read_sql(
                f"SELECT [index], fl_date, arr_delay, [crs_dep_time(mins)], [crs_arr_time(mins)], origin_taxi, DL, Saturday, o_windspeedKmph, o_visibility, o_cloudcover, o_humidity, o_precipMM, o_totalSnow_cm, o_sunrise, o_sunset, d_windspeedKmph, d_visibility, d_cloudcover, d_pressure, d_humidity, d_precipMM, d_totalSnow_cm FROM flights_final WHERE rowid BETWEEN {inv[0]} AND {inv[1]};", cnx, index_col=['index'])

        elif mode == 'A30':
            data = pd.read_sql(
                f"SELECT [index], fl_date, arr_delay, [crs_dep_time(mins)], [crs_arr_time(mins)], crs_elapsed_time, origin_traffic, dest_traffic, origin_taxi, dest_taxi, WN, DL, Monday, Thursday, Saturday, Friday, o_tempC, o_windspeedKmph, o_winddirDegree, o_visibility, o_cloudcover, o_humidity, o_precipMM, o_totalSnow_cm, o_sunrise, o_sunset, d_windspeedKmph, d_visibility, d_cloudcover, d_pressure, d_humidity, d_precipMM, d_totalSnow_cm FROM flights_final WHERE rowid BETWEEN {inv[0]} AND {inv[1]};", cnx, index_col=['index'])

        y = data['arr_delay']
        X = data.iloc[:, 2:]

        # scale data
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X).astype(float))

        # load model
        if os.path.exists(filepath):
            model = pickle.load(open(filepath, 'rb'))

        # partial fit
        model.partial_fit(X_scaled, y)

        # save model
        save_model(model, filepath)

        count += 1
        print(f'partial-fit progress: {count//invs*100}% or {count}/{invs}')

    return f"model trained and saved to: {filepath}"


def get_weather_data(df, how):
    """merge weather data onto a SINGLE interval of flights data. Both origin, and destination weather.

    Args:
        df (DataFrame): flights data (inverval of total)
        how (str): {'left', 'inner'} How to merge weather (right df) on flights (left df)

    Returns:
        (DataFrame): resulting merged, weather & flights dataframe
    """

    # load weather_data
    weather_cnx = create_connection('data/weather2.sqlite')
    weather = pd.read_sql("SELECT * from weather_data",
                          weather_cnx, index_col='index')

    # merge ORIGIN weather_data onto flights_data interval (df)
    df = df.merge(weather, how=how, left_on=[
                  'fl_date', 'origin_city_name', 'dep_hr'], right_on=['date', 'city', 'time'])

    # rename weather cols with o_ prefix (for origin)
    cols = weather.columns.tolist()
    new_cols = weather[cols].add_prefix('o_').columns.tolist()
    d = {cols[i]: new_cols[i] for i in range(len(cols))}
    df = df.rename(columns=d)

    # merge DESTINATION weather_data onto flights_data interval (df)
    df = df.merge(weather, how=how, left_on=[
                  'fl_date', 'dest_city_name', 'arr_hr'], right_on=['date', 'city', 'time'])

    # rename weather cols with d_ prefix (for destination)
    new_cols = weather[cols].add_prefix('d_').columns.tolist()
    d = {cols[i]: new_cols[i] for i in range(len(cols))}
    df = df.rename(columns=d)

    # drop columns used for merge
    df = df.drop(columns=['origin_city_name', 'dest_city_name',
                 'o_date', 'o_city', 'o_time', 'd_date', 'd_city', 'd_time'])

    return df


def merge_weather(target, table_name, how, upload_mode):
    """merge weather data on ALL segmented flights data

    Args:
        target (str): {'training','test'}. Flight data for merge
        table_name (str): target table name to save to
        how (str): {'inner','left'}. How to merge weather (right) on flights (left) data
        upload_mode (str): {'interval','none'}. Upload via intervals or not

    Returns:
        msg (str): confirmation message, all merges complete.
    """

    import numpy as np
    import pandas as pd
    from functions import create_connection, send_data_to_db

    # import weather data
    weather_cnx = create_connection('data/weather2.sqlite')
    weather = pd.read_sql("SELECT * from weather_data",
                          weather_cnx, index_col='index')

    # connect to segmented flight data
    fl_cnx = create_connection('data/fl_merge.sqlite')

    # list table names for pull (depending on target arguement)
    if target == 'training':
        fl_tables = [f'flight_{i}' for i in np.arange(60)]
    elif target == 'test':
        fl_tables = [f'flight_test_{i}' for i in np.arange(60)]
    else:
        return "invalid target chosen"

    # incrementally merge weather data onto segmented flight data & save to DB
    count = 0
    for flights in fl_tables:
        flights = pd.read_sql(
            f"SELECT * FROM {flights};", fl_cnx, index_col=['index'])
        flights.index.name = None
        flights = get_weather_data(flights, how=how)
        send_data_to_db(flights, table_name=table_name,
                        db_path='data/fl_final.sqlite', if_exist='append', mode=upload_mode)

        count += 1
        print(f'progress: {count/60*100}% or {count}/60')

    return "COMPLETE!!!! WOOOHOOOOOOO!!!!"
