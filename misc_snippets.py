
#!/usr/bin/env python
# %%

''' MISC: WEATHER & FLIGHTS MERGING (notes & code snippets) '''
# %%
from functions import *
import re
from sqlite3 import Error
import sqlite3
from IPython.display import JSON  # JSON(response.json())
import json
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
# %%
fl_1 = fl_1.merge(weather, how='inner', left_on='mkt_carrier_fl_num',
                  right_on='mkt_carrier_fl_num')  # rsuffix='_copy')
# %%
fl_1.columns
# %%
fl_1[['mkt_carrier_fl_num', 'mkt_carrier_fl_num_copy']].head()
# %%
# read-in flights data
flights.drop(columns='Unnamed: 0', inplace=True)
# %%
# reset origin & dest cols as city name (without state)
flights['origin'] = flights['origin_city_name'].apply(
    lambda x: x.split(', ')[0])
flights['dest'] = flights['dest_city_name'].apply(lambda x: x.split(', ')[0])
# %%
# round up times to nearest hour
flights['crs_dep_time'] = np.round(flights['crs_dep_time'], -2)
flights['crs_arr_time'] = np.round(flights['crs_arr_time'], -2)
# %%
# create subset of flights - based on cities we have weather reports for
cities = df.value_counts('city').index.tolist()
sub = flights[flights['origin'].isin(cities)]
len(sub)
# %%
# check head of subset from flights (columns for joining on)
sub[['origin', 'dest', 'crs_dep_time', 'crs_arr_time', 'fl_date']].head()
# %%
# convert time column in weather data to int64 (to match type in flights data)
df['time'] = df['time'].astype('int64')
# %%
# merge weather data into flights data on ORIGINS
sub = pd.merge(sub, df, how='left', left_on=['origin', 'crs_dep_time', 'fl_date'], right_on=['city', 'time', 'date'])
# %%
# rename merged columns with "o_" prefix (for origin_{weather data})
new_cols = sub[cols].add_prefix('o_').columns.tolist()
d = {cols[i]: new_cols[i] for i in range(len(cols))}
sub.rename(columns=d, inplace=True)
# %%
# merge weather data into flights data on DESTINATIONS
sub = pd.merge(sub, df, how='left', left_on=['dest', 'crs_arr_time', 'fl_date'], right_on=['city', 'time', 'date'])
# %%
# rename merged columns with "d_" prefix (for destination_{weather data})
cols = df.columns.tolist()
new_cols = sub[cols].add_prefix('d_').columns.tolist()
d = {cols[i]: new_cols[i] for i in range(len(cols))}
sub.rename(columns=d, inplace=True)
# %%
sub.head()

############################ function to join & send to db ######################
send_data_to_db(flights, flights_final,
                'data/fl_final.sqlite', if_exist='append')
fl_cnx = create_connection('data/fl_merge.sqlite')

fl_tables = [f'flight_{i}' for i in np.arange(60)]
fl_tables

for flights in fl_tables:
    flights = pd.read_sql(
        f"SELECT * FROM {flights};", fl_cnx, index_col=['index'])
    flights.index.name = None
    flights = flights.reset_index().set_index('mkt_carrier_fl_num')
    flights = flights.join(weather, how='left')
    flights = flights.reset_index().set_index('index')
    flights.index.name = None
    send_data_to_db(flights, 'flights_final',
                    db_path='data/fl_final.sqlite')

# %%
''' MISC: Creating Functions, request_weather & add_jsons_to_db '''

# create & connect to weather database

# Set columns for table
cols = ['city', 'date', 'time', 'tempC', 'windspeedKmph', 'winddirDegree', 'WindGustKmph', 'visibility',
        'visibilityMiles', 'cloudcover', 'pressure', 'humidity', 'precipMM', 'totalSnow_cm', 'sunrise', 'sunset']

# instantiate DataFrame to build on
df = pd.DataFrame(data=None, columns=cols)


# get all unique city names
origins = pd.read_csv('data/origin_cities.csv')
origins = origins.origin_city_name.tolist()


path = ['data', 'weather', 'hourly']
meta = [['data', 'weather', 'date'],
        ['data', 'weather', 'totalSnow_cm'],
        ['data', 'weather', 'astronomy'],
        ['data', 'request']]

# create dataframe
for idx in range(len(jsons)):
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
    data = data[cols]

    df = pd.concat([df, data], axis=0)
    del data


df.head()

# convert time to int64 (to match flights data)
df['time'] = df['time'].astype('int64')

df.to_sql('weather_data', cnx, if_exists='append')

# function for normalizing JSONs and concatenating to df


def construct_df(jsons):

  path = ['data', 'weather', 'hourly']
  meta = [['data', 'weather', 'date'],
          ['data', 'weather', 'totalSnow_cm'],
          ['data', 'weather', 'astronomy'],
          ['data', 'request']]

  for idx in range(len(jsons)):
    try:
      if list(jsons[idx]['data'].keys())[0] == 'error':
        del jsons[idx]
    except Exception:
      pass
    else:
      data = pd.json_normalize(
          data=jsons[idx], record_path=path, meta=meta, errors='ignore')
      data = data.rename(columns={'data.weather.astronomy': 'sun',
                                  'data.weather.totalSnow_cm': 'totalSnow_cm',
                                  'data.weather.date': 'date',
                                  'data.request': 'city'})
      data['city'] = data.city.apply(lambda x: x['query'].split(', ')[0])
      data['sunrise'] = data.sun.apply(lambda x: x['sunrise'])
      data['sunset'] = data.sun.apply(lambda x: x['sunset'])
      data['sunrise'] = pd.to_datetime(
          data['sunrise'], format='%H:%M %p').dt.time
      data['sunset'] = pd.to_datetime(
          data['sunrise'], format='%H:%M %p').dt.time
      data = data.drop(columns='sun')
      data = data[cols]

      df = pd.concat([df, data], axis=0)

  return df


construct_df(jsons=jsons)

# %%
''' MISC: DATA EXPLORATORY ANALYSIS '''

%reset - f
# %%

# %%
cnx = create_connection('data/fl_final.sqlite')
fl_final = pd.read_sql('SELECT * FROM flights_final', cnx, index_col=['index'])
# %%
fl = pd.read_csv('data/flights.csv', low_memory=False)
# %%
len(fl)
# %%
fl.dtypes
# %%
fl = fl.drop(columns=['Unnamed: 0', 'no_name', ])
# %%
obj = fl.dtypes[fl.dtypes == object].index
# %%
nunq = [(i, fl[i].nunique()) for i in obj]
# %%
na_cols = fl.isna().any()
# %%
na_cols = fl.columns[na_cols]
# %%
na_cols = fl[na_cols].isnull().sum()
# %%
na_cols
# %%
delay_cols = na_cols[na_cols.index.str.contains('delay')]
# %%
delay_cols
# %%
fl[delay_cols.index].info(verbose=True)
# %%
fl[delay_cols.index].describe(include='all')
# %%
fl[delay_cols.index].head()
# %%
medians = [delay_cols.index, np.median(i) for i in fl[delay_cols.index]]
# %%
# %%
null_cols_counts = null_counts[null_counts != 0]
# %%
null_cols = [fl.isnull().sum() != 0]
# %%
null_cols
# %%
null_counts = fl.isnull().sum()
# %%
null_cols_counts = null_counts[null_counts != 0]
null_cols = null_cols_counts.index.tolist()
# %%
null_cols
# %%
obj = fl.dtypes[fl.dtypes == object].index
# %%
obj_0 = [i for i in obj if (i in null_cols)]
# %%
obj_vals = [fl[i].value_counts() for i in obj]
# %%
obj_vals = {obj[i]: obj_vals[i] for i in range(len(obj))}
# %%
obj_vals[['tail_num', 'cancellation_code']]
# %%
obj_0 = fl[obj][fl[obj].isnull().sum() != 0].index
# %%
obj_0
# %%


''' MISC: DIRTY DATA CLEANING - merged flights & weather data '''
# %%
# %%
df = pd.read_csv('data/flight_ready.csv', low_memory=False)
df.drop(columns='Unnamed: 0', inplace=True)
df.dropna(how='any', axis=0, inplace=True)  # drop all rows with na

# %%
obj = df.dtypes[df.dtypes == object].index.tolist()
# %%
obj
# %%
idxs = [pd.to_numeric(df[col], errors='coerce').isna() for col in obj]
# %%
idx_nums = [df[i].index for i in idxs]
# %%
idx_nums
# %%
del idx_nums[1]
# %%
idx_n = [i[0] for i in idx_nums]
# %%
obj2 = obj.copy()
# %%
del obj2[1]
# %%
tups = zip(obj2, idx_n)
# %%
tups = [i for i in tups]
del obj2
# %%
tups
# %%
idx_n = list(set(idx_n))
# %%
len(df)
# %%
df = df.drop(idx_n)
# %%
len(df)
# %%
errors = []
for i in obj:
    try:
        df[i] = df[i].astype(float)
    except Exception as e:
        errors.append(e)
        pass
# %%
df[['precipMM_origin', 'totalSnow_cm_dest']].dtypes
# %%
obj2 = obj.copy()
# %%
obj2
# %%
del obj2[-2]
# %%
df[obj2].head()
# %%
errors = []
for i in obj2:
    try:
        df[i] = df[i].astype(int)
    except Exception as e:
        errors.append(e)
        pass
# %%
[str(i).split(' ')[-1] for i in errors]
# %%
df['humidity_origin'] = df.humidity_origin.astype(float)
# %%
tups = [i for i in zip(obj2, ['0.0', 'York', '1\\x1014', 'May'])]
# %%
ind = []
ind.append(df[df[tups[1][0]] == 'New York'].index[0])
# %%
[print(i) for i in errors]
# %%
ind.append(df[df['pressure_dest'] == r"1\x1014"].index[0])
# %%
df['pressure_dest'].apply(lambda x: int(x) if str(x).isdigit() else None)
# %%
df['pressure_dest'].apply(type).value_counts()
# %%
obj = df.dtypes[df.dtypes == object].index.tolist()
# %%
tups = [i for i in zip(obj, errors)]
# %%
tups
# %%
df['sunset_origin'] = pd.to_numeric(df['sunset_origin'], errors='coerce')
# %%
df['pressure_dest'] = pd.to_numeric(df['pressure_dest'], errors='coerce')
# %%
df['totalSnow_cm_dest'] = pd.to_numeric(
    df['totalSnow_cm_dest'], errors='coerce')
# %%
df['sunset_dest'] = pd.to_numeric(df['sunset_dest'], errors='coerce')
# %%
len(df)
# %%
df = df.dropna()
# %%
len(df)
# %%
obj
# %%
errors = []
for i in obj:
    try:
        df[i] = df[i].astype(float)
    except Exception as e:
        errors.append(e)
        pass
# %%
df.dtypes
# %%
# DID ITTTTTTTTTTTTT !!!!!
# %%
df.to_csv('data/flight_almost.csv', index=False)

# %%

''' MISC: DIRTY DATA CLEANING - merged flights & weather data '''
# %%
# %%
df = pd.read_csv('data/flight_ready.csv', low_memory=False)
df.drop(columns='Unnamed: 0', inplace=True)
df.dropna(how='any', axis=0, inplace=True)  # drop all rows with na

# %%
obj = df.dtypes[df.dtypes == object].index.tolist()
# %%
obj
# %%
idxs = [pd.to_numeric(df[col], errors='coerce').isna() for col in obj]
# %%
idx_nums = [df[i].index for i in idxs]
# %%
idx_nums
# %%
del idx_nums[1]
# %%
idx_n = [i[0] for i in idx_nums]
# %%
obj2 = obj.copy()
# %%
del obj2[1]
# %%
tups = zip(obj2, idx_n)
# %%
tups = [i for i in tups]
del obj2
# %%
tups
# %%
idx_n = list(set(idx_n))
# %%
len(df)
# %%
df = df.drop(idx_n)
# %%
len(df)
# %%
errors = []
for i in obj:
    try:
        df[i] = df[i].astype(float)
    except Exception as e:
        errors.append(e)
        pass
# %%
df[['precipMM_origin', 'totalSnow_cm_dest']].dtypes
# %%
obj2 = obj.copy()
# %%
obj2
# %%
del obj2[-2]
# %%
df[obj2].head()
# %%
errors = []
for i in obj2:
    try:
        df[i] = df[i].astype(int)
    except Exception as e:
        errors.append(e)
        pass
# %%
[str(i).split(' ')[-1] for i in errors]
# %%
df['humidity_origin'] = df.humidity_origin.astype(float)
# %%
tups = [i for i in zip(obj2, ['0.0', 'York', '1\\x1014', 'May'])]
# %%
ind = []
ind.append(df[df[tups[1][0]] == 'New York'].index[0])
# %%
[print(i) for i in errors]
# %%
ind.append(df[df['pressure_dest'] == r"1\x1014"].index[0])
# %%
df['pressure_dest'].apply(lambda x: int(x) if str(x).isdigit() else None)
# %%
df['pressure_dest'].apply(type).value_counts()
# %%
obj = df.dtypes[df.dtypes == object].index.tolist()
# %%
tups = [i for i in zip(obj, errors)]
# %%
tups
# %%
df['sunset_origin'] = pd.to_numeric(df['sunset_origin'], errors='coerce')
# %%
df['pressure_dest'] = pd.to_numeric(df['pressure_dest'], errors='coerce')
# %%
df['totalSnow_cm_dest'] = pd.to_numeric(
    df['totalSnow_cm_dest'], errors='coerce')
# %%
df['sunset_dest'] = pd.to_numeric(df['sunset_dest'], errors='coerce')
# %%
len(df)
# %%
df = df.dropna()
# %%
len(df)
# %%
obj
# %%
errors = []
for i in obj:
    try:
        df[i] = df[i].astype(float)
    except Exception as e:
        errors.append(e)
        pass
# %%
df.dtypes
# %%
# DID ITTTTTTTTTTTTT !!!!!
# %%
df.to_csv('data/flight_almost.csv', index=False)


# %%

''' MISC: WEATHER & FLIGHTS MERGING (notes & code snippets) '''
# %%
# %%
fl_1 = fl_1.merge(weather, how='inner', left_on='mkt_carrier_fl_num',
                  right_on='mkt_carrier_fl_num')  # rsuffix='_copy')
# %%
fl_1.columns
# %%
fl_1[['mkt_carrier_fl_num', 'mkt_carrier_fl_num_copy']].head()
# %%
# read-in flights data
flights.drop(columns='Unnamed: 0', inplace=True)
# %%
# reset origin & dest cols as city name (without state)
flights['origin'] = flights['origin_city_name'].apply(
    lambda x: x.split(', ')[0])
flights['dest'] = flights['dest_city_name'].apply(lambda x: x.split(', ')[0])
# %%
# round up times to nearest hour
flights['crs_dep_time'] = np.round(flights['crs_dep_time'], -2)
flights['crs_arr_time'] = np.round(flights['crs_arr_time'], -2)
# %%
# create subset of flights - based on cities we have weather reports for
cities = df.value_counts('city').index.tolist()
sub = flights[flights['origin'].isin(cities)]
len(sub)
# %%
# check head of subset from flights (columns for joining on)
sub[['origin', 'dest', 'crs_dep_time', 'crs_arr_time', 'fl_date']].head()
# %%
# convert time column in weather data to int64 (to match type in flights data)
df['time'] = df['time'].astype('int64')
# %%
# merge weather data into flights data on ORIGINS
sub = pd.merge(sub, df, how='left', left_on=['origin', 'crs_dep_time', 'fl_date'], right_on=['city', 'time', 'date'])
# %%
# rename merged columns with "o_" prefix (for origin_{weather data})
new_cols = sub[cols].add_prefix('o_').columns.tolist()
d = {cols[i]: new_cols[i] for i in range(len(cols))}
sub.rename(columns=d, inplace=True)
# %%
# merge weather data into flights data on DESTINATIONS
sub = pd.merge(sub, df, how='left', left_on=['dest', 'crs_arr_time', 'fl_date'], right_on=['city', 'time', 'date'])
# %%
# rename merged columns with "d_" prefix (for destination_{weather data})
cols = df.columns.tolist()
new_cols = sub[cols].add_prefix('d_').columns.tolist()
d = {cols[i]: new_cols[i] for i in range(len(cols))}
sub.rename(columns=d, inplace=True)
# %%
sub.head()

############################ function to join & send to db ######################
send_data_to_db(flights, flights_final,
                'data/fl_final.sqlite', if_exist='append')
fl_cnx = create_connection('data/fl_merge.sqlite')

fl_tables = [f'flight_{i}' for i in np.arange(60)]
fl_tables

for flights in fl_tables:
    flights = pd.read_sql(
        f"SELECT * FROM {flights};", fl_cnx, index_col=['index'])
    flights.index.name = None
    flights = flights.reset_index().set_index('mkt_carrier_fl_num')
    flights = flights.join(weather, how='left')
    flights = flights.reset_index().set_index('index')
    flights.index.name = None
    send_data_to_db(flights, 'flights_final',
                    db_path='data/fl_final.sqlite')
