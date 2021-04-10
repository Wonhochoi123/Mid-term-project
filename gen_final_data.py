
# %% 
%reset -f
from functions import *
import numpy as np
import pandas as pd

# %%
''' CREATE FINAL DATASETS FOR ML '''

# %%

''' ** training datasets ** '''
# %%
cnx = create_connection('data/fl_final.sqlite')
df = pd.read_sql("SELECT [index], fl_date, arr_delay, Jan, [crs_dep_time(mins)], [crs_arr_time(mins)], o_visibility, o_precipMM, o_totalSnow_cm, d_visibility, d_cloudcover, d_pressure, d_humidity, d_precipMM FROM flights_final;", cnx, index_col=['index'])
# %%
df.to_csv('data/flights_10.csv', index=False) # top 10 fts ALL data

# %%
cnx = create_connection('data/fl_final.sqlite')
df = pd.read_sql("SELECT [index], fl_date, arr_delay, [crs_dep_time(mins)], [crs_arr_time(mins)], crs_elapsed_time, origin_traffic, dest_traffic, origin_taxi, dest_taxi, WN, DL, Monday, Thursday, Saturday, Friday, o_tempC, o_windspeedKmph, o_winddirDegree, o_visibility, o_cloudcover, o_humidity, o_precipMM, o_totalSnow_cm, o_sunrise, o_sunset, d_windspeedKmph, d_visibility, d_cloudcover, d_pressure, d_humidity, d_precipMM, d_totalSnow_cm FROM flights_final WHERE Jan = 1;", cnx, index_col=['index'])
# %%
df.to_csv('data/flights_30J.csv', index=False) # top 30 fts ONLY Jan

# %%
cnx = create_connection('data/fl_final.sqlite')
df = pd.read_sql("SELECT [index], fl_date, arr_delay, [crs_dep_time(mins)], [crs_arr_time(mins)], origin_taxi, DL, Saturday, o_windspeedKmph, o_visibility, o_cloudcover, o_humidity, o_precipMM, o_totalSnow_cm, o_sunrise, o_sunset, d_windspeedKmph, d_visibility, d_cloudcover, d_pressure, d_humidity, d_precipMM, d_totalSnow_cm FROM flights_final;", cnx, index_col=['index'])
# %%
df.to_csv('data/flights_20J.csv', index=False) # top 20 fts ONLY Jan

# %%
cnx = create_connection('data/fl_final.sqlite')
df = pd.read_sql("SELECT [index], fl_date, arr_delay, [crs_dep_time(mins)], [crs_arr_time(mins)], o_visibility, o_precipMM, o_totalSnow_cm, d_visibility, d_cloudcover, d_pressure, d_humidity, d_precipMM FROM flights_final WHERE Jan = 1;", cnx, index_col=['index'])
# %%
df.to_csv('data/flights_10J.csv', index=False) # top 10 fts ONLY Jan
# %%



''' ** test datasets ** '''

cnx = create_connection('data/fl_final.sqlite')
df = pd.read_sql("SELECT [index], fl_date, mkt_carrier, mkt_carrier_fl_num, origin, dest, [crs_dep_time(mins)], [crs_arr_time(mins)], crs_elapsed_time, origin_traffic, dest_traffic, origin_taxi, dest_taxi, WN, DL, Monday, Thursday, Saturday, Friday, o_tempC, o_windspeedKmph, o_winddirDegree, o_visibility, o_cloudcover, o_humidity, o_precipMM, o_totalSnow_cm, o_sunrise, o_sunset, d_windspeedKmph, d_visibility, d_cloudcover, d_pressure, d_humidity, d_precipMM, d_totalSnow_cm FROM flights_final_test;", cnx, index_col=['index'])
# %%
df.to_csv('data/flights_30J_test.csv', index=False) # top 30 fts ONLY Jan
# %%
cnx = create_connection('data/fl_final.sqlite')
df = pd.read_sql("SELECT [index], fl_date, mkt_carrier, mkt_carrier_fl_num, origin, dest, [crs_dep_time(mins)], [crs_arr_time(mins)], origin_taxi, DL, Saturday, o_windspeedKmph, o_visibility, o_cloudcover, o_humidity, o_precipMM, o_totalSnow_cm, o_sunrise, o_sunset, d_windspeedKmph, d_visibility, d_cloudcover, d_pressure, d_humidity, d_precipMM, d_totalSnow_cm FROM flights_final_test;", cnx, index_col=['index'])
# %%
df.to_csv('data/flights_20J_test.csv', index=False) # top 20 fts ONLY Jan
# %%
cnx = create_connection('data/fl_final.sqlite')
df = pd.read_sql("SELECT [index], fl_date, mkt_carrier, mkt_carrier_fl_num, origin, dest, [crs_dep_time(mins)], [crs_arr_time(mins)], o_visibility, o_precipMM, o_totalSnow_cm, d_visibility, d_cloudcover, d_pressure, d_humidity, d_precipMM FROM flights_final_test;", cnx, index_col=['index'])
# %%
df.to_csv('data/flights_10J_test.csv', index=False) # top 10 fts ONLY Jan
# %%



''' EXAMINE: Flights_Test Data for NA values '''
# %%
df = pd.read_csv('data/flights_30J_test.csv', index=False) # top 30 fts ONLY Jan
# %%
df.dtypes # all feature columns are numerical
# %%
df.isna().sum() # 71463 null vals in weather cols
# %%
len(df) # 793918 # total row count
# %%
71463/793918*100 # 9% of data is missing weather details 
# %%
del df