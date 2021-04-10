
#!/usr/bin/env python
# %%
import numpy as np
import pandas as pd
import os
from datetime import datetime
from functions import *

''' DATA CLEANING '''

# open DB connection
cnx = create_connection('data/weather.sqlite')
# %%

''' CLEAN ** WEATHER ** DATA & SEND TO DB ''' 

# read-in weather data
weather = pd.read_sql("SELECT * from weather_data", cnx, index_col='index')

# convert sunset/sunrise times to 24h integers
weather['sunset'] = weather.sunset.apply(lambda x: int(datetime.strptime(x,'%I:%M %p').strftime('%H%M')) if any(i in x for i in ['AM','PM']) else x)
weather['sunrise'] = weather.sunrise.apply(lambda x: int(datetime.strptime(x,'%I:%M %p').strftime('%H%M')) if any(i in x for i in ['AM','PM']) else x)

# convert "No sunrise" or "No sunset" values to median
rise_med = int(np.median(weather.sunrise[weather.sunrise.map(type) == int]))
set_med = int(np.median(weather.sunset[weather.sunset.map(type) == int]))

# convert remaining columns from objects to integers / floats
weather['sunset'] = weather.sunset.apply(lambda x: set_med if type(x) != int else int(x))
weather['sunrise'] = weather.sunrise.apply(lambda x: rise_med if type(x) != int else int(x))
weather.iloc[:,3:-4] = weather.iloc[:,3:-4].astype(int)
weather.iloc[:,-4:-2] = weather.iloc[:,-4:-2].astype(float)

# open connection to NEW database
cnx = create_connection('data/weather2.sqlite')

# export cleaned data to NEW database
weather.to_sql('weather_data', cnx, if_exists='replace')

# %%
''' CLEAN ** FLIGHTS_TRAINING ** DATA & SEND TO DB '''

import numpy as np
import pandas as pd
import re
pd.options.display.max_columns = 50

flights = pd.read_csv('data/flights_merge.csv', low_memory=False).drop(columns=['Unnamed: 0'])
# %%
len(flights)
# %%
flights.dtypes
# %%
obj = flights.dtypes[flights.dtypes == object].index
# %%
nunq = [(i, flights[i].nunique()) for i in obj]
# %%
na_cols = flights.isna().any()
# %%
na_cols = flights.columns[na_cols]
# %%
na_cols = flights[na_cols].isnull().sum()
# %%
na_cols
# %%
flights = flights.dropna()
# %%
flights.isna().any()
# %%
flights.to_csv('data/flights_merge.csv', index=False)
# %%


''' CLEAN ** FLIGHTS_TEST ** DATA & SEND TO DB '''
import numpy as np
import pandas as pd
import re
pd.options.display.max_columns = 50
# %%

flights_test = pd.read_csv('data/flights_test_merge.csv', low_memory=False)
# %%
len(flights_test)
# %%
flights_test.dtypes
# %%
obj = flights_test.dtypes[flights_test.dtypes == object].index
# %%
nunq = [(i, flights_test[i].nunique()) for i in obj]
# %%
na_cols = flights_test.isna().any()
# %%
na_cols = flights_test.columns[na_cols]
# %%
na_cols = flights_test[na_cols].isnull().sum()
# %%
na_cols
# %%
'''nothing to clean! :)'''
# %%

# %%
''' QUICK EXPLORATORY ANALYSIS '''
# %%
flights.head()
# %%
flights[~flights.arr_delay.isnull()].head(50)
# %%
flights.crs_elapsed_time.value_counts()
# %%
np.mean(flights.crs_elapsed_time)
# %%
np.median(flights.crs_elapsed_time)
# %%
delay_cols = na_cols[na_cols.index.str.contains('delay')]
# %%
delay_cols
# %%
flights[delay_cols.index].info(verbose=True)
# %%
flights[delay_cols.index].describe(include='all')
# %%
flights[delay_cols.index].head()
# %%
medians = [delay_cols.index, np.median(i) for i in flights[delay_cols.index]]
# %%
null_cols_counts = null_counts[null_counts != 0]
# %%
null_cols = [flights.isnull().sum() != 0]
# %%
null_cols


''' CREATE FLIGHT INTERVALS: FOR MERGING WITH WEATHER DATA & SEND TO DB '''
# %%
%reset -f
from functions import *
import numpy as np
import pandas as pd

# %%
''' ** training data ** '''

flights = pd.read_csv('data/flights_merge.csv', low_memory=False)
# %%
flights.head()
# %%
intervals = interval_calc(0, len(flights), 60)
# %%
intervals
# %%
flight_invs = [flights.iloc[iv[0]:iv[1],:] for iv in intervals]
# %%
fl_cnx = create_connection('data/fl_merge.sqlite')
# %%
[flight_invs[inv].to_sql(f'flight_{inv}', fl_cnx, if_exists='replace') for inv in range(len(flight_invs))]
# %%
del flights
del flight_invs

#%%
''' ** test data ** '''

flights_test = pd.read_csv('data/flights_test_merge.csv', low_memory=False)
# %%
flights_test.head()
# %%
intervals = interval_calc(0, len(flights_test), 60)
# %%
intervals
# %%
flight_invs = [flights_test.iloc[iv[0]:iv[1],:] for iv in intervals]
# %%
fl_cnx = create_connection('data/fl_merge.sqlite')
# %%
[flight_invs[inv].to_sql(f'flight_test_{inv}', fl_cnx, if_exists='replace') for inv in range(len(flight_invs))]
# %%
del flight_invs
# %%

''' CLEAN WEATHER_DATA & SEND TO DB '''

weather_cnx = create_connection('data/weather2.sqlite')
weather = pd.read_sql("SELECT * from weather_data", weather_cnx, index_col='index')
# %%
weather.head()
# %%
[print(i) for i in weather.dtypes.index.sort_values(key=lambda col: col.str.lower())]
print(f'\nfl row count: {len(weather)}')
print(f'\nfl column count: {len(weather.columns)}')
# %%
flights_test.head()
# %%
# city names in flights_data
fl_cities=['Adak Island',
'Saginaw/Bay City/Midland',
'Allentown/Bethlehem/Easton',
'Bloomington/Normal',
'Sarasota/Bradenton',
'Branson',
'Bullhead City',
'Jacksonville/Camp Lejeune',
'Guam',
'Cedar Rapids/Iowa City',
'Champaign/Urbana',
'Clarksburg/Fairmont',
'College Station/Bryan',
'North Bend/Coos Bay',
'Elmira/Corning',
'Ithaca/Cortland',
'Montrose/Delta',
'Dillingham',
'Charleston/Dunbar',
'Arcata/Eureka',
'Lawton/Fort Sill',
'Dallas/Fort Worth',
'Gulfport/Biloxi',
'Gustavus',
'Hancock/Houghton',
'Hattiesburg/Laurel',
'Greensboro/High Point',
'Hilton Head',
'Hyannis',
'Iron Mountain/Kingsfd',
'Bristol/Johnson City/Kingsport',
'Kona',
'Pasco/Kennewick/Richland',
'King Salmon',
'Lanai',
'Bismarck/Mandan',
'Manhattan/Ft. Riley',
'Mission/McAllen/Edinburg',
'Nantucket',
'New Bern/Morehead/Beaufort',
'Newport News/Williamsburg',
'Midland/Odessa',
'West Palm Beach/Palm Beach',
'Beaumont/Port Arthur',
'Newburgh/Poughkeepsie',
'Presque Isle/Houlton',
'Raleigh/Durham',
'Bend/Redmond',
'Rota',
'St. Cloud',
'Harlingen/San Benito',
'St. Louis',
'Sun Valley/Hailey/Ketchum',
'Unalaska',
'Jackson/Vicksburg',
"Martha's Vineyard",
'West Yellowstone',
'Scranton/Wilkes-Barre',
'Youngstown/Warren']
# %%
# replace city names in weather data with city names in FLIGHTS data
uniq_w_cities = set(weather.city.unique()) # set of uniq_weather cities
uniq_f_cities = set(flights_test.dest_city_name.unique()) # set of uniq_flight cities
wea_cities = sorted(list(uniq_w_cities.difference(uniq_f_cities)))
# %%
# view if any cities are missing after changing above names
set(flights_test.origin_city_name.unique()).difference(set(weather.city.unique()))
# missing cities == {'Riverton/Lander', 'Sheridan', 'St. Petersburg'}
# %%
weather.city.replace(wea_cities,fl_cities)
# %%
weather.isna().sum()
# %%
weather[weather.duplicated()]
# %%
weather = weather.drop_duplicates()
# %%
send_data_to_db(weather, 'weather_data', 'data/weather2.sqlite', if_exist='replace')
# %%

''' MERGE WEATHER AND FLIGHTS DATA '''

'''flights['fl_date', 'origin_city_name', 'dest_city_name', 'dep_hr', 'arr_hr'] ON'''
'''weather['date', 'city', 'time']'''
# %%
%reset -f
# %%
from functions import *
import numpy as np
import pandas as pd
# %%
''' TEST FUNCTION: for merging weather data on SINGLE interval of flights data'''
# %%
fl_cnx = create_connection('data/fl_merge.sqlite')
fl_1 = pd.read_sql("SELECT * FROM flight_1;", fl_cnx, index_col=['index'])
fl_1.index.name = None
# %%
weather_cnx = create_connection('data/weather2.sqlite')
weather = pd.read_sql("SELECT * from weather_data", weather_cnx, index_col='index')
# %%
fl_1.head()
# %%
weather.head()
# %%
# examine data
print(fl_1.dtypes)
print(f'\nfl row count: {len(fl_1)}')
print(f'\nfl column count: {len(fl_1.columns)}\n')
print(weather.dtypes)
print(f'\nfl row count: {len(weather)}')
print(f'\nfl column count: {len(weather.columns)}')
##################################################################################
# %%
''' compare columns (used for merging) prior to testing ''' 
# %%
fl_1[['fl_date', 'origin_city_name', 'dest_city_name', 'dep_hr', 'arr_hr']].dtypes
# %%
weather[['date', 'city', 'time']].head()
# %%
print(set(fl_1['fl_date'].unique()).difference(set(weather['date'].unique())))
print(set(fl_1['origin_city_name'].unique()).difference(set(weather['city'].unique())))
print(set(fl_1['dest_city_name'].unique()).difference(set(weather['city'].unique())))
print(set(fl_1['dep_hr'].unique()).difference(set(weather['time'].unique())))
print(set(fl_1['arr_hr'].unique()).difference(set(weather['time'].unique())))
# %%

''' ** testing function ** '''

fl_1w = get_weather_data(fl_1)
# %%
pd.options.display.max_rows = None
pd.options.display.max_columns = None
# %%
fl_1w.head()
# %%
fl_1w.dtypes

# %%
''' MERGE WEATHER_DATA on ALL intervals of FLIGHTS_DATA & SEND TO DB'''
############################ function to join & send to db ######################
# %%
%reset -f
# %%
from functions import *
import numpy as np
import pandas as pd

# %%
# function moved to functions.py
merge_weather('training', 'flights_final', 'inner', 'None')
# %%
merge_weather('test', 'flights_final_test', 'left', 'None')

# %%