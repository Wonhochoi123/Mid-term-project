#!/usr/bin/env python

''' REQUEST WEATHER JSONS: for every city in dataset on the hour 2018-2020 '''

# import modules
from data.keys import weather_keys
import calendar
import numpy as np
import pandas as pd
import requests
import os
import json
from IPython.display import JSON  # JSON(response.json())
from functions import interval_calc, request_weather, add_jsons_to_db

# List of all City Names in Dataset
origins = pd.read_csv('data/origin_cities.csv')
origins = origins.origin_city_name.tolist()


# List [first day, last day] for each month (from 2018 to 02-2020)
month_days_2018 = [[f'2018-{month}-01',
                    f'2018-{month}-{[days for days in calendar.monthrange(2018, month)][1]}'] for month in np.linspace(1, 12, 12, dtype=int)]

month_days_2019 = [[f'2019-{month}-01',
                    f'2019-{month}-{[days for days in calendar.monthrange(2019, month)][1]}'] for month in np.linspace(1, 12, 12, dtype=int)]

month_days_2020 = [[f'2020-{month}-01',
                    f'2020-{month}-{[days for days in calendar.monthrange(2020, month)][1]}'] for month in np.linspace(1, 2, 2, dtype=int)]


# compile dates into list
month_days = month_days_2018 + month_days_2019 + month_days_2020

# list of lists [city, first day, last day] for every month of the year
querys = []

for city in origins:
    for month in month_days:
        querys.append([city, month[0], month[1]])


# set GET request parameters
url = 'https://api.worldweatheronline.com/premium/v1/past-weather.ashx'

# calc intervals for running GET requests
all_intervals = interval_calc(0, len(querys), intervals=10, roundup=False)

# run all GET requests & save to file
for interval in all_intervals:
    request_weather(url=url, querys=querys,
                    keys=weather_keys, interval=interval)

# JSON files
json_files = ["data/weather_1000_f.txt",
              "data/weather_1000-2000_f.txt",
              "data/weather_2000-4000_f.txt", 
              "data/weather_4000-5156_f.txt", 
              "data/weather_2000_2018_f.txt",
              "data/weather_2000-4416_2018_f.txt"]

# for each JSON file append to database
for file in json_files:
    jsons = json.load(open(file, 'r'))
    add_jsons_to_db(jsons=jsons)
    del jsons
    print(f'added file to database: {file}')
