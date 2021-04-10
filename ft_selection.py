# %%
# import functions
from sklearn.feature_selection import VarianceThreshold, f_regression, SelectKBest
import pandas as pd
import numpy as np
from functions import *
%reset - f
# %%
''' FEATURE SELECTION: from random sampling of 7-mill rows (out of ~16 mill) '''
# %%
# 7 mill rows is the max that would fit in RAM memory.
# TOP 30 fts, random 7,000,000 rows from final flights data.

cnx = create_connection('data/fl_final.sqlite')
df = pd.read_sql("SELECT [index], arr_delay, [crs_dep_time(mins)], [crs_arr_time(mins)], crs_elapsed_time, origin_traffic, dest_traffic, origin_taxi, dest_taxi, WN, DL, Monday, Thursday, Saturday, Friday, o_tempC, o_windspeedKmph, o_winddirDegree, o_visibility, o_cloudcover, o_humidity, o_precipMM, o_totalSnow_cm, o_sunrise, o_sunset, d_windspeedKmph, d_visibility, d_cloudcover, d_pressure, d_humidity, d_precipMM, d_totalSnow_cm FROM flights_final ORDER BY RANDOM() LIMIT 7000000;", cnx, index_col=['index'])
# %%
df.to_csv('data/flights_f_half.csv', index=False)
# %%
y = df['arr_delay']  # save-out y_test
df = df.iloc[:, 1:]  # df without arr_delay (y)
# %%
df.dtypes
# %%
# calc intervals so RAM can handle ft selection
inv = interval_calc(0, len(df.columns), 6)
# %%
print(inv)
# %%
df_invs = [df.iloc[:, iv[0]:iv[1]] for iv in inv]  # section cols for df into 6 sets
# %%
''' (1) REMOVE UNINFORMATIVE COLUMNS (if very small variance between observations) '''
# instantiate VarianceThresh, save select_cols to variable
select_cols = []

for interval in df_invs:
    vt = VarianceThreshold(0.1)
    df_transformed = vt.fit_transform(interval)

    # get columns with higher variance
    cols = interval.columns[vt.get_support()].tolist()

    [select_cols.append(i) for i in cols]
# %%
print(len(select_cols))
print(select_cols)
# %%
# subset dataframe with select_cols
df = pd.DataFrame(df, columns=select_cols)
# %%
above_thresh_42: ['crs_dep_time(mins)', 'crs_arr_time(mins)', 'crs_elapsed_time', 'distance', 'origin_traffic', 'dest_traffic', 'origin_taxi', 'dest_taxi', 'WN', 'DL', 'AA', 'Monday', 'Tuesday', 'Thursday', 'Saturday', 'Wednesday', 'Friday', 'Sunday', 'arr_hr', 'o_tempC', 'o_windspeedKmph', 'o_winddirDegree','o_WindGustKmph', 'o_visibility', 'o_visibilityMiles', 'o_cloudcover', 'o_humidity', 'o_precipMM', 'o_totalSnow_cm', 'o_sunrise', 'o_sunset', 'd_tempC', 'd_windspeedKmph', 'd_winddirDegree', 'd_visibility', 'd_visibilityMiles', 'd_cloudcover', 'd_pressure', 'd_humidity', 'd_precipMM', 'd_totalSnow_cm', 'd_sunrise']
# %%
del df_transformed  # save some RAM


# %%
''' (2) REMOVE HIGHLY CORRELATIVE COLUMNS (keep one of two cols that strongly correlate) '''
df_corr = df.corr().abs()

indices = np.where(df_corr > 0.8)  # thresh 0.8
indices = [(df_corr.index[x], df_corr.columns[y]) for x, y in zip(*indices) if x != y and x < y]

# %%
# View correlated indices
print(indices)

corr_indices = [('crs_arr_time(mins)', 'arr_hr'), ('crs_elapsed_time', 'distance'), ('o_windspeedKmph', 'o_WindGustKmph'), ('o_visibility', 'o_visibilityMiles'), ('o_sunrise', 'd_sunrise'), ('d_visibility', 'd_visibilityMiles')]
# %%
# Remove 1 of 2 paired cols (that strongly correlate > 0.8 pearsons r^2)
for idx in indices:  # each pair
    try:
        df.drop(idx[1], axis=1, inplace=True)
    except KeyError:
        pass
# %%
print(len(df.columns))
print(df.columns)
# %%
top_36 = ['crs_dep_time(mins)', 'crs_arr_time(mins)', 'crs_elapsed_time',  'origin_traffic', 'dest_traffic', 'origin_taxi', 'dest_taxi', 'WN', 'DL', 'AA', 'Monday', 'Tuesday', 'Thursday', 'Saturday', 'Wednesday', 'Friday', 'Sunday', 'o_tempC', 'o_windspeedKmph', 'o_winddirDegree',       'o_visibility', 'o_cloudcover', 'o_humidity', 'o_precipMM', 'o_totalSnow_cm', 'o_sunrise', 'o_sunset', 'd_tempC', 'd_windspeedKmph', 'd_winddirDegree', 'd_visibility', 'd_cloudcover', 'd_pressure', 'd_humidity', 'd_precipMM', 'd_totalSnow_cm']


# %%
''' (3) FORWARD REGRESSION (find k-best features for predicting target (forward wrapper)) '''
skb = SelectKBest(f_regression, k=10)
X = skb.fit_transform(df, y)

# %%
cols = df.columns[skb.get_support()].tolist()
print(len(cols))
print(cols)
# %%
cols
# %%
top_10 = ['crs_dep_time(mins)', 'crs_arr_time(mins)', 'o_visibility', 'o_precipMM','o_totalSnow_cm', 'd_visibility', 'd_cloudcover', 'd_pressure', 'd_humidity', 'd_precipMM']

top_20 = ['crs_dep_time(mins)', 'crs_arr_time(mins)', 'origin_taxi', 'DL', 'Saturday','o_windspeedKmph', 'o_visibility', 'o_cloudcover', 'o_humidity', 'o_precipMM','o_totalSnow_cm', 'o_sunrise', 'o_sunset', 'd_windspeedKmph', 'd_visibility', 'd_cloudcover', 'd_pressure', 'd_humidity', 'd_precipMM', 'd_totalSnow_cm']

top_30 = ['crs_dep_time(mins)', 'crs_arr_time(mins)', 'crs_elapsed_time', 'origin_traffic', 'dest_traffic', 'origin_taxi', 'dest_taxi', 'WN', 'DL', 'Monday', 'Thursday', 'Saturday', 'Friday', 'o_tempC', 'o_windspeedKmph','o_winddirDegree', 'o_visibility', 'o_cloudcover', 'o_humidity', 'o_precipMM', 'o_totalSnow_cm', 'o_sunrise', 'o_sunset', 'd_windspeedKmph', 'd_visibility', 'd_cloudcover', 'd_pressure', 'd_humidity', 'd_precipMM', 'd_totalSnow_cm']
