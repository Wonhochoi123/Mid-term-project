#!/usr/bin/env python

''' CREATE & SAVE MODELS '''
# TESTING INCREMENTAL LEARNERS

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from functions import incr_train

incr_train(model=SGDRegressor(), filepath='models/SGDReg_A20', db_path='data/fl_final.sqlite', table='flights_final', invs=60, mode='A20')

incr_train(model=SGDRegressor(), filepath='models/SGDReg_A30', db_path='data/fl_final.sqlite', table='flights_final', invs=60, mode='A30')

incr_train(model=PassiveAggressiveRegressor(), filepath='models/PAReg_A20', db_path='data/fl_final.sqlite', table='flights_final', invs=60, mode='A20')

incr_train(model=PassiveAggressiveRegressor(), filepath='models/PAReg_A30', db_path='data/fl_final.sqlite', table='flights_final', invs=60, mode='A30')

# %%

''' IMPORT & SPLIT DATA INTO: TRAINING & TEST '''
from sklearn.model_selection import train_test_split
from functions import load_model
from sklearn import metrics
# %%
X.dtypes
# %%
train_data = pd.read_csv('data/flights_30J.csv')
test_data = pd.read_csv('data/flights_30J_test.csv') 
# %%
X = train_data.iloc[:,2:]
y = train_data['arr_delay']
Xt = test_data.iloc[:,5:]
Xt_noNA = Xt.dropna()
# %%
len(X.columns)
# %%

''' SPLIT DATA '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
# %%

''' LOAD MODELS '''
#SGD_A20 = load_model('models/SGDReg_A20.sav')
SGD_A30 = load_model('models/SGDReg_A30.sav')
#PAR_A20 = load_model('models/PAReg_A20.sav')
PAR_A30 = load_model('models/PAReg_A30.sav')
# %%
algs = [SGD_A30, PAR_A30]
results = [alg.predict(X_test) for alg in algs]
results_T = [alg.predict(Xt_noNA) for alg in algs]
# %%

''' ASSESS MODELS '''

for y_pred in results:
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  print('R2_score:', metrics.r2_score(y_test, y_pred))
  print('Explained Variance Score:', metrics.explained_variance_score(y_test,y_pred))
# %%
for y_pred in results_T:
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  print('R2_score:', metrics.r2_score(y_test, y_pred))
  print('Explained Variance Score:', metrics.explained_variance_score(y_test,y_pred))
