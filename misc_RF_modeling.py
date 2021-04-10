#!/usr/bin/env python
# %%
''' RANDOM FOREST '''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from functions import create_connection, interval_calc, save_model

''' TEST SAMPLE '''
# %%
# Pull random subset from database
cnx = create_connection('data/training.sqlite')
sample = pd.read_sql_query(
    'SELECT * FROM flight_ready_nonull ORDER BY RANDOM() LIMIT 200000;', cnx, index_col='index')

# %%
sample.head()
# %%
# select continuous data columns
X = sample.iloc[:, :-1]
y = sample.iloc[:, -1:]
# %%
# standardize data
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X).astype(float))

# %%
# split into training & test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=123)
# %%
# instantiate RandomForest
RF = RandomForestRegressor()
# %%
# params_grid for GridSearch
RF_params = {
    'n_estimators': np.array([8, 16, 32, 64, 100]),
    'max_features': np.array(['auto', 'sqrt', 'log2']),
    'max_depth': np.array(list(range(1, 32, 4))),
    'min_samples_split': np.array([0.05, 0.1, 0.2, 0.5]),
    'min_samples_leaf': np.array([0.1, 0.5, 1])
}

# %%
# run GridSearch
RF_grid = GridSearchCV(
    RF, RF_params, scoring='neg_mean_squared_error', n_jobs=-1, cv=5, verbose=1)

# %%
# train model and find best parameters for dataset
RF_result = RF_grid.fit(X_train, y_train)
best = RF_result.best_estimator_
#RandomForestRegressor(max_depth=25, min_samples_leaf=0.1, min_samples_split=0.1, n_estimators=64)

# %%
RF_result.best_score_
#best score: -2466.7403907120406

# %%
save_model(RF_result, 'RF_X_200K')
# %%
