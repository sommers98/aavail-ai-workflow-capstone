import time, os, re, csv, sys, uuid, joblib
from datetime import date
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from logger import update_predict_log, update_train_log
from cslib import fetch_ts, engineer_features

# model specific variables (iterate the version and note with each change)
MODEL_DIR = "models"
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learning model for time-series"


def _model_train(df, tag, test=False):
    """
    example function to train model

    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file

    """

    # start timer for runtime
    time_start = time.time()

    x, y, dates = engineer_features(df)

    if test:
        n_samples = int(np.round(0.3 * x.shape[0]))
        subset_indices = np.random.choice(np.arange(x.shape[0]), n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(y.size), subset_indices)
        y = y[mask]
        x = x[mask]
        dates = dates[mask]

    # Perform a train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                        shuffle=True, random_state=42)

    print("... training random forest for {}".format(tag))
    # train a random forest model
    param_grid_rf = {
        'rf__criterion': ['mse', 'mae'],
        'rf__n_estimators': [10, 15, 20, 25]
    }

    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                              ('rf', RandomForestRegressor())])

    grid_rf = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=5, n_jobs=1)
    grid_rf.fit(x_train, y_train)
    y_pred_rf = grid_rf.predict(x_test)
    eval_rmse_rf = round(np.sqrt(mean_squared_error(y_test, y_pred_rf)))

    # retrain using all data
    grid_rf.fit(x, y)
    model_name = re.sub("\.", "_", str(MODEL_VERSION))
    if test:
        saved_model = os.path.join(MODEL_DIR,
                                   "test-{}-{}.joblib".format(tag, model_name))
        print("... saving test version of model: {}".format(saved_model))
    else:
        saved_model = os.path.join(MODEL_DIR,
                                   "rf-{}-{}.joblib".format(tag, model_name))
        print("... saving model: {}".format(saved_model))

    joblib.dump(grid_rf, saved_model)

    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)

    # update log
    update_train_log(tag, (str(dates[0]), str(dates[-1])), {'rmse': eval_rmse_rf}, runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE)

    print("... training XGBRegressor for {}".format(tag))
    # training an XGBoost model
    pipe_xgb = Pipeline(steps=[('scaler', StandardScaler()),
                               ('xgb_model', xgb.XGBRegressor())])

    param_grid_xgb = {
        'xgb_model__subsample': np.arange(.05, 1, .05),
        'xgb_model__max_depth': np.arange(3, 20, 1),
        'xgb_model__colsample_bytree': np.arange(.1, 1.05, .05)
    }

    grid_xgb = RandomizedSearchCV(estimator=pipe_xgb, param_distributions=param_grid_xgb,
                                  n_iter=10, scoring='neg_mean_squared_error', cv=4)

    grid_xgb.fit(x_train, y_train)

    y_pred_xgb = grid_xgb.predict(x_test)
    eval_rmse_xgb = round(np.sqrt(mean_squared_error(y_test, y_pred_xgb)))

    # retrain using all data
    grid_xgb.fit(x, y)

    model_name = re.sub("\.", "_", str(MODEL_VERSION))
    if test:
        saved_model = os.path.join(MODEL_DIR,
                                   "test-{}-{}.joblib".format(tag, model_name))
        print("... saving test version of model: {}".format(saved_model))
    else:
        saved_model = os.path.join(MODEL_DIR,
                                   "xgb-{}-{}.joblib".format(tag, model_name))
        print("... saving model: {}".format(saved_model))

    joblib.dump(grid_xgb, saved_model)

    update_train_log(tag, (str(dates[0]), str(dates[-1])), {'rmse': eval_rmse_xgb}, runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE)


def model_train(data_dir, test=False):
    """
    function to train model given a df

    'mode' -  can be used to subset data essentially simulating a train
    """

    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if test:
        print("... test flag on")
        print("...... subsetting data")
        print("...... subsetting countries")

    # fetch time-series formatted data
    ts_data = fetch_ts(data_dir)

    # train a different model for each data sets
    for country, df in ts_data.items():

        if test and country not in ['all', 'united_kingdom']:
            continue

        _model_train(df, country, test=test)


def model_load(prefix='test', data_dir=None, training=True):
    """
    example function to load model

    The prefix allows the loading of different models
    """

    if not data_dir:
        data_dir = os.path.join("..", "cs-train")

    models = [f for f in os.listdir(os.path.join("models")) if re.search(prefix, f)]

    if len(models) == 0:
        raise Exception("Models with prefix '{}' cannot be found did you train?".format(prefix))

    all_models = {}
    for model in models:
        all_models[re.split("-", model)[1]] = joblib.load(os.path.join("models", model))

    # load data
    ts_data = fetch_ts(data_dir)
    all_data = {}
    for country, df in ts_data.items():
        x, y, dates = engineer_features(df, training=training)
        dates = np.array([str(d) for d in dates])
        all_data[country] = {"x": x, "y": y, "dates": dates}

    return all_data, all_models


def model_predict(country, year, month, day, prefix, all_models=None, all_data=None, test=False):
    """
    example function to predict from model
    """

    # start timer for runtime
    time_start = time.time()

    # load model if needed
    if not all_models:
        all_data, all_models = model_load(training=False, prefix=prefix)

    # input checks
    if country not in all_models.keys():
        raise Exception("ERROR (model_predict) - model for country '{}' could not be found".format(country))

    for d in [year, month, day]:
        if re.search("\D", d):
            raise Exception("ERROR (model_predict) - invalid year, month or day")

    # load data
    model = all_models[country]
    data = all_data[country]

    # check date
    target_date = "{}-{}-{}".format(year, str(month).zfill(2), str(day).zfill(2))
    print(target_date)

    if target_date not in data['dates']:
        raise Exception("ERROR (model_predict) - date {} not in range {}-{}".format(target_date,
                                                                                    data['dates'][0],
                                                                                    data['dates'][-1]))
    date_indx = np.where(data['dates'] == target_date)[0][0]
    query = data['x'].iloc[[date_indx]]

    # sainty check
    if data['dates'].shape[0] != data['x'].shape[0]:
        raise Exception("ERROR (model_predict) - dimensions mismatch")

    # make prediction and gather data for log entry
    y_pred = model.predict(query)
    y_proba = None
    if 'predict_proba' in dir(model) and 'probability' in dir(model):
        if model.probability:
            y_proba = model.predict_proba(query)

    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)

    # update predict log
    update_predict_log(country, y_pred, y_proba, target_date,
                       runtime, MODEL_VERSION, test=test)

    if not y_proba:
        return {'y_pred': y_pred, 'y_proba': 'none'}
    else:
        return {'y_pred': y_pred, 'y_proba': y_proba}


if __name__ == "__main__":
    """
    basic test procedure for model.py
    """

    # train the model
    print("TRAINING MODELS")
    data_dir = os.path.join("..", "cs-train")
    model_train(data_dir, test=True)

    # load the model
    # print("LOADING MODELS")
    # all_data, all_models = model_load(prefix='xgb')
    # print("... models loaded: ", ",".join(all_models.keys()))

    # test predict
    test_country = 'all'
    test_year = '2018'
    test_month = '02'
    test_day = '17'
    test_prefix = 'test'
    result = model_predict(test_country, test_year, test_month, test_day, test_prefix)
    print(result)
