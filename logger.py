#!/usr/bin/env python
"""
module with functions to enable logging
"""

import time, os, re, csv, sys, uuid, joblib
from datetime import date

if not os.path.exists('logs'):
    os.mkdir('logs')


def update_train_log(tag, date_eval, eval_test, runtime, model_version, model_version_note, test=False) -> object:
    """
    update train log file
    """

    # name the logfile including the date
    today = date.today()
    if test:
        logfile = os.path.join('logs', 'train-test.log')
    else:
        logfile = os.path.join('logs', 'train-{}-{}-{}.log'.format(today.year, today.month, today.day))

    # write the data to a csv file
    header = ['unique_id',
              'timestamp',
              'tag',
              'data',
              'eval_test',
              'model_version',
              'model_version_note',
              'runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,
                       [uuid.uuid4(),
                        time.time(),
                        tag,
                        date_eval,
                        eval_test,
                        model_version,
                        model_version_note,
                        runtime])
        writer.writerow(to_write)


def update_predict_log(country, y_pred, y_proba, target_date, runtime, model_version, test=False) -> object:
    """
    update predict log file
    """

    # name the logfile including the date
    today = date.today()
    if test:
        logfile = os.path.join('logs', 'predict-test.log')
    else:
        logfile = os.path.join('logs', 'predict-{}-{}-{}.log'.format(today.year, today.month, today.day))

    # write the data to a csv file
    header = ['unique_id',
              'timestamp',
              'country',
              'y_pred',
              'y_proba',
              'target_date',
              'model_version',
              'runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,
                       [uuid.uuid4(),
                        time.time(),
                        country,
                        y_pred,
                        y_proba,
                        target_date,
                        model_version,
                        runtime])
        writer.writerow(to_write)


if __name__ == "__main__":
    """
    basic test procedure for logger.py
    """

    from model import MODEL_VERSION, MODEL_VERSION_NOTE

    # train logger
    tag = 'test'
    data_shape = (100, 10)
    eval_test = {'rmse': 0.5}
    runtime = "00:00:01"
    model_version = 0.1
    model_version_note = "test model"
    update_train_log(tag, data_shape, eval_test, runtime, model_version, model_version_note, test=True)

    # predict logger
    country = 'united states'
    y_pred = [0]
    y_proba = [0.6, 0.4]
    target_date = 24
    runtime = "00:00:02"
    model_version = 0.1
    query = ['united_states', 24, 'aavail_basic', 8]
    update_predict_log(country, y_pred, y_proba, target_date, runtime, model_version, test=True)
