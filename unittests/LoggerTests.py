#!/usr/bin/env python
"""
model tests
"""

import os, sys
import csv
import unittest
from ast import literal_eval
import pandas as pd
from datetime import date

sys.path.insert(1,
                'C:/Users/IsaacSommers/Desktop/Apr2021/Capstone/ai-workflow-capstone-master/ai-workflow-capstone-master/Capstone')

# import model specific functions and variables
from logger import update_train_log, update_predict_log


class LoggerTest(unittest.TestCase):
    """
    test the essential functionality
    """

    def test_01_train(self):
        """
        ensure log file is created
        """

        log_file = os.path.join("logs", "train-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        # update the log
        tag = 'test'
        data_shape = (100, 10)
        eval_test = {'rmse': 0.5}
        runtime = "00:00:01"
        model_version = 0.1
        model_version_note = "test model"

        update_train_log(tag, data_shape, eval_test, runtime,
                         model_version, model_version_note, test=True)

        self.assertTrue(os.path.exists(log_file))

    def test_02_train(self):
        """
        ensure that content can be retrieved from log file
        """

        log_file = os.path.join("logs", "train-test.log")

        ## update the log
        tag = 'test'
        data_shape = (100, 10)
        eval_test = {'rmse': 0.5}
        runtime = "00:00:01"
        model_version = 0.1
        model_version_note = "test model"

        update_train_log(tag, data_shape, eval_test, runtime,
                         model_version, model_version_note, test=True)

        df = pd.read_csv(log_file)
        logged_eval_test = [literal_eval(i) for i in df['eval_test'].copy()][-1]
        self.assertEqual(eval_test, logged_eval_test)

    def test_03_predict(self):
        """
        ensure log file is created
        """

        log_file = os.path.join("logs", "predict-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        ## update the log
        country = 'united states'
        y_pred = [0]
        y_proba = [0.6, 0.4]
        target_date = 24
        runtime = "00:00:02"
        model_version = 0.1

        update_predict_log(country, y_pred, y_proba, target_date, runtime,
                           model_version, test=True)

        self.assertTrue(os.path.exists(log_file))

    def test_04_predict(self):
        """
        ensure that content can be retrieved from log file
        """

        log_file = os.path.join("logs", "predict-test.log")

        ## update the log
        country = 'united states'
        y_pred = [0]
        y_proba = [0.6, 0.4]
        target_date = 24
        runtime = "00:00:02"
        model_version = 0.1

        update_predict_log(country, y_pred, y_proba, target_date, runtime,
                           model_version, test=True)

        df = pd.read_csv(log_file)
        logged_y_pred = [literal_eval(i) for i in df['y_pred'].copy()][-1]
        self.assertEqual(y_pred, logged_y_pred)


### Run the tests
if __name__ == '__main__':
    unittest.main()
