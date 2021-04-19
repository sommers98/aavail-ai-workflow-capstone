#!/usr/bin/env python
"""
model tests
"""

import unittest
import sys, os, re

sys.path.insert(1,
                'C:/Users/IsaacSommers/Desktop/Apr2021/Capstone/ai-workflow-capstone-master/ai-workflow-capstone-master/Capstone')
# import model specific functions and variables
from model import *
from model import MODEL_VERSION


class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """

    def test_01_train(self):
        """
        test the train functionality
        """
        data_dir = os.path.join("..", "cs-train")
        # train the model
        model_train(data_dir, test=True)
        model_name = re.sub("\.", "_", str(MODEL_VERSION))
        self.assertTrue(os.path.exists(os.path.join("models", "test-all-{}.joblib".format(model_name))))

    def test_02_load(self):
        """
        test the train functionality
        """

        # train the model
        data, model = model_load()

        self.assertTrue('predict' in dir(model['all']))
        self.assertTrue('fit' in dir(model['all']))

    def test_03_predict(self):
        """
        test the predict function input
        """

        # ensure that a list can be passed
        country = 'all'
        year = '2018'
        month = '02'
        day = '17'
        prefix = 'test'

        result = model_predict(country, year, month, day, prefix)
        y_pred = result['y_pred']
        self.assertTrue(y_pred[0] > 0)


# Run the tests
if __name__ == '__main__':
    unittest.main()
