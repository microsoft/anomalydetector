import sys
sys.path.append('../')

import unittest
from unittest import mock
import numpy as np
import pandas as pd
import shutil
import os
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
import invoker


class TestErrorInput(unittest.TestCase):
    def setUp(self):
        self.__input_path = './functional_test_input_data_frame_directory'
        self.__input_file = './functional_test_input_file.csv'
        self.__detect_mode = 'AnomalyOnly'
        self.__timestamp_column = '%7B%22isFilter%22%3Atrue%2C%22rules%22%3A%5B%7B%22exclude%22%3Afalse%2C%22ruleType%22%3A%22ColumnNames%22%2C%22columns%22%3A%5B%22timestamp%22%5D%7D%5D%7D'
        self.__value_column = '%7B%22isFilter%22%3Atrue%2C%22rules%22%3A%5B%7B%22exclude%22%3Afalse%2C%22ruleType%22%3A%22ColumnNames%22%2C%22columns%22%3A%5B%22value%22%5D%7D%5D%7D'
        self.__batch_size = 2000
        self.__threshold = 0.3
        self.__sensitivity = 99
        self.__append_mode = True
        self.compute_stats_in_visualization = True,
        self.__output_path = './functional_test_output_data_frame_directory'

    def tearDown(self):
        self.deleteDataFrameDirectory()

    def deleteDataFrameDirectory(self):
        if os.path.exists(self.__input_path):
            shutil.rmtree(self.__input_path)

        if os.path.exists(self.__input_file):
            os.remove(self.__input_file)

        if os.path.exists(self.__output_path):
            shutil.rmtree(self.__output_path)

    def generate_input_data_frame(self):
        df = pd.DataFrame()
        df['timestamp'] = pd.date_range(start='2020-01-01', periods=200, freq='1D')
        df['value'] = np.sin(np.linspace(1, 20, 200))
        return df

    def testAnomalyOnlyModeAnyDirectory(self):
        df = self.generate_input_data_frame()
        df.to_csv(self.__input_file)
        invoker.invoke(self.__input_file, self.__detect_mode, self.__timestamp_column, self.__value_column,
                        self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode,
                        self.compute_stats_in_visualization, self.__output_path)
        result = load_data_frame_from_directory(self.__output_path).data
        self.assertEqual(result.shape[0], 200)
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' not in result.columns)
        self.assertTrue('upperBoundary' not in result.columns)
        self.assertTrue('lowerBoundary' not in result.columns)

    def testAnomalyAndMarginAnyDirectory(self):
        df = self.generate_input_data_frame()
        df.to_csv(self.__input_file)
        invoker.invoke(self.__input_file, "AnomalyAndMargin", self.__timestamp_column, self.__value_column,
                        self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode,
                        self.compute_stats_in_visualization, self.__output_path)
        result = load_data_frame_from_directory(self.__output_path).data
        self.assertEqual(result.shape[0], 200)
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' in result.columns)
        self.assertTrue('upperBoundary' in result.columns)
        self.assertTrue('lowerBoundary' in result.columns)

    def testBatchModeAnyDirectory(self):
        df = self.generate_input_data_frame()
        df.to_csv(self.__input_file)
        invoker.invoke(self.__input_file, "AnomalyAndMargin", self.__timestamp_column, self.__value_column,
                        66, self.__threshold, self.__sensitivity, self.__append_mode,
                        self.compute_stats_in_visualization, self.__output_path)
        result = load_data_frame_from_directory(self.__output_path).data

        self.assertEqual(result.shape[0], 200)
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' in result.columns)
        self.assertTrue('upperBoundary' in result.columns)
        self.assertTrue('lowerBoundary' in result.columns)

    def testAnomalyOnlyModeDFD(self):
        df = self.generate_input_data_frame()
        save_data_frame_to_directory(self.__input_path, df)
        invoker.invoke(self.__input_path, self.__detect_mode, self.__timestamp_column, self.__value_column,
                        self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode,
                        self.compute_stats_in_visualization, self.__output_path)
        result = load_data_frame_from_directory(self.__output_path).data
        self.assertEqual(result.shape[0], 200)
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' not in result.columns)
        self.assertTrue('upperBoundary' not in result.columns)
        self.assertTrue('lowerBoundary' not in result.columns)

    def testAnomalyAndMarginDFD(self):
        df = self.generate_input_data_frame()
        save_data_frame_to_directory(self.__input_path, df)
        invoker.invoke(self.__input_path, "AnomalyAndMargin", self.__timestamp_column, self.__value_column,
                        self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode,
                        self.compute_stats_in_visualization, self.__output_path)
        result = load_data_frame_from_directory(self.__output_path).data
        self.assertEqual(result.shape[0], 200)
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' in result.columns)
        self.assertTrue('upperBoundary' in result.columns)
        self.assertTrue('lowerBoundary' in result.columns)

    def testBatchModeDFD(self):
        df = self.generate_input_data_frame()
        save_data_frame_to_directory(self.__input_path, df)
        invoker.invoke(self.__input_path, "AnomalyAndMargin", self.__timestamp_column, self.__value_column,
                        66, self.__threshold, self.__sensitivity, self.__append_mode,
                        self.compute_stats_in_visualization, self.__output_path)
        result = load_data_frame_from_directory(self.__output_path).data

        self.assertEqual(result.shape[0], 200)
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' in result.columns)
        self.assertTrue('upperBoundary' in result.columns)
        self.assertTrue('lowerBoundary' in result.columns)


if __name__ == '__main__':
    unittest.main()
