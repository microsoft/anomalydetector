import sys
sys.path.append('../')

import unittest
import numpy as np
import pandas as pd
import shutil
import os
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
import invoker


class TestErrorInput(unittest.TestCase):
    def setUp(self):
        self.__input_path = './functional_test_input_folder'
        self.__input_file = './functional_test_input_file.csv'
        self.__detect_mode = 'AnomalyOnly'
        self.__timestamp_column = 'timestamp'
        self.__value_column = 'value'
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

        if os.path.exists(self.__output_path):
            shutil.rmtree(self.__output_path)

    def generate_input_data_frame(self, start_date: str = '2020-01-01'):
        df = pd.DataFrame()
        df['timestamp'] = pd.date_range(start=start_date, periods=200, freq='1D')
        df['value'] = np.sin(np.linspace(1, 20, 200))
        return df

    def generate_input_folder(self):
        start_dates = ['2018-01-01', '2019-01-01', '2020-01-01']
        for start_date in start_dates:
            df = self.generate_input_data_frame(start_date)
            df.to_csv(".".join(["/".join([self.__input_path, start_date]), 'csv']))

    def testAnomalyOnlyModeCSVFile(self):
        df = self.generate_input_data_frame()
        df.to_csv(self.__input_file, index=False)
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

    def testAnomalyOnlyModeCSVFolder(self):
        self.generate_input_folder()
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

    def testAnomalyAndMargin(self):
        df = pd.DataFrame()
        df['timestamp'] = pd.date_range(start='2020-01-01', periods=200, freq='1D')
        df['value'] = np.sin(np.linspace(1, 20, 200))
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

    def testBatchMode(self):
        df = pd.DataFrame()
        df['timestamp'] = pd.date_range(start='2020-01-01', periods=200, freq='1D')
        df['value'] = np.sin(np.linspace(1, 20, 200))
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
