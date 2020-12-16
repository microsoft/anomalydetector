import sys
sys.path.append('../')

import unittest
import numpy as np
import pandas as pd
import shutil
import os
import invoker


class TestErrorInput(unittest.TestCase):
    def setUp(self):
        self.__input_path = './functional_test_input_folder'
        self.__input_csv_file = './functional_test_input_file.csv'
        self.__input_parquet_file = './functional_test_input_file.parquet'
        self.__detect_mode = 'AnomalyOnly'
        self.__timestamp_column = 'timestamp'
        self.__value_column = 'value'
        self.__batch_size = 2000
        self.__threshold = 0.3
        self.__sensitivity = 99
        self.__append_mode = True
        self.__output_path = './functional_test_output_directory'

    def tearDown(self):
        self.deleteDataFrameDirectory()

    def deleteDataFrameDirectory(self):
        if os.path.exists(self.__input_path):
            shutil.rmtree(self.__input_path)

        if os.path.exists(self.__input_csv_file):
            os.remove(self.__input_csv_file)

        if os.path.exists(self.__input_parquet_file):
            os.remove(self.__input_parquet_file)

        if os.path.exists(self.__output_path):
            shutil.rmtree(self.__output_path)

    def generate_input_data_frame(self, start_date: str = '2020-01-01'):
        df = pd.DataFrame()
        df['timestamp'] = pd.date_range(start=start_date, periods=200, freq='1D')
        df['value'] = np.sin(np.linspace(1, 20, 200))
        return df

    def generate_input_folder(self, file_type: str = 'csv'):
        if not os.path.isdir(self.__input_path):
            os.mkdir(self.__input_path)
        start_dates = ['2018-01-01', '2019-01-01', '2020-01-01']
        for start_date in start_dates:
            df = self.generate_input_data_frame(start_date)
            if file_type == 'csv':
                df.to_csv(f"{self.__input_path}/{start_date}.csv", index=False)
            elif file_type == 'parquet':
                df.to_parquet(f"{self.__input_path}/{start_date}.parquet", index=False)
            else:
                raise Exception(f'Unsupported input data type {file_type}, only csv and parquet file are allowed')

    def testAnomalyOnlyModeCsvFile(self):
        df = self.generate_input_data_frame()
        df.to_csv(self.__input_csv_file, index=False)
        invoker.invoke(self.__input_csv_file, self.__detect_mode, self.__timestamp_column, self.__value_column,
                        self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode, self.__output_path)
        result = pd.read_csv(f"{self.__output_path}/output.csv")
        self.assertEqual(result.shape[0], 200)
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' not in result.columns)
        self.assertTrue('upperBoundary' not in result.columns)
        self.assertTrue('lowerBoundary' not in result.columns)

    def testAnomalyOnlyModeCsvFolder(self):
        self.generate_input_folder()
        invoker.invoke(self.__input_path, self.__detect_mode, self.__timestamp_column, self.__value_column,
                        self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode, self.__output_path)
        result = pd.read_csv(f"{self.__output_path}/output.csv")
        self.assertEqual(result.shape[0], 600)
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' not in result.columns)
        self.assertTrue('upperBoundary' not in result.columns)
        self.assertTrue('lowerBoundary' not in result.columns)

    def testAnomalyOnlyModeParquetFile(self):
        df = self.generate_input_data_frame()
        df.to_parquet(self.__input_parquet_file, index=False)
        invoker.invoke(self.__input_parquet_file, self.__detect_mode, self.__timestamp_column, self.__value_column,
                        self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode, self.__output_path)
        result = pd.read_csv(f"{self.__output_path}/output.csv")
        self.assertEqual(result.shape[0], 200)
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' not in result.columns)
        self.assertTrue('upperBoundary' not in result.columns)
        self.assertTrue('lowerBoundary' not in result.columns)

    def testAnomalyOnlyModeParquetFolder(self):
        self.generate_input_folder('parquet')
        invoker.invoke(self.__input_path, self.__detect_mode, self.__timestamp_column, self.__value_column,
                        self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode, self.__output_path)
        result = pd.read_csv(f"{self.__output_path}/output.csv")
        self.assertEqual(result.shape[0], 600)
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' not in result.columns)
        self.assertTrue('upperBoundary' not in result.columns)
        self.assertTrue('lowerBoundary' not in result.columns)

    def testAnomalyAndMarginCsvFile(self):
        df = self.generate_input_data_frame()
        df.to_csv(self.__input_csv_file, index=False)
        invoker.invoke(self.__input_csv_file, "AnomalyAndMargin", self.__timestamp_column, self.__value_column,
                        self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode, self.__output_path)
        result = pd.read_csv(f"{self.__output_path}/output.csv")
        self.assertEqual(result.shape[0], 200)
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' in result.columns)
        self.assertTrue('upperBoundary' in result.columns)
        self.assertTrue('lowerBoundary' in result.columns)

    def testAnomalyAndMarginCsvFolder(self):
        self.generate_input_folder()
        invoker.invoke(self.__input_path, "AnomalyAndMargin", self.__timestamp_column, self.__value_column,
                        self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode, self.__output_path)
        result = pd.read_csv(f"{self.__output_path}/output.csv")
        self.assertEqual(result.shape[0], 600)
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' in result.columns)
        self.assertTrue('upperBoundary' in result.columns)
        self.assertTrue('lowerBoundary' in result.columns)

    def testAnomalyAndMarginParquetFile(self):
        df = self.generate_input_data_frame()
        df.to_parquet(self.__input_parquet_file, index=False)
        invoker.invoke(self.__input_parquet_file, "AnomalyAndMargin", self.__timestamp_column, self.__value_column,
                        self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode, self.__output_path)
        result = pd.read_csv(f"{self.__output_path}/output.csv")
        self.assertEqual(result.shape[0], 200)
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' in result.columns)
        self.assertTrue('upperBoundary' in result.columns)
        self.assertTrue('lowerBoundary' in result.columns)

    def testAnomalyAndMarginParquetFolder(self):
        self.generate_input_folder('parquet')
        invoker.invoke(self.__input_path, "AnomalyAndMargin", self.__timestamp_column, self.__value_column,
                        self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode, self.__output_path)
        result = pd.read_csv(f"{self.__output_path}/output.csv")
        self.assertEqual(result.shape[0], 600)
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' in result.columns)
        self.assertTrue('upperBoundary' in result.columns)
        self.assertTrue('lowerBoundary' in result.columns)

    def testBatchModeCsvFile(self):
        df = self.generate_input_data_frame()
        df.to_csv(self.__input_csv_file, index=False)
        invoker.invoke(self.__input_csv_file, "AnomalyAndMargin", self.__timestamp_column, self.__value_column,
                        66, self.__threshold, self.__sensitivity, self.__append_mode, self.__output_path)
        result = pd.read_csv(f"{self.__output_path}/output.csv")
        self.assertEqual(result.shape[0], 200)
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' in result.columns)
        self.assertTrue('upperBoundary' in result.columns)
        self.assertTrue('lowerBoundary' in result.columns)

    def testBatchModeCsvFolder(self):
        self.generate_input_folder()
        invoker.invoke(self.__input_path, "AnomalyAndMargin", self.__timestamp_column, self.__value_column,
                        66, self.__threshold, self.__sensitivity, self.__append_mode, self.__output_path)
        result = pd.read_csv(f"{self.__output_path}/output.csv")
        self.assertEqual(result.shape[0], 600)
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' in result.columns)
        self.assertTrue('upperBoundary' in result.columns)
        self.assertTrue('lowerBoundary' in result.columns)

    def testBatchModeParquetFile(self):
        df = self.generate_input_data_frame()
        df.to_parquet(self.__input_parquet_file, index=False)
        invoker.invoke(self.__input_parquet_file, "AnomalyAndMargin", self.__timestamp_column, self.__value_column,
                        66, self.__threshold, self.__sensitivity, self.__append_mode, self.__output_path)
        result = pd.read_csv(f"{self.__output_path}/output.csv")
        self.assertEqual(result.shape[0], 200)
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' in result.columns)
        self.assertTrue('upperBoundary' in result.columns)
        self.assertTrue('lowerBoundary' in result.columns)

    def testBatchModeParquetFolder(self):
        self.generate_input_folder('parquet')
        invoker.invoke(self.__input_path, "AnomalyAndMargin", self.__timestamp_column, self.__value_column,
                        66, self.__threshold, self.__sensitivity, self.__append_mode, self.__output_path)
        result = pd.read_csv(f"{self.__output_path}/output.csv")
        self.assertEqual(result.shape[0], 600)
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' in result.columns)
        self.assertTrue('upperBoundary' in result.columns)
        self.assertTrue('lowerBoundary' in result.columns)

if __name__ == '__main__':
    unittest.main()
