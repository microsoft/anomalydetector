import sys
sys.path.append('../')

import unittest
import numpy as np
import pandas as pd
import shutil
import os
from azureml.studio.core.error import UserError
import invoker


class TestErrorInput(unittest.TestCase):
    def setUp(self):
        self.__input_path = '../../samples/sample_error.csv'
        self.__detect_mode = 'AnomalyOnly'
        self.__timestamp_column = '%7B%22isFilter%22%3Atrue%2C%22rules%22%3A%5B%7B%22exclude%22%3Afalse%2C%22ruleType%22%3A%22ColumnNames%22%2C%22columns%22%3A%5B%22timestamp%22%5D%7D%5D%7D'
        self.__value_column = '%7B%22isFilter%22%3Atrue%2C%22rules%22%3A%5B%7B%22exclude%22%3Afalse%2C%22ruleType%22%3A%22ColumnNames%22%2C%22columns%22%3A%5B%22value%22%5D%7D%5D%7D'
        self.__batch_size = 2000
        self.__threshold = 0.3
        self.__sensitivity = 99
        self.__append_mode = True
        self.compute_stats_in_visualization = False
        self.__output_path = './test_output_data_frame_directory'

    def tearDown(self):
        self.deleteDataFrameDirectory()

    def deleteDataFrameDirectory(self):
        if os.path.exists(self.__input_path):
            os.remove(self.__input_path)

        if os.path.exists(self.__output_path):
            shutil.rmtree(self.__output_path)

    def test_empty_input(self):
        df = pd.DataFrame()
        df.to_csv(self.__input_path)
        self.assertRaisesRegexp(UserError, "The dataset should contain at leaslt 12 points to run this module.",
                                invoker.invoke,
                                self.__input_path, self.__detect_mode, self.__timestamp_column, self.__value_column,
                                self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode,
                                self.compute_stats_in_visualization, self.__output_path)

    def test_invalid_timestamp(self):
        df = pd.DataFrame()
        df['timestamp'] = 'invalid'
        df['value'] = np.ones(20)
        df.to_csv(self.__input_path)
        self.assertRaisesRegexp(UserError, "The timestamp column specified is malformed.",
                                invoker.invoke,
                                self.__input_path, self.__detect_mode, self.__timestamp_column, self.__value_column,
                                self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode,
                                self.compute_stats_in_visualization, self.__output_path)

    def test_invalid_series_order(self):
        df = pd.DataFrame()
        timestamps = pd.date_range(start='2020-01-01', periods=20, freq='1D')[::-1]
        df['timestamp'] = timestamps
        df['value'] = np.ones(20)
        df.to_csv(self.__input_path)
        self.assertRaisesRegexp(UserError, "The timestamp column specified is not in ascending order.",
                                invoker.invoke,
                                self.__input_path, self.__detect_mode, self.__timestamp_column, self.__value_column,
                                self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode,
                                self.compute_stats_in_visualization, self.__output_path)

    def test_dunplicate_sereis(self):
        df = pd.DataFrame()
        df['value'] = np.ones(20)
        df['timestamp'] = '2020-01-01'
        df.to_csv(self.__input_path)
        self.assertRaisesRegexp(UserError, "The timestamp column specified has duplicated timestamps.",
                                invoker.invoke,
                                self.__input_path, self.__detect_mode, self.__timestamp_column, self.__value_column,
                                self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode,
                                self.compute_stats_in_visualization, self.__output_path)

    def test_invalid_value_format(self):
        df = pd.DataFrame()
        timestamps = pd.date_range(start='2020-01-01', periods=20, freq='1D')
        df['timestamp'] = timestamps
        df['value'] = 'invalid'
        df.to_csv(self.__input_path)
        self.assertRaisesRegexp(UserError, 'The data in column "value" can not be parsed as float values.',
                                invoker.invoke,
                                self.__input_path, self.__detect_mode, self.__timestamp_column, self.__value_column,
                                self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode,
                                self.compute_stats_in_visualization, self.__output_path)

    def test_invalid_series_value(self):
        df = pd.DataFrame()
        timestamps = pd.date_range(start='2020-01-01', periods=20, freq='1D')
        df['timestamp'] = timestamps
        df['value'] = np.nan
        df.to_csv(self.__input_path)
        self.assertRaisesRegexp(UserError, 'The data in column "value" contains nan values.',
                                invoker.invoke,
                                self.__input_path, self.__detect_mode, self.__timestamp_column, self.__value_column,
                                self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode,
                                self.compute_stats_in_visualization, self.__output_path)

    def test_value_overflow(self):
        df = pd.DataFrame()
        timestamps = pd.date_range(start='2020-01-01', periods=20, freq='1D')
        df['timestamp'] = timestamps
        df['value'] = 1e200
        df.to_csv(self.__input_path)
        self.assertRaisesRegexp(UserError, 'The magnitude of data in column "value" exceeds limitation.',
                                invoker.invoke,
                                self.__input_path, self.__detect_mode, self.__timestamp_column, self.__value_column,
                                self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode,
                                self.compute_stats_in_visualization, self.__output_path)

    def test_not_enough_points(self):
        df = pd.DataFrame()
        timestamps = pd.date_range(start='2020-01-01', periods=10, freq='1D')
        df['timestamp'] = timestamps
        df['value'] = np.sin(np.linspace(1, 10, 10))
        df.to_csv(self.__input_path)
        self.assertRaisesRegexp(UserError, "The dataset should contain at leaslt 12 points to run this module.",
                                invoker.invoke,
                                self.__input_path, self.__detect_mode, self.__timestamp_column, self.__value_column,
                                self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode,
                                self.compute_stats_in_visualization, self.__output_path)

    def test_invalid_batch_size(self):
        df = pd.DataFrame()
        timestamps = pd.date_range(start='2020-01-01', periods=20, freq='1D')
        df['timestamp'] = timestamps
        df['value'] = np.sin(np.linspace(1, 10, 20))
        df.to_csv(self.__input_path)
        self.assertRaisesRegexp(UserError, 'The "batchSize" parameter should be at least 12 or 0 that indicates to run all data in a batch',
                                invoker.invoke,
                                self.__input_path, self.__detect_mode, self.__timestamp_column, self.__value_column,
                                5, self.__threshold, self.__sensitivity, self.__append_mode,
                                self.compute_stats_in_visualization, self.__output_path)

    def test_timestamp_column_missing(self):
        df = pd.DataFrame()
        timestamps = pd.date_range(start='2020-01-01', periods=20, freq='1D')
        df['time'] = timestamps
        df['value'] = np.sin(np.linspace(1, 10, 20))
        df.to_csv(self.__input_path)
        self.assertRaisesRegexp(Exception, 'Column with name or index "timestamp" not found.',
                                invoker.invoke,
                                self.__input_path, self.__detect_mode, self.__timestamp_column, self.__value_column,
                                self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode,
                                self.compute_stats_in_visualization, self.__output_path)

    def test_value_column_missing(self):
        df = pd.DataFrame()
        timestamps = pd.date_range(start='2020-01-01', periods=20, freq='1D')
        df['timestamp'] = timestamps
        df['missed'] = np.sin(np.linspace(1, 10, 20))
        df.to_csv(self.__input_path)
        self.assertRaisesRegexp(Exception, 'Column with name or index "value" not found.',
                                invoker.invoke,
                                self.__input_path, self.__detect_mode, self.__timestamp_column, self.__value_column,
                                self.__batch_size, self.__threshold, self.__sensitivity, self.__append_mode,
                                self.compute_stats_in_visualization, self.__output_path)


if __name__ == '__main__':
    unittest.main()
