import unittest
import pandas as pd
import numpy as np
from msanomalydetector import SpectralResidual, DetectMode


class FunctionalyTest(unittest.TestCase):
    def test_anomaly_only_mode(self):
        frame = pd.DataFrame({'timestamp': pd.date_range('2020-01-01', periods=100, freq='1D'),
                              'value': np.linspace(1, 100, 100)})
        model = SpectralResidual(frame, threshold=0.3, mag_window=3, score_window=21, sensitivity=99,
                                 detect_mode=DetectMode.anomaly_only, batch_size=0)
        result = model.detect()
        self.assertEqual(result.shape[0], frame.shape[0])
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' not in result.columns)
        self.assertTrue('upperBoundary' not in result.columns)
        self.assertTrue('lowerBoundary' not in result.columns)

    def test_anomaly_and_margin_mode(self):
        frame = pd.DataFrame({'timestamp': pd.date_range('2020-01-01', periods=100, freq='1D'),
                              'value': np.linspace(1, 100, 100)})
        model = SpectralResidual(frame, threshold=0.3, mag_window=3, score_window=21, sensitivity=99,
                                 detect_mode=DetectMode.anomaly_and_margin, batch_size=0)
        result = model.detect()
        self.assertEqual(result.shape[0], frame.shape[0])
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' in result.columns)
        self.assertTrue('upperBoundary' in result.columns)
        self.assertTrue('lowerBoundary' in result.columns)

    def test_batch_mode(self):
        frame = pd.DataFrame({'timestamp': pd.date_range('2020-01-01', periods=100, freq='1D'),
                              'value': np.linspace(1, 100, 100)})
        model = SpectralResidual(frame, threshold=0.3, mag_window=3, score_window=21, sensitivity=99,
                                 detect_mode=DetectMode.anomaly_and_margin, batch_size=33)
        result = model.detect()
        self.assertEqual(result.shape[0], frame.shape[0])
        self.assertTrue('value' in result.columns)
        self.assertTrue('isAnomaly' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('expectedValue' in result.columns)
        self.assertTrue('upperBoundary' in result.columns)
        self.assertTrue('lowerBoundary' in result.columns)


if __name__ == '__main__':
    unittest.main()
