from msanomalydetector import SpectralResidual
from msanomalydetector import THRESHOLD, MAG_WINDOW, SCORE_WINDOW, DetectMode
import os
import pandas as pd


def detect_anomaly(series, threshold, mag_window, score_window, sensitivity, detect_mode):
    detector = SpectralResidual(series=series, threshold=threshold, mag_window=mag_window, score_window=score_window,
                                sensitivity=sensitivity, detect_mode=detect_mode)
    print(detector.detect())


if __name__ == '__main__':
    sample_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "samples"))
    for sample_file in os.listdir(sample_dir):
        sample = pd.read_csv(os.path.join(sample_dir, sample_file))
        detect_anomaly(sample, THRESHOLD, MAG_WINDOW, SCORE_WINDOW, 99, DetectMode.anomaly_only)
