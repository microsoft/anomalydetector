import pandas as pd
from msanomalydetector import SpectralResidual, DetectMode
from azureml.studio.core.logger import module_logger as logger


def sr_detect(frame, detect_mode, batch_size, threshold, sensitivity):
    model = SpectralResidual(frame, threshold=threshold, mag_window=3, score_window=21,
                             sensitivity=sensitivity, detect_mode=DetectMode(detect_mode),  batch_size=batch_size)
    result = model.detect()

    if detect_mode == DetectMode.anomaly_and_margin.value:
        return result[['isAnomaly', 'mag', 'score', 'expectedValue', 'lowerBoundary', 'upperBoundary']]
    return result[['isAnomaly', 'mag', 'score']]


def detect(timestamp, data_to_detect, detect_mode, batch_size, threshold=0.3, sensitivity=99):

    column_length = len(data_to_detect.columns)
    if column_length == 1:
        logger.debug('single column to detect')

        frame = pd.DataFrame(columns=['timestamp', 'value'])
        frame['timestamp'] = timestamp
        frame['value'] = data_to_detect.iloc[:, 0]
        output = sr_detect(frame, detect_mode, batch_size, threshold, sensitivity)
    else:
        logger.debug(f'detect {column_length} columns')
        output = pd.DataFrame()

        for col in data_to_detect.columns:
            frame = pd.DataFrame(columns=['timestamp', 'value'])
            frame['timestamp'] = timestamp
            frame['value'] = data_to_detect[col]
            result = sr_detect(frame, detect_mode, batch_size, threshold, sensitivity)
            result.columns = [f'{rc}_{col}' for rc in result.columns]
            output = pd.concat((output, result), axis=1)

    return output
