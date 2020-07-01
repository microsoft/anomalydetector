import pandas as pd
from msanomalydetector import SpectralResidual, DetectMode
import matplotlib
import matplotlib.pyplot as plt
import logging
from azureml.core.run import Run
import os


def log_plot_result(input_df, output_df, col_name, mode):
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(211)
    if mode == 'AnomalyAndMargin':
        ax1.fill_between(output_df.index, output_df['lowerBoundary'], output_df['upperBoundary'], color='grey', alpha=0.2, zorder=1)
        ax1.plot(output_df.index, output_df['expectedValue'], alpha=0.5, label='expected value', zorder=8)
    ax1.plot(input_df.index, input_df['value'], label='value', zorder=5)
    ax1.legend()
    anomalies = input_df[output_df['isAnomaly']]
    ax1.scatter(anomalies.index, anomalies['value'], c='red', zorder=10)
    ax1.set_title(col_name)

    ax2 = fig.add_subplot(212)
    ax2.plot(output_df.index, output_df['mag'])
    ax2.set_title('mag')

    run = Run.get_context()
    run.log_image(col_name, plot=plt)


def sr_detect(frame, detect_mode, batch_size, threshold, sensitivity):
    model = SpectralResidual(frame, threshold=threshold, mag_window=3, score_window=40,
                             sensitivity=sensitivity, detect_mode=DetectMode(detect_mode),  batch_size=batch_size)
    result = model.detect()

    if detect_mode == DetectMode.anomaly_and_margin.value:
        return result[['isAnomaly', 'mag', 'score', 'expectedValue', 'lowerBoundary', 'upperBoundary']]
    return result[['isAnomaly', 'mag', 'score']]


def detect(timestamp, data_to_detect, detect_mode, batch_size, threshold=0.3, sensitivity=99):

    column_length = len(data_to_detect.columns)
    if column_length == 1:
        logging.debug('single column to detect')

        frame = pd.DataFrame(columns=['timestamp', 'value'])
        frame['timestamp'] = timestamp
        frame['value'] = data_to_detect.iloc[:, 0]
        output = sr_detect(frame, detect_mode, batch_size, threshold, sensitivity)
        log_plot_result(frame, output, data_to_detect.columns[0], detect_mode)
    else:
        logging.debug(f'detect {column_length} columns')
        output = pd.DataFrame()

        for col in data_to_detect.columns:
            frame = pd.DataFrame(columns=['timestamp', 'value'])
            frame['timestamp'] = timestamp
            frame['value'] = data_to_detect[col]
            result = sr_detect(frame, detect_mode, batch_size, threshold, sensitivity)
            log_plot_result(frame, result, col, detect_mode)
            result.columns = [f'{rc}_{col}' for rc in result.columns]
            output = pd.concat((output, result), axis=1)

    return output
