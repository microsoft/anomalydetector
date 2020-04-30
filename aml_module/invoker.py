import argparse
from urllib.parse import unquote
import pandas as pd
import numpy as np
from azureml.studio.core.logger import module_logger as logger
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.studio.core.utils.column_selection import ColumnSelection
from azureml.studio.internal.error_handler import error_handler
import sr_detector
from error_messages import *
from constants import *

PACKAGE_NAME = 'spectral_residual_anomaly_detection_module'
VERSION = '0.0.1'


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def is_timestamp_ascending(timestamps):
    count = len(timestamps)

    for i in range(count - 1):
        if timestamps[i] > timestamps[i + 1]:
            return -1
        elif timestamps[i] == timestamps[i + 1]:
            return -2
    return 0


@error_handler
def invoke(input_path, detect_mode, timestamp_column, value_column, batch_size, threshold, sensitivity,
            appendMode, compute_stats_in_visualization, output_path):
    data_frame_directory = load_data_frame_from_directory(input_path)

    logger.debug(f"Shape of loaded DataFrame: {data_frame_directory.data.shape}")

    if data_frame_directory.data.shape[0] < MIN_POINTS:
        raise Exception(NotEnoughPoints.format(MIN_POINTS))

    if 0 < batch_size < MIN_POINTS:
        raise Exception(InvalidBatchSize.format(MIN_POINTS))

    query_string = unquote(timestamp_column)
    timestamp_column_selector = ColumnSelection(query_string)
    timestamp = timestamp_column_selector.select_dataframe_directory(data_frame_directory).data

    timestamps = pd.to_datetime(timestamp.iloc[:, 0].values)

    if np.any(np.isnat(timestamps)):
        raise Exception(InvalidTimestamps)

    res = is_timestamp_ascending(timestamps)
    if res == -1:
        raise Exception(InvalidSeriesOrder)
    elif res == -2:
        raise Exception(DuplicateSeriesTimestamp)


    query_string = unquote(value_column)
    data_column_selector = ColumnSelection(query_string)
    data_columns = data_column_selector.select_dataframe_directory(data_frame_directory).data

    for col in data_columns.columns:
        try:
            float_data = data_columns[col].apply(float)
        except Exception as e:
            raise Exception(InvalidValueFormat.format(col))

        if not np.all(np.isfinite(float_data)):
            raise Exception(InvalidSeriesValue.format(col))

        if np.any(np.less(float_data, VALUE_LOWER_BOUND)) or np.any(np.greater(float_data, VALUE_UPPER_BOUND)):
            raise Exception(ValueOverflow.format(col))

        data_columns[col] = float_data

    result = sr_detector.detect(timestamps, data_columns, detect_mode=detect_mode,
                                batch_size=batch_size, threshold=threshold, sensitivity=sensitivity)

    if appendMode is True:
        result = pd.merge(data_frame_directory.data, result, left_index=True, right_index=True)

    save_data_frame_to_directory(output_path, result, compute_stats_in_visualization=compute_stats_in_visualization)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-path',
        help='Input Dataframe path'
    )

    parser.add_argument(
        '--detect-mode',
        choices=['AnomalyOnly', 'AnomalyAndMargin'],
        help='Specify the detect mode.'
    )

    parser.add_argument(
        '--timestamp-column',
        help='Choose the column that contains timestamps.'
    )

    parser.add_argument(
        '--value-column',
        help='Choose the column that contains values.'
    )

    parser.add_argument(
        '--batch-size', type=int,
        help='This parameter specifies the size of each batch that the detection is perfomed.'
    )

    parser.add_argument(
        '--threshold', type=float,
        help='This parameter specifies the threshold anomaly score that a point is judged as anomaly.'
    )

    parser.add_argument(
        '--sensitivity', type=float,
        help='This parameter is used in AnomalyAndMargin mode to control the width of margin.'
    )

    parser.add_argument(
        '--append-mode', type=str2bool, default=False,
        help='This parameter is used in AnomalyAndMargin mode to control the width of margin.'
    )

    parser.add_argument(
        '--compute-stats-in-visualization', type=str2bool, default=False,
        help='Enable this parameter to get stats visualization.'
    )

    parser.add_argument(
        '--output-path',
        help='Output Dataframe path'
    )

    args, _ = parser.parse_known_args()

    logger.info(f"Hello world from {PACKAGE_NAME} {VERSION}")

    logger.debug("Received parameters:")
    logger.debug(f"input: {args.input_path}")
    logger.debug(f"detect mode: {args.detect_mode}")
    logger.debug(f"timestamp column: {args.timestamp_column}")
    logger.debug(f"value column: {args.value_column}")
    logger.debug(f"batch size: {args.batch_size}")
    logger.debug(f"threshold: {args.threshold}")
    logger.debug(f"sensitivity: {args.sensitivity}")
    logger.debug(f"appendMode: {args.append_mode}")
    logger.debug(f"appendMode: {args.compute_stats_in_visualization}")
    logger.debug(f"output path: {args.output_path}")

    invoke(args.input_path, args.detect_mode, args.timestamp_column, args.value_column,
        args.batch_size, args.threshold, args.sensitivity, args.append_mode,
        args.compute_stats_in_visualization, args.output_path)
