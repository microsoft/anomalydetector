import numpy as np

IsAnomaly = "isAnomaly"
AnomalyId = "id"
AnomalyScore = "score"
Value = "value"
Timestamp = "timestamp"

MAX_RATIO = 0.25
EPS = 1e-8
THRESHOLD = 3
MAG_WINDOW = 3
SCORE_WINDOW = 100


def average_filter(values, n=3):
    """
    Calculate the sliding window average for the give time series.
    Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
    :param values: list.
        a list of float numbers
    :param n: int, default 3.
        window size.
    :return res: list.
        a list of value after the average_filter process.
    """

    if n >= len(values):
        n = len(values)

    res = np.cumsum(values, dtype=float)
    res[n:] = res[n:] - res[:-n]
    res[n:] = res[n:] / n

    for i in range(1, n):
        res[i] /= (i + 1)

    return res
