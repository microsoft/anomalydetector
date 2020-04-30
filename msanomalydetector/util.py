"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from enum import Enum
import numpy as np

IsAnomaly = "isAnomaly"
AnomalyId = "id"
AnomalyScore = "score"
Value = "value"
Timestamp = "timestamp"
Mag = "mag"
ExpectedValue = "expectedValue"
UpperBoundary = "upperBoundary"
LowerBoundary = "lowerBoundary"

MAX_RATIO = 0.25
EPS = 1e-8
THRESHOLD = 0.3
MAG_WINDOW = 3
SCORE_WINDOW = 40


class DetectMode(Enum):
    anomaly_only = 'AnomalyOnly'
    anomaly_and_margin = 'AnomalyAndMargin'


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


def leastsq(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(np.multiply(x, x))
    sum_xy = np.sum(np.multiply(x, y))
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    b = (sum_xx * sum_y - sum_x * sum_xy) / (n * sum_xx - sum_x * sum_x)
    return a, b


def deanomaly_entire(values, entire_anomalies):
    deanomaly_data = np.copy(values)
    min_points_to_fit = 4
    for idx in entire_anomalies:
        step = 1
        start = max(idx - step, 0)
        end = min(len(values) - 1, idx + step)
        fit_values = [(i, values[i]) for i in range(start, end+1) if i not in entire_anomalies]
        while len(fit_values) < min_points_to_fit and (start > 0 or end < len(values)-1):
            step = step + 2
            start = max(idx - step, 0)
            end = min(len(values) - 1, idx + step)
            fit_values = [(i, values[i]) for i in range(start, end+1) if i not in entire_anomalies]

        if len(fit_values) > 1:
            x, y = tuple(zip(*fit_values))
            a, b = leastsq(x, y)
            deanomaly_data[idx] = a * idx + b

    return deanomaly_data
