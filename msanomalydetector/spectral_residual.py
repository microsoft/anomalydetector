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

from msanomalydetector.util import AnomalyId, AnomalyScore, IsAnomaly, Value, Timestamp, EPS, average_filter
import pandas as pd
import numpy as np


class SpectralResidual:
    def __init__(self, series, threshold, mag_window, score_window):
        self.__series__ = series
        self.__threshold__ = threshold
        self.__mag_window = mag_window
        self.__score_window = score_window

    def detect(self):
        anomaly_scores = self.generate_spectral_score(series=self.__series__['value'].tolist())
        anomaly_frame = pd.DataFrame({Timestamp: self.__series__['timestamp'],
                                      Value: self.__series__['value'],
                                      AnomalyId: list(range(0, len(anomaly_scores))),
                                      AnomalyScore: anomaly_scores})
        anomaly_frame[IsAnomaly] = np.where(anomaly_frame[AnomalyScore] >= self.__threshold__, True, False)
        anomaly_frame.set_index(AnomalyId, inplace=True)

        return anomaly_frame

    def generate_spectral_score(self, series):
        extended_series = SpectralResidual.extend_series(series)
        mag = self.spectral_residual_transform(extended_series)[:len(series)]
        ave_mag = average_filter(mag, n=self.__score_window)
        ave_mag = [EPS if np.isclose(x, EPS) else x for x in ave_mag]

        return abs(mag - ave_mag) / ave_mag

    def spectral_residual_transform(self, values):
        """
        This method transform a time series into spectral residual series
        :param values: list.
            a list of float values.
        :return: mag: list.
            a list of float values as the spectral residual values
        """

        trans = np.fft.fft(values)
        mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)

        mag_log = [np.log(item) if abs(item) > EPS else 0 for item in mag]

        spectral = np.exp(mag_log - average_filter(mag_log, n=self.__mag_window))

        trans.real = [i_real * i_spectral / i_mag if abs(i_mag) > EPS else 0
                      for i_real, i_spectral, i_mag in zip(trans.real, spectral, mag)]
        trans.imag = [i_imag * i_spectral / i_mag if abs(i_mag) > EPS else 0
                      for i_imag, i_spectral, i_mag in zip(trans.imag, spectral, mag)]

        wave_r = np.fft.ifft(trans)
        mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)
        return mag

    @staticmethod
    def predict_next(values):
        """
        Predicts the next value by sum up the slope of the last value with previous values.
        Mathematically, g = 1/m * sum_{i=1}^{m} g(x_n, x_{n-i}), x_{n+1} = x_{n-m+1} + g * m,
        where g(x_i,x_j) = (x_i - x_j) / (i - j)
        :param values: list.
            a list of float numbers.
        :return : float.
            the predicted next value.
        """

        if len(values) <= 1:
            raise ValueError(f'data should contain at least 2 numbers')

        v_last = values[-1]
        n = len(values)

        slopes = [(v_last - v) / (n - 1 - i) for i, v in enumerate(values[:-1])]

        return values[1] + sum(slopes)

    @staticmethod
    def extend_series(values, extend_num=5, look_ahead=5):
        """
        extend the array data by the predicted next value
        :param values: list.
            a list of float numbers.
        :param extend_num: int, default 5.
            number of values added to the back of data.
        :param look_ahead: int, default 5.
            number of previous values used in prediction.
        :return: list.
            The result array.
        """

        if look_ahead < 1:
            raise ValueError('look_ahead must be at least 1')

        extension = [SpectralResidual.predict_next(values[-look_ahead - 2:-1])] * extend_num
        return values + extension
