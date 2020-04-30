import unittest
import numpy as np
from msanomalydetector import boundary_utils


class TestBoundaryUnit(unittest.TestCase):
    def test_calculate_boundary_unit(self):
        data = [139809.0, 139706.0, 140562.0, 140534.0, 140568.0, 139934.0, 139392.0, 141714.0, 144167.0, 147127.0,
                147450.0, 147991.0, 151621.0, 154912.0, 158443.0, 160899.0, 164170.0, 164339.0, 165780.0, 167373.0,
                167654.0, 168863.0, 169472.0, 169830.0, 169632.0, 169028.0, 165843.0, 162517.0, 159335.0, 156503.0,
                151731.0, 151612.0, 151911.0, 157120.0, 157027.0, 159949.0, 160263.0, 160073.0, 160001.0, 159721.0,
                160138.0, 160292.0, 160280.0, 159822.0, 159482.0, 159384.0, 159038.0, 158901.0, 158899.0, 156036.0]

        is_anomaly = [False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                      False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                      False, False, True, True, True, False, False, False, False, False, False, False, False, False, False,
                      False, False, False, False, False, False, False]

        expected_output = \
            [148560.58510638, 148567.58510638, 148574.58510638, 148576.08510638, 148577.58510638, 148864.08510638,
                149150.58510638, 149763.83510638, 150377.08510638, 151857.08510638, 152018.58510638, 152289.08510638,
                154104.08510638, 155749.58510638, 157515.08510638, 158743.08510638, 160378.58510638, 160463.08510638,
                161183.58510638, 161183.58510638, 161183.58510638, 161183.58510638, 161183.58510638, 161183.58510638,
                161183.58510638, 161183.58510638, 161183.58510638, 159552.08510638, 158425.08510638, 158330.08510638,
                158294.08510638, 158268.08510638, 158268.08510638, 158268.08510638, 158268.08510638, 158204.58510638,
                158154.08510638, 158154.08510638, 158154.08510638, 158154.08510638, 158154.08510638, 158154.08510638,
                158179.33510638, 158204.58510638, 158179.33510638, 158154.08510638, 158094.33510638, 158034.58510638,
                158010.08510638, 157985.58510638]

        actual_output = boundary_utils.calculate_boundary_unit_entire(np.asarray(data, dtype=float), is_anomaly)
        for e, v in zip(expected_output, actual_output):
            self.assertAlmostEqual(e, v)

        expected_last_unit = 156748.27551020408
        actual_last_unit = boundary_utils.calculate_boundary_unit_last(np.asarray(data, dtype=float))
        self.assertAlmostEqual(expected_last_unit, actual_last_unit)

    def test_calculate_boundary_unit_negative(self):
        data = [-21901.0, -31123.0, -33203.0, -33236.0, -54681.0, -112808.0, -5368.0, -40021.0, -35.0, -72593.0,
                -30880.0, -34597.0, -6210.0, -5508.0, -28892.0, -41091.0, -34916.0, -31941.0, -31084.0, -7379.0,
                -4883.0, -32563.0, -29919.0, -33599.0, -33019.0, -35218.0, -9520.0, -4454.0, -39660.0, -29634.0,
                -35751.0, -39912.0, -46940.0, -28969.0, -20196.0, -57031.0, -45264.0, -44059.0, -29180.0, -34642.0,
                -11041.0, -10455.0, -40181.0, -43345.0, -37045.0, -33232.0, -37800.0, -9240.0, -12108.0, -34654.0]

        is_anomaly = [False, False, False, False, False, True, False, False, False, True, False, False, False, False, False,
                False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                False, False, False, False, False]

        expected_output = [
            33250.48958333333, 33258.73958333333, 33250.48958333333, 33258.73958333333, 33250.48958333333,
            32730.489583333332, 32210.489583333332, 32730.489583333332, 33250.48958333333, 33250.48958333333,
            33250.48958333333, 32619.489583333332, 32190.989583333332, 32190.989583333332, 32088.989583333332,
            32190.989583333332, 32190.989583333332, 32619.489583333332, 32190.989583333332, 32190.989583333332,
            32190.989583333332, 32190.989583333332, 32619.489583333332, 32930.48958333333, 32930.48958333333,
            32619.489583333332, 32190.989583333332, 32930.48958333333, 33158.48958333333, 33448.48958333333,
            33448.48958333333, 33969.98958333333, 33969.98958333333, 33969.98958333333, 33969.98958333333,
            34524.48958333333, 35171.48958333333, 34524.48958333333, 35171.48958333333, 35171.48958333333,
            33969.98958333333, 33969.98958333333, 33972.98958333333, 33975.98958333333, 33972.98958333333,
            33969.98958333333, 33617.48958333333, 33969.98958333333, 33620.48958333333, 33975.98958333333]

        actual_output = boundary_utils.calculate_boundary_unit_entire(np.asarray(data), is_anomaly)
        for e, v in zip(expected_output, actual_output):
            self.assertAlmostEqual(e, v)

        expected_last_unit = 33197.17346938775
        actual_last_unit = boundary_utils.calculate_boundary_unit_last(np.asarray(data))
        self.assertAlmostEqual(expected_last_unit, actual_last_unit)

    def test_calculate_margin(self):
        self.assertAlmostEqual(boundary_utils.calculate_margin(10, 0), 1843316.2871148242)
        self.assertAlmostEqual(boundary_utils.calculate_margin(10, 5), 502228.4038287002)
        self.assertAlmostEqual(boundary_utils.calculate_margin(10, 25), 3359.7473532360186)
        self.assertAlmostEqual(boundary_utils.calculate_margin(10, 95), 0.0014700521929794912)
        self.assertAlmostEqual(boundary_utils.calculate_margin(10, 99), 0.00016994687082728675)
        self.assertAlmostEqual(boundary_utils.calculate_margin(10, 100), 0.0)
        self.assertAlmostEqual(boundary_utils.calculate_margin(345969.3476, 79.7333448252325), 3762.3800000299298)

    def test_calculate_anomaly_score(self):
        self.assertAlmostEqual(boundary_utils.calculate_anomaly_score(10, 15, 5, False), 0)
        self.assertAlmostEqual(boundary_utils.calculate_anomaly_score(10, 15, 5, True), 0.5)
        self.assertAlmostEqual(boundary_utils.calculate_anomaly_score(10+1e-5, 10, 1, True), 0.005884191895350754)
        self.assertAlmostEqual(boundary_utils.calculate_anomaly_score(10+1e-7, 10, 1, True), 5.884191859812512e-05)


if __name__ == '__main__':
    unittest.main()
