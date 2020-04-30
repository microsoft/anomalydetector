import bisect
import numpy as np
from msanomalydetector._anomaly_kernel_cython import median_filter


# pseudo - code to generate the factors.
# factors = [1]
# for i in range(50):
#     if i < 40:
#         factors.append(factors[-1] / (1.15 + 0.001 * i))
#     else:
#         factors.append(factors[-1] / (1.25 + 0.01 * i))
# for i in range(50):
#     factors.insert(0, factors[0] * (1.25 + 0.001 * i))

factors = [
    184331.62871148242, 141902.71648305038, 109324.12672037778, 84289.9974713784, 65038.57829581667, 50222.84038287002,
    38812.08684920403, 30017.081863266845, 23233.035497884553, 17996.15452973242, 13950.50738738947, 10822.736530170265,
    8402.745753237783, 6528.939979205737, 5076.93622022219, 3950.92312857758, 3077.042935029268, 2398.318733460069,
    1870.7634426365591, 1460.393007522685, 1140.9320371270976, 892.0500681212648, 698.0047481387048, 546.5972968979678,
    428.36778753759233, 335.97473532360186, 263.71643275007995, 207.16137686573444, 162.8627176617409, 128.13746472206208,
    100.8956415134347, 79.50799173635517, 62.70346351447568, 49.48971074544253, 39.09139869308257, 30.90229145698227,
    24.448015393182175, 19.35709849024717, 15.338429865489042, 12.163703303322, 9.653732780414286, 7.667778221139226,
    6.095213212352326, 4.8490160798347866, 3.8606815922251485, 3.076240312529999, 2.4531421949999994, 1.9578149999999996,
    1.5637499999999998, 1.25, 1.0, 0.8695652173913044, 0.7554867223208555, 0.655804446459076, 0.5687809596349316,
    0.4928777813127657, 0.4267340097946024, 0.36914706729636887, 0.3190553736355825, 0.27552277516026125, 0.23772456873189068,
    0.20493497304473338, 0.17651591132190647, 0.1519069804835684, 0.13061649224726435, 0.11221348131208278, 0.09632058481723846,
    0.08260770567516164, 0.0707863801843716, 0.06060477755511267, 0.051843265658779024, 0.0443104834690419, 0.03783986632710667,
    0.03228657536442549, 0.027524787181948417, 0.02344530424356765, 0.019953450420057577, 0.01696721974494692, 0.014415649740821513,
    0.012237393667929978, 0.010379468759906684, 0.008796159966022614, 0.0074480609365136455, 0.006301235986898177,
    0.00532648857725966, 0.004498723460523362, 0.0037963911059268884, 0.0032010043051660104, 0.002696718032995797,
    0.0022699646742388863, 0.0019091376570554135, 0.0011570531254881296, 0.000697019955113331, 0.00041737721863073713,
    0.000248438820613534, 0.00014700521929794912, 8.647365841055832e-05, 5.056939088336744e-05, 2.9400808653120604e-05,
    1.6994687082728674e-05, 9.767061541798089e-06
]


def calculate_boundary_unit_last(data):
    if len(data) == 0:
        return 0

    calculation_size = len(data) - 1
    window = int(min(calculation_size // 3, 512))
    trends = np.abs(np.asarray(median_filter(data[:calculation_size], window, need_two_end=True), dtype=float))

    unit = max(np.mean(trends), 1.0)

    if not np.isfinite(unit):
        raise Exception('Not finite unit value')

    return unit


def calculate_boundary_unit_entire(data, is_anomaly):
    if len(data) == 0:
        return []

    window = int(min(len(data)//3, 512))
    trend_fraction = 0.5
    trends = np.abs(np.asarray(median_filter(data, window, need_two_end=True), dtype=float))
    valid_trend = [t for a, t in zip(is_anomaly, trends) if not a]

    if len(valid_trend) > 0:
        average_part = np.mean(valid_trend)
        units = trend_fraction * trends + average_part * (1 - trend_fraction)
    else:
        units = trends

    if not np.all(np.isfinite(units)):
        raise Exception('Not finite unit values')

    units = np.clip(units, 1.0, max(np.max(units), 1.0))

    return units


def calculate_margin(unit, sensitivity):

    def calculate_margin_core(unit, sensitivity):
        lb = int(sensitivity)
        # if lb == sensitivity:
        #     return unit * factors[lb]

        return (factors[lb + 1] + (factors[lb] - factors[lb + 1]) * (1 - sensitivity + lb)) * unit

    if 0 > sensitivity or sensitivity > 100:
        raise Exception('sensitivity should be integer in [0, 100]')

    if unit <= 0:
        raise Exception('unit should be a positive number')

    if sensitivity == 100:
        return 0.0

    return calculate_margin_core(unit, sensitivity)


def calculate_anomaly_score(value, expected_value, unit, is_anomaly):
    if not is_anomaly:
        return 0.0

    distance = np.abs(expected_value - value)
    margins = [calculate_margin(unit, i) for i in range(101)][::-1]
    lb = bisect.bisect_left(margins, distance)

    if lb == 0:
        return 0
    elif lb >= 100:
        return 1.0
    else:
        a, b = margins[lb-1], margins[lb]
        score = lb - 1 + (distance - a) / (b - a)

    return score / 100.0


def calculate_anomaly_scores(values, expected_values, units, is_anomaly):
    scores = [calculate_anomaly_score(value, exp, unit, anomaly)
              for value, exp, unit, anomaly in zip(values, expected_values, units, is_anomaly)]
    return scores
