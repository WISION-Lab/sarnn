import numpy as np


def peak_accuracy(acc_snn):
    """
    Computes the peak accuracy (PA) metric.

    :param numpy.ndarray acc_snn: A time series of the SNN accuracy
    :return float: The value of the PA metric
    """

    return np.max(acc_snn)


def power_above_curve(acc_snn, acc_ann, power):
    """
    Computes the power above curve (PAC) metric.

    If acc_ann is an array, computes the relative power above curve
    (R-PAC) metric.

    :param numpy.ndarray acc_snn: A time series of the SNN accuracy
    :param acc_ann: Either a single value (for PAC) or a time series
        with length equal to acc_snn (for R-PAC) of the ANN accuracy
    :param numpy.ndarray power: A time series of the SNN power
        consumption, containing the same number of elements as acc_snn
    :return float: The value of the PAC or R-PAC metric
    """

    area = np.sum((acc_ann - acc_snn) * power)
    return area / np.max(acc_ann)


def power_to_accuracy(acc_snn, threshold, power):
    """
    Computes the power to accuracy (PTA) metric.

    If acc_snn never exceeds the threshold, the sum of the power array
    is returned.

    :param numpy.ndarray acc_snn: A time series of the SNN accuracy
    :param float threshold: The accuracy threshold
    :param numpy.ndarray power: A time series of the SNN power
        consumption, containing the same number of elements as acc_snn
    :return float: The value of the PTA metric
    """

    if not acc_snn.size == power.size:
        s = "acc_ann size {} does not equal power size {}"
        raise ValueError(s.format(acc_snn.size, power.size))
    t_first = time_to_accuracy(acc_snn, threshold)
    return np.sum(power[:t_first])


def time_above_curve(acc_snn, acc_ann):
    """
    Computes the time above curve (TAC) metric.

    If acc_ann is an array, computes the relative time above curve
    (R-TAC) metric.

    :param numpy.ndarray acc_snn: A time series of the SNN accuracy
    :param acc_ann: Either a single value (for TAC) or a time series
        with length equal to acc_snn (for R-TAC) of the ANN accuracy
    :return float: The value of the TAC or R-TAC metric
    """

    area = np.sum(acc_ann - acc_snn)
    return area / np.max(acc_ann)


def time_to_accuracy(acc_snn, threshold):
    """
    Computes the time to accuracy (TTA) metric.
    
    If acc_snn never exceeds the threshold, the length of the acc_snn
    array is returned.

    :param numpy.ndarray acc_snn: A time series of the SNN accuracy
    :param float threshold: The accuracy threshold
    :return float: The value of the TTA metric
    """

    above = np.where(acc_snn >= threshold)[0]
    if above.size == 0:
        return acc_snn.size + 1
    else:
        return above[0] + 1
