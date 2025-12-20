'''Module to compute the rolling median on a curve'''
import numpy as np


def rolling_median_mad(x: np.ndarray, window: int):
    """
    Calcola mediana e MAD su finestra mobile.
    """
    half = window // 2
    med = np.zeros_like(x)
    mad = np.zeros_like(x)

    for i in range(len(x)):
        start = max(0, i - half)
        end = min(len(x), i + half)

        segment = x[start:end]
        m = np.median(segment)
        med[i] = m
        mad[i] = np.median(np.abs(segment - m))

    return med, mad
