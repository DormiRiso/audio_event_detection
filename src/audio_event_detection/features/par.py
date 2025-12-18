'''Module to compute the PAR of an audio signal'''
import numpy as np


def compute_par(stft):
    '''Function to compute the PAR of an audio signal
    Args:
        stft (np.ndarray): The amplitude of the STFT of an audio signal.
    Returns:
        np.ndarray: The PAR values.
    '''

    par = np.max(stft, axis=0)/np.mean(stft, axis=0)

    return par
