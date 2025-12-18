'''Module to compute the Max Pooling of an audio signal'''


def compute_maxp(stft):
    '''Function to compute the Max Pooling of an audio signal
    Args:
        stft (np.ndarray): The amplitude of the STFT of an audio signal.
    Returns:
        np.ndarray: The Max Pooling values.
    '''

    maxp = stft.max(axis=0)

    return maxp
