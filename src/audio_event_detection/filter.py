'''Module for applying high-pass and low-pass filters to audio data.'''
from scipy.signal import butter, filtfilt
from audio_event_detection.classes import AudioPipelineConfig


def highpass_filter(data, sr, cutoff=250):
    '''Function to apply a high-pass filter to the audio data.
    Args:
        data (np.ndarray): The audio data to be filtered.
        sr (int): The sample rate of the audio data.
        cutoff (float): The cutoff frequency for the high-pass filter in Hz.
    Returns:
        np.ndarray: The filtered audio data.
    '''

    b, a = butter(8, cutoff / (sr / 2), btype='high')
    return filtfilt(b, a, data)


def lowpass_filter(data, sr, cutoff=1100):
    '''Function to apply a low-pass filter to the audio data.
    Args:
        data (np.ndarray): The audio data to be filtered.
        sr (int): The sample rate of the audio data.
        cutoff (float): The cutoff frequency for the low-pass filter in Hz.
    Returns:
        np.ndarray: The filtered audio data.
    '''

    b, a = butter(8, cutoff / (sr / 2), btype='low')
    return filtfilt(b, a, data)


def apply_filters(data, config: AudioPipelineConfig):
    '''Function to apply both high-pass and low-pass filters to the audio data.
    Args:
        data (np.ndarray): The audio data to be filtered.
    Returns:
        np.ndarray: The filtered audio data.
    '''

    filtered_data = highpass_filter(
        data=data,
        sr=config.audio.sr,
        cutoff=config.features.fmin
    )
    filtered_data = lowpass_filter(
        data=filtered_data,
        sr=config.audio.sr,
        cutoff=config.features.fmax
    )

    return filtered_data
