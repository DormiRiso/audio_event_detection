'''Module to compute the STFT of an audio signal.'''
import librosa
import numpy as np
from audio_event_detection.classes import AudioPipelineConfig


def compute_stft(audio_data, config: AudioPipelineConfig):
    '''Function to compute the STFT of an audio signal.
    Args:
        audio_data (np.ndarray): The audio data.
    Returns:
        np.ndarray: The amplitude of the STFT.
    '''

    stft = librosa.stft(
        y=audio_data,
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
    )

    stft = np.abs(stft)

    if (config.features.stft_min != 0) and (config.features.stft_max != 0):

        freqs = librosa.fft_frequencies(sr=config.audio.sr, n_fft=config.audio.n_fft)

        idx_min = np.argmin(np.abs(freqs - config.features.stft_min))
        idx_max = np.argmin(np.abs(freqs - config.features.stft_max))

        if idx_min > idx_max:
            idx_min, idx_max = idx_max, idx_min

        stft = stft[idx_min:idx_max, :]

    return stft
