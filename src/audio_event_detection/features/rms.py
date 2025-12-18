'''Module to compute the Root Mean Square (RMS) energy of an audio signal.'''
import librosa
from audio_event_detection.classes import AudioPipelineConfig


def compute_rms(audio_data, config: AudioPipelineConfig):
    '''Function to compute the Root Mean Square (RMS) energy of an audio signal.
    Args:
        audio_data (np.ndarray): The audio data.
    Returns:
        np.ndarray: The RMS energy values.
    '''

    rms = librosa.feature.rms(
        y=audio_data,
        frame_length=config.audio.n_fft,
        hop_length=config.audio.hop_length
    )[0]

    return rms
