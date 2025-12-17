'''Module to compute and visualize the Root Mean Square (RMS) energy of an audio signal.'''
import librosa


def compute_rms(audio_data):
    '''Function to compute the Root Mean Square (RMS) energy of an audio signal.
    Args:
        audio_data (np.ndarray): The audio data.
        frame_size (int): The size of each frame for RMS calculation.
        hop_length (int): The hop length between frames.
    Returns:
        np.ndarray: The RMS energy values.
    '''

    rms = librosa.feature.rms(y=audio_data)[0]

    return rms
