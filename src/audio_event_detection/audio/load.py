'''Module for loading audio files for AED tests.'''
import librosa


def load_audio(file_path, sr=22050, mono=True):
    """Load an audio file.

    Parameters:
    - file_path: str, path to the audio file
    - sr: int, target sampling rate
    - mono: bool, whether to convert the signal to mono

    Returns:
    - y: np.ndarray, audio time series
    """

    y, _ = librosa.load(file_path, sr=sr, mono=mono)

    return y
