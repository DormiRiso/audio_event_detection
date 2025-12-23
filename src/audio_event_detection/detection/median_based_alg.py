'''Module to apply a median-based event detection algorithm'''
import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from audio_event_detection.classes import AudioPipelineConfig
from audio_event_detection.classes import Peaks


def compute_peaks(curve, config: AudioPipelineConfig):
    '''Function to compute peaks in the input function using adaptive threshold.'''

    # Curve smoothing
    curve_smoothing_window = int(
        round(
            config.detection.curve_smoothing_window * config.audio.sr / config.audio.hop_length
        )
    )
    curve_smoothing_window = max(1, curve_smoothing_window)

    curve_smooth = uniform_filter1d(curve, size=curve_smoothing_window)

    # Median and MAD
    med = np.median(curve_smooth)
    mad = np.median(np.abs(med-curve_smooth))

    # Adaptive threshold
    threshold = med + config.detection.threshold_coefficient * mad

    # Peaks and times
    peaks, _ = find_peaks(curve_smooth, height=threshold)
    times = librosa.frames_to_time(
        peaks,
        sr=config.audio.sr,
        hop_length=config.audio.hop_length,
    )

    return Peaks(
        peak_times=times,
        threshold=threshold,
        detection_curve_smoothed=curve_smooth
    )
