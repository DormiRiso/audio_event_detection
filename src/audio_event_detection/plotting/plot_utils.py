'''Module for plotting audio event detection results.'''
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
from audio_event_detection.classes import AudioPipelineConfig


def plot_events(pipeline, config: AudioPipelineConfig, plot_save_path=None):
    """Plot spectrogram + detection curve timeline with marked events."""


    if plot_save_path is None:
        return


    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

    # STFT
    pipeline.features.compute_stft()
    stft_db = pipeline.features.stft
    stft_db = librosa.amplitude_to_db(np.abs(stft_db), ref=np.max)


    n_freqs = stft_db.shape[0]
    freqs = np.linspace(
        config.features.stft_min,
        config.features.stft_max,
        n_freqs
    )


    times_stft = librosa.frames_to_time(
        np.arange(stft_db.shape[1]), sr=config.audio.sr, hop_length=config.audio.hop_length
    )
    times_curve = librosa.frames_to_time(
        np.arange(len(pipeline.detect.events.detection_curve)),
        sr=config.audio.sr,
        hop_length=config.audio.hop_length
    )


    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Limit time axis to audio duration
    audio_duration = len(pipeline.y) / config.audio.sr
    ax1.set_xlim(0, audio_duration)
    ax2.set_xlim(0, audio_duration)

    # Plot spectrogram
    ax1.imshow(
        stft_db,
        aspect='auto',
        origin='lower',
        extent=[times_stft[0], times_stft[-1], freqs[0], freqs[-1]],
        cmap='magma'
    )
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title("Spectrogram")

    # Plot RMS
    ax2.plot(
        times_curve,
        pipeline.detect.events.detection_curve,
        label="Detection curve",
        color='blue'
    )
    ax2.axhline(
        pipeline.detect.events.threshold,
        color="green",
        linestyle="--",
        label="Threshold"
    )


    for i, (s, e) in enumerate(pipeline.detect.events.event_timestamps):
        ax2.axvspan(s, e, alpha=0.3, color='cyan')
        ax2.text((s + e) / 2, max(pipeline.detect.events.detection_curve), str(i + 1), ha='center')


    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Detection Curve Value")
    ax2.legend()


    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
