'''Audio event detection pipeline class'''
from dataclasses import dataclass
import os
from pathlib import Path

from audio_event_detection import (
    load,
    filter as flt,
    rms,
)


@dataclass
class AudioPipelineConfig:
    '''Audio event detection pipeline configuration parameters'''

    # Audio parameters
    sr: int = 22050 #sample rate
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 64

    # Filter parameters
    fmin: float = 250.0 #[Hz]
    fmax: float = 1100.0 #[Hz]

    # Detection parameters

    # Merging parameters

    # Segmentation parameters

    # OS parameters
    output_folder_path: Path = None
    verbose: bool = True

    def __post_init__(self):
        '''Parameters validation'''

        if self.sr <= 0:
            raise ValueError("Sample rate must be positive")
        if self.fmin >= self.fmax:
            raise ValueError("fmin must be less than fmax")
        if self.output_folder_path:
            os.makedirs(self.output_folder_path, exist_ok=True)


class AudioPipeline:
    '''Audio event detection pipeline class'''


    def __init__(self, config: AudioPipelineConfig):
        self.cfg = config
        self.y = None

        # Cache
        self._filtered = False
        self._mel = None
        self._stft = None
        self._rms = None
        self._maxp = None
        self._par = None
        self._events = None


    def __repr__(self):
        lines = ["AudioPipeline configuration:"]
        for k, v in vars(self.cfg).items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines) + "\n"


    def load_audio_file(self, file_path: str):
        '''Function to load a audio file to be processed'''
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Audio file not found: {file_path}')

        try:
            self.y = load.load_audio(file_path=file_path, sr=self.cfg.sr, mono=True)
        except Exception as e:
            raise RuntimeError(f'Failed to load audio file {file_path}: {e}') from e

        if self.cfg.verbose:
            print(f'Audio file: {file_path} loaded successfully\n')


    def _filter_audio_file(self):
        '''Function to band-filter a audio file'''
        if self.y is None:
            raise RuntimeError('Error occured while filtering audio file: no audio file loaded')

        try:
            self.y = flt.apply_filters(
                data=self.y,
                sr=self.cfg.sr,
                highpass_cutoff=self.cfg.fmin,
                lowpass_cutoff=self.cfg.fmax
            )
            self._filtered = True
        except Exception as e:
            raise RuntimeError(f'Failed to filter audio file: {e}') from e

        if self.cfg.verbose:
            print('Audio file filtered successfully\n')


    def _compute_rms(self):
        '''Function to compute the RMS of the audio file'''
        if self._rms is not None:
            pass
        if not self._filtered:
            AudioPipeline._filter_audio_file(self)

        try:
            self._rms = rms.compute_rms(self.y)
        except Exception as e:
            raise RuntimeError(f'Failed to compute the RMS: {e}') from e

        if self.cfg.verbose:
            print('RMS computed successfully\n')


    def rms_detection(self):
        '''Run audio event detection algorithm based on RMS value'''
        if self._rms is None:
            AudioPipeline._compute_rms(self)
        print("pipeline rms")
