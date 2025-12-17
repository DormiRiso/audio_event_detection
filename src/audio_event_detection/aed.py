'''Audio event detection pipeline class'''
from dataclasses import dataclass
import os

from audio_event_detection import (
    load
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
    output_folder_path: os.path = None
    verbose: bool = True


class AudioPipeline:
    '''Audio event detection pipeline class'''

    def __init__(self, config: AudioPipelineConfig):
        self.cfg = config
        self.y = None

        # Cache
        self._mel = None
        self._stft = None
        self._rms = None
        self._maxp = None
        self._par = None


    def __repr__(self):
        lines = ["AudioPipeline configuration:"]
        for k, v in vars(self.cfg).items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


    def load_audio_file(self, file_path: str):
        '''Function to load a audio file to be processed'''
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            self.y = load.load_audio(file_path=file_path, sr=self.cfg.sr, mono=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {file_path}: {e}") from e

        if self.cfg.verbose:
            print(f'\nAudio file: {file_path} loaded successfully\n')
