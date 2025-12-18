'''Dataclasses for the configuration of the audio event detection pipeline'''
from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class AudioConfig:
    '''Audio loading parameters'''
    sr: int = 22050
    n_fft: int = 2048
    hop_length: int = 512


@dataclass
class FeatureConfig:
    '''Feature extraction parameters'''
    n_mels: int = 64
    fmin: float = 250.0
    fmax: float = 1100.0
    stft_min: float = 250.0
    stft_max: float = 1100.0

    def __post_init__(self):
        if self.fmin >= self.fmax:
            raise ValueError("fmin must be < fmax")


@dataclass
class DetectionConfig:
    '''Event detection parameters'''


@dataclass
class MergingConfig:
    '''Event merging parameters'''


@dataclass
class SegmentationConfig:
    '''Audio segmentation parameters'''


@dataclass
class OutputConfig:
    '''Program output parameters'''
    output_folder_path: Path | None = None
    verbose: bool = True

    def __post_init__(self):
        if self.output_folder_path:
            os.makedirs(self.output_folder_path, exist_ok=True)


@dataclass
class AudioPipelineConfig:
    '''Audio event detection pipeline configuration parameters'''

    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    merging: MergingConfig = field(default_factory=MergingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
