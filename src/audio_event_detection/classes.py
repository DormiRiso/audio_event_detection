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
    threshold_coefficient: float = 1.0
    curve_smoothing_window: float = 10.0


@dataclass
class MergingConfig:
    '''Event merging parameters'''
    merging_time_window: float = 1.0
    minimum_event_duration: float = 0.5
    head_tail_extension: float = 1.0


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


@dataclass
class Peaks:
    '''Dataclass to store detected peaks'''
    peak_times: list
    threshold: float
    detection_curve_smoothed: list


@dataclass
class Events:
    '''Dataclass to store detected events'''
    event_timestamps: list
    detection_curve: list
    threshold: float
    detection_curve_smoothed: list
