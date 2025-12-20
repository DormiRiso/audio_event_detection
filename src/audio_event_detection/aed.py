'''Audio event detection pipeline class'''
import os

from audio_event_detection.classes import AudioPipelineConfig
from audio_event_detection import (
    load,
)
from audio_event_detection.features.features import (
    Feature,
)
from audio_event_detection.detection.detect import (
    Detect,
)


class AudioPipeline:
    '''Audio event detection pipeline class'''


    def __init__(self, config: AudioPipelineConfig):
        self.cfg = config
        self.y = None

        # Initialize feature class
        self.features = Feature(self)
        self.detect = Detect(self)

        # Cache
        self._events = None


    def __repr__(self):
        def format_block(name, cfg, indent=2):
            lines = [f"{' ' * indent}{name}:"]
            for k, v in vars(cfg).items():
                lines.append(f"{' ' * (indent + 2)}{k}: {v}")
            return lines

        lines = ["AudioPipeline configuration:"]
        for section, cfg in vars(self.cfg).items():
            lines.extend(format_block(section, cfg))

        return "\n".join(lines) + "\n"


    def load_audio_file(self, file_path: str):
        '''Function to load a audio file to be processed'''
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Audio file not found: {file_path}')

        try:
            self.y = load.load_audio(file_path=file_path, sr=self.cfg.audio.sr, mono=True)
        except Exception as e:
            raise RuntimeError(f'Failed to load audio file {file_path}: {e}') from e

        if self.cfg.output.verbose:
            print(f'Audio file: {file_path} loaded successfully\n')
