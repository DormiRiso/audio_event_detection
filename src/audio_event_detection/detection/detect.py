'''Audio event detection class'''
from dataclasses import dataclass
from audio_event_detection.detection import (
    median_based_alg,
)
from audio_event_detection.merging.merge import merge_peaks_into_events


@dataclass
class Events:
    '''Dataclass to store detected events'''
    event_timestamps: list
    detection_curve: list
    threshold: float


class Detect:
    '''Audio event detection class'''

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.events: Events = None

        # Cache
        self.rms_detection = None
        self.maxp_detection = None
        self.par_detection = None


    def detect_by_rms(self):
        '''Run the median-based audio event detection algorithm using the rms'''
        if self.pipeline.y is None:
            raise RuntimeError('Error occured while detecting events: no audio file loaded')
        if self.rms_detection is not None:
            pass

        if self.pipeline.features.rms is None:
            self.pipeline.features.compute_rms()

        try:
            self.rms_detection = median_based_alg.compute_peaks(
                curve=self.pipeline.features.rms,
                config=self.pipeline.cfg
            )
            self.events = Events(
                event_timestamps=merge_peaks_into_events(
                    peak_times=self.rms_detection[0],
                    config=self.pipeline.cfg
                ),
                detection_curve=self.pipeline.features.rms,
                threshold=self.rms_detection[1]
            )
        except Exception as e:
            raise RuntimeError(f'Failed to detect events by RMS: {e}') from e

        if self.pipeline.cfg.output.verbose:
            print('Audio event computed successfully\n')


    def detect_by_maxp(self):
        '''Run the median-based audio event detection algorithm using the Max Pooling'''
