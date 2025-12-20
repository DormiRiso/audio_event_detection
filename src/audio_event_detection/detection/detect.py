'''Audio event detection class'''
from audio_event_detection.detection import (
    median_based_alg,
)


class Detect:
    '''Audio event detection class'''

    def __init__(self, pipeline):
        self.pipeline = pipeline

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
        except Exception as e:
            raise RuntimeError(f'Failed to detect events by RMS: {e}') from e

        if self.pipeline.cfg.output.verbose:
            print('Audio event computed successfully\n')


    def detect_by_maxp(self):
        '''Run the median-based audio event detection algorithm using the Max Pooling'''
