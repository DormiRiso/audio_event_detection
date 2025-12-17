'''Audio features extraction class'''
from audio_event_detection import (
    filter as flt,
    rms,
)


class Feature:
    '''Audio feature extractor class'''

    def __init__(self, pipeline):
        self.pipeline = pipeline

        # Cache
        self._filtered = False
        self.mel = None
        self.stft = None
        self.rms = None
        self.maxp = None
        self.par = None


    def _filter_audio_file(self):
        '''Function to band-filter a audio file'''
        if self.pipeline.y is None:
            raise RuntimeError('Error occured while filtering audio file: no audio file loaded')

        try:
            self.pipeline.y = flt.apply_filters(
                data=self.pipeline.y,
                sr=self.pipeline.cfg.audio.sr,
                highpass_cutoff=self.pipeline.cfg.features.fmin,
                lowpass_cutoff=self.pipeline.cfg.features.fmax
            )
            self._filtered = True
        except Exception as e:
            raise RuntimeError(f'Failed to filter audio file: {e}') from e

        if self.pipeline.cfg.output.verbose:
            print('Audio file filtered successfully\n')


    def compute_rms(self):
        '''Function to compute the RMS of the audio file'''
        if self.rms is not None:
            pass
        if not self._filtered:
            Feature._filter_audio_file(self)

        try:
            self.rms = rms.compute_rms(self.pipeline.y)
        except Exception as e:
            raise RuntimeError(f'Failed to compute the RMS: {e}') from e

        if self.pipeline.cfg.output.verbose:
            print('RMS computed successfully\n')


    def plot(self):
        '''Function to plot the RMS of the audio file'''
        if self.rms is None:
            Feature.compute_rms(self)
