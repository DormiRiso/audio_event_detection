'''Audio features extraction class'''
from audio_event_detection import (
    filter as flt,
)
from audio_event_detection.features import (
    rms,
    stft,
    par,
    maxp,
)


class Feature:
    '''Audio feature extractor class'''

    def __init__(self, pipeline):
        self.pipeline = pipeline

        # Cache
        self._filtered = False
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
                config=self.pipeline.cfg,
            )
            self._filtered = True
        except Exception as e:
            raise RuntimeError(f'Failed to filter audio file: {e}') from e

        if self.pipeline.cfg.output.verbose:
            print('Audio file filtered successfully\n')


    def compute_rms(self):
        '''Function to compute the RMS of the audio file'''
        if self.pipeline.y is None:
            raise RuntimeError('Error occured while computing RMS: no audio file loaded')
        if self.rms is not None:
            pass
        if not self._filtered:
            Feature._filter_audio_file(self)

        try:
            self.rms = rms.compute_rms(self.pipeline.y, self.pipeline.cfg)
        except Exception as e:
            raise RuntimeError(f'Failed to compute the RMS: {e}') from e

        if self.pipeline.cfg.output.verbose:
            print('RMS computed successfully\n')


    def compute_stft(self):
        '''Function to compute the STFT of the audio file'''
        if self.pipeline.y is None:
            raise RuntimeError('Error occured while computing STFT: no audio file loaded')
        if self.stft is not None:
            pass

        try:
            self.stft = stft.compute_stft(self.pipeline.y, self.pipeline.cfg)
        except Exception as e:
            raise RuntimeError(f'Failed to compute the STFT: {e}') from e

        if self.pipeline.cfg.output.verbose:
            print('STFT computed successfully\n')


    def compute_par(self):
        '''Function to compute the PAR of the audio file'''
        if self.pipeline.y is None:
            raise RuntimeError('Error occured while computing PAR: no audio file loaded')
        if self.par is not None:
            pass

        if self.stft is None:
            Feature.compute_stft(self)

        try:
            self.par = par.compute_par(self.stft)
        except Exception as e:
            raise RuntimeError(f'Failed to compute the PAR: {e}') from e

        if self.pipeline.cfg.output.verbose:
            print('PAR computed successfully\n')


    def compute_maxp(self):
        '''Function to compute the Max Pooling of the audio file'''
        if self.pipeline.y is None:
            raise RuntimeError('Error occured while computing Max Pooling: no audio file loaded')
        if self.maxp is not None:
            pass

        if self.stft is None:
            Feature.compute_stft(self)

        try:
            self.maxp = maxp.compute_maxp(self.stft)
        except Exception as e:
            raise RuntimeError(f'Failed to compute the Max Pooling: {e}') from e

        if self.pipeline.cfg.output.verbose:
            print('Max Pooling computed successfully\n')
