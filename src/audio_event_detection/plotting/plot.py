'''Plotting functions for audio event detection'''
from audio_event_detection.plotting import (
    plot_utils,
)


class Plot:
    '''Plotting functions for audio event detection'''

    def __init__(self, pipeline):
        self.pipeline = pipeline


    def plot_events(self, plot_save_path=None):
        '''Function to plot detected audio events'''
        if self.pipeline.y is None:
            raise RuntimeError('Error occured while plotting events: no audio file loaded')
        if self.pipeline.detect.rms_detection is None \
        and self.pipeline.detect.maxp_detection is None \
        and self.pipeline.detect.par_detection is None:
            raise RuntimeError('Error occured while plotting events: no detection algorithm run')
        if self.pipeline.detect.events is None:
            raise RuntimeError('Error occured while plotting events: no events detected')

        if plot_save_path is None:
            raise RuntimeError('Plot save path must be provided to save the plot')

        try:
            plot_utils.plot_events(
                pipeline=self.pipeline,
                config=self.pipeline.cfg,
                plot_save_path=plot_save_path,
            )
        except Exception as e:
            raise RuntimeError(f'Failed to plot audio events: {e}') from e

        if self.pipeline.cfg.output.verbose:
            print(f'Audio events plot saved successfully at: {plot_save_path}\n')
