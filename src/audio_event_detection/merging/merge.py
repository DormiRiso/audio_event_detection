'''Module for merging detected peaks into audio events.'''
from audio_event_detection.classes import AudioPipelineConfig


def merge_peaks_into_events(peak_times: list, config: AudioPipelineConfig):
    '''Function to merge peaks into events based on time window and minimum duration.'''

    if len(peak_times) == 0:
        return []

    events = []
    start = peak_times[0]
    end = peak_times[0]

    for t in peak_times[1:]:
        if t - end < config.merging.merging_time_window:
            end = t
        else:
            if (end - start) >= config.merging.minimum_event_duration:
                events.append(
                    (start - config.merging.head_tail_extension,
                     end + config.merging.head_tail_extension)
                )
            start = t
            end = t

    if (end - start) >= config.merging.minimum_event_duration:
        events.append(
            (start - config.merging.head_tail_extension,
             end + config.merging.head_tail_extension)
        )

    return events
