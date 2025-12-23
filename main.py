'''Main function for the audio event detection script'''
from audio_event_detection import aed
from audio_event_detection.classes import (
    AudioPipelineConfig,
    AudioConfig,
    FeatureConfig,
    DetectionConfig,
    MergingConfig,
    OutputConfig,
)


config = AudioPipelineConfig(

    audio = AudioConfig(
        sr = 22050,
        n_fft = 2048,
        hop_length = 512,
    ),

    features = FeatureConfig(
        n_mels = 64,
        fmin = 250,
        fmax = 1100,
        stft_min = 250,
        stft_max = 1100,
    ),

    detection = DetectionConfig(
        threshold_coefficient = 0.75,
        curve_smoothing_window = 1.0,
    ),

    merging = MergingConfig(
        merging_time_window = 5.0,
        minimum_event_duration = 1.0,
        head_tail_extension = 0.5,
    ),

    output = OutputConfig(
        output_folder_path = "output/",
        verbose = True,
    ),

)


pipeline = aed.AudioPipeline(config=config)

print(pipeline)

# Folder system setup
input_file = "data/277566242.wav"
output_folder = "output/" + input_file.split('/')[-1].split('.')[0] + '/'
plot_save_path = output_folder + "plot.jpeg"

# Pipeline start
pipeline.load_audio_file("data/277566242.wav")
pipeline.detect.detect_by_rms()
pipeline.plot.plot_events(plot_save_path=plot_save_path)
