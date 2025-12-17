'''Main function for the audio event detection script'''
from audio_event_detection import aed
from audio_event_detection.classes import (
    AudioPipelineConfig,
    AudioConfig,
    FeatureConfig,
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
    ),

    output = OutputConfig(
        output_folder_path = "output/",
        verbose = True,
    ),

)


pipeline = aed.AudioPipeline(config=config)

print(pipeline)

# Pipeline start
pipeline.load_audio_file("data/277566242.wav")
pipeline.rms_detection()
