class VideoHyperParams:
    FRAME_WIDTH = 512
    FRAME_HEIGHT = 215
    NUMBER_OF_FRAMES = 215
    NUMBER_OF_CONVOLUTION_LAYERS = 3
    EMBENDING_DIM = 512
    FPS = '21.5'


class AudioHyperParams:
    SAMPLING_RATE = '22050'
    MEL_SAMPLES = 430  # temporal bins of audio representation
    SIGNAL_LENGTH = 220500  # Length in samples. It's SR (22050 samples per second) multiply by audio duration (10 seconds)
    NUMBER_OF_MEL_BANDS = 80
    FRAME_SIZE = 2048  # Frame size for STFT
    EMBENDING_DIM = 215


class TrainParams:
    LEARNING_RATE = 0.0002
    EPOCHS = 100
    BETA1 = 0.5
    ADDITIONAL_LOSS_COEFF = 100
    BATCH_SIZE = 6
