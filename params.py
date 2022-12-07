class VideoHyperParams:
    FRAME_WIDTH = 224
    FRAME_HEIGHT = 224
    NUMBER_OF_FRAMES = 215
    NUMBER_OF_CONVOLUTION_LAYERS = 3
    EMBENDING_DIM = 500
    FPS = '21.5'


class AudioHyperParams:
    SAMPLING_RATE = '22050'
    MEL_SAMPLES = 430  # temporal bins of audio representation
    SIGNAL_LENGTH = 220500  # Length in samples. It's SR (22050 samples per second) multiply by audio duration (10 seconds)
    NUMBER_OF_MEL_BANDS = 80
    FRAME_SIZE = 2048  # Frame size for STFT
    EMBENDING_DIM = 500


class TrainParams:
    LEARNING_RATE = 0.0002
    EPOCHS = 1
    BETA1 = 0.5
    ADDITIONAL_LOSS_COEFF = 0.001
    BATCH_SIZE = 1
