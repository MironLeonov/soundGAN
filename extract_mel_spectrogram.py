import numpy as np
import os
import librosa
import os.path as P
from multiprocessing import Pool
from functools import partial
from glob import glob
from params import AudioHyperParams


def get_spectrogram(audio_path, save_dir):
    wav, _ = librosa.load(audio_path, sr=None)
    length = AudioHyperParams.SIGNAL_LENGTH
    y = np.zeros(length)
    if wav.shape[0] < length:
        y[:len(wav)] = wav
    else:
        y = wav[:length]

    mel_spectrogram = librosa.feature.melspectrogram(y, sr=int(AudioHyperParams.SAMPLING_RATE),
                                                     n_fft=int(AudioHyperParams.FRAME_SIZE),
                                                     hop_length=int(AudioHyperParams.FRAME_SIZE / 4),
                                                     n_mels=int(AudioHyperParams.NUMBER_OF_MEL_BANDS))

    print(mel_spectrogram.shape)

    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    finally_mel_spectrogram = log_mel_spectrogram[:, :AudioHyperParams.MEL_SAMPLES]
    audio_name = os.path.basename(audio_path).split('.')[0]
    os.makedirs(P.join(save_dir, audio_name), exist_ok=True)
    np.save(P.join(save_dir, audio_name, "mel.npy"), finally_mel_spectrogram)


if __name__ == '__main__':
    OUTPUT_DIR = 'features_data/hammer/audio_features'
    INPUT_DIR = 'converted_data/hammer/audio_10s_22050hz'

    audio_paths = glob(P.join(INPUT_DIR, "*.wav"))
    audio_paths.sort()

    with Pool(1) as p:
        p.map(partial(get_spectrogram, save_dir=OUTPUT_DIR), audio_paths)
