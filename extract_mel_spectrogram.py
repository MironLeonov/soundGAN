import numpy as np
import os
import librosa
import argparse
import os.path as P
from multiprocessing import Pool
from functools import partial
from glob import glob
from params import AudioHyperParams

mel_basis = librosa.filters.mel(int(AudioHyperParams.SAMPLING_RATE), n_fft=1024, fmin=125, fmax=7600, n_mels=80)


def get_spectrogram(audio_path, save_dir, length):
    wav, _ = librosa.load(audio_path, sr=None)
    y = np.zeros(length)
    if wav.shape[0] < length:
        y[:len(wav)] = wav
    else:
        y = wav[:length]
    spectrogram = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    mel_spec = np.dot(mel_basis, spectrogram)
    mel_spec = 20 * np.log10(np.maximum(1e-5, mel_spec)) - 20
    mel_spec = np.clip((mel_spec + 100) / 100, 0, 1.0)
    print(mel_spec.shape)
    mel_spec = mel_spec[:, :860]
    os.makedirs(save_dir, exist_ok=True)
    audio_name = os.path.basename(audio_path).split('.')[0]
    np.save(P.join(save_dir, audio_name + "_mel.npy"), mel_spec)
    np.save(P.join(save_dir, audio_name + "_audio.npy"), y)


if __name__ == '__main__':
    OUTPUT_DIR = 'features_data/hammer/audio_features'
    INPUT_DIR = 'converted_data/hammer/audio_10s_22050hz'

    audio_paths = glob(P.join(INPUT_DIR, "*.wav"))
    audio_paths.sort()

    with Pool(1) as p:
        p.map(partial(get_spectrogram, save_dir=OUTPUT_DIR, length=int(AudioHyperParams.SAMPLING_RATE)), audio_paths)
