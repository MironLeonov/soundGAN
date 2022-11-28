import os
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import torch
import numpy as np

from params import VideoHyperParams, AudioHyperParams


class VideoDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.transform = transform
        self.frames_dir = root_dir + '/video_features'
        self.videos = os.listdir(self.frames_dir)

        self.audio_dir = root_dir + '/audio_features'
        self.audios = os.listdir(self.audio_dir)

        assert len(self.videos) == len(self.audios)

    def __len__(self):
        return len(self.videos)

    def get_frames(self, idx):
        video_dir = self.frames_dir + '/' + self.videos[idx]

        images_paths = glob(video_dir + "/*.jpg")

        images_paths = images_paths[:VideoHyperParams.NUMBER_OF_FRAMES]

        frames = []
        for img_path in images_paths:
            frame = Image.open(img_path)
            frames.append(frame)

        frames_tr = []
        for frame in frames:
            frame = self.transform(frame)
            frames_tr.append(frame)

        if len(frames_tr) > 0:
            frames_tr = torch.stack(frames_tr)

        return frames_tr

    def get_mel(self, idx):

        audio_dir = self.audio_dir + '/' + self.videos[idx]

        mel_path = glob(audio_dir + "/*.npy")

        assert len(mel_path) == 1

        mel_spec = np.load(mel_path[0])

        if mel_spec.shape[1] < AudioHyperParams.MEL_SAMPLES:
            mel_spec_padded = np.zeros((mel_spec.shape[0], AudioHyperParams.MEL_SAMPLES))
            mel_spec_padded[:, 0:mel_spec.shape[1]] = mel_spec
        else:
            mel_spec_padded = mel_spec[:, 0:AudioHyperParams.MEL_SAMPLES]

        mel_spec_padded = torch.from_numpy(mel_spec_padded).float()

        return mel_spec_padded

    def __getitem__(self, idx):

        mel_spec = self.get_mel(idx)
        frames = self.get_frames(idx)

        return mel_spec, frames