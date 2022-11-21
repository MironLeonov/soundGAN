import os
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import torch

from params import VideoHyperParams


class VideoDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.transform = transform
        self.frames_dir = root_dir + '/video_features'
        self.videos = os.listdir(self.frames_dir)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_dir = self.frames_dir + '/' + self.videos[idx]
        print(video_dir)

        images_paths = glob(video_dir + "/*.jpg")
        print(images_paths)
        images_paths = images_paths[:VideoHyperParams.NUMBER_OF_FRAMES]
        print(images_paths)

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
