import torch
import torch.nn as nn
from params import AudioHyperParams, VideoHyperParams


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.frames_conv = nn.Sequential(
            nn.Conv3d(in_channels=int(AudioHyperParams.MEL_SAMPLES / 2),
                      out_channels=1,
                      kernel_size=3, stride=1),
        )
        self.frames_post_conv = nn.Sequential(
            nn.Conv1d(in_channels=222,
                      out_channels=int(AudioHyperParams.NUMBER_OF_MEL_BANDS),
                      kernel_size=5, stride=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=int(AudioHyperParams.NUMBER_OF_MEL_BANDS),
                      out_channels=int(AudioHyperParams.NUMBER_OF_MEL_BANDS),
                      kernel_size=4, stride=1),
            nn.ReLU(True)
        )

        self.mel_conv = nn.Conv1d(int(AudioHyperParams.NUMBER_OF_MEL_BANDS), int(AudioHyperParams.NUMBER_OF_MEL_BANDS),
                                  kernel_size=3, stride=2, padding=1)

        self.down_sampling = nn.Sequential(
            nn.Conv2d(2, 64,
                      kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128,
                      kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 256,
                      kernel_size=4, stride=3),
            nn.ReLU(True),
            nn.Conv2d(256, 512,
                      kernel_size=4, stride=3),
            nn.ReLU(True),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
        )

        self.output = nn.Sequential(
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

    def forward(self, frames, mel_spec):
        mel = self.mel_conv(mel_spec)
        frames = self.frames_conv(frames)
        frames = frames.squeeze()
        frames = self.frames_post_conv(frames)
        concat = torch.stack([frames, mel], dim=0)
        concat = self.down_sampling(concat)
        out = concat.squeeze()
        out = self.output(out)
        return out
