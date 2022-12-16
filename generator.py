import torch.nn as nn
import torch
from params import AudioHyperParams


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.conv1d_fisrt = nn.Sequential(
            nn.Conv1d(in_channels=int(AudioHyperParams.EMBENDING_DIM),
                      out_channels=125,
                      kernel_size=3),
            nn.BatchNorm1d(125),
            nn.LeakyReLU(True),

            nn.Conv1d(in_channels=125,
                      out_channels=80,
                      kernel_size=3),
            nn.BatchNorm1d(80),
            nn.LeakyReLU(True)
        )

        self.conv1d_fisrt.apply(init_weights)

        self.conv1d_second = nn.Sequential(
            nn.Conv1d(in_channels=588,
                      out_channels=int(AudioHyperParams.MEL_SAMPLES),
                      kernel_size=3, padding=1),
            nn.LeakyReLU(True),

            nn.Conv1d(in_channels=int(AudioHyperParams.MEL_SAMPLES),
                      out_channels=int(AudioHyperParams.MEL_SAMPLES),
                      kernel_size=3, padding=1),
        )

        self.conv1d_second.apply(init_weights)

    def forward(self, x):
        x = self.conv1d_fisrt(x)
        x = x.transpose(1, 2)

        out = self.conv1d_second(x)
        out = out.transpose(1, 2)

        return out
