import torch.nn as nn
from params import AudioHyperParams


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=int(AudioHyperParams.EMBENDING_DIM),
                      out_channels=int(AudioHyperParams.EMBENDING_DIM / 2),
                      kernel_size=2, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=int(AudioHyperParams.EMBENDING_DIM / 2),
                      out_channels=int(AudioHyperParams.NUMBER_OF_MEL_BANDS),
                      kernel_size=2, stride=1),
            nn.ReLU(True)
        )

        self.conv1d_transpose = nn.Sequential(
            nn.ConvTranspose1d(in_channels=int(AudioHyperParams.NUMBER_OF_MEL_BANDS),
                               out_channels=int(AudioHyperParams.NUMBER_OF_MEL_BANDS),
                               kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(int(AudioHyperParams.MEL_SAMPLES / 2)),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=int(AudioHyperParams.NUMBER_OF_MEL_BANDS),
                               out_channels=int(AudioHyperParams.NUMBER_OF_MEL_BANDS),
                               kernel_size=2, stride=2),
            nn.ReLU(True),
        )

    def forward(self, x):
        out = self.conv2d(x)
        out = out.squeeze()
        out = self.conv1d_transpose(out)

        return out
