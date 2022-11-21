import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from params import VideoHyperParams


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.resnet = models.resnet18(pretrained=True)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),

            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),

            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4 , stride=1, padding='same'),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=True)
        )

        self.conv_layers.apply(init_weights)

        self.BiLSTM = nn.LSTM(input_size=int(VideoHyperParams.EMBENDING_DIM),
                              hidden_size=int(VideoHyperParams.EMBENDING_DIM / 2), num_layers=2,
                              batch_first=True, bidirectional=True)

        self.BiLSTM_proj = nn.Linear(in_features=int(VideoHyperParams.EMBENDING_DIM),
                                     out_features=int(VideoHyperParams.EMBENDING_DIM / 2))

    def forward(self, images):
        batch_size, numbers_of_frames, c, h, w = images.shape

        x = images[:, 0]
        x = self.resnet(x)
        x = self.conv_layers(x)

        out, (hn, cn) = self.BiLSTM(x)

        for i in range(1, numbers_of_frames):
            x = images[:, i]
            x = self.resnet(x)
            x = self.conv_layers(x)
            out, (hn, cn) = self.BiLSTM(x, (hn, cn))

        out = self.BiLSTM_proj(out)
        return out
