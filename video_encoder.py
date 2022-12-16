import torch
import torch.nn as nn
import torch.nn.functional as F

from params import VideoHyperParams


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class VideoEncoder(nn.Module):

    def __init__(self):
        super(VideoEncoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)

        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0.01)

        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1)

        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0.01)

        self.BiLSTM = nn.LSTM(input_size=int(VideoHyperParams.EMBENDING_DIM),
                              hidden_size=int(VideoHyperParams.EMBENDING_DIM), num_layers=1,
                              batch_first=True)

        self.BiLSTM_proj = nn.Linear(in_features=int(VideoHyperParams.EMBENDING_DIM),
                                     out_features=int(VideoHyperParams.EMBENDING_DIM))

        torch.nn.init.xavier_uniform(self.BiLSTM_proj.weight)
        self.BiLSTM_proj.bias.data.fill_(0.01)

    def forward(self, images):
        batch_size, numbers_of_frames, c, h, w = images.shape

        x = images[:, 0]
        x = self.conv1(x)
        x = F.leaky_relu(self.conv2(x))
        x = torch.squeeze(x)
        out, (hn, cn) = self.BiLSTM(x)

        for i in range(1, numbers_of_frames):
            x = images[:, i]

            x = self.conv1(x)
            x = F.leaky_relu(self.conv2(x))
            x = torch.squeeze(x)
            out, (hn, cn) = self.BiLSTM(x, (hn, cn))

        res = self.BiLSTM_proj(out)
        return res