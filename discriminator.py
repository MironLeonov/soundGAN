import torch
import torch.nn as nn
from params import AudioHyperParams, VideoHyperParams

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.frames_post_conv = nn.Sequential(
            nn.Conv2d(in_channels=int(VideoHyperParams.NUMBER_OF_FRAMES),
                      out_channels=int(VideoHyperParams.NUMBER_OF_FRAMES),
                      kernel_size=5, stride=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=int(VideoHyperParams.NUMBER_OF_FRAMES),
                      out_channels=int(AudioHyperParams.NUMBER_OF_MEL_BANDS),
                      kernel_size=6, stride=3),
            nn.ReLU(True),
            nn.Conv2d(in_channels=int(AudioHyperParams.NUMBER_OF_MEL_BANDS),
                      out_channels=int(AudioHyperParams.NUMBER_OF_MEL_BANDS),
                      kernel_size=7, stride=3),
        )

        self.frames_post_conv.apply(init_weights)

        self.mel_conv = nn.Conv1d(int(AudioHyperParams.NUMBER_OF_MEL_BANDS), int(AudioHyperParams.NUMBER_OF_MEL_BANDS),
                                  kernel_size=3, stride=2, padding=1)

        torch.nn.init.xavier_uniform(self.mel_conv.weight)
        self.mel_conv.bias.data.fill_(0.01)

        self.down_sampling = nn.Sequential(
            nn.Conv1d(80, 80,
                      kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv1d(80, 120,
                      kernel_size=6, stride=3),
            nn.ReLU(True),
            nn.Conv1d(120, 160,
                      kernel_size=6, stride=3),
            nn.Conv1d(160, 200,
                      kernel_size=6, stride=3),
            nn.Conv1d(200, 1,
                      kernel_size=6, stride=3),
        )

        self.down_sampling.apply(init_weights)

        self.sigmoid = nn.Sigmoid()


    def forward(self, frames, mel_spec):

        mel = self.mel_conv(mel_spec)

        frames = torch.squeeze(frames)
        bs, _, _, _ = frames.shape
        frames = self.frames_post_conv(frames)
        frames = torch.reshape(frames, (bs, 80, -1))

        concat = torch.cat([frames, mel], dim=2)
        out = self.down_sampling(concat)
        out = torch.squeeze(out)
        out = self.sigmoid(out) #!

        return out
