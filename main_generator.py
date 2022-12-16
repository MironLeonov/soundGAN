import torch
import torch.nn as nn

from video_encoder import VideoEncoder
from audio_encoder import AudioEncoder
from generator import Generator


class MainGenerator(nn.Module):
    def __init__(self):
        super(MainGenerator, self).__init__()

        self.v_encoder = VideoEncoder()
        self.a_encoder = AudioEncoder()
        self.generator = Generator()

    def forward(self, video, audio):
        v_encoded = self.v_encoder(video)
        if self.training:
            a_encoded = self.a_encoder(audio)
        else:
            a_encoded = self.a_encoder(audio * 0)
        data_for_gen = torch.cat([v_encoded, a_encoded], dim=2)
        out = self.generator(data_for_gen)

        return out
