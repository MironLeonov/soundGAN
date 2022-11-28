import torch
import torch.nn as nn
from params import AudioHyperParams, VideoHyperParams


class AudioEncoder(nn.Module):

    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.BiLSTM = nn.LSTM(int(AudioHyperParams.NUMBER_OF_MEL_BANDS), int(AudioHyperParams.EMBENDING_DIM / 2), 2,
                              batch_first=True, bidirectional=True)
        self.BiLSTM_proj = nn.Linear(int(AudioHyperParams.MEL_SAMPLES), int(VideoHyperParams.NUMBER_OF_FRAMES))

    def forward(self, spec):
        batch_size, numbers_of_mels, number_of_frames = spec.shape
        spec = spec.transpose(1, 2)  # batch_size, number_of_frames, numbers_of_mels
        x = spec[:, 0]

        res, (hn, cn) = self.BiLSTM(x)

        for i in range(1, number_of_frames):
            x = spec[:, i]
            out, (hn, cn) = self.BiLSTM(x, (hn, cn))
            res = torch.cat([res, out], dim=0)

        res = res.transpose(0, 1)
        res = self.BiLSTM_proj(res)

        return res
