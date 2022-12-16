import torch
import torch.nn as nn
from params import AudioHyperParams


class AudioEncoder(nn.Module):

    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.BiLSTM = nn.LSTM(int(AudioHyperParams.MEL_SAMPLES), int(AudioHyperParams.EMBENDING_DIM), num_layers=2,
                              batch_first=True)

        self.BiLSTM_proj = nn.Linear(int(AudioHyperParams.EMBENDING_DIM), int(AudioHyperParams.EMBENDING_DIM))

    def forward(self, spec):
        res, (hn, cn) = self.BiLSTM(spec)
        res = self.BiLSTM_proj(res)
        res = res.transpose(1, 2)
        return res
