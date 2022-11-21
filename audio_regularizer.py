import torch.nn as nn


class Auxiliary_lstm_last(nn.Module):

    def __init__(self):
        super(Auxiliary_lstm_last, self).__init__()
        self.BiLSTM = nn.LSTM(80, int(500), 2,
                           batch_first=True, bidirectional=True)
        self.BiLSTM_proj = nn.Linear(500, 250)

    def forward(self, x):
        print(x.shape)
        x = x.transpose(1, 2)
        print(x.shape)
        x, (h, c) = self.BiLSTM(x)
        print(x.shape)
        x = self.BiLSTM_proj(h[-1])
        print(x.shape)
        print(x)
        bs, c = x.shape
        x = x.unsqueeze(1).expand(bs, 215, c)
        return x