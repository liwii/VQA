import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 300
LSTM_OUT_DIM = 1024
class VQANN(nn.Module):
    def __init__(self, image_dim, out_dim):
        super(VQANN, self).__init__()
        self.lstm = nn.LSTM(EMBEDDING_DIM, LSTM_OUT_DIM, 2)
        self.fcim = nn.Linear(image_dim, LSTM_OUT_DIM )
        self.fc1 = nn.Linear(LSTM_OUT_DIM, out_dim)

    def forward(self, words, image, word_lengths):
        x = nn.utils.rnn.pack_padded_sequence(words, word_lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.lstm(x)
        lstm_out = hidden[0][1].view(-1, LSTM_OUT_DIM)
        im_out = F.relu(self.fcim(image))
        im_lstm = lstm_out + im_out
        out = self.fc1(im_lstm)
        return out

class MultiClassCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(MultiClassCrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        eps = 1e-8
        return - torch.sum(target * torch.log(output + eps))