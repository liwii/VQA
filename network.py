import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 300
LSTM_OUT_DIM = 512
class VQANN(nn.Module):
    def __init__(self, image_dim, out_dim):
        super(VQANN, self).__init__()
        self.lstm = nn.LSTM(EMBEDDING_DIM, LSTM_OUT_DIM, 3)
        self.fclstm = nn.Linear(LSTM_OUT_DIM * 6, LSTM_OUT_DIM * 2)
        self.fcim = nn.Linear(image_dim, LSTM_OUT_DIM * 2)
        self.fc1 = nn.Linear(LSTM_OUT_DIM * 4, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim)

    def forward(self, words, image, word_lengths):
        x = nn.utils.rnn.pack_padded_sequence(words, word_lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.lstm(x)
        lstm_out = torch.cat((hidden[0][0], hidden[0][1], hidden[0][2], hidden[1][0], hidden[1][1], hidden[1][2]), 1)
        lstm_out = self.fclstm(lstm_out)
        im_out = F.relu(self.fcim(image))
        im_lstm = torch.cat((im_out, lstm_out), 1)
        out = F.relu(self.fc1(im_lstm))
        out = F.relu(self.fc2(out))
        return self.fc3(out)

class MultiClassCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(MultiClassCrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        eps = 1e-8
        return - torch.sum(target * torch.log(output + eps))