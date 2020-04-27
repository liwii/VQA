import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 300
LSTM_OUT_DIM = 512
class VQANN(nn.Module):
    def __init__(self, word_dim, image_dim, out_dim):
        super(VQANN, self).__init__()
        self.fc1 = nn.Linear(word_dim, EMBEDDING_DIM)
        self.lstm1 = nn.LSTM(EMBEDDING_DIM, LSTM_OUT_DIM)
        self.lstm2 = nn.LSTM(LSTM_OUT_DIM, LSTM_OUT_DIM)
        self.fc2 = nn.Linear(LSTM_OUT_DIM * 4, LSTM_OUT_DIM * 2)
        self.fcim = nn.Linear(image_dim, LSTM_OUT_DIM * 2)
        self.fclast = nn.Linear(LSTM_OUT_DIM * 2, out_dim)

    def forward(self, words, image):
        x = F.relu(self.fc1(words))
        x = x.transpose(0, 1)
        out, hidden1 = self.lstm1(x)
        _ , hidden2 = self.lstm2(out, hidden1)
        lstm_out = torch.cat((*hidden1, *hidden2), dim=2).view(-1, LSTM_OUT_DIM * 4)
        lstm_out = F.relu(self.fc2(lstm_out))
        im_out = F.relu(self.fcim(image))
        out = nn.Softmax(dim=1)(self.fclast(lstm_out * im_out))
        return out
