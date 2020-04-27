import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 300
LSTM_OUT_DIM = 512
class VQANN(nn.Module):
    def __init__(self, image_dim, out_dim):
        super(VQANN, self).__init__()
        self.lstm1 = nn.LSTM(EMBEDDING_DIM, LSTM_OUT_DIM)
        self.lstm2 = nn.LSTM(LSTM_OUT_DIM, LSTM_OUT_DIM)
        self.fcw = nn.Linear(LSTM_OUT_DIM * 4, LSTM_OUT_DIM * 2)
        self.fcim = nn.Linear(image_dim, LSTM_OUT_DIM * 2)
        self.fclast = nn.Linear(LSTM_OUT_DIM * 2, out_dim)

    def forward(self, words, image):
        x = words.transpose(0, 1)
        out, hidden1 = self.lstm1(x)
        _ , hidden2 = self.lstm2(out, hidden1)
        lstm_out = torch.cat((*hidden1, *hidden2), dim=2).view(-1, LSTM_OUT_DIM * 4)
        lstm_out = F.relu(self.fcw(lstm_out))
        im_out = F.relu(self.fcim(image))
        im_lstm = lstm_out * im_out
        out = nn.Softmax(dim=1)(self.fclast(im_lstm))
        return out

class MultiClassCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(MultiClassCrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        eps = 1e-8
        return - torch.sum(target * torch.log(output + eps))