import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 300
LSTM_OUT_DIM = 512
class VQANN(nn.Module):
    def __init__(self, image_dim, out_dim):
        super(VQANN, self).__init__()
        self.lstm = nn.LSTM(EMBEDDING_DIM, LSTM_OUT_DIM)
        self.fcim = nn.Linear(image_dim, LSTM_OUT_DIM * 2)
        self.fc1 = nn.Linear(LSTM_OUT_DIM * 2, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim)

    def forward(self, words, image):
        x = words.transpose(0, 1)
        _, hidden = self.lstm(x)
        lstm_out = torch.cat(hidden, dim=2).view(-1, LSTM_OUT_DIM * 2)
        im_out = F.relu(self.fcim(image))
        im_lstm = lstm_out * im_out
        im_lstm = F.relu(self.fc1(im_lstm))
        im_lstm = F.relu(self.fc2(im_lstm))
        out = nn.Softmax(dim=1)(self.fc3(im_lstm))
        return out

class MultiClassCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(MultiClassCrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        eps = 1e-8
        return - torch.sum(target * torch.log(output + eps))