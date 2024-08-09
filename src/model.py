import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.bmm = nn.Bilinear(hidden_size, hidden_size, 1)

    def forward(self, lstm_output, lstm_hidden):
        lstm_hidden = lstm_hidden.squeeze(0)
        lstm_hidden = self.fc(lstm_hidden)
        lstm_hidden = lstm_hidden.unsqueeze(2)
        lstm_output = self.fc(lstm_output)
        attention_scores = self.bmm(lstm_output, lstm_hidden)
        attention_scores = F.softmax(attention_scores, dim=1)
        attention_output = torch.bmm(lstm_output.permute(0, 2, 1), attention_scores).squeeze(2)
        return attention_output

class DGCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)
        return x
    

class Model(nn.Module):
    def __init__(self, opt) -> None:
        super(Model, self).__init__()

        self.dropout_rate = opt.dropout_rate
        self.upsampling_num = opt.upsampling_num
        self.rnn_size = opt.rnn_size

        self.BiLSTM = BiLSTM(
            input_size = opt.input_size,
            hidden_size = opt.rnn_size,
            num_layers = opt.rnn_layers,
            num_classes = opt.n_classes
        )

        self.attention = Attention(opt.rnn_size)
        self.dgcnn = DGCNN(opt.rnn_size, opt.hidden_dim, opt.rnn_size)
        self.fc = nn.Linear(opt.rnn_size, opt.n_classes)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(opt.dropout_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:

        # run through bidirectional LSTM
        rnn_out, _ = self.BiLSTM(input_tensor)

        # attention module
        H = rnn_out[:, :, :self.rnn_size] + rnn_out[:, :, self.rnn_size:]
        r = self.attention(H)

        # DGCNN
        new_r = self.dgcnn(r)

        h = self.tanh(new_r)

        scores = self.fc(self.dropout(h))

        return scores
