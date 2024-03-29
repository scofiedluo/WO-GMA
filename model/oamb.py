import torch
import torch.nn as nn
import torch.nn.functional as F
from model.activation import activation_factory


class OAMB(nn.Module):
    """
    Online Action Modeling Branch

    Input: x,state = N*C*T,(h_0,c_0),
           where h_0 and c_0 of shape (num_layers * num_directions, batch, hidden_size)
    Output: output = N*T*hidden_size, per_frame_socres = N*T*(num_class+1),
            the last N*T*1 is background score
    and start_scores = N*T*2
    here we have num_layers * num_directions = 1

    param input_size(int): The number of expected features in the input x
    param hidden_size(int): The number of features in the hidden state h
    param linear_hidden_channels(list): the hidden channels of linear layers to project output of LSTM
    param batch_size(int): batch size
    param num_class(int): action classes num and a background class
    parma num_layers(int): Number of recurrent layers
    param M(int): max pooling window size
    """
    def __init__(self,input_size, hidden_size, linear_hidden_channels, num_class,
                batch_size, device, num_layers=1, dropout=0, batch_first=True, activation='relu'):

        super(OAMB, self).__init__()
        self.device = device
        self.dropout = dropout
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=batch_first, dropout=0)

        self.state = (torch.zeros(num_layers,batch_size, hidden_size).to(self.device),
                      torch.zeros(num_layers,batch_size, hidden_size).to(self.device))

        linear_channels = [hidden_size] + linear_hidden_channels + [num_class+1]
        self.scores_layers = nn.ModuleList()
        for i in range(1,len(linear_channels)-1):
            if dropout>0:
                self.scores_layers.append(nn.Dropout(p=dropout))
            self.scores_layers.append(nn.Linear(linear_channels[i-1], linear_channels[i]))
            self.scores_layers.append(activation_factory(activation))
        self.scores_layers.append(nn.Linear(linear_channels[len(linear_channels)-2],
                                            linear_channels[len(linear_channels)-1]))

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, training=True):
        # x: N*C*T
        x = x.permute(0, 2, 1).contiguous()

        # change hidden state for inference
        num_samples = x.shape[0]
        if training is False:
            if num_samples < self.state[0].shape[1]:
                hn = self.state[0][:, :num_samples, :].to(self.device)
                cn = self.state[1][:, :num_samples, :].to(self.device)
                tmp_state = (hn, cn)
            else:
                tmp_state = self.state
        if training:
            output, self.state = self.rnn(x, self.state)
        else:
            output, _ = self.rnn(x, tmp_state)
        # state = (hn, cn)   # both(num_layers * num_directions, batch, hidden_size)

        for layer in self.scores_layers:
            output = layer(output)
        # per-frame action scores over classes and background class, shape N*T*(num_class+1)
        per_frame_scores = self.softmax(output)

        return per_frame_scores
