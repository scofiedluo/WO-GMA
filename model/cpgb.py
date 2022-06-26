import torch.nn as nn
import numpy as np
from model.activation import activation_factory


class CPGB(nn.Module):
    """
    Clip-level Pseudo Labels Generating Branch

    Input: N*C*T,
    output: N*T*num_class, where C denote feature demension,
            T denote time block number, num_class=out_channels,is the action

    @param:
    in_channels(int): input feature channels
    hidden_channels(list of int): hidden layer channels
    num_class(int): action class and a background class
    kappa(int): a parameter to contral topk in proportion time length
    """
    def __init__(self, in_channels, hidden_channels, num_class,
                 kappa, dropout=0, activation='relu'):
        super(CPGB, self).__init__()
        self.kappa = kappa
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.channels = [in_channels] + hidden_channels + [num_class]

        # conv1d
        self.layers1 = nn.ModuleList()
        for i in range(1,len(self.channels)-1):
            self.layers1.append(nn.Conv1d(self.channels[i-1], self.channels[i], 5, stride = 1, padding=2))
            self.layers1.append(nn.BatchNorm1d(self.channels[i]))
            self.layers1.append(activation_factory(activation))
        self.out_fc = nn.Linear(self.channels[len(self.channels)-2], self.channels[len(self.channels)-1])

    def topK(self, scores):
        # scores.shape = N*T*num_class
        T = scores.shape[1]
        k = np.ceil(T/self.kappa).astype('int32')
        topk,_ = scores.topk(k, dim=1, largest=True, sorted = True)
        # video_level_score.shape = N*num_class
        video_level_score = topk.mean(axis=1,keepdim=False)
        return video_level_score

    def proposal_gen(self,scores,video_level_score,labels,video_thres=0.5,frame_thres=0.5):
        """
        Two stage threshold strategy

        @param:
        video_thres(float):
        frame_thres(float):
        scores(tensor): N*T*num_class, T is temporal block num, action socres of each temporal block
        video_level_score(tensor): N*num_class
        labels(tensor): N*num_class, video level labels

        output(tensor): N*T*(num_class)
        """

        # stage 1
        mask = video_level_score.ge(video_thres)
        mask = mask.unsqueeze(1)
        # stage 2
        proposals = mask*scores
        proposals = proposals.ge(frame_thres)

        # filtering, use the groud truth to filter the wrong proposal
        labels = labels.unsqueeze(1)
        proposals = proposals* labels
        return proposals

    def forward(self, x):
        # x: N*C*T

        # for conv1d
        for layer in self.layers1:
            x = layer(x)
        x = x.permute(0,2,1).contiguous()
        x = self.out_fc(x)

        if x.shape[2]==1:
            x = self.sigmoid(x)
        else:
            x = self.softmax(x)

        return x
