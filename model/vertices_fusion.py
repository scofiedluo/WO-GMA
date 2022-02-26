import torch
import torch.nn as nn

from model.activation import activation_factory


class vertices_fusion(nn.Module):
    """
    fuse the skeleton features into one vertice
    we use fully connnected network for fusion by default, other method can be developed in future work
    input: N*C_in*T*V, 
    output: N*C_out*T, corrsponding to batch size, channels, nums of time block

    @param:
    in_channels:
    out_channels:
    fusion_type: string, 
    node_nums: vertices num of one skeleton
    activation: 

    """
    def __init__(self,in_channels, out_channels, node_nums = 18, 
                fusion_type = 'nodeConv',activation='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_nums = node_nums
        self.act = activation_factory(activation)

        if fusion_type == 'nodeConv':
            self.fusion_method = nn.Conv2d(self.in_channels,self.out_channels,kernel_size=(1,self.node_nums))
        else:
            # TODO: there can be adjust in future work
            self.fusion_method = nn.Identity()
        
        self.BN = nn.BatchNorm1d(self.out_channels)
    
    def forward(self,x):
        x = self.fusion_method(x)
        x = x.squeeze(dim=3)
        x = self.BN(x)
        # print("vertices_fusion out shape:", x.shape)

        # here batchNorm

        return self.act(x)
