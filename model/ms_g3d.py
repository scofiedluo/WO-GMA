import torch.nn as nn
import torch.nn.functional as F

from model.mlp import MLP
from model.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows


class MS_G3D(nn.Module):
    """
    This module is for multi-scale 3D graph convolution

    param in_channels: the channels of skeleton vertices features
    param out_channels: the channels of skeleton vertices features
    param A_binary: Adjacency matrix of pre-defined skeleton
    param num_scales: the scales of multi scale graph convlution
    param window_size: window of time axis, the skeleton numbers in a window is window_size.
                       this param will not change the output time length
    param window_stride: slide length of between two windows.
                         The output time length T1 = [T/window_stride], [] means rounded up.
    param window_dilation: gap between skeletons in one window, eg: sk1---sk2---sk3,
                           window_size = 3, dilation = 4
    embed_factor: before gcn3d, set 1D conv layer, something like bottle neck structure,
                  embed_factor may bigger than 1

    If input size is N*in_C*T*V, then the output size is N*out_C*[T/window_stride]*V
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 window_stride,
                 window_dilation,
                 embed_factor=1,
                 activation='relu'):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor

        # self.in1x1 means 1x1 conv layer
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            # The first STGC block changes channels right away; others change at collapse
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            SpatialTemporal_MS_GCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=True
            )
        )

        self.out_conv = nn.Conv3d(self.embed_channels_out,
                                  out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        batch_size, _, tem_len, vertices = x.shape
        x = self.in1x1(x)     # x.shape = (N,C,T,V)
        # Construct temporal windows and apply MS-GCN
        x = self.gcn3d(x)     # x.shape = ()

        # Collapse the window dimension
        x = x.view(batch_size, self.embed_channels_out, -1, self.window_size, vertices)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)

        # no activation
        return x
