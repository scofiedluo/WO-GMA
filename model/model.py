import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import import_class, count_params
from model.MS_G3D import MS_G3D
from model.vertices_fusion import vertices_fusion
from model.TPG import TPG
from model.OAR import OAR


class feature_preprocess(nn.Module):
    """
    process original skeleton feature
    (1) local spatial temporal feature fusion, (2) vertices fusion

    input: N*in_C*T*V; output: N*out_C*[T/window_stride], 
    where out_C corrsponding to args.vertices_fusion_out_dim.

    @param
    args: dict, mean of it's key can be found in class MS_G3D and class vertices_fusion
    """
    def __init__(self,args):
        super(feature_preprocess,self).__init__()
        Graph = import_class(args.graph)
        A_binary = Graph().A_binary
        # instance of local spatial temporal feature fusion
        self.MS_G3D = MS_G3D(
            in_channels = args.skeleton_feature_dim,
            out_channels = args.conv3D_out_dim,
            A_binary = A_binary,
            num_scales = args.num_scales,
            window_size = args.window_size,
            window_stride = args.window_stride,
            window_dilation = args.window_dilation
            )
        # instance of vertices fusion
        self.vertices_fusion = vertices_fusion(
            in_channels = args.conv3D_out_dim,
            out_channels = args.vertices_fusion_out_dim,
            )
    def forward(self,x):
        x = self.MS_G3D(x)
        x = self.vertices_fusion(x)
        return x


class model(nn.Module):
    """
    model integrating

    @param:

    """
    def __init__(self,args, device):
        super(model,self).__init__()
        self.feature_preprocess = feature_preprocess(args)
        self.TPG = TPG(in_channels = args.vertices_fusion_out_dim,
                       hidden_channels = args.TPG_hidden_channels,
                       num_class = args.num_class,
                       kappa = args.kappa,
                       dropout = args.TPG_dropout,
                       activation = args.TPG_activation
                       )
        self.OAR = OAR(input_size = args.vertices_fusion_out_dim,
                       hidden_size = args.LSTM_hidden_channels,
                       linear_hidden_channels = args.OAR_linear_hidden_channels,
                       batch_size=args.batch_size,
                       num_class = args.num_class,
                       device = device,
                       num_layers = 1,
                       M = args.M,
                       dropout = args.OAR_dropout,
                       batch_first=True
                       )
    
    def forward_TPG(self,x):
        preprocessed_feature = self.feature_preprocess(x)  # N*C*T
        TPG_out_scores = self.TPG(preprocessed_feature)    # N*T*num_class
        video_level_score = self.TPG.topK(TPG_out_scores)
        TPG_output = {"preprocessed_feature":preprocessed_feature, "TPG_out_scores":TPG_out_scores, "video_level_score":video_level_score}
        return TPG_output
    
    # def forward_OAR(self,preprocessed_feature,state):
    #     start_scores,state,per_frame_scores = self.OAR(preprocessed_feature,state)
    #     OAR_output = {"start_scores":start_scores, "state":state, "per_frame_scores":per_frame_scores}
        
    #     return OAR_output

    # def forward(self,x):
    #     preprocessed_feature = self.feature_preprocess(x)
    #     return preprocessed_feature

    def inference_forward(self,x,training=True):
        preprocessed_feature = self.feature_preprocess(x)
        start_scores,per_frame_scores, LSTM_feature = self.OAR(preprocessed_feature,training)
        OAR_out = {"start_scores":start_scores, "per_frame_scores":per_frame_scores,"preprocessed_feature": preprocessed_feature, "LSTM_feature":LSTM_feature}
        return OAR_out
    
    def forward_OAR(self,preprocessed_feature,training=True):
        start_scores, per_frame_scores, LSTM_feature = self.OAR(preprocessed_feature,training)
        OAR_out = {"start_scores":start_scores, "per_frame_scores":per_frame_scores, "LSTM_feature":LSTM_feature}
        return OAR_out

