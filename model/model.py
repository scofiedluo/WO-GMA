import torch.nn as nn
import torch.nn.functional as F
from utils import import_class
from model.ms_g3d import MS_G3D
from model.vertices_fusion import VerticesFusion
from model.cpgb import CPGB
from model.oamb import OAMB


class LocalFeature(nn.Module):
    """
    process original skeleton feature
    (1) local spatial temporal feature fusion, (2) vertices fusion

    input: N*in_C*T*V; output: N*out_C*[T/window_stride],
    where out_C corrsponding to args.vertices_fusion_out_dim.

    @param
    args: dict, mean of it's key can be found in class MS_G3D and class vertices_fusion
    """
    def __init__(self, args):
        super(LocalFeature, self).__init__()
        graph = import_class(args.graph)
        adj_binary = graph().A_binary
        # instance of local spatial temporal feature fusion
        self.ms_g3d = MS_G3D(
            in_channels = args.skeleton_feature_dim,
            out_channels = args.conv3D_out_dim,
            A_binary = adj_binary,
            num_scales = args.num_scales,
            window_size = args.window_size,
            window_stride = args.window_stride,
            window_dilation = args.window_dilation
            )
        # instance of vertices fusion
        self.vertices_fusion = VerticesFusion(
            in_channels = args.conv3D_out_dim,
            out_channels = args.vertices_fusion_out_dim,
            )

    def forward(self, x):
        x = self.ms_g3d(x)
        x = self.vertices_fusion(x)
        return x


class Model(nn.Module):
    """
    model integrating

    @param: args, device

    """
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.feature_preprocess = LocalFeature(args)
        self.CPGB = CPGB(in_channels = args.vertices_fusion_out_dim,
                       hidden_channels = args.CPGB_hidden_channels,
                       num_class = args.num_class,
                       kappa = args.kappa,
                       dropout = args.CPGB_dropout,
                       activation = args.CPGB_activation
                       )
        self.OAMB = OAMB(input_size = args.vertices_fusion_out_dim,
                       hidden_size = args.LSTM_hidden_channels,
                       linear_hidden_channels = args.OAMB_linear_hidden_channels,
                       batch_size = args.batch_size,
                       num_class = args.num_class,
                       device = device,
                       num_layers = 1,
                       dropout = args.OAMB_dropout,
                       batch_first=True
                       )

    def forward_CPGB(self, x):
        preprocessed_feature = self.feature_preprocess(x)  # N*C*T
        CPGB_out_scores = self.CPGB(preprocessed_feature)    # N*T*num_class
        video_level_score = self.CPGB.topK(CPGB_out_scores)
        CPGB_output = {"preprocessed_feature": preprocessed_feature,
                       "CPGB_out_scores": CPGB_out_scores, "video_level_score": video_level_score}
        return CPGB_output

    def inference_forward(self, x, training=True):
        preprocessed_feature = self.feature_preprocess(x)
        per_frame_scores = self.OAMB(preprocessed_feature, training)
        OAMB_out = {"per_frame_scores":per_frame_scores}
        return OAMB_out

    def forward_OAMB(self, preprocessed_feature, training=True):
        per_frame_scores = self.OAMB(preprocessed_feature, training)
        OAMB_out = {"per_frame_scores": per_frame_scores}
        return OAMB_out
