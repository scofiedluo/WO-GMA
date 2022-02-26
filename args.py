import argparse
import os
import random

def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='WOAT')

    parser.add_argument(
        '--work-dir',
        type=str,
        default='training_results/',
        # required=True,
        help='the work folder for storing results')
    parser.add_argument('--model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/train_config.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--assume-yes',
        action='store_true',
        help='Say yes to every prompt')

    parser.add_argument(
        '--phase',
        default='train',
        help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    parser.add_argument(
        '--seed',
        type=int,
        default=8,
        help='random seed')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--eval-start',
        type=int,
        default=1,
        help='The epoch number to start evaluating models')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')

    parser.add_argument(
        '--feeder',
        default='feeder.feeder',
        help='data loader will be used')
    parser.add_argument(
        '--experiment',
        default='demo',
        type = str,
        help='sub folder parameter')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=os.cpu_count(),
        help='the number of worker for data loader')

    # data
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/GMs_data/all_v3/processed_data_joint.npy',
        help='the path to all data')
    parser.add_argument(
        '--label_path',
        type=str,
        default='data/GMs_data/all_v3/processed_label.pkl',
        help='the path to all labels')
    parser.add_argument(
        '--video_info_path',
        type=str,
        default='data/video_info/video_info.csv',
        help='the path to video info')
    parser.add_argument(
        '--split_test_csv_path',
        type=str,
        default='data/GMs_data/split_v5/fold3_val.csv',
        help='the data split csv')
    parser.add_argument(
        '--split_train_csv_path',
        type=str,
        default='data/GMs_data/split_v5/fold3_train.csv',
        help='the data split csv')
    parser.add_argument(
        '--frame_annotations',
        type=str,
        default='data/frame_level_annotations/temporal_labels_v2.csv',
        help='path to frame annotations .csv file'
    )
    
    parser.add_argument(
        '--action_classes',
        type=str,
        default=['F+'],
        nargs='+',
        help='action classes list of datasets'
    )
    parser.add_argument(
        '--TPG_train_step',
        type=int,
        default=[0,10,20,30,40,50,61,85],
        nargs='+',
        help = 'intervals to train TPG'
    )
    parser.add_argument(
        '--OAR_train_step',
        type=int,
        default=[10,20,51,60,93,100],
        nargs='+',
        help = 'intervals to train OAR'
    )
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model parameters
    parser.add_argument(
        '--skeleton_feature_dim',
        type=int,
        default=3,
        help='the arguments of original skeleton feature dim, 3 or 2')
    parser.add_argument(
        "--conv3D_out_dim",
        type=int,
        default=256,
        help='the output dim of MS_G3D block')
    parser.add_argument(
        '--graph',
        type=str,
        default='graph.kinetics.AdjMatrixGraph',
        help='graph class of skeleton')
    parser.add_argument(
        '--num_scales',
        type=int,
        default=8,
        help='the scales of multi scale graph convlution')
    parser.add_argument(
        '--window_size',
        type=int,
        default=32,
        # nargs='+',
        help='window of time axis, the skeleton numbers in a window is window_size. this param will not change the output time length')
    parser.add_argument(
        '--window_stride',
        type=int,
        default=32,
        help='slide length of between two windows. The output time length T1 = [T/window_stride], [] means rounded up.')
    parser.add_argument(
        '--window_dilation',
        type=int,
        default=1,
        help='gap between skeletons in one window, eg: sk1---sk2---sk3, window_size = 3, dilation = 4')
    parser.add_argument(
        '--vertices_fusion_out_dim',
        type=int,
        default=2048,
        help='the out channel of vertices_fusion module')
    parser.add_argument(
        '--TPG_hidden_channels',
        type=int,
        default=[512,128,64],
        nargs='+',
        help = 'fully connected layers in temporal proposals module')
    parser.add_argument(
        '--num_class',
        type=int,
        default=1,
        help='num of action classes in dataset')
    parser.add_argument(
        '--kappa',
        type=int,
        default=8,
        help='a parameter to contral topk in proportion time length')
    parser.add_argument(
        '--TPG_dropout',
        type=float,
        default=0.5,
        help='the dropout rate in TPG')
    parser.add_argument(
        '--TPG_activation',
        type=str,
        default='relu',
        help='the activation function of TPG')
    parser.add_argument(
        '--LSTM_hidden_channels',
        type=int,
        default=1024,
        help='the hidden channel of LSTM in OAR')
    parser.add_argument(
        '--OAR_linear_hidden_channels',
        type=int,
        default=[512,128,64],
        nargs='+',
        help='fully connected layers in OAR module')
    parser.add_argument(
        '--M',
        type=int,
        default=5,
        help='the maxpooling window size of start point learner in OAR')
    parser.add_argument(
        '--OAR_dropout',
        type=float,
        default=0.5,
        help='the dropout rate in OAR LSTM')
    parser.add_argument(
        '--OAR_thresh',
        type=float,
        default=0.5,
        help='OAR action score threshold'
    )
    parser.add_argument(
        '--OAR_precent',
        type=float,
        default=0.05,
        help='OAR video level action judge precentage, if num(s>OAR_thresh)/len > OAR_precent, prediction True'
    )
    
    parser.add_argument(
        '--model',
        default=None,
        help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.00005,
        help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--optimizer',
        default='SGD',
        help='type of optimizer')
    parser.add_argument(
        '--nesterov',
        type=str2bool,
        default=False,
        help='use nesterov or not')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='training batch size')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=256,
        help='test batch size')
    parser.add_argument(
        '--forward-batch-size',
        type=int,
        default=16,
        help='Batch size during forward pass, must be factor of --batch-size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=100,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--optimizer-states',
        type=str,
        help='path of previously saved optimizer states')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='training_results/formal_train_v3.13--16/model.pt',
        help='path of previously saved training checkpoint')
    parser.add_argument(
        '--debug',
        type=str2bool,
        default=False,
        help='Debug mode; default false')

    return parser

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')