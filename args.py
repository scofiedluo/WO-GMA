import argparse
import os


def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='WO_GMA')

    parser.add_argument('--work-dir', type=str, default='training_results/',
        # required=True,
        help='the work folder for storing results')
    parser.add_argument('--config', default='./config/train_config.yaml',
        help='path to the configuration file')

    parser.add_argument('--save-score', type=str2bool, default=False,
        help='if ture, the classification score will be stored')

    parser.add_argument('--seed', type=int, default=8,
        help='random seed')

    parser.add_argument('--feeder', default='feeder.feeder',
        help='data loader will be used')
    parser.add_argument('--experiment', default='demo', type = str,
        help='sub folder parameter')
    parser.add_argument('--num-worker', type=int, default=os.cpu_count(),
        help='the number of worker for data loader')

    # data
    parser.add_argument('--data_path', type=str,
        default='data/GMs_data/all_v3/processed_data_joint.npy',
        help='the path to all data')
    parser.add_argument('--label_path', type=str,
        default='data/GMs_data/all_v3/processed_label.pkl',
        help='the path to all labels')
    parser.add_argument('--video_info_path', type=str,
        default='data/video_info/video_info.csv',
        help='the path to video info')
    parser.add_argument('--split_test_csv_path', type=str,
        default='data/GMs_data/split_v5/fold3_val.csv',
        help='the data split csv')
    parser.add_argument('--split_train_csv_path', type=str,
        default='data/GMs_data/split_v5/fold3_train.csv',
        help='the data split csv')
    parser.add_argument('--frame_annotations', type=str,
        default='data/frame_level_annotations/temporal_labels_v2.csv',
        help='path to frame annotations .csv file')

    parser.add_argument('--action_classes', type=str, default=['F+'], nargs='+',
        help='action classes list of datasets')

    # model parameters
    parser.add_argument('--skeleton_feature_dim', type=int, default=3,
        help='the arguments of original skeleton feature dim, 3 or 2')
    parser.add_argument("--conv3D_out_dim", type=int, default=256,
        help='the output dim of MS_G3D block')
    parser.add_argument('--graph', type=str, default='graph.kinetics.AdjMatrixGraph',
        help='graph class of skeleton')
    parser.add_argument('--num_scales', type=int, default=8,
        help='the scales of multi scale graph convlution')
    parser.add_argument('--window_size', type=int, default=20,
        help='window of time axis, the skeleton numbers in a window is window_size. this param will not change the output time length')
    parser.add_argument('--window_stride', type=int, default=20,
        help='slide length of between two windows. The output time length T1 = [T/window_stride], [] means rounded up.')
    parser.add_argument('--window_dilation', type=int, default=1,
        help='gap between skeletons in one window, eg: sk1---sk2---sk3, window_size = 3, dilation = 4')
    parser.add_argument('--vertices_fusion_out_dim', type=int, default=2048,
        help='the out channel of vertices_fusion module')
    parser.add_argument('--CPGB_hidden_channels', type=int, default=[512,128,64], nargs='+',
        help = 'fully connected layers in temporal proposals module')
    parser.add_argument('--num_class', type=int, default=1,
        help='num of action classes in dataset')
    parser.add_argument('--kappa', type=int, default=8,
        help='a parameter to contral topk in proportion time length')
    parser.add_argument('--CPGB_dropout', type=float, default=0.5,
        help='the dropout rate in CPGB')
    parser.add_argument('--CPGB_activation', type=str, default='relu',
        help='the activation function of CPGB')
    parser.add_argument('--LSTM_hidden_channels', type=int, default=1024,
        help='the hidden channel of LSTM in OAMB')
    parser.add_argument('--OAMB_linear_hidden_channels', type=int, default=[512,128,64], nargs='+',
        help='fully connected layers in OAMB module')
    parser.add_argument('--OAMB_dropout', type=float, default=0.5,
        help='the dropout rate in OAMB LSTM')
    parser.add_argument('--OAMB_thresh', type=float, default=0.5,
        help='OAMB action score threshold')
    parser.add_argument('--OAMB_precent', type=float, default=0.05,
        help='OAMB video level action judge precentage, if num(s>OAMB_thresh)/len > OAMB_precent, prediction True')

    parser.add_argument('--weights', default=None,
        help='the weights for network initialization')
    parser.add_argument('--base-lr', type=float, default=0.00005,
        help='initial learning rate')
    parser.add_argument('--device', type=int, default=0,
        help='the index of GPU for training or testing')
    parser.add_argument('--batch-size', type=int, default=16,
        help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=16,
        help='test batch size')
    parser.add_argument('--start_epoch', type=int, default=0,
        help='stop training in which epoch')
    parser.add_argument('--num-epoch', type=int, default=100,
        help='stop training in which epoch')
    parser.add_argument('--checkpoint', type=str,
        default='training_results/formal_train_v3.13--16/model.pt',
        help='path of previously saved training checkpoint')

    return parser


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
