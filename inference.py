import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import os
import codecs
from memory_profiler import profile

from model.model import model
from utils import perf_measure,init_seed,plot_time_prediction,save_arg,plot_time_AUC,compute_PAP_result,compute_FAP_result,start_scores_processor
from args import get_parser
from feeder import Feeder
from test import eval

parser = get_parser()
p = parser.parse_args()
if p.config is not None:
    with open(p.config, 'r') as f:
        default_arg = yaml.load(f,Loader=yaml.FullLoader)
    key = vars(p).keys()
    for k in default_arg.keys():
        if k not in key:
            print('WRONG ARG:', k)
            assert (k in key)
    parser.set_defaults(**default_arg)

args = parser.parse_args()
experiments = args.experiment
if not os.path.exists(args.work_dir+experiments):
    os.makedirs(args.work_dir+experiments)
save_arg(args,experiments)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

net = model(args, device).to(device)
net.load_state_dict(torch.load(args.checkpoint))

init_seed(args.seed)
test_loader = torch.utils.data.DataLoader(
                dataset=Feeder(args.data_path,args.label_path,args.split_test_csv_path,args.frame_annotations),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_worker,
                drop_last=False,
                worker_init_fn=np.random.seed(args.seed))

EVAL = eval(args=args, experiment=experiments, device=device, plot=True)
F1, pap, fap = EVAL.forward(model=net, data_loader=test_loader, epoch='1', state='best_')
# acc, pap, fap = eval(model=net, data_loader=test_loader, args=args, 
#                     experiment=experiments, device=device, state='best', plot=False)
