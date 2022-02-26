import os
import time
import copy
import codecs
from typing import Optional
from matplotlib import lines
from torch.nn.modules.container import T
import yaml
import matplotlib.pyplot as plt

import torch
import numpy as np
import random
from tqdm import tqdm
from train_GPU import train_epoch
from model.model import model
from args import get_parser
from feeder import Feeder
from test import test,eval
from utils import plot_loss_curve,init_seed,perf_measure,save_arg,compute_PAP_result,compute_FAP_result,start_scores_processor


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
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

experiments = args.experiment
if not os.path.exists(args.work_dir+experiments):
    os.makedirs(args.work_dir+experiments)
save_arg(args,experiments)

init_seed(args.seed)
net = model(args, device)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr,
        betas=(0.9, 0.999), weight_decay=0.0005)

train_loader = torch.utils.data.DataLoader(
                dataset=Feeder(args.data_path,args.label_path,args.split_train_csv_path,args.frame_annotations),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_worker,
                drop_last=True,
                worker_init_fn=np.random.seed(args.seed))

test_loader = torch.utils.data.DataLoader(
                dataset=Feeder(args.data_path,args.label_path,args.split_test_csv_path,args.frame_annotations),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_worker,
                drop_last=False,
                worker_init_fn=np.random.seed(args.seed))

global_step = 0
Loss = []
F1_best = 0
pap_best = 0
fap_best = 0

TPG_epoch = []
OAR_epoch = []
for i in range(0, len(args.TPG_train_step), 2):
    TPG_epoch += [idx for idx in range(args.TPG_train_step[i], args.TPG_train_step[i+1])]
for i in range(0, len(args.OAR_train_step), 2):
    OAR_epoch += [idx for idx in range(args.OAR_train_step[i], args.OAR_train_step[i+1])]
# print("TPG_epoch",TPG_epoch)
# print("OAR_epoch",OAR_epoch)
EVAL = eval(args=args, experiment=experiments, device=device, plot=False)
for epoch in range(args.start_epoch, args.num_epoch):
    net.OAR.state = (torch.zeros(1,args.batch_size,args.LSTM_hidden_channels).to(device),torch.zeros(1,args.batch_size,args.LSTM_hidden_channels).to(device))
    loss = train_epoch(args=args,model=net,global_step=global_step,device=device).train_epoch_E2E(optimizer,train_loader,experiments,epoch)
    print(f"loss of epoch {epoch} = {loss.item()}")
    Loss.append(loss.item())
    F1, pap, fap = EVAL.forward(model=net, data_loader=test_loader, epoch=f'OAR {epoch}', state='')
    if F1>F1_best:
        F1_best = F1
        EVAL.save_results(f'OAR {epoch}', state='best_', plot=True)

plot_loss_curve(args.work_dir,experiments,len(Loss), Loss,'E2E')
