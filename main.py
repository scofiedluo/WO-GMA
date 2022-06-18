import os
import time
from torch.nn.modules.container import T
import yaml

import torch
import numpy as np
import random
from tqdm import tqdm
from train import train_epoch
from model.model import Model
from args import get_parser
from feeder import Feeder
from test import eval
from utils import plot_loss_curve,init_seed,save_arg


parser = get_parser()
p = parser.parse_args()
# if p.config is not None:
#     with open(p.config, 'r') as f:
#         default_arg = yaml.load(f,Loader=yaml.FullLoader)
#     key = vars(p).keys()
#     for k in default_arg.keys():
#         if k not in key:
#             print(k)
#             print('WRONG ARG:', k)
#             assert (k in key)
#     parser.set_defaults(**default_arg)

args = parser.parse_args()
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

experiments = args.experiment
if not os.path.exists(args.work_dir+experiments):
    os.makedirs(args.work_dir+experiments)
save_arg(args,experiments)

init_seed(args.seed)
net = Model(args, device)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr,
        betas=(0.9, 0.999), weight_decay=0.0005)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 20, gamma = 0.1)

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
ACC_best = 0
pap_best = 0
fap_best = 0

EVAL = eval(args=args, experiment=experiments, device=device, plot=False)
for epoch in range(args.start_epoch, args.num_epoch):
    # scheduler.step()
    net.OAMB.state = (torch.zeros(1,args.batch_size,args.LSTM_hidden_channels).to(device),torch.zeros(1,args.batch_size,args.LSTM_hidden_channels).to(device))
    loss = train_epoch(args=args,model=net,global_step=global_step,device=device).train_epoch_E2E(optimizer,train_loader,experiments,epoch)
    print(f"loss of epoch {epoch} = {loss.item()}")
    Loss.append(loss.item())
    ACC, dmap = EVAL.forward(model=net, data_loader=test_loader, epoch=f'OAMB {epoch}', state='')
    if ACC>ACC_best:
        ACC_best = ACC
        EVAL.save_results(f'OAMB {epoch}', state=f'best_{epoch}_', plot=True)

plot_loss_curve(args.work_dir,experiments,len(Loss), Loss,'E2E')
