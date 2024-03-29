import numpy as np
import torch
import torch.nn.functional as F
import yaml
import os

from model.model import Model
from utils import init_seed,save_arg
from args import get_parser
from feeder import Feeder
from test import Eval


def main():
    parser = get_parser()
    p = parser.parse_args()
    # if p.config is not None:
    #     with open(p.config, 'r') as f:
    #         default_arg = yaml.load(f,Loader=yaml.FullLoader)
    #     key = vars(p).keys()
    #     for k in default_arg.keys():
    #         if k not in key:
    #             print('WRONG ARG:', k)
    #             assert (k in key)
    #     parser.set_defaults(**default_arg)

    args = parser.parse_args()
    experiments = args.experiment
    if not os.path.exists(args.work_dir + experiments):
        os.makedirs(args.work_dir + experiments)
    save_arg(args,experiments)


    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    net = Model(args, device).to(device)
    net.load_state_dict(torch.load(args.checkpoint))

    init_seed(args.seed)
    test_loader = torch.utils.data.DataLoader(
                    dataset=Feeder(args.data_path,args.label_path,args.split_test_csv_path,args.frame_annotations,window_size=6000),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_worker,
                    drop_last=False,
                    worker_init_fn=np.random.seed(args.seed))

    evaluation = Eval(args=args, experiment=experiments, device=device, plot=True)
    acc, dmap = evaluation.forward(model=net, data_loader=test_loader, epoch='1', state='best_')


if __name__ == '__main__':
    main()
