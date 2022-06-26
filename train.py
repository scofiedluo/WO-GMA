import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import topK


class train_epoch():
    """
    @param
    """
    def __init__(self, args, model, global_step, device):
        super().__init__()
        self.model = model
        self.global_step = global_step
        self.args = args
        self.device = device

    def mill_loss(self, topk_score, labels, binary=True):
        """
        topk_score has the same shape with labels, (N,num_class), where N means batch size
        for dataset has one action class only, sigmoid should replace log_softmax
        """
        if binary:
            milloss = nn.BCELoss()(topk_score,labels)
        else:
            # topk.shape = (N,1)
            milloss = -torch.mean(torch.sum(labels * F.log_softmax(topk_score, dim=1), dim=1),
                                  dim=0)
        return milloss

    def frame_loss(self, proposals, per_frame_scores):
        """
        frame loss of online action modeling branch

        @param
        proposals: N*T*num_class, proposals of CPGB, note that if all num_class are zero
                   before a time point, then this time point is start point.
        per_frame_scores: N*T*(num_class+1)
        start_point: shape: N*T*2
        """
        # append the backgraound lable in proposals
        tmp,_ = torch.max(proposals,dim=2,keepdim=True)   # shape N*T*1
        inv_tmp = tmp
        inv_tmp = torch.ones_like(tmp,dtype=float).to(self.device) - inv_tmp
        proposals = torch.cat([proposals,inv_tmp],dim=2)  # N*T*(num_class+1)

        # frame_loss
        total_T = proposals.shape[0] * proposals.shape[1]
        frame_loss = -torch.sum(proposals * torch.log(per_frame_scores)) / total_T

        return frame_loss

    def train_epoch_E2E(self, optimizer, data_loader, experiment, epoch):
        self.model.train()
        dataset_video_names = data_loader.dataset.sample_name
        batch_loss = 0
        process = tqdm(data_loader, dynamic_ncols=True)
        for batch_idx, (data, label, index) in enumerate(process):
            index = index.cpu().numpy()
            video_names = []
            for i in index:
                video_names.append(dataset_video_names[i])
            self.global_step += 1
            optimizer.zero_grad()
            with torch.no_grad():
                data = data.float().to(self.device)
                label = label.float().to(self.device)
            if len(label.shape)==1:
                label = label.unsqueeze(1)

            tmp_loss = torch.zeros(1, requires_grad=True).to(self.device)
            CPGB_output = self.model.forward_CPGB(data)  # CPGB_output is a dic
            proposals = self.model.CPGB.proposal_gen(CPGB_output["CPGB_out_scores"],
                                                     CPGB_output["video_level_score"],
                                                     label, video_thres=0.4, frame_thres=0.3)

            if label.shape[1]==1:
                binary = True
            else:
                binary = False
            milloss1 = self.mill_loss(CPGB_output["video_level_score"], label, binary)
            self.model.OAMB.state = detach(self.model.OAMB.state)
            OAMB_output = self.model.inference_forward(data)
            framloss = self.frame_loss(proposals, OAMB_output["per_frame_scores"])
            scores = OAMB_output["per_frame_scores"][:,:,:-1]  # drop background demension
            video_level_score = topK(scores, kappa=8)
            milloss2 = self.mill_loss(video_level_score, label, binary)

            tmp_loss = milloss1 + framloss + milloss2
            print("batch:", batch_idx, ", MIL Loss1 = ", milloss1.item(),
                  ", MIL Loss2 = ", milloss2.item(), ", frame Loss = ", framloss.item(),
                  ", loss_sum = ", tmp_loss.item())
            batch_loss += tmp_loss
            tmp_loss.backward()
            optimizer.step()
        return batch_loss / len(process)


def detach(states):
    """
    pytorch Computational graph gradient detach
    """
    return [state.detach() for state in states]
