import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import plot_score


class train_epoch():
    """
    @param
    
    """
    def __init__(self,args,model,global_step,device):
        super().__init__()
        self.model = model
        self.global_step = global_step
        self.args = args
        self.device = device

    def MILoss(self,topk_score,labels,binary=True):
        """
        topk_score has the same shape with labels, (N,num_class), where N means batch size
        for dataset has one action class only, sigmoid should replace log_softmax
        """
        # print("topk_score.shape:",topk_score.shape, "      labels.shape:",labels.shape)
        # print(topk_score)
        # print(labels)
        if binary:
            # milloss = F.mse_loss(topk_score,labels)
            milloss = nn.BCELoss()(topk_score,labels)
            # milloss = torch.mean(torch.sum(labels * torch.sigmoid(topk_score), dim=1), dim=0)
            # milloss = torch.mean(torch.sum(labels * torch.sigmoid(topk_score), dim=1), dim=0)
        else:
            milloss = -torch.mean(torch.sum(labels * F.log_softmax(topk_score,dim=1), dim=1), dim=0) # topk.shape = (N,1)
        return milloss

    def CASLoss(self,features, scores, labels, num_similar):
        """
        Co-Activity Similarity loss
        # TODO: this loss may be refactored to data with different time length in one batch

        @param
        features(tensor): N*f*T, the output of original skeleton data processed by feature_preprocess 
        scores(tensor): N*T*num_class, T is temporal block num, action socres of each temporal block
        labels(tensors): N*num_class, video level labels
        num_simialr(int): num of similar pair to compute CASLoss in a batch.
        device: compute on cpu or GPU
        """
        
        attention = F.softmax(scores,dim=1)         #N*T*num_class

        # print("attention.shape = ",attention.shape)
        # print("attention=",attention)
        # # print("features.shape = ",features.shape)
        # print("scores.shape = ",scores.shape)
        # print("scores=",scores)
        # print("labels.shape = ",labels.shape)
        
        agg_act = torch.bmm(features,attention)    # N*f*num_class = (N*f*T) \times (N*T*num_class)
        agg_back = torch.bmm(features,(1-attention)/attention.shape[1])    #N

        # print("agg_act.shape = ",agg_act.shape, agg_act)
        # print("agg_back.shape = ",agg_back.shape, agg_back)

        co_loss = torch.zeros(1).to(self.device)
        n_tmp = torch.zeros(1).to(self.device)       # num of classes in all pair
        for i in range(0,int(num_similar*2),2):      # num_similar*2 <= N
            Hf1 = agg_act[i]
            Hf2 = agg_act[i+1]
            Lf1 = agg_back[i]
            Lf2 = agg_back[i+1]

            d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))
            d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            co_loss = co_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).to(self.device))*labels[i,:]*labels[i+1,:])
            co_loss = co_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).to(self.device))*labels[i,:]*labels[i+1,:])
            n_tmp = n_tmp + torch.sum(labels[i,:]*labels[i+1,:])
        # print("n_tmp=",n_tmp)
        if n_tmp == 0:
            co_loss = co_loss
        else:
            co_loss = co_loss / n_tmp
        return co_loss

    def OARLoss(self, proposals, per_frame_scores, start_point):
        """
        Onlion action recognizer loss = frame loss + start loss

        @param
        proposals: N*T*num_class, proposals of TPG, note that if all num_class are zero before a time point, then this time point is start point.
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
        frame_loss = -torch.sum(proposals*torch.log(per_frame_scores))/total_T

        # start loss
        head = torch.zeros(proposals.shape[0],1,1).to(self.device)
        tmp = torch.cat([head,tmp],dim=1)   # shape N*(T+1)*1
        
        start_label = torch.zeros(proposals.shape[0],proposals.shape[1],2).to(self.device)
        for b in range(proposals.shape[0]):
            for t in range(1,proposals.shape[1]):
                # if previous time block has not action, 
                # and current time block has action, then current time block is action start point
                if tmp[b][t-1][0] == 0 and tmp[b][t][0]==1:
                    start_label[b][t][0] = 1.
                else:
                    start_label[b][t][1] = 1.
        start_loss = -torch.sum(start_label*torch.log(start_point))/total_T

        return frame_loss+start_loss
    
    def train_epoch(self, stage, optimizer, data_loader,experiment,epoch):
        """
        @param
        stage(str)：training stage
        """
        self.model.train()
        # self.model.OAR.state = (torch.zeros(1,self.args.batch_size,self.args.LSTM_hidden_channels),torch.zeros(1,self.args.batch_size,self.args.LSTM_hidden_channels))
        # device = self.model.device

        if stage == 'TPG':
            print("------------------------------Train TPG------------------------------")
            for param in self.model.feature_preprocess.parameters():
                param.requires_grad = True
            for param in self.model.TPG.parameters():
                param.requires_grad = True
        elif stage == 'OAR_full':
            print("------------------------------Train full OAR------------------------------")
            for param in self.model.feature_preprocess.parameters():
                param.requires_grad = True
            for param in self.model.TPG.parameters():
                param.requires_grad = True
        else:
            print("------------------------------Train OAR------------------------------")
            # freeze the feature_preprocess module while training OAR
            for param in self.model.feature_preprocess.parameters():
                param.requires_grad = False
            for param in self.model.TPG.parameters():
                param.requires_grad = False
        
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
                data = data.float().to(self.device)   # ? for data paralall, this will be modified
                label = label.float().to(self.device)
            
            # batch_video_names = dataset_video_names[index]
            # print(batch_video_names)
            
            if len(label.shape)==1:
                label = label.unsqueeze(1)

            tmp_loss = torch.zeros(1, requires_grad=True).to(self.device)
            TPG_output = self.model.forward_TPG(data)  #TPG_output is a dic
            proposals = self.model.TPG.proposal_gen(TPG_output["TPG_out_scores"],TPG_output["video_level_score"],label,video_thres=0.4,frame_thres=0.3)

            if stage == 'TPG':
                casLoss = self.CASLoss(TPG_output["preprocessed_feature"],TPG_output["TPG_out_scores"],label,self.args.batch_size/2)
                if label.shape[1]==1:
                    binary = True
                else:
                    binary = False
                milLoss = self.MILoss(TPG_output["video_level_score"],label,binary)
                print("MIL Loss = ",milLoss.item(),",\tCAS Loss = ",casLoss.item())
                tmp_loss = casLoss + milLoss
                # plot_score(TPG_output["TPG_out_scores"].detach().cpu().numpy(), video_names, self.args.action_classes, self.args.work_dir, experiment,epoch,type='TPG_scores')
            elif stage == 'OAR_full':
                self.model.OAR.state = detach(self.model.OAR.state)
                OAR_output = self.model.inference_forward(data)
                oarLoss = self.OARLoss(proposals,OAR_output["per_frame_scores"],OAR_output["start_scores"])
                print("OAR Loss = ", oarLoss.item())
                tmp_loss = oarLoss
            else:
                self.model.OAR.state = detach(self.model.OAR.state)
                OAR_output = self.model.forward_OAR(TPG_output["preprocessed_feature"])
                # state = OAR_output["state"]
                # proposals = self.model.TPG.proposal_gen(TPG_output["TPG_out_scores"],TPG_output["video_level_score"],label,video_thres=0.5,frame_thres=0.5)
                # print("sum of proposals = ", torch.sum(proposals))
                # print(proposals)
                oarLoss = self.OARLoss(proposals,OAR_output["per_frame_scores"],OAR_output["start_scores"])
                print("OAR Loss = ", oarLoss.item())
                tmp_loss = oarLoss
            
            print("batch:",batch_idx, ",\ttmp_loss = ",tmp_loss.item())
            batch_loss += tmp_loss
            # with torch.autograd.detect_anomaly():
            tmp_loss.backward()
            optimizer.step()
            # tmp_loss = 0
            # if (batch_idx+1) % self.args.batch_size == 0:
            #     batch_loss.backward()
            #     optimizer.step()
            #     batch_loss = 0
        
        return batch_loss/len(process)
    
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
                data = data.float().to(self.device)   # ? for data paralall, this will be modified
                label = label.float().to(self.device)
            if len(label.shape)==1:
                label = label.unsqueeze(1)

            tmp_loss = torch.zeros(1, requires_grad=True).to(self.device)
            TPG_output = self.model.forward_TPG(data)  #TPG_output is a dic
            proposals = self.model.TPG.proposal_gen(TPG_output["TPG_out_scores"],TPG_output["video_level_score"],label,video_thres=0.4,frame_thres=0.3)

            casLoss = self.CASLoss(TPG_output["preprocessed_feature"],TPG_output["TPG_out_scores"],label,self.args.batch_size/2)
            if label.shape[1]==1:
                binary = True
            else:
                binary = False
            milLoss = self.MILoss(TPG_output["video_level_score"],label,binary)
            print("MIL Loss = ",milLoss.item(),",\tCAS Loss = ",casLoss.item())
            self.model.OAR.state = detach(self.model.OAR.state)
            OAR_output = self.model.inference_forward(data)
            oarLoss = self.OARLoss(proposals,OAR_output["per_frame_scores"],OAR_output["start_scores"])
            print("OAR Loss = ", oarLoss.item())

            tmp_loss = milLoss + 0.1*casLoss + oarLoss
            print("batch:",batch_idx, ",\ttmp_loss = ",tmp_loss.item())
            batch_loss += tmp_loss
            # with torch.autograd.detect_anomaly():
            tmp_loss.backward()
            optimizer.step()
        return batch_loss/len(process)

def detach(states):
    return [state.detach() for state in states] 