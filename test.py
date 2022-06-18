import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import codecs
from tqdm import tqdm
from utils import proposal_map,plot_score,video_level_prediction, plot_distributed
from utils import plot_time_prediction, plot_time_AUC, perf_measure, compute_AUC, plot_gt_prediction_compare
from metrics import detection_map

def test(model,x,label,video_names,args,experiment,plot=False):
    x = x.to(torch.float32)
    CPGB_output = model.forward_CPGB(x)
    proposals = model.CPGB.proposal_gen(CPGB_output["CPGB_out_scores"],CPGB_output["video_level_score"],label,video_thres=0.4,frame_thres=0.3)
    proposal_map(args.video_info_path,proposals,video_names,args.action_classes,args.window_size,args.window_stride,args.work_dir+experiment+'/proposals.csv')
    OAMB_out = model.inference_forward(x,training=False)
    if plot:
        plot_score(OAMB_out["per_frame_scores"].detach().cpu().numpy(), video_names, args.action_classes, args.work_dir, experiment,type='OAMB_scores')

    prediction = video_level_prediction(OAMB_out["per_frame_scores"],threshold=args.OAMB_thresh,kappa=8)
    return prediction, OAMB_out["per_frame_scores"].detach().cpu().numpy()

class eval():
    def __init__(self, args, experiment, device, plot=False):
        self.args = args
        self.experiment = experiment
        self.device = device
        self.plot = plot

    def forward(self, model, data_loader, epoch='OAMB_1', state='best'):
        self.model = model
        self.state = state
        self.data_loader = data_loader
        self.dataset_video_names = self.data_loader.dataset.sample_name
        self.prediction = []
        self.label = []
        self.video_names = []
        self.scores = []

        self.model.eval()
        process = tqdm(self.data_loader, dynamic_ncols=True)
        with torch.no_grad():
            for batch_idx, (batch_data, batch_label, index) in enumerate(process):
                batch_video_names = []
                with torch.no_grad():
                    batch_data = batch_data.float().to(self.device)   # ? for data paralall, this will be modified
                    batch_label = batch_label.float().to(self.device)
                for i in index.numpy():
                    batch_video_names.append(self.dataset_video_names[i])
                self.video_names += batch_video_names
                if len(batch_label.shape)==1:
                    batch_label = batch_label.unsqueeze(1)

                batch_prediction, batch_scores = test(self.model,batch_data,batch_label,batch_video_names,self.args,self.experiment,self.plot)
                self.scores.append(batch_scores[:,:,0])
                self.prediction.append(batch_prediction.cpu().numpy())
                self.label.append(batch_label.cpu().numpy())

        self.scores = np.array(np.concatenate(self.scores))
        self.prediction = np.array(np.concatenate(self.prediction))
        self.label = np.array(np.concatenate(self.label))
        
        # frame level performance
        self.dmap, self.iou = detection_map(self.scores, self.data_loader.dataset.frame_annotations, 
                                self.video_names)
        print('Test at IOU: ', self.iou)
        print('DetectionMAP: ', self.dmap)

        # video level performance
        self.matrix, self.results, self.wrong_indexs = perf_measure(self.label,self.prediction)
        self.auc = compute_AUC(self.scores,self.label,self.args.work_dir,self.experiment,kappa=8,final=True)
        print('*************************************************')
        print(self.matrix)
        print(self.results)
        print("AUC: ",self.auc)
        print('*************************************************')

        self.save_results(epoch, self.state, self.plot)

        return self.results['acc'], self.dmap

    def save_results(self,epoch, state, plot=False):
        # save video level results
        np.save(self.args.work_dir +self.experiment+ f'/{state}scores.npy',self.scores)
        np.save(self.args.work_dir +self.experiment+ f'/{state}video_names.npy',np.array(self.video_names))
        np.save(self.args.work_dir +self.experiment+ f'/{state}labels.npy',self.label)

        if plot:
            plot_distributed(self.scores, self.args, state=state, threshold=self.args.OAMB_thresh)
            plot_time_AUC(self.scores,self.label,kappa=8,work_dir=self.args.work_dir,state=state,experiment=self.experiment)
            plot_time_prediction(self.scores,self.label,self.args.work_dir,self.experiment,
                                threshold=self.args.OAMB_thresh,state=state,precentage=self.args.OAMB_precent)     # change from 0.1 to 0.05
        
            plot_gt_prediction_compare(self.data_loader.dataset.frame_annotations, self.scores, self.video_names, self.args)

        video_level_results = self.args.work_dir +self.experiment+ f'/{state}video_level_results.txt'
        f = codecs.open(video_level_results, mode='a', encoding='utf-8')
        out = [f'\nresults of {epoch}\n',
            'TP, FP, TN, FN = {},{},{},{}\n'.format(self.matrix['TP'], self.matrix['FP'], self.matrix['TN'], self.matrix['FN']),
            f"Accuracy = {self.results['acc']}\n",
            f"Precision = {self.results['precision']}\n",
            f"Recall = {self.results['recall']}\n",
            f"Specificity = {self.results['Specificity']}\n",
            f"F1 = {self.results['F1']}\n",
            f'auc = {self.auc}\n']
        tplt = "{0:^30}\t{1:^15}\t{2:^15}\n"
        out.append(tplt.format("video name", "ground truth","prediction",chr(12288)))
        for i in self.wrong_indexs:
            out.append(tplt.format(str(self.video_names[i]),str(self.label[i]),str(self.prediction[i]),chr(12288)))
        f.writelines(out)
        f.close()

        # save frame level results
        frame_level_results = self.args.work_dir +self.experiment+ f'/{state}frame_level_results.txt'
        f = codecs.open(frame_level_results, mode='a', encoding='utf-8')
        out = [f'\nresults of {epoch}\n',
            f"Test at IOU: {self.iou}\n",
            f"DetectionMAP: {self.dmap}\n"]
        f.writelines(out)
        f.close()

        torch.save(self.model.state_dict(),self.args.work_dir+self.experiment+f'/{state}model.pt')
