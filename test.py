import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import codecs
from tqdm import tqdm
from utils import proposal_map,plot_score,video_level_prediction,start_scores_processor,compute_PAP_result,compute_FAP_result
from utils import plot_time_prediction, plot_time_AUC, perf_measure, compute_AUC, plot_gt_prediction_compare, tsne_vis, getDetectionMAP1

# @profile
def test(model,x,label,video_names,args,experiment,plot=False):
    x = x.to(torch.float32)
    TPG_output = model.forward_TPG(x)
    proposals = model.TPG.proposal_gen(TPG_output["TPG_out_scores"],TPG_output["video_level_score"],label,video_thres=0.4,frame_thres=0.3)
    proposal_map(args.video_info_path,proposals,video_names,args.action_classes,args.window_size,args.window_stride,args.work_dir+experiment+'/proposals.csv')
    OAR_out = model.inference_forward(x,training=False)
    if plot:
        plot_score(OAR_out["per_frame_scores"].detach().cpu().numpy(), video_names, args.action_classes, args.work_dir, experiment,type='OAR_scores')
        plot_score(OAR_out["start_scores"].detach().cpu().numpy(), video_names, args.action_classes, args.work_dir, experiment,type='OAR_start')

    prediction = video_level_prediction(OAR_out["per_frame_scores"],threshold=args.OAR_thresh,precentage=args.OAR_precent)
    return prediction, OAR_out["per_frame_scores"].detach().cpu().numpy(), OAR_out["start_scores"].detach().cpu().numpy(), OAR_out["preprocessed_feature"].detach().cpu().numpy(), OAR_out["LSTM_feature"].detach().cpu().numpy()

class eval():
    def __init__(self, args, experiment, device, plot=False):
        self.args = args
        self.experiment = experiment
        self.device = device
        self.plot = plot

    def forward(self, model, data_loader, epoch='OAR_1', state='best'):
        self.model = model
        self.state = state
        self.data_loader = data_loader
        self.dataset_video_names = self.data_loader.dataset.sample_name
        self.prediction = []
        self.label = []
        self.video_names = []
        self.scores = []
        self.start_scores = []
        self.clip_features = []
        self.LSTM_features = []

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

                batch_prediction, batch_scores, batch_start_scores,batch_features, batch_LSTM_feature = test(self.model,batch_data,batch_label,batch_video_names,self.args,self.experiment,self.plot)
                self.scores.append(batch_scores[:,:,0])
                self.start_scores.append(batch_start_scores)
                self.prediction.append(batch_prediction.cpu().numpy())
                self.label.append(batch_label.cpu().numpy())
                self.clip_features.append(batch_features)
                self.LSTM_features.append(batch_LSTM_feature)

        self.scores = np.array(np.concatenate(self.scores))
        self.start_scores = np.array(np.concatenate(self.start_scores))[:,:,0]
        self.prediction = np.array(np.concatenate(self.prediction))
        self.label = np.array(np.concatenate(self.label))
        self.clip_features = np.array(np.concatenate(self.clip_features))
        self.LSTM_features = np.array(np.concatenate(self.LSTM_features))
        
        # frame level performance
        start_scores_processed = start_scores_processor(self.start_scores, 0.2)
        self.pap = {}
        dist_ths = [5, 10, 15, 20, 25, 30, 35, 40]
        total = 0.
        for dist_th in dist_ths:
            self.pap[dist_th] = compute_PAP_result(self.data_loader.dataset.frame_annotations, start_scores_processed, self.video_names, self.args, dist_th)
            total += self.pap[dist_th]
        self.pap["mean"] = total/len(dist_ths)
        self.fap = compute_FAP_result(self.data_loader.dataset.frame_annotations, self.scores, self.video_names, self.args)
        self.dmap, self.iou = getDetectionMAP1(self.scores, self.data_loader.dataset.frame_annotations, 
                                self.video_names)
        print('Test point mAP = ', self.pap)
        print('Test frame mAP = {:.5f}'.format(self.fap))
        print('Test at IOU: ', self.iou)
        print('DetectionMAP: ', self.dmap)

        # video level performance
        self.matrix, self.results, self.wrong_indexs = perf_measure(self.label,self.prediction)
        self.auc = compute_AUC(self.scores,self.label,self.args.work_dir,self.experiment,kappa=8,final=True)

        self.save_results(epoch, self.state, self.plot)

        return self.results['F1'], self.pap["mean"], self.fap

    def save_results(self,epoch, state, plot=False):
        # save video level results
        np.save(self.args.work_dir +self.experiment+ f'/{state}scores.npy',self.scores)
        np.save(self.args.work_dir +self.experiment+ f'/{state}video_names.npy',np.array(self.video_names))
        np.save(self.args.work_dir +self.experiment+ f'/{state}labels.npy',self.label)

        if plot:
            plot_time_AUC(self.scores,self.label,kappa=8,work_dir=self.args.work_dir,state=state,experiment=self.experiment)
            plot_time_prediction(self.scores,self.label,self.args.work_dir,self.experiment,
                                threshold=self.args.OAR_thresh,state=state,precentage=self.args.OAR_precent)     # change from 0.1 to 0.05
        
            plot_gt_prediction_compare(self.data_loader.dataset.frame_annotations, self.scores, self.video_names, self.args)

            clip_features = self.clip_features.swapaxes(1,2) # (n_sample*clip_nums*channel)
            clip_features = clip_features.reshape(-1,clip_features.shape[2])  # ((n_sample*clip_nums)*channel)
            LSTM_features = self.LSTM_features.reshape(-1,self.LSTM_features.shape[2])  # ((n_sample*clip_nums)*channel)
            tsne_vis(clip_features,self.scores.reshape(-1)>self.args.OAR_thresh,self.args, label_list = [True, False],state = 'local')
            tsne_vis(LSTM_features,self.scores.reshape(-1)>self.args.OAR_thresh,self.args, label_list = [True, False],state = 'LSTM')
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
        print('*************************************************')
        print(self.matrix)
        print(self.results)
        print("AUC: ",self.auc)
        print('*************************************************')

        # save frame level results
        frame_level_results = self.args.work_dir +self.experiment+ f'/{state}frame_level_results.txt'
        f = codecs.open(frame_level_results, mode='a', encoding='utf-8')
        out = [f'\nresults of {epoch}\n',
            f"Test frame mAP = {self.fap}\n",
            f"Test point mAP @ dist_th 5 = {self.pap[5]}\n",
            f"Test point mAP @ dist_th 10 = {self.pap[10]}\n",
            f"Test point mAP @ dist_th 15 = {self.pap[15]}\n",
            f"Test point mAP @ dist_th 20 = {self.pap[20]}\n",
            f"Test point mAP @ dist_th 25 = {self.pap[25]}\n",
            f"Test point mAP @ dist_th 30 = {self.pap[30]}\n",
            f"Test point mAP @ dist_th 35 = {self.pap[35]}\n",
            f"Test point mAP @ dist_th 40 = {self.pap[40]}\n",
            f"Test point mAP mean = {self.pap['mean']}\n",
            f"Test at IOU: {self.iou}\n",
            f"DetectionMAP: {self.dmap}\n"]
        f.writelines(out)
        f.close()

        torch.save(self.model.state_dict(),self.args.work_dir+self.experiment+f'/{state}model.pt')