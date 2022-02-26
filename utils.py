import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import csv
import torch
import os
import numpy as np
import random
import warnings
import pickle
from torch._C import _resolve_type_from_object
import yaml
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
from collections import OrderedDict
from sklearn.metrics import average_precision_score
from sklearn.manifold import TSNE


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def proposal_map(video_info_path,proposals,video_names,action_types, clip_length,stride,result_path):
    """
    map TPG proposals to real video time size

    @param:
    video_info_path(str): the path of video info csv, eg: ./video_info.csv
    proposals(numpy array or tensor): N*time_block, element with 0 or 1, shape N*T*num_class
    video_names(list): video names corrsponding to each proposal in proposals
    action_types(list): action class in datasets
    clip_length(int): num of frames in a time block
    stride(int): stride of time block
    result_path(str): eg: ./results.csv
    """
    assert len(video_names) == proposals.shape[0], 'proposal nums should equal to video nums'
    assert len(action_types) == proposals.shape[2], 'action classes nums should eaual'
    video_info = pd.read_csv(video_info_path, encoding='gbk')
    results = []
    for i in range(len(video_names)):
        video = video_names[i]
        video = os.path.splitext(video)[0]
        index = 0
        while index<len(list(video_info.video_name)):
            if video in list(video_info.video_name)[index]:
                break
            index += 1
        # index = list(video_info.video_name).index(video)
        clip_time = clip_length / video_info.FPS[index]
        stride_time = stride / video_info.FPS[index]
        
        for action_id in range(proposals.shape[2]):
            left = 0
            right = 0
            proposals_time = []
            while right < proposals.shape[1]:
                if proposals[i][right][action_id] == 1:
                    right += 1
                    if right==proposals.shape[1]:
                        proposals_time.append((left * stride_time, (right - 1) * stride_time + clip_time))
                        break
                else:
                    if right > left:
                        proposals_time.append((left * stride_time, (right - 1) * stride_time + clip_time))
                    while right < proposals.shape[1] and proposals[i][right][action_id] == 0:
                        right += 1
                    left = right
            results.append([video] +[action_types[action_id]]+ [len(proposals_time)] + proposals_time)
    with open(result_path, 'a', encoding='gbk', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        writer.writerows(results)


def plot_score(per_frame_scores, video_names, action_types, work_dir, experiment='test', epoch=1, type='TPG_scores'):
    """
    plot action score respect to time

    @param:
    per_frame_scores(tensor or numpy array): the output of OAR, shape N*T*(num_class+1)
    video_names(list): video names corrsponding to each scores in per_frame_scores, shape N
    action_types(list): all action type, inclue a background, shape N
    work_dir(str): eg:'/training_results'
    experiment: eg:'test_v1.0'
    final(bool): if plot is used for inference
    """
    if type == 'TPG_scores':
        results_dir = work_dir+experiment+f'/{type}'+f'/epoch{epoch}'
        num_action = per_frame_scores.shape[2]
    elif type == 'OAR_scores':
        results_dir = work_dir+experiment+f'/{type}'
        num_action = per_frame_scores.shape[2]-1
    elif type == 'OAR_start':
        results_dir = work_dir+experiment+f'/{type}'
        num_action = per_frame_scores.shape[2]-1
    else:
        print('wrong parameter type!')
        results_dir = work_dir+experiment+'/scores'
        num_action = 1
        
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # plt.rcParams["font.sans-serif"] = "SimHei"
    plt.rcParams['font.sans-serif']=['simhei']
    plt.rcParams['axes.unicode_minus']=False
    plt.figure(figsize=(10, 3),clear=True)
    for i in range(per_frame_scores.shape[0]):
        for j in range(num_action):
            video_name = video_names[i]
            video_name = os.path.splitext(video_name)[0]
            action_type = action_types[j]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.plot(range(per_frame_scores.shape[1]), per_frame_scores[i, :, j], lw=2, label=video_name + action_type)
                plt.title(video_name + '\taction:' +action_type)
                plt.savefig(results_dir + f'/{video_name}_{action_type}.png', format='png')
            plt.clf()
    plt.close('all')


def plot_loss_curve(work_dir, experiments, epoch,loss_list,branch = 'TPG'):
    plt.figure()
    lw = 2
    # plt.figure(figsize=(10,10))
    plt.plot([i for i in range(epoch)], loss_list, color='darkorange',
         lw=lw, label='loss curve') 
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'{branch}')
    plt.legend(loc="lower right")
    plt.savefig(work_dir + experiments + f'/{branch}_cost.png', format='png')


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def video_level_prediction(scores,threshold,precentage=0.2):
    """
    get video prediction on OAR output scores
    """
    scores = scores[:,:,:-1]
    scores = scores.ge(threshold)
    time_length = scores.shape[1]
    video_level_scores = torch.sum(scores,dim=1)
    video_level_scores = video_level_scores/time_length
    prediction = video_level_scores.ge(precentage)

    return prediction

def perf_measure(label, prediction):
    TP, FP, TN, FN = 0, 0, 0, 0

    index = []
    for i in range(label.size):
        if label[i] == 1 and prediction[i] == 1:
            TP += 1
        if label[i] == 0 and prediction[i] == 1:
            FP += 1
            index.append(i)
        if label[i] == 0 and prediction[i] == 0:
            TN += 1
        if label[i] == 1 and prediction[i] == 0:
            FN += 1
            index.append(i)
    matrix = {"TP":TP, "FP":FP, "TN":TN, "FN":FN}
    acc = (TP+TN)/(TP+TN+FP+FN)
    if TP+FP==0:
        precision = 0
    else:
        precision = TP/(TP+FP)
    if TP+FN==0:
        recall = 0
    else:
        recall = TP/(TP+FN)
    if TN+FP == 0:
        Specificity = 0
    else:
        Specificity = TN/(TN+FP)
    
    if precision!=0 or recall != 0:
        f1 = 2*precision*recall/(precision+recall)
    else:
        f1 = 0

    results = {"acc":acc,"precision":precision,
             "recall":recall,"Specificity":Specificity,'F1':f1}
    return matrix, results, index


def label_inv(label_path):
    with open(label_path, 'rb') as f:
        sample_name, label = pickle.load(f, encoding='latin1')
    label_inv = [1-i for i in label]
    with open(label_path, 'wb') as f:
        pickle.dump((sample_name, list(label_inv)), f)


def plot_time_prediction(scores,labels,work_dir,experiment,threshold,state,precentage=0.2):
    time_step = [i for i in range(10,scores.shape[1],5)]
    if scores.shape[1]%10:
        time_step.append(scores.shape[1])

    scores = torch.from_numpy(scores)

    acc_list = []
    precision_list = []
    recall_list = []
    specificity_list = []
    f1_list = []
    for i in time_step:
        tmp_score = scores[:,:i]
        tmp_score = tmp_score.ge(threshold)
        time_length = tmp_score.shape[1]
        video_level_scores = torch.sum(tmp_score,dim=1)
        video_level_scores = video_level_scores/time_length
        prediction = video_level_scores.ge(precentage)

        matrix, results, index = perf_measure(labels,prediction)
        acc_list.append(results['acc'])
        precision_list.append(results['precision'])
        recall_list.append(results['recall'])
        specificity_list.append(results['Specificity'])
        f1_list.append(results["F1"])
    

    results_dir = work_dir+experiment
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    csv_dic = {'time':time_step,'acc':acc_list,'precision':precision_list,'recall':recall_list,'specificity':specificity_list,"F1":f1_list}
    dataframe = pd.DataFrame(csv_dic)
    dataframe.to_csv(os.path.join(results_dir, f'{state}time_prediction.csv'),index=False,sep=',',header=True, encoding='gbk')

    plt.figure()
    plt.figure(figsize=(10, 3))
    # plt.ylim([0.2, 1.0])
    plt.plot(time_step,acc_list)
    plt.plot(time_step,precision_list)
    plt.plot(time_step,recall_list)
    plt.plot(time_step,specificity_list)
    plt.plot(time_step,f1_list)
    plt.legend(['accuracy','precision','recall','specificity','F1'])
    plt.savefig(results_dir + f'/{state}time_prediction.png', format='png')


def compute_AUC(scores,labels,work_dir,experiment,kappa,final=False, plot = False):
    if final:
        scores = torch.from_numpy(scores)
        if len(scores.shape)==2:
            scores = scores.unsqueeze(2)
        video_level_scores = topK(scores,kappa)    #N*num_class
        if video_level_scores.shape[1]==1:
            video_level_scores = torch.sigmoid(video_level_scores)
        else:
            video_level_scores = F.softmax(video_level_scores,dim=1)
        video_level_scores = video_level_scores.numpy()
        scores = video_level_scores
    fpr,tpr,threshold = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr,tpr)

    if plot:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) 
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        results_dir = work_dir+experiment
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        plt.savefig(results_dir + '/Roc.png', format='png')
    return roc_auc


def plot_time_AUC(scores,labels,kappa,work_dir,experiment,state):
    time_step = [i for i in range(10,scores.shape[1],5)]
    if scores.shape[1]%10:
        time_step.append(scores.shape[1])
    scores = torch.from_numpy(scores)
    if len(scores.shape)==2:
        scores = scores.unsqueeze(2)
    
    auc_list = []
    for i in time_step:
        tmp_score = scores[:,:i,:]
        video_level_scores = topK(tmp_score,kappa)    #N*num_class
        if video_level_scores.shape[1]==1:
            video_level_scores = torch.sigmoid(video_level_scores)
        else:
            video_level_scores = F.softmax(video_level_scores,dim=1)
        
        video_level_scores = video_level_scores.numpy()
        auc = compute_AUC(video_level_scores,labels,work_dir,experiment,kappa,plot = False)
        auc_list.append(auc)
    
    results_dir = work_dir+experiment
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.figure()
    plt.figure(figsize=(10, 3))
    plt.plot(time_step,auc_list)
    df = pd.DataFrame({'auc':auc_list})
    df.to_csv(os.path.join(results_dir,f'{state}time_auc.csv'),index=False,sep=',',header=True, encoding='gbk')
    plt.savefig(results_dir + f'/{state}time_auc.png', format='png')


def topK(scores,kappa):
    # scores.shape = N*T*num_class 
    T = scores.shape[1]
    k = np.ceil(T/kappa).astype('int32')
    topk,_ = scores.topk(k, dim=1, largest=True, sorted = True)
    # video_level_score.shape = N*num_class
    video_level_score = topk.mean(axis=1,keepdim=False)
    return video_level_score


def save_arg(args,experiments):
        # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.work_dir+experiments):
        os.makedirs(args.work_dir+experiments)
    with open(os.path.join(args.work_dir+experiments, 'config.yaml'), 'w') as f:
        yaml.dump(arg_dict, f)


def compute_FAP_result(GTs, score_metrics, video_names, args, verbose=False):
    """
    frame-based average precision (F-AP)
    @param:
    num_classes(int): class num in datasets
    score_metrics(np.array): prediction socres, (n_samples, n_clips)
    target_metrics(np.array): groud truth, (n_samples, n_clips)
    """
    result = OrderedDict()

    # restore to original size
    tmp = np.zeros((score_metrics.shape[0], score_metrics.shape[1]*args.window_size))
    for i in range(score_metrics.shape[1]):
        for j in range(args.window_size):
            tmp[:,args.window_size*i+j] = score_metrics[:,i]
    score_metrics = tmp

    # get inner join of test set and GTs
    target_metrics = np.zeros_like(score_metrics)
    sele_prediction_idx = []
    origin_vid_frames_num = []
    for k, item in enumerate(video_names):
        if item in list(GTs['video_name']):
            sele_prediction_idx.append(k)
            idx = list(GTs['video_name']).index(item)
            segments_frame = decode_str2list(GTs['segments_frame'][idx])
            end = score_metrics.shape[1]
            for segment in segments_frame:
                start = segment[0]
                end = min(segment[1], score_metrics.shape[1])
                for frame in range(int(start), int(end)):
                    target_metrics[k][frame] = 1
            origin_vid_frames_num.append(int(min(end,GTs['nframes'][idx])))
    score_metrics = np.array(score_metrics[sele_prediction_idx])
    target_metrics = np.array(target_metrics[sele_prediction_idx]).astype(np.int)

    # Compute AP
    res = []
    for i,frames_num in enumerate(origin_vid_frames_num):
        if np.sum(target_metrics[i][:frames_num])>0:   # ignore negative samples
            res.append(average_precision_score(target_metrics[i][:frames_num], score_metrics[i][:frames_num]))
    result['AP'] = np.mean(res)

    if verbose:
        print('mAP: {:.5f}'.format(result['AP']))
    return result['AP']


def compute_PAP_result(GTs, prediction, video_names, args, dist_th=5, rec_th =1.0):
    """

    param
    @GTs(dict): contains ground truth segments annotations
    @prediction(ndarray): start point prediction score, shape (num_video, n_clips), may contains more video than GTs
    @video_names(list): video name list corrsponding to prediction

    return
    mean average precision
    """
    result = OrderedDict()
    result['pointAP'] = OrderedDict()
    result['mAP'] = OrderedDict()

    npos = 0
    Res = dict()
    for k, v in enumerate(list(GTs['video_name'])):
        posct = 0
        segments_frame = decode_str2list(GTs['segments_frame'][k])
        for idx in range(len(segments_frame)):
            if segments_frame[idx][0] < min(6000, GTs['nframes'][k]):  # start frame shorter than video_len or frames
                posct += 1
        npos += posct
        Res[k] = [0 for _ in range(len(segments_frame))]
    
    # restore to original size
    tmp = np.zeros((prediction.shape[0], prediction.shape[1]*args.window_size))
    for i in range(prediction.shape[1]):
        tmp[:,args.window_size*i] = prediction[:,i]
    prediction = tmp
    
    videoIds = []
    times = []
    sele_prediction_idx = []
    gt_idx = []
    for k, item in enumerate(video_names):
        if item in list(GTs['video_name']):
            sele_prediction_idx.append(k)
            idx = list(GTs['video_name']).index(item)
            gt_idx.append(idx)
            frame_time = 1.0/GTs['FPS'][idx]
            for i in range(min(int(GTs['nframes'][idx]), prediction.shape[1])):  # ignore zero paddings
                videoIds.append(idx)
                times.append(frame_time*(i+1))
    
    confidence = []
    for i in range(len(sele_prediction_idx)):
        p_idx = sele_prediction_idx[i]
        g_idx = gt_idx[i]
        confidence.append(prediction[p_idx][:min(int(GTs['nframes'][g_idx]), prediction.shape[1])])
    confidence = np.concatenate(confidence)
    sorted_ind = np.argsort(-confidence)
    times = np.array(times)[sorted_ind]
    videoIds = np.array(videoIds)[sorted_ind]
    nd = len(times)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):       # each frame of each video
        # print(f"videoIds[{d}] = ", videoIds[d])
        ASs = decode_str2list(GTs['segments_time'][videoIds[d]])
        ASs = np.array([item[0] for item in ASs])
        time = times[d].astype(float)
        dist_min = np.inf
        if len(ASs) > 0:
            # compute absolute distance
            dists = np.abs(time - ASs)
            dist_min = np.min(dists)
            jmin = np.argmin(dists)
            # print(jmin, dist_min)
        if dist_min <= dist_th:
            if Res[videoIds[d]][jmin] == 0:
                tp[d] = 1.
                Res[videoIds[d]][jmin] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, True, rec_th)

    rec,prec,result['pointAP'] = rec, prec, ap
    # result['mAP'] = np.mean(list(result['pointAP'].values()))
    return ap


def decode_str2list(s):
    """
    s = "[(65.03, 77.31), (81.02, 87.02), (186.22, 197.14)]"
    """
    s = s[1:-1]
    res = []
    if not s:
        # print(res)
        return res
    s_list = s.split(',')
    for i, tup in enumerate(s_list):
        tup = tup.strip().strip('(').strip(')')
        if not i%2:
            start = float(tup)
        else:
            res.append((start,float(tup)))
    # print(res)
    return res

        
def voc_ap(rec, prec, use_07_metric=True, rec_th=1.0):
    """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., rec_th+rec_th/10.0, rec_th/10.0):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    return ap


def start_scores_processor(start_scores, threshold):
    """
    gen start point scores

    param
    @start_scores(ndarray): shape(n_videos, n_frames)

    return
    @start_scores_processed(ndarray): shape(n_videos, n_frames)
    """
    start_scores_processed = np.zeros_like(start_scores)
    for video_idx in range(start_scores.shape[0]):
        pre = 0
        frame_idx = 0
        while frame_idx < start_scores.shape[1]:
            if start_scores[video_idx][frame_idx] >= threshold and pre == 0:
                start_scores_processed[video_idx][frame_idx] = start_scores[video_idx][frame_idx]
                pre = 1
                while frame_idx < start_scores.shape[1] and start_scores[video_idx][frame_idx] >= threshold:
                    frame_idx += 1
            pre = 0
            while frame_idx < start_scores.shape[1] and start_scores[video_idx][frame_idx] < threshold:
                frame_idx += 1
    return start_scores_processed


def plot_gt_prediction_compare(GTs, prediction, video_names, args):
    """
    param
    @GTs(dict): contains ground truth segments annotations
    @prediction(ndarray): start point prediction score, shape (num_video, n_clips), may contains more video than GTs
    @video_names(list): video name list corrsponding to prediction

    """
    pre_indexs = []
    gt_labels = []
    gt_indexs = []
    for k, v in enumerate(list(GTs['video_name'])):
        if v in video_names:
            pre_indexs.append(video_names.index(v))
            gt_indexs.append(k)
            gt_labels.append(decode_str2list(GTs['segments_frame'][k]))
    
    # process gt_labels
    gt_npy = np.zeros((len(gt_indexs),6000))
    for k, vid_gt in enumerate(gt_labels):
        for segement in vid_gt:
            if segement[0] > gt_npy.shape[1]:
                break
            gt_npy[k][int(segement[0]):min(int(gt_npy.shape[1]), int(segement[1]))] = 1
    
    # squeeze gt_npy according to window_size and window_stride
    gt_scores = np.zeros((len(gt_indexs), prediction.shape[1]))
    for i in range(prediction.shape[1]-1):
        gt_scores[:,i] = np.mean(gt_npy[:,args.window_stride * i:args.window_stride*(i+1)], axis=1)
    gt_scores[:,-1] = np.mean(gt_npy[:,args.window_stride * (prediction.shape[1]-1):], axis=1)

    results_dir = os.path.join(args.work_dir, args.experiment) + f'/GT_pred_compare'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    plt.rcParams['font.sans-serif']=['simhei']
    plt.rcParams['axes.unicode_minus']=False
    plt.figure(figsize=(10, 3),clear=True)
    action_type = "F+"
    for k, gt_index in enumerate(gt_indexs):
        video_name = list(GTs['video_name'])[gt_index]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.plot(range(gt_scores.shape[1]), gt_scores[k,:], lw=2, label=video_name+action_type)
            plt.plot(range(prediction.shape[1]), prediction[pre_indexs[k],:], lw=2, label=video_name+action_type)
            plt.legend(['Ground Truth','Prediction'])
            plt.title(video_name + '\taction:' + action_type)
            plt.savefig(results_dir + f'/{video_name}_{action_type}.png', format='png')
        plt.clf()

def tsne_vis(feature, labels, args, n_components=2, label_list = ['F+', 'F-'], state = 'local'):
    """
    param
    @feature(numpy array): (num_samples, feature_dim)
    @labels(list or numpy array): (num_samples)
    """
    labels = np.array(labels)
    labels = labels[:20000]
    feature = feature[:20000,:]
    tsne = TSNE(n_components=n_components, init='pca', random_state=0)
    result = tsne.fit_transform(feature)

    # normalization and outlier remove
    topk = int(0.001*result.shape[0])
    index = np.argsort(-result[:,0])
    result = result[index[topk:]]
    labels = labels[index[topk:]]
    index = np.argsort(-result[:,1])
    result = result[index[topk:]]
    labels = labels[index[topk:]]

    result_min, result_max = np.min(result, 0), np.max(result, 0)
    result = (result - result_min) / (result_max - result_min)

    results_dir = os.path.join(args.work_dir, args.experiment)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    dot_colore = ['#FF8C00', '#4169E1']

    plt.figure(figsize=(10, 10),clear=True)
    for i, cls in enumerate(label_list):
        d = result[labels == label_list[i]]
        if label_list[i]:
            dot_label = "F+"
        else:
            dot_label = "F-"
        plt.scatter(d[:, 0], d[:, 1], c = dot_colore[i], label=dot_label)
    plt.legend(loc='upper left')
    plt.savefig(results_dir + f'/{state}T_SNE.png', format='png')


def getDetectionMAP1(predictions, GTs, video_names):
   iou_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
   dmap_list = []
   for iou in iou_list:
      # print('Testing for IoU %f' %iou)
      dmap_list.append(getLocMAP1(predictions, iou, GTs, video_names))
   return dmap_list, iou_list


def getLocMAP1(predictions, th, GTs, video_names):
   classlist = ['F+', 'F-']

   sele_prediction_idx = []
   gt_idx = []
   gts, gtl, vn, pred = [], [], [], []
   for k, item in enumerate(video_names):
      if item in list(GTs['video_name']):
         sele_prediction_idx.append(k)
         idx = list(GTs['video_name']).index(item)
         gt_idx.append(idx)
         segments_frame = decode_str2list(GTs['segments_frame'][idx])
         gts.append(segments_frame)
         gtl.append('F+')
         vn.append(item)
         pred.append(predictions[k])
   # print("test anno num: ", len(sele_prediction_idx))
   gtsegments = gts
   gtlabels = gtl
   videoname = vn
   predictions = pred

   templabelidx = [0]
   # process the predictions such that classes having greater than a certain threshold are detected only
   predictions_mod = []
   c_score = []
   for p in predictions:
      if p.shape[0]>300:      
         p = p[:300]
      p = np.expand_dims(p,1)   # ! this need modify for multiclass, i.e p = (T,num_class)
    #   pp = - p; pp.sort(); pp=-pp
      pp = - p; [pp[:,i].sort() for i in range(np.shape(pp)[1])]; pp=-pp # 
      c_s = np.mean(pp[:int(np.shape(pp)[0]/8),:],axis=0)   # 
      ind = c_s > 0.2       #? 0.0 -> 0.2
      c_score.append(c_s)
      predictions_mod.append(p*ind)
   predictions = predictions_mod

   detection_results = []
   for i,vn in enumerate(videoname):
      detection_results.append([])
      detection_results[i].append(vn)
   
   ap = []
   for c in templabelidx:  # c = 0
      segment_predict = []
      # Get list of all predictions for class c
      for i in range(len(predictions)):
         tmp = predictions[i][:,c]
         threshold = np.max(tmp) - (np.max(tmp) - np.min(tmp))*0.5
         vid_pred = np.concatenate([np.zeros(1),(tmp>threshold).astype('float32'),np.zeros(1)], axis=0)
         vid_pred_diff = [vid_pred[idt]-vid_pred[idt-1] for idt in range(1,len(vid_pred))]
         s = [idk for idk,item in enumerate(vid_pred_diff) if item==1]
         e = [idk for idk,item in enumerate(vid_pred_diff) if item==-1]
         for j in range(len(s)):
            aggr_score = np.max(tmp[s[j]:e[j]]) + 0.7*c_score[i][c]
            if e[j]-s[j]>=2:
               segment_predict.append([i,s[j],e[j], aggr_score])       
               detection_results[i].append([classlist[c], s[j], e[j], aggr_score])
      segment_predict = np.array(segment_predict)

      # Sort the list of predictions for class c based on score
      if len(segment_predict) == 0:
         return 0
      
      segment_predict = segment_predict[np.argsort(-segment_predict[:,3])]  # sorted by aggr_score

      # Create gt list
      segment_gt = []
      for i in range(len(gtsegments)):  # i is the index of video
         for j in range(len(gtsegments[i])):   # j is the index of segment
            if gtsegments[i][j][0]>=6000:
               break
            segment_gt.append([i, int(gtsegments[i][j][0]/20), int(gtsegments[i][j][1]/20)])
      gtpos = len(segment_gt)

      # Compare predictions and gt
      tp, fp = [], []
      for i in range(len(segment_predict)):
         flag = 0.
         for j in range(len(segment_gt)):
            if segment_predict[i][0]==segment_gt[j][0]:
               gt = range(int(segment_gt[j][1]), int(segment_gt[j][2]))
               p = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
               IoU = float(len(set(gt).intersection(set(p))))/float(len(set(gt).union(set(p))))
               if IoU >= th:
                  flag = 1.
                  del segment_gt[j]
                  break
         tp.append(flag)
         fp.append(1.-flag)
      tp_c = np.cumsum(tp)
      fp_c = np.cumsum(fp)
      rec = tp_c/float(gtpos)
      prec = tp_c / np.maximum(tp_c + tp_c, np.finfo(np.float64).eps)
      # if sum(tp)==0:
      #    prc = 0.
      # else:
      #    prc = np.sum((tp_c/(fp_c+tp_c))*tp)/gtpos
      ap.append(voc_ap(rec, prec))
   
   return 100*np.mean(ap)