import os
import csv
import random
import warnings
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import yaml
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from collections import OrderedDict
from sklearn.manifold import TSNE


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def proposal_map(video_info_path, proposals, video_names,
                 action_types, clip_length, stride, result_path):
    """
    map CPGB proposals to real video time size

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
        while index < len(list(video_info.video_name)):
            if video in list(video_info.video_name)[index]:
                break
            index += 1
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


def plot_score(per_frame_scores, video_names, action_types, work_dir,
               experiment='test', epoch=1, choice='CPGB_scores'):
    """
    plot action score respect to time

    @param:
    per_frame_scores(tensor or numpy array): the output of OAMB, shape N*T*(num_class+1)
    video_names(list): video names corrsponding to each scores in per_frame_scores, shape N
    action_types(list): all action type, inclue a background, shape N
    work_dir(str): eg:'/training_results'
    experiment: eg:'test_v1.0'
    final(bool): if plot is used for inference
    """
    if choice == 'CPGB_scores':
        results_dir = work_dir+experiment + f'/{choice}' + f'/epoch{epoch}'
        num_action = per_frame_scores.shape[2]
    elif choice == 'OAMB_scores':
        results_dir = work_dir + experiment + f'/{choice}'
        num_action = per_frame_scores.shape[2] - 1
    elif choice == 'OAMB_start':
        results_dir = work_dir + experiment+f'/{choice}'
        num_action = per_frame_scores.shape[2] - 1
    else:
        print(f'Warning! wrong parameter {choice}!')
        results_dir = work_dir + experiment+'/scores'
        num_action = 1

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # plt.rcParams['font.sans-serif']=['simhei']
    # plt.rcParams['axes.unicode_minus']=False
    plt.figure(figsize=(10, 3), clear=True)
    for i in range(per_frame_scores.shape[0]):
        for j in range(num_action):
            video_name = video_names[i]
            video_name = os.path.splitext(video_name)[0]
            action_type = action_types[j]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.plot(range(per_frame_scores.shape[1]), per_frame_scores[i, :, j],
                         lw=2, label=video_name + action_type)
                plt.title(video_name + '\taction:' + action_type)
                plt.savefig(results_dir + f'/{video_name}_{action_type}.png', format='png')
            plt.clf()
    plt.close('all')


def plot_loss_curve(work_dir, experiments, epoch, loss_list, branch = 'CPGB'):
    plt.figure()
    lw = 2
    plt.plot([i for i in range(epoch)], loss_list, color='darkorange',
              lw=lw, label='loss curve') 
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
    # os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def video_level_prediction(scores, threshold=0.5, kappa=8):
    """
    get video prediction on OAMB output scores
    """
    scores = scores[:, :, :-1]  # drop background demension
    video_level_score = topK(scores, kappa)
    prediction = video_level_score.ge(threshold)

    return prediction


def label_inv(label_path):
    with open(label_path, 'rb') as f:
        sample_name, label = pickle.load(f, encoding='latin1')
    label_inv = [1-i for i in label]
    with open(label_path, 'wb') as f:
        pickle.dump((sample_name, list(label_inv)), f)


def perf_measure(label, prediction):
    """
    video level performance
    """
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
    acc = (TP + TN) / (TP + TN + FP + FN)
    if TP+FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP+FN==0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    if TN+FP == 0:
        Specificity = 0
    else:
        Specificity = TN / (TN + FP)
    if precision != 0 or recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    results = {"acc": acc, "precision": precision,
             "recall": recall, "Specificity": Specificity, 'F1': f1}
    return matrix, results, index


def compute_AUC(scores, labels, work_dir, experiment,
                kappa, final=False, plot = False):
    if final:
        scores = torch.from_numpy(scores)
        if len(scores.shape) == 2:
            scores = scores.unsqueeze(2)
        video_level_scores = topK(scores, kappa)    # N*num_class
        if video_level_scores.shape[1] == 1:
            video_level_scores = torch.sigmoid(video_level_scores)
        else:
            video_level_scores = F.softmax(video_level_scores, dim=1)
        video_level_scores = video_level_scores.numpy()
        scores = video_level_scores
    fpr, tpr, threshold = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

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


def plot_time_prediction(scores, labels, work_dir, experiment,
                         threshold, state, precentage=0.2, kappa=8):
    time_step = [i for i in range(10, scores.shape[1], 5)]
    time_step.append(scores.shape[1])
    if scores.shape[1] % 10:
        time_step.append(scores.shape[1])

    scores = torch.from_numpy(scores)

    acc_list = []
    precision_list = []
    recall_list = []
    specificity_list = []
    f1_list = []
    for i in time_step:
        tmp_score = scores[:, :i]
        video_level_score = topK(tmp_score, kappa)
        prediction = video_level_score.ge(threshold)
        # tmp_score = tmp_score.ge(threshold)
        # time_length = tmp_score.shape[1]
        # video_level_scores = torch.sum(tmp_score,dim=1)
        # video_level_scores = video_level_scores/time_length
        # prediction = video_level_scores.ge(precentage)

        matrix, results, index = perf_measure(labels, prediction)
        acc_list.append(results['acc'])
        precision_list.append(results['precision'])
        recall_list.append(results['recall'])
        specificity_list.append(results['Specificity'])
        f1_list.append(results["F1"])

    results_dir = work_dir + experiment
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    csv_dic = {'time': time_step, 'acc': acc_list, 'precision': precision_list,
               'recall': recall_list, 'specificity': specificity_list, "F1": f1_list}
    dataframe = pd.DataFrame(csv_dic)
    dataframe.to_csv(os.path.join(results_dir, f'{state}time_prediction.csv'),
                     index=False, sep=',', header=True, encoding='gbk')

    plt.figure()
    plt.figure(figsize=(10, 3))
    # plt.ylim([0.2, 1.0])
    plt.plot(time_step,acc_list)
    plt.plot(time_step,precision_list)
    plt.plot(time_step,recall_list)
    plt.plot(time_step,specificity_list)
    plt.plot(time_step,f1_list)
    plt.legend(['accuracy', 'precision', 'recall', 'specificity', 'F1'])
    plt.savefig(results_dir + f'/{state}time_prediction.png', format='png')


def plot_time_AUC(scores, labels, kappa, work_dir, experiment, state):
    time_step = [i for i in range(10,scores.shape[1], 5)]
    time_step.append(scores.shape[1])
    if scores.shape[1] % 10:
        time_step.append(scores.shape[1])
    scores = torch.from_numpy(scores)
    if len(scores.shape) == 2:
        scores = scores.unsqueeze(2)

    auc_list = []
    for i in time_step:
        tmp_score = scores[:, :i, :]
        video_level_scores = topK(tmp_score, kappa)    #N*num_class
        if video_level_scores.shape[1] == 1:
            video_level_scores = torch.sigmoid(video_level_scores)
        else:
            video_level_scores = F.softmax(video_level_scores, dim=1)

        video_level_scores = video_level_scores.numpy()
        auc = compute_AUC(video_level_scores, labels, work_dir, experiment, kappa, plot=False)
        auc_list.append(auc)

    results_dir = work_dir + experiment
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.figure()
    plt.figure(figsize=(10, 3))
    plt.plot(time_step, auc_list)
    df = pd.DataFrame({'auc': auc_list})
    df.to_csv(os.path.join(results_dir, f'{state}time_auc.csv'),
              index=False,sep=',', header=True, encoding='gbk')
    plt.savefig(results_dir + f'/{state}time_auc.png', format='png')


def topK(scores, kappa):
    # scores.shape = N*T*num_class
    T = scores.shape[1]
    k = np.ceil(T/kappa).astype('int32')
    topk, _ = scores.topk(k, dim=1, largest=True, sorted = True)
    # video_level_score.shape = N*num_class
    video_level_score = topk.mean(axis=1, keepdim=False)
    return video_level_score


def save_arg(args, experiments):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.work_dir + experiments):
        os.makedirs(args.work_dir + experiments)
    with open(os.path.join(args.work_dir + experiments, 'config.yaml'), 'w') as f:
        yaml.dump(arg_dict, f)


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
            res.append((start, float(tup)))
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
    gt_npy = np.zeros((len(gt_indexs), 6000))
    for k, vid_gt in enumerate(gt_labels):
        for segement in vid_gt:
            if segement[0] > gt_npy.shape[1]:
                break
            gt_npy[k][int(segement[0]):min(int(gt_npy.shape[1]), int(segement[1]))] = 1
    
    # squeeze gt_npy according to window_size and window_stride
    gt_scores = np.zeros((len(gt_indexs), prediction.shape[1]))
    for i in range(prediction.shape[1] - 1):
        gt_scores[:,i] = np.mean(gt_npy[:, args.window_stride * i:args.window_stride * (i+1)],
                                 axis=1)
    gt_scores[:,-1] = np.mean(gt_npy[:, args.window_stride * (prediction.shape[1]-1):], axis=1)

    results_dir = os.path.join(args.work_dir, args.experiment) + '/GT_pred_compare'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    np.save(results_dir + '/gt_score.npy', gt_scores)
    np.save(results_dir + '/pred_score.npy', prediction[pre_indexs])

    # plt.rcParams['font.sans-serif']=['simhei']
    # plt.rcParams['axes.unicode_minus']=False
    plt.figure(figsize=(10, 3),clear=True)
    action_type = "F+"
    for k, gt_index in enumerate(gt_indexs):
        video_name = list(GTs['video_name'])[gt_index]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.plot(range(gt_scores.shape[1]), gt_scores[k,:], lw=2, label=video_name+action_type)
            plt.plot(range(prediction.shape[1]), prediction[pre_indexs[k],:],
                     lw=2, label=video_name+action_type)
            plt.legend(['Ground Truth', 'Prediction'])
            plt.title(video_name + '\taction:' + action_type)
            plt.savefig(results_dir + f'/{video_name}_{action_type}.png', format='png')
        plt.clf()


def plot_distributed(scores, args, state, threshold=0.5):
    """
    scores(ndarray):(num_video, num_clip)
    """
    count = [0 for i in range(scores.shape[1])]

    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            if scores[i][j] > threshold:
                count[j] += 1
    plt.figure()
    plt.plot(range(len(count)), count)
    results_dir = os.path.join(args.work_dir, args.experiment)
    plt.savefig(results_dir + '/prediction_distributing.png', format='png')
    count = np.array(count)
    np.save(results_dir + f'/{state}prediction_distributing.npy', count)
