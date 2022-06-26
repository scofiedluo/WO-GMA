import numpy as np
import pandas as pd
from utils import decode_str2list


def detection_map(predictions, gts_df, video_names):
    """
    param:
    gts_df: (dataframe) contains columns: video_name,nframes,FPS,duration (second),label,
         segments_frame,segments_time
    predictions: (list of numpy array)
    video_names: (list of str)

    return:
    mean ap over class, list, last element is average over IoU
    """
    iou_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    dmap_list = getlocmap(predictions, gts_df, video_names, iou_list, thres=0.7)
    dmap_list = list(dmap_list)
    dmap_list.append([sum(dmap_list)/len(dmap_list)])  # average over IoU
    return dmap_list, iou_list


def getlocmap(predictions, gts_df, video_names, iou_list, thres=0.7):
    """
    param:
    gts_df: (dataframe) contains columns: video_name,nframes,FPS,duration (second),label,
         segments_frame,segments_time
    predictions: (list of numpy array)
    video_names: (list of str)
    thres: (float)

    return:
    mean ap over class, np.array
    """
    classlist = ['F+', 'F-']

    sele_prediction_idx = []
    gt_idx = []
    gts, gtl, vns, pred = [], [], [], []
    for k, item in enumerate(video_names):
        if item in list(gts_df['video_name']):
            sele_prediction_idx.append(k)
            idx = list(gts_df['video_name']).index(item)
            gt_idx.append(idx)
            segments_frame = decode_str2list(gts_df['segments_frame'][idx])
            gts.append(segments_frame)
            gtl.append(gts_df['label'][idx])
            vns.append(item)
            pred.append(predictions[k])
    # print("test anno num: ", len(sele_prediction_idx))
    gtsegments = gts
    videoname = vns
    predictions = pred

    templabelidx = [0]   # modify this for multi-class
    # process the predictions such that classes
    # having greater than a certain threshold are detected only
    predictions_mod = []
    c_score = []
    for pred in predictions:
        if pred.shape[0] > 300:
            pred = pred[:300]
        pred = np.expand_dims(pred,1)   # ! this need modify for multiclass, i.e p = (T,num_class)
        # pp = - p; pp.sort(); pp=-pp
        desc_pred = np.sort(pred, axis=0)[::-1]
        # pp = - p; [pp[:,i].sort() for i in range(np.shape(pp)[1])]; pp=-pp
        c_s = np.mean(desc_pred[:int(np.shape(desc_pred)[0]/8),:],axis=0)
        ind = c_s > 0.2       #? 0.0 -> 0.2
        c_score.append(c_s)
        predictions_mod.append(pred * ind)
    predictions = predictions_mod

    detection_results = []
    for i, vidn in enumerate(videoname):
        detection_results.append([])
        detection_results[i].append(vidn)
    ap = []
    for cls_idx in templabelidx:  # c = 0
        segment_predict = []
        # Get list of all predictions for class c
        for i in range(len(predictions)):
            tmp = predictions[i][:, cls_idx]
            if thres is None:
                threshold = np.max(tmp) - (np.max(tmp) - np.min(tmp))*0.5
            else:
                threshold = thres
            vid_pred = np.concatenate([np.zeros(1), (tmp>threshold).astype('float32'), np.zeros(1)],
                                      axis=0)
            vid_pred_diff = [vid_pred[idt]-vid_pred[idt-1] for idt in range(1,len(vid_pred))]
            start = [idk for idk,item in enumerate(vid_pred_diff) if item==1]
            end = [idk for idk,item in enumerate(vid_pred_diff) if item==-1]
            for j in range(len(start)):
                aggr_score = np.max(tmp[start[j]:end[j]]) + 0.7*c_score[i][cls_idx]
                if end[j]-start[j]>=2:
                    segment_predict.append([i,start[j],end[j], aggr_score])     # i is the video id
                    detection_results[i].append([classlist[cls_idx], start[j], end[j], aggr_score])
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
        # gtpos = len(segment_gt)
        print("Number of ground truth instances:", len(segment_gt))
        print("Number of prediction instances:",len(segment_predict))
        segment_gt_df = pd.DataFrame({
            "video-id": list(np.array(segment_gt)[:,0]),
            "t-start": list(np.array(segment_gt)[:,1]),
            "t-end": list(np.array(segment_gt)[:,2])
        })
        segment_predict_df = pd.DataFrame({
            "video-id": list(np.array(segment_predict)[:,0]),
            "t-start": list(np.array(segment_predict)[:,1]),
            "t-end": list(np.array(segment_predict)[:,2]),
            "score": list(np.array(segment_predict)[:,3])
        })
        ap_iou = compute_average_precision_detection(segment_gt_df,
                                                     segment_predict_df, tiou_thresholds=iou_list)
        ap.append(ap_iou)
    return np.mean(ap, axis=0)*100   # mean over class


def compute_average_precision_detection(
    ground_truth,
    prediction,
    tiou_thresholds=np.linspace(0.1, 0.5, 5)
):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)
    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly ground truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap
