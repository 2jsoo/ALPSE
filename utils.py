## Import Library
from import_library import *

import warnings
warnings.filterwarnings(action="ignore")

## Code reference : 
    # https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/evaluate.py


## Utils
# IOU
def cal_iou(pred_box, target_box):
    """
    pred_box  : (#, 2) -> 2 : x_mid, width
    """
    box1_xmin = pred_box[..., 0:1] - pred_box[..., 1:2] / 2
    box2_xmin = target_box[..., 0:1] - target_box[..., 1:2] / 2

    box1_xmax = pred_box[..., 0:1] + pred_box[..., 1:2] / 2
    box2_xmax = target_box[..., 0:1] + target_box[..., 1:2] / 2

    xmin = torch.max(box1_xmin, box2_xmin)
    xmax = torch.min(box1_xmax, box2_xmax)

    intersect_length = (xmax - xmin).clamp(0)
    
    union_length = (box2_xmax - box2_xmin) + (box1_xmax - box1_xmin) - intersect_length

    iou = intersect_length / (union_length + 1e-6)

    return iou


def cal_diou(pred_box, target_box):
    """
    pred_box  : (#, 2) -> 2 : x_mid, width
    """
    iou = cal_iou(pred_box, target_box) 

    euclidean_distance = torch.abs(pred_box[..., 0:1] - target_box[..., 0:1]) # 중점 간 거리

    box1_xmin = pred_box[..., 0:1] - pred_box[..., 1:2] / 2
    box2_xmin = target_box[..., 0:1] - target_box[..., 1:2] / 2
    box1_xmax = pred_box[..., 0:1] + pred_box[..., 1:2] / 2
    box2_xmax = target_box[..., 0:1] + target_box[..., 1:2] / 2
    xmin = torch.min(box1_xmin, box2_xmin)
    xmax = torch.max(box1_xmax, box2_xmax)
    
    c = torch.abs(xmax - xmin)

    penalty_term = euclidean_distance / (c + 1e-6)

    diou = iou - penalty_term

    return diou

# NMS
def nms(data_boxes, iou_threshold, confidence_threshold, diou=False):
    # data_boxes : [class, confidence_score, xmid, width]
    data_boxes = [sub_box for sub_box in data_boxes if sub_box[1] > confidence_threshold]
    data_boxes = sorted(data_boxes, key = lambda x: x[1], reverse=True)

    data_boxes_after_nms = []
    while data_boxes:
        cur_box = data_boxes.pop(0)
        data_boxes_after_nms.append(cur_box)

        if diou :
            data_boxes = [sub_box for sub_box in data_boxes
                        if (cal_diou(torch.tensor(sub_box[2:]), torch.tensor(cur_box[2:])) < iou_threshold)
                        or (sub_box[0] != cur_box[0])] 
        else:
            data_boxes = [sub_box for sub_box in data_boxes
                            if (cal_iou(torch.tensor(sub_box[2:]), torch.tensor(cur_box[2:])) < iou_threshold)
                            or (sub_box[0] != cur_box[0])]
    return data_boxes_after_nms

def compute_average_precision(recall, precision):
    """ Compute AP for one class.
    Args:
        recall: (numpy array) recall values of precision-recall curve.
        precision: (numpy array) precision values of precision-recall curve.
    Returns:
        (float) average precision (AP) for the class.
    """
    # AP (AUC of precision-recall curve) computation using all points interpolation.
    # For mAP computation, you can find a great explaination below.
    # https://github.com/rafaelpadilla/Object-Detection-Metrics

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i -1], precision[i])

    ap = 0.0 # average precision (AUC of the precision-recall curve).
    for i in range(precision.size - 1):
        ap += (recall[i + 1] - recall[i]) * precision[i + 1]

    return ap

def mean_average_precision(preds, targets, num_classes=1, iou_threshold=0.5, savepath=''):
    """ Compute mAP metric.
    Args:
        pred_boxes (list): list of lists containing all bboxes with each bboxes : [train_idx, class_prediction, prob_score, x_mid, width]
        threshold: (float) threshold for IoU to separate TP from FP.
    Returns:
        (list of float) list of average precision (AP) for each class.
    """
    aps = [] # list of average precisions (APs) for each class.
    precisions = []
    recalls = []
    f1s = []
    ious = []
    fig, ax = plt.subplots(figsize=(7, 8))
    for class_lbl in range(num_classes):
        class_preds = [] # all predicted objects for this class.
        ground_truths = []
        
        for detection in preds:
            if detection[1] == class_lbl:
                class_preds.append(detection)
        for true_box in targets:
            if true_box[1] == class_lbl:
                ground_truths.append(true_box)

        if len(class_preds) == 0:
            ap = 0.0 # if no box detected, assigne 0 for AP of this class.
            print('---class {} AP {}---'.format('AFIB', ap))
            aps.append(ap)
            break

        data_idxs = [pred[0]  for pred in class_preds]
        probs        = [pred[2]  for pred in class_preds]
        boxes        = [pred[3:] for pred in class_preds]

        # Sort lists by probs.
        sorted_idxs = np.argsort(probs)[::-1]
        data_idxs = [data_idxs[i] for i in sorted_idxs]
        boxes        = [boxes[i] for i in sorted_idxs]

        # Compute total number of ground-truth boxes. This is used to compute precision later.
        num_gt_boxes = 0
        num_gt_boxes += len([gt for gt in targets if (gt[1] == class_lbl)])

        # Go through sorted lists, classifying each detection into TP or FP.
        num_detections = len(boxes)
        tp = np.zeros(num_detections) # if detection `i` is TP, tp[i] = 1. Otherwise, tp[i] = 0.
        fp = np.ones(num_detections)  # if detection `i` is FP, fp[i] = 1. Otherwise, fp[i] = 0.

        for det_idx, (data_idx, box) in tqdm(enumerate(zip(data_idxs, boxes))):

            if len([gt for gt in targets if (gt[0] == data_idx) and (gt[1] == class_lbl)]) != 0:
                boxes_gt = [gt for gt in targets if (gt[0] == data_idx) and (gt[1] == class_lbl)]
                det_gt_iou = []
                for box_gt in boxes_gt:
                    # Compute IoU b/w/ predicted and groud-truth boxes.
                    iou_with_gt = cal_iou(torch.tensor(box), torch.tensor(box_gt[3:]))

                    if iou_with_gt >= iou_threshold:
                        ious.append(iou_with_gt.item())

                        tp[det_idx] = 1.0
                        fp[det_idx] = 0.0

                        boxes_gt.remove(box_gt) # each ground-truth box can be assigned for only one detected box.
                        if len(boxes_gt) == 0:
                            for remove_ in [gt for gt in targets if (gt[0] == data_idx) and (gt[1] == class_lbl)]:
                                targets.remove(remove_) # remove empty element from the dictionary.
                        break
                    else:
                        det_gt_iou.append(iou_with_gt.item())
                if len(det_gt_iou) != 0:
                    ious.append(max(det_gt_iou))
            else:
                ious.append(0)
                pass # this detection is FP.

        # Compute AP from `tp` and `fp`.
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        eps = np.finfo(np.float64).eps
        precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, eps)
        recall = tp_cumsum / float(num_gt_boxes)

        precision_ = (np.sum(tp)/(np.sum(tp)+np.sum(fp)))
        recall_ = (np.sum(tp)/(num_gt_boxes))
        f1_ = np.round(2 * (precision_ * recall_) / (precision_ + recall_ + 1e-6), 3)

        ap = compute_average_precision(recall, precision)
        if savepath != '':
            print('---class {} AP {}---'.format('AFIB', ap))
            print('---class {} Precision {}---'.format('AFIB', precision_))
            print('---class {} Recall {}---'.format('AFIB', recall_))
            print('---class {} F1 {}---'.format('AFIB', f1_))
        aps.append(ap)
        precisions.append(precision_)
        recalls.append(recall_)
        f1s.append(f1_)

        if savepath != '':
            display = PrecisionRecallDisplay(
                recall=recall,
                precision=precision,
                average_precision=np.trapz(precision, recall),
            )
            display.plot(ax=ax, name=f"Precision-recall for class label {class_lbl} AP {np.round(ap, 3)} Precision {np.round(precision_, 3)} Recall {np.round(recall_, 3)} F1 {np.round(f1_, 3)}")

            with open(f'{savepath}.txt', 'a') as f:
                f.write(f'class label {class_lbl} AP {ap} Precision {precision_} Recall {recall_} F1 {f1_}\n')
                f.write(f'class label {class_lbl} TP {max(tp_cumsum)} FP {max(fp_cumsum)} total_true_bboxes {num_gt_boxes}\n')

    if savepath != '':
        ax.set_xlim((-0.05, 1.05))
        ax.set_ylim((-0.05, 1.05))
        fig.savefig(f'{savepath}.png')
        plt.clf()
        plt.cla()

    # Compute mAP by averaging APs for all classes.
    if savepath != '':
        print('---mAP {}---'.format(np.mean(aps)))
        print('---mPrecision {}---'.format(np.mean(precisions)))
        print('---mRecall {}---'.format(np.mean(recalls)))
        print('---mF1 {}---'.format(np.mean(f1s)))
        print('---mIoU {}---'.format(np.mean(ious)))

    return np.mean(aps), np.mean(precisions), np.mean(recalls), np.mean(f1s), np.mean(ious)

def get_bboxes(loader, model, S, iou_threshold, confidence_threshold, class_threshold, diou, device="cuda"):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        with torch.no_grad():
            x = x.unsqueeze(1).float().to(device)
            labels = labels.to(device)

        
            predictions = model(x)

            batch_size = x.shape[0]
            true_bboxes = cellboxes_to_boxes(labels, S=S, class_threshold=class_threshold)
            bboxes = cellboxes_to_boxes(predictions, S=S, class_threshold=class_threshold)

            for idx in range(batch_size):
                nms_boxes = nms(
                    bboxes[idx],
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold,
                    diou=diou,
                )

                for nms_box in nms_boxes:
                    all_pred_boxes.append([train_idx] + nms_box)

                for box in true_bboxes[idx]:
                    # many will get converted to 0 pred
                    if box[1] > confidence_threshold:
                        all_true_boxes.append([train_idx] + box)

                train_idx += 1

    # model.train()
    return all_pred_boxes, all_true_boxes

def convert_cellboxes(predictions, S=9):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    bboxes1 = predictions[..., 2:4]
    bboxes2 = predictions[..., 5:7]
    scores = torch.cat((predictions[..., 1].unsqueeze(0), predictions[..., 4].unsqueeze(0)), dim=0)
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(S).repeat(batch_size, 1).unsqueeze(-1)
    x_mid = 1 / S * (best_boxes[..., :1] + cell_indices)
    width = 1 / S * best_boxes[..., 1:2]
    converted_bboxes = torch.cat((x_mid, width), dim=-1)
    predicted_class = predictions[..., :1].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 1], predictions[..., 4]).unsqueeze(-1)
    converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)
    return converted_preds


def cellboxes_to_boxes(out, S=15):
    converted_pred = convert_cellboxes(out, S,)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes
