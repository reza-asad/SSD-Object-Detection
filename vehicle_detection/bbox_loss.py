import torch.nn as nn
import torch.nn.functional as F
import torch

def hard_negative_mining(predicted_prob, gt_label, neg_pos_ratio=3.0):
    """
    The training sample has much more negative samples, the hard negative mining produces balanced
    positive and negative examples.
    :param predicted_prob: predicted probability for each prior item, dim: (N, H*W*num_prior)
    :param gt_label: ground_truth label, dim: (N, H*W*num_prior)
    :param neg_pos_ratio:
    :return:
    """
    pos_flag = gt_label > 0                                        # 0 = negative label

    # Sort the negative samples
    predicted_prob[pos_flag] = -1.0                                # temporarily remove positive by setting -1
    _, indices = predicted_prob.sort(dim=1, descending=True)       # sort by descend order, the positives are at the end
    _, orders = indices.sort(dim=1)                                # sort the negative samples by its original index

    # Remove the extra negative samples
    num_pos = pos_flag.sum(dim=1, keepdim=True)                     # compute the num. of positive examples
    num_neg = neg_pos_ratio * num_pos                               # determine of neg. examples, should < neg_pos_ratio
    neg_flag = orders < num_neg                                     # retain the first 'num_neg' negative samples index.

    return pos_flag, neg_flag


class MultiboxLoss(nn.Module):

    def __init__(self, iou_threshold=0.5, neg_pos_ratio=1.0):
        super(MultiboxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.neg_label_idx = 0

    def forward(self, confidence, pred_loc, gt_class_labels, gt_bbox_loc):
        """
         Compute the Multibox joint loss:
            L = (1/N) * L_{loc} + L_{class}
        :param confidence: predicted class probability, dim: (N, H*W*num_prior, num_classes)
        :param pred_loc: predicted prior bounding boxes, dim: (N, H*W*prior_num, 4)
        :param gt_class_labels: ground-truth class label, dim:(N, H*W*num_prior)
        :param gt_bbox_loc: ground-truth bounding box for prior, dim: (N, H*W*num_prior, 4)
        :return:
        """
        # Do the hard negative mining and produce balanced positive and negative examples
        with torch.no_grad():
            neg_class_prob = -F.log_softmax(confidence, dim=2)[:, :, self.neg_label_idx]      # select neg. class prob.
            pos_flag, neg_flag = hard_negative_mining(neg_class_prob, gt_class_labels, neg_pos_ratio=self.neg_pos_ratio)
            sel_flag = pos_flag | neg_flag
            num_pos = pos_flag.sum(dim=1, keepdim=True).float().sum()

        # Loss for the classification
        if num_pos == 0:
            print("No Positive")
        num_classes = confidence.shape[2]
        sel_conf = confidence[sel_flag]
        # print("Label Prediction:")
        # print(sel_conf.reshape(-1, num_classes))
        # print("Labe Ground Truth")
        # print(gt_class_labels[sel_flag])
        conf_loss = F.cross_entropy(sel_conf.view(-1, num_classes),
                                    gt_class_labels[sel_flag],
                                    reduction='sum') / num_pos

        # Loss for the bounding box prediction
        # TODO: implementation on bounding box regression
        pos_idx = pos_flag.unsqueeze(2).expand_as(pred_loc)
        # print("Location Prediction:")
        # print(pred_loc[pos_idx].reshape(-1, 4))
        # print("Location Ground Truth")
        # print(gt_bbox_loc[pos_idx].reshape(-1, 4))
        loc_huber_loss = F.smooth_l1_loss(pred_loc[pos_idx].view(-1, 4),
                                          gt_bbox_loc[pos_idx].view(-1, 4),
                                          reduction='sum') / num_pos


        return conf_loss, loc_huber_loss