import torch
import numpy as np

''' Prior Bounding Box  ------------------------------------------------------------------------------------------------
'''


def generate_prior_bboxes(prior_layer_cfg=None, smin=0.2, smax=0.9):
    """
    Generate prior bounding boxes on different feature map level. This function used in 'cityscape_dataset.py'

    Use VGG_SSD 300x300 as example:
    Feature map dimension for each output layers:
       Layer    | Map Dim (h, w) | Single bbox size that covers in the original image
    1. Conv4    | (38x38)        | (30x30) (unit. pixels)
    2. Conv7    | (19x19)        | (60x60)
    3. Conv8_2  | (10x10)        | (111x111)
    4. Conv9_2  | (5x5)          | (162x162)
    5. Conv10_2 | (3x3)          | (213x213)
    6. Conv11_2 | (1x1)          | (264x264)
    NOTE: The setting may be different using MobileNet v3, you have to set your own implementation.
    Tip: see the reference: 'Choosing scales and aspect ratios for default boxes' in original paper page 5.
    :param prior_layer_cfg: configuration for each feature layer, see the 'example_prior_layer_cfg' in the following.
    :return prior bounding boxes with form of (cx, cy, w, h), where the value range are from 0 to 1, dim (1, num_priors, 4)
    """
    if prior_layer_cfg is None:
        prior_layer_cfg = [
            {'layer_name': 'Conv11', 'feature_dim_hw': (19, 19), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 1 / 5, 2.0, 3.0, 1.0)},
            {'layer_name': 'Conv13', 'feature_dim_hw': (10, 10), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 1 / 5, 2.0, 3.0, 1.0)},
            {'layer_name': 'Conv13_2', 'feature_dim_hw': (19, 19), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 1 / 5, 2.0, 3.0, 1.0)},
            # TODO: define your feature map settings
            {'layer_name': 'Conv14', 'feature_dim_hw': (5, 5), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 1 / 5, 2.0, 3.0, 1.0)},
            {'layer_name': 'Conv15', 'feature_dim_hw': (3, 3), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 1 / 5, 2.0, 3.0, 1.0)},
            {'layer_name': 'Conv16', 'feature_dim_hw': (2, 2), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 1 / 5, 2.0, 3.0, 1.0)},
            {'layer_name': 'Conv17', 'feature_dim_hw': (1, 1), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 1 / 5, 2.0, 3.0, 1.0)}
        ]

    priors_bboxes = []
    for feat_level_idx in range(0, len(prior_layer_cfg)):               # iterate each layers
        layer_cfg = prior_layer_cfg[feat_level_idx]
        layer_feature_dim = layer_cfg['feature_dim_hw']
        layer_aspect_ratio = layer_cfg['aspect_ratio']

        # Todo: compute S_{k} (reference: SSD Paper equation 4.)
        sk = smin + (smax - smin) / (len(prior_layer_cfg) - 1) * feat_level_idx
        fk = layer_feature_dim

        for y in range(0, layer_feature_dim[0]):
            for x in range(0,layer_feature_dim[0]):

                # Todo: compute bounding box center
                cx = (y + 0.5) / fk[0]
                cy = (x + 0.5) / fk[1]

                # Todo: generate prior bounding box with respect to the aspect ratio
                for aspect_ratio in layer_aspect_ratio[:-1]:
                    h = sk / np.sqrt(aspect_ratio)
                    w = sk * np.sqrt(aspect_ratio)
                    priors_bboxes.append([cx, cy, w, h])

                # Add additional prior bbox for the aspect ratio 1
                sk1 = smin + (smax - smin) / (len(prior_layer_cfg) - 1) * (feat_level_idx + 1)
                sk_prime = np.sqrt(sk * sk1)
                h = sk_prime
                w = sk_prime
                priors_bboxes.append([cx, cy, w, h])



    # Convert to Tensor
    priors_bboxes = torch.tensor(priors_bboxes)
    priors_bboxes = torch.clamp(priors_bboxes, 0.0, 1.0)
    num_priors = priors_bboxes.shape[0]

    # [DEBUG] check the output shape
    assert priors_bboxes.dim() == 2
    assert priors_bboxes.shape[1] == 4
    return priors_bboxes


def iou(gt_bboxes, prior_bboxes):
    """
    # Compute the Intersection over Union
    Note: function iou(a, b) used in match_priors
    :param a: bounding boxes, dim: (n_items, 4)
    :param b: bounding boxes, dim: (n_items, 4) or (1, 4) if b is a reference
    :return: iou value: dim: (n_item)
    """
    # [DEBUG] Check if input is the desire shape
    assert gt_bboxes.dim() == 2
    assert gt_bboxes.shape[1] == 4
    assert prior_bboxes.dim() == 2
    assert prior_bboxes.shape[1] == 4

    # TODO: implement IoU of two bounding box
    # Expand dimensions for both gt and prior bboxes
    N = gt_bboxes.shape[0]
    M = prior_bboxes.shape[0]
    prior = torch.ones((N, M, 4)) * prior_bboxes.unsqueeze(0)
    gt = gt_bboxes.unsqueeze(1)
    # Compute the top left and bottom right coordinates for intersection
    bottom_right = torch.min(gt[..., 2:], prior[..., 2:])
    top_left = torch.max(gt[..., :2], prior[..., :2])

    # Compute the intersection
    intersection = torch.clamp((bottom_right - top_left), min=0.0)
    intersection = torch.prod(intersection, dim=2)

    # Compute union
    area_gt = torch.prod((gt[..., 2:] - gt[..., :2]), dim=2)
    area_prior = torch.prod((prior[..., 2:] - prior[..., :2]), dim=2)
    union = area_gt + area_prior - intersection

    # Compute the iou
    iou = intersection / union

    # [DEBUG] Check if output is the desire shape
    assert iou.shape[0] == N
    assert iou.shape[1] == M
    return iou

def match_priors(prior_bboxes, gt_bboxes, gt_labels, iou_threshold=0.5):
    """
    Match the ground-truth boxes with the priors.
    Note: Use this function in your ''cityscape_dataset.py', see the SSD paper page 5 for reference.

    :param gt_bboxes: ground-truth bounding boxes, dim:(n_samples, 4)
    :param gt_labels: ground-truth classification labels, negative (background) = 0, dim: (n_samples)
    :param prior_bboxes: prior bounding boxes on different levels, dim:(num_priors, 4)
    :param iou_threshold: matching criterion
    :return matched_boxes: real matched bounding box, dim: (num_priors, 4)
    :return matched_labels: real matched classification label, dim: (num_priors)
    """
    # [DEBUG] Check if input is the desire shape
    assert gt_bboxes.dim() == 2
    assert gt_bboxes.shape[1] == 4
    assert gt_labels.dim() == 1
    assert gt_labels.shape[0] == gt_bboxes.shape[0]
    assert prior_bboxes.dim() == 2
    assert prior_bboxes.shape[1] == 4

    matched_boxes = torch.zeros_like(prior_bboxes)
    matched_labels = torch.zeros(len(prior_bboxes), dtype=torch.long)

    # TODO: implement prior matching
    # Compute the Jaccard's Similarity using the iou function.
    jaccard_sim = iou(gt_bboxes, center2corner(prior_bboxes))

    # Find the best prior matching gt and vice versa along with the
    # corresponding indices.
    best_prior_sim, best_prior_idx = jaccard_sim.max(1)
    best_gt_sim, best_gt_idx = jaccard_sim.max(0)

    # Make sure every best prior that was selected for a ground truth
    # is not eliminated.
    best_gt_sim.index_fill_(0, best_prior_idx, 1)

    # Make sure the ground truth that needs the prior more is selected.
    for i in range(len(best_prior_idx)):
        best_gt_idx[best_prior_idx[i]] = i

    # Remove the priors that do no meet the threshold overlap
    best_prior_idx = torch.arange(len(best_gt_idx))
    best_prior_idx = (best_prior_idx + 1) * (best_gt_sim > iou_threshold).long()
    best_prior_idx = best_prior_idx[best_prior_idx != 0]
    best_prior_idx -= 1

    # Remove the ground truth corresponding to the removed prior
    best_gt_idx = best_gt_idx[best_gt_sim > iou_threshold]

    # Convert the ground truth bounding boxes to center format
    matched_boxes[best_prior_idx] = corner2center(gt_bboxes[best_gt_idx])

    # Convert the ground truth bounding boxes to ssd locations.
    matched_boxes[best_prior_idx] = bbox2loc(matched_boxes[best_prior_idx],
                                             prior_bboxes[best_prior_idx])

    # Extract the ground truth labels for each prior
    matched_labels[best_prior_idx] = gt_labels[best_gt_idx]

    # [DEBUG] Check if output is the desire shape
    assert matched_boxes.dim() == 2
    assert matched_boxes.shape[1] == 4
    assert matched_labels.dim() == 1
    assert matched_labels.shape[0] == matched_boxes.shape[0]

    return matched_boxes, matched_labels


''' NMS ----------------------------------------------------------------------------------------------------------------
'''
def nms_bbox(bbox_loc, bbox_confid_scores, overlap_threshold=0.5, prob_threshold=0.445):
    """
    Non-maximum suppression for computing best overlapping bounding box for a object
    Use this function when testing the samples.

    :param bbox_loc: bounding box loc and size, dim: (num_priors, 4)
    :param bbox_confid_scores: bounding box confidence probabilities, dim: (num_priors, num_classes)
    :param overlap_threshold: the overlap threshold for filtering out outliers
    :return: selected bounding box with classes
    """

    # [DEBUG] Check if input is the desire shape
    assert bbox_loc.dim() == 2
    assert bbox_loc.shape[1] == 4
    assert bbox_confid_scores.dim() == 2
    assert bbox_confid_scores.shape[0] == bbox_loc.shape[0]

    sel_bbox = []

    # Todo: implement nms for filtering out the unnecessary bounding boxes
    num_classes = bbox_confid_scores.shape[1]
    for class_idx in range(1, num_classes):
        # Select only boxes with a high confidence score
        high_confid = bbox_confid_scores[:, class_idx] > prob_threshold
        remaining_boxes = bbox_loc[high_confid].clone()
        remaining_confid = bbox_confid_scores[high_confid, class_idx].clone()

        # Sort the remaining confidence scores in ascending order.
        remaining_confid, indices = remaining_confid.sort(descending=False)
        remaining_boxes = remaining_boxes[indices]

        # Discard bounding boxes with overlap of at least overlap_threshold
        while len(remaining_boxes) > 0:
            # Take the bbox and it's confidence score
            sel_bbox.append((remaining_boxes[-1], class_idx))
            # Compute Jaccard similarity
            jaccard_sim = iou(remaining_boxes[-1].unsqueeze(0), remaining_boxes)
            # Discard the bounding boxes with too much overlap
            remaining_boxes = remaining_boxes[(jaccard_sim < overlap_threshold).squeeze()]
            remaining_confid = remaining_confid[(jaccard_sim < overlap_threshold).squeeze()]

    return sel_bbox


''' Bounding Box Conversion --------------------------------------------------------------------------------------------
'''


def loc2bbox(loc, priors, center_var=0.1, size_var=0.2):
    """
    Compute SSD predicted locations to boxes(cx, cy, h, w).
    :param loc: predicted location, dim: (N, num_priors, 4)
    :param priors: default prior boxes, dim: (1, num_prior, 4)
    :param center_var: scale variance of the bounding box center point
    :param size_var: scale variance of the bounding box size
    :return: boxes: (cx, cy, h, w)
    """
    # assert priors.shape[0] == 1
    # assert priors.dim() == 3

    # prior bounding boxes
    p_center = priors[..., :2]
    p_size = priors[..., 2:]

    # locations
    l_center = loc[..., :2]
    l_size = loc[..., 2:]

    # real bounding box
    return torch.cat([center_var * l_center * p_size + p_center,       # b_{center}
                      p_size * torch.exp(size_var * l_size)], dim=-1)  # b_{size}


def bbox2loc(bbox, priors, center_var=0.1, size_var=0.2):
    """
    Compute boxes (cx, cy, h, w) to SSD locations form.
    :param bbox: bounding box (cx, cy, h, w) , dim: (N, num_priors, 4)
    :param priors: default prior boxes, dim: (1, num_prior, 4)
    :param center_var: scale variance of the bounding box center point
    :param size_var: scale variance of the bounding box size
    :return: loc: (cx, cy, h, w)
    """
    # assert priors.shape[0] == 1
    # assert priors.dim() == 3
    assert priors.shape[0] == bbox.shape[0]
    assert priors.shape[1] == bbox.shape[1]

    # prior bounding boxes
    p_center = priors[..., :2]
    p_size = priors[..., 2:]

    # locations
    b_center = bbox[..., :2]
    b_size = bbox[..., 2:]

    c =  torch.cat([1 / center_var * ((b_center - p_center) / p_size),
                      torch.log(b_size / p_size) / size_var], dim=-1)
    return c


def center2corner(center):
    """
    Convert bounding box in center form (cx, cy, w, h) to corner form (x,y) (x+w, y+h)
    :param center: bounding box in center form (cx, cy, w, h)
    :return: bounding box in corner form (x,y) (x+w, y+h)
    """
    return torch.cat([center[..., :2] - center[..., 2:]/2.0,
                      center[..., :2] + center[..., 2:]/2.0], dim=1)


def corner2center(corner):
    """
    Convert bounding box in corner form (x,y) (x+w, y+h) to center form (cx, cy, w, h)
    :param: bounding box in corner form (x,y) (x+w, y+h)
    :return center: bounding box in center form (cx, cy, w, h)
    """
    return torch.cat([(corner[..., :2] + corner[..., 2:])/2.0,
                      corner[..., 2:] - corner[..., :2]], dim=1)