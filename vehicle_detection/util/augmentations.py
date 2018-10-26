import numpy as np


def iou(gt, prior):
    """
    # Compute the Intersection over Union
    Note: function iou(a, b) used in match_priors
    :param a: bounding boxes, dim: (n_items, 4)
    :param b: bounding boxes, dim: (n_items, 4) or (1, 4) if b is a reference
    :return: iou value: dim: (n_item)
    """

    # Compute the top left and bottom right coordinates for intersection
    bottom_right = np.minimum(gt[:, 2:], prior[2:])
    top_left = np.maximum(gt[:, :2], prior[:2])

    # Compute the intersection
    intersection = np.clip((bottom_right - top_left), a_min=0.0, a_max=np.inf)
    intersection = np.prod(intersection, axis=1)

    # Compute union
    area_gt = np.prod((gt[:, 2:] - gt[:, :2]), axis=1)
    area_prior = np.prod((prior[2:] - prior[:2]))
    union = area_gt + area_prior - intersection

    # Compute the iou
    jaccard_sim = intersection / union

    return jaccard_sim


def expansion(img, sample_bboxes, sample_labels, mean_img):

    if np.random.randint(2):
        return img, sample_bboxes, sample_labels

    h, w, c = img.shape
    exp_ratio = np.random.uniform(1, 4.0)
    x = np.random.uniform(0, w * exp_ratio - w)
    y = np.random.uniform(0, h * exp_ratio - h)

    expand_img = np.zeros((int(h * exp_ratio),
                           int(w * exp_ratio), c),
                          dtype=img.dtype)
    expand_img[:, :, :] = mean_img
    expand_img[int(y):int(y + h), int(x):int(x + w)] = img

    sample_bboxes = sample_bboxes.copy()
    sample_bboxes[:, :2] += (int(x), int(y))
    sample_bboxes[:, 2:] += (int(x), int(y))

    return expand_img, sample_bboxes, sample_labels


# This will randomly crop the input image
def random_crop(img, sample_bboxes, sample_labels, max_itr=50, min_crop_ratio=0.3):
    while True:
        # Choose min jaccard similarity randomly
        sim_thresholds = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        min_sim = np.random.choice(sim_thresholds)
        # if min_sim is None:
            # return img, sample_bboxes, sample_labels

        h, w, _ = img.shape
        for i in range(max_itr):
            # choose width and height for the crop
            length = np.minimum(h, w)
            if np.random.randint(2):
                min_crop_ratio = 1
            crop_length = np.random.uniform(min_crop_ratio * length, length)
            
            # h_crop = np.random.uniform(0.3 * h, h)
            # if (w_crop / h_crop < 0.5) or (w_crop / h_crop > 2):
                # continue

            # Find the coordinates of the crop
            x = np.random.uniform(w-crop_length)
            y = np.random.uniform(h-crop_length)
            crop_coord = np.asarray([x, y, x+crop_length, y+crop_length])

            # Check if the jaccard similarity meets minimum requirement
            # pos_labels = sample_labels > 0
            # jaccard_sim = iou(sample_bboxes[pos_labels, :], crop_coord)
            # mask = jaccard_sim.min() > min_sim
            # if not mask.any():
                # continue
            # Crop the image
            new_img = img[int(y):int(y+crop_length), int(x):int(x+crop_length), :]

            # Compute the center of the objects
            pos_labels = sample_labels > 0
            centers = (sample_bboxes[:, :2] + sample_bboxes[:, 2:]) / 2.0

            # Take the ground truth bounding box if the center is in the
            # cropped image
            I1 = (centers[:, 0] > crop_coord[0]) & (centers[:, 0] < crop_coord[2])
            I2 = (centers[:, 1] > crop_coord[1]) & (centers[:, 1] < crop_coord[3])
            mask = I1 * I2 * pos_labels
            if not mask.any():
                continue
            bboxes_masked = sample_bboxes[mask, :].copy()
            labels_masked = sample_labels[mask]

            # Take the overlap as the new ground truth bounding box
            bboxes_masked[:, :2] = np.maximum(bboxes_masked[:, :2], crop_coord[:2])
            bboxes_masked[:, 2:] = np.minimum(bboxes_masked[:, 2:], crop_coord[2:])

            # Adjust the ground truth coordinates wrt to the cropped image
            bboxes_masked[:, :2] -= crop_coord[:2]
            bboxes_masked[:, 2:] -= crop_coord[:2]
            # print(min_sim)

            return new_img, bboxes_masked, labels_masked
