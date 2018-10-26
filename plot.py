from vehicle_detection.ssd_net import *

from preprocess import label_dict
from vehicle_detection.cityscape_dataset import *
from vehicle_detection.solver import *
from vehicle_detection.bbox_helper import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def main():
    # Model
    num_classes = len(set(label_dict.values())) + 1
    ssd_net = SSD(num_classes).cuda()
    net_state = torch.load('vehicle_detection/ssd_net', map_location='cuda')
    ssd_net.load_state_dict(net_state)
    prior_bboxes = generate_prior_bboxes()

    # Validation data
    valid_list = np.load('valid.npy')
    train_list = np.load('train.npy')
    train_dataset = CityScapeDataset(train_list, input_dim=(300, 300), mode='train')
    valid_dataset = CityScapeDataset(valid_list, input_dim=(300, 300), mode='test')
    solver = Solver(train_dataset, valid_dataset, batch_size=16)

    # Plotting sample bounding boxes
    NUM_IMGS = 5
    fig, ax = plt.subplots(NUM_IMGS)
    mean_img = np.asarray((127, 127, 127), dtype=np.float32).reshape(3, 1, 1)
    std_img = 128.0

    with torch.no_grad():
        # Now I'm going to read a few data examples from the DataLoader and plot them
        ssd_net.eval()
        idx, (x, gt_locs, gt_labels) = next(enumerate(solver.valid_loader))

        # Move to the right device
        x = Variable(x.cuda())
        gt_locs = Variable(gt_locs.cuda())
        gt_labels = Variable(gt_labels.cuda())

        # Predict the classes and bbox coordinates for the input.
        confidences, locs = ssd_net.forward(x)

        img_sample = x[:NUM_IMGS].clone().cpu()
        loc_sample = locs[:NUM_IMGS, :, :].clone().cpu()
        conf_sample = confidences[:NUM_IMGS, :, :].clone().cpu()
        for i in range(0, NUM_IMGS):
            img = np.asarray(img_sample[i])
            img = (img * std_img) + mean_img
            img = np.ascontiguousarray(img.transpose(1, 2, 0).astype('uint8'))
            h, w, c = img.shape
            # Convert the locations to bbox
            bbox = loc2bbox(loc_sample[i], prior_bboxes)
            # Convert bbox to corner format
            bbox = center2corner(bbox)
            bbox[:, [0, 2]] *= w
            bbox[:, [1, 3]] *= h
            # Apply nms on the bounding boxes
            sel_boxes = nms_bbox(bbox, conf_sample[i])
            # Draw bounding boxes on the image
            colors = ['g', 'b', 'r']
            for (bbox, label) in sel_boxes:
                (x, y, xw, yh) = bbox
                rect = patches.Rectangle([x, y], xw - x, yh - y, linewidth=1, edgecolor=colors[label-1], facecolor='none')
                ax[i].add_patch(rect)
            ax[i].imshow(img)
        plt.show()


if __name__ == "__main__":
    main()

