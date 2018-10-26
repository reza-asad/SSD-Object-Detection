import torch.nn
from PIL import Image
import sys
from vehicle_detection.ssd_net import *
from vehicle_detection.bbox_helper import *
from preprocess import label_dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.autograd import Variable

img_dims = (300, 300)
img_mean = np.asarray((127, 127, 127), dtype=np.float32).reshape(3, 1, 1)
img_std = 128.0


def main():
    # Model
    num_classes = len(set(label_dict.values())) + 1
    ssd_net = SSD(num_classes).cuda()
    net_state = torch.load('vehicle_detection/ssd_net', map_location='cuda')
    ssd_net.load_state_dict(net_state)
    prior_bboxes = generate_prior_bboxes()

    # Load and process img
    img_path = sys.argv[1]
    print("Evaluating ", img_path)
    orig_img = Image.open(img_path).resize(img_dims)

    img = np.asarray(orig_img, dtype=np.float32).transpose(2, 0, 1)
    img = (img - img_mean) / img_std
    img = torch.from_numpy(img).unsqueeze(0)

    with torch.no_grad():
        ssd_net.eval()

        # Forward once
        x = Variable(img.cuda())
        confidences, locs = ssd_net.forward(x)

        # Translate bbox cords
        bbox = loc2bbox(locs[0].cpu(), prior_bboxes)
        bbox = center2corner(bbox)
        bbox[:, [0, 2]] *= img_dims[1]
        bbox[:, [1, 3]] *= img_dims[0]
        print(bbox.shape)
        # Apply nms
        sel_boxes = nms_bbox(bbox, confidences[0].cpu())

        # Draw bboxes on the image
        fig, ax = plt.subplots(1)
        colors = ['g', 'b', 'r']
        for (bbox, label) in sel_boxes:
            (x, y, xw, yh) = bbox
            rect = patches.Rectangle([x, y], xw - x, yh - y, linewidth=1, edgecolor=colors[label-1], facecolor='none')
            ax.add_patch(rect)
        ax.imshow(orig_img)
        plt.show()


if __name__ == "__main__":
    main()