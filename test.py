import torch.nn
from vehicle_detection.ssd_net import *
from vehicle_detection.bbox_loss import *
from vehicle_detection.cityscape_dataset import *
from vehicle_detection.solver import *
from preprocess import label_dict

improve_model = True


def main():
    # Load the train and validation data lists
    train_list = np.load('train.npy')
    valid_list = np.load('valid.npy')

    # Create a dataset using the data list you prepared
    input_dim=(300, 300)
    train_dataset = CityScapeDataset(train_list, input_dim=input_dim, mode='train')
    valid_dataset = CityScapeDataset(valid_list, input_dim=input_dim, mode='test')

    # Create a solver to train the model
    solver = Solver(train_dataset, valid_dataset, batch_size=64)

    # Instantiate the model
    num_classes = len(set(label_dict.values())) + 1
    ssd_net = SSD(num_classes).cuda()

    if improve_model:
        net_state = torch.load('vehicle_detection/ssd_net', map_location='cuda')
        ssd_net.load_state_dict(net_state)

    # Create an optimizer
    optimizer = torch.optim.Adam(ssd_net.parameters(), lr=1e-3)

    # Initialize loss function
    loss_function = MultiboxLoss()

    train_bbox_losses = []
    train_class_losses = []
    valid_bbox_losses = []
    valid_class_losses = []

    # Training the model
    lr = 1e-7
    for num_epochs in [40, 40]:
        optimizer.param_groups[0]['lr'] = lr
        tclass_loss, tbbox_loss, vclass_loss, vbbox_loss = solver.train(ssd_net,
                                                                        optimizer,
                                                                        loss_function,
                                                                        num_epochs=num_epochs,
                                                                        print_every=230)
        train_class_losses += tclass_loss
        train_bbox_losses += tbbox_loss
        valid_class_losses += vclass_loss
        valid_bbox_losses += vbbox_loss

        np.save('train_class_losses', train_class_losses)
        np.save('train_bbox_losses', train_bbox_losses)
        np.save('valid_class_losses', valid_class_losses)
        np.save('valid_bbox_losses', valid_bbox_losses)

        lr *= 1e-1


if __name__ == "__main__":
    main()


