import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np


class Solver:
    def __init__(self, train_data_set, valid_data_set, batch_size):
        self.train_data_set = train_data_set
        self.valid_data_set = valid_data_set
        self.batch_size = batch_size

        self.train_loader = DataLoader(self.train_data_set, batch_size=self.batch_size, shuffle=True, num_workers=6)
        self.valid_loader = DataLoader(self.valid_data_set, batch_size=self.batch_size, shuffle=True, num_workers=6)

        self.mean_img = np.asarray((127, 127, 127), dtype=np.float32).reshape(3, 1, 1)
        self.std_img = 128.0

    def evaluate_model(self, model, loss_criteria, loader=None):
        # Put the model in evaluation mode
        model.eval()
        class_losses = []
        bbox_losses = []
        num_samples = 0
        num_detected = 0
        if loader is None:
            loader = self.valid_loader
        with torch.no_grad():
            for (x, gt_locs, gt_labels) in loader:
                # Move to the right device
                x = Variable(x.cuda())
                gt_locs = Variable(gt_locs.cuda())
                gt_labels = Variable(gt_labels.cuda())

                # Predict the classes and bbox coordinates for the input.
                confidences, locs = model.forward(x)

                # Compute the validation loss
                conf_loss, loc_huber_loss = loss_criteria.forward(confidences, locs, gt_labels, gt_locs)
                class_losses.append(conf_loss.item())
                bbox_losses.append(loc_huber_loss.item())

                # Find accuracy on classification
                not_background = gt_labels != 0
                _, predicted_class = confidences.max(2)
                num_samples += len(gt_labels[not_background])
                num_detected += (predicted_class[not_background] == gt_labels[not_background]).sum()

            accuracy = num_detected.item() / float(num_samples) * 100
        return accuracy, class_losses, bbox_losses

    def train(self, model, optimizer, loss_criteria, num_epochs=1, print_every=100):
        # Cache the loss history
        train_class_losses = []
        train_bbox_losses = []
        avg_train_class_losses = []
        avg_train_bbox_losses = []
        avg_valid_class_losses = []
        avg_valid_bbox_losses = []
        for i in range(num_epochs):
            print("This is epoch %d" % i)
            for batch_idx, (x, gt_locs, gt_labels) in enumerate(self.train_loader):
                # set the model in training mode
                model.train()

                # Move to the right device
                x = Variable(x.cuda())
                gt_labels = Variable(gt_labels.cuda(), requires_grad=False)
                gt_locs = Variable(gt_locs.cuda(), requires_grad=False)

                # Compute the coordinates
                confidences, locs = model.forward(x)

                # Compute the loss and save it.
                conf_loss, loc_huber_loss = loss_criteria.forward(confidences, locs, gt_labels, gt_locs)
                total_loss = conf_loss + loc_huber_loss
                train_class_losses.append(conf_loss.item())
                train_bbox_losses.append(loc_huber_loss.item())

                # Zero out the gradients before optimization
                optimizer.zero_grad()

                # Backprop
                total_loss.backward()

                # Take an optimization step
                optimizer.step()

                if ((batch_idx % print_every) == 0) and (batch_idx != 0):
                    # Calculate avg training losses since last print
                    avg_train_class_loss = np.mean(train_class_losses)
                    avg_train_bbox_loss = np.mean(train_bbox_losses)
                    avg_total_loss = avg_train_class_loss + avg_train_bbox_loss
                    avg_train_class_losses.append(avg_train_class_loss)
                    avg_train_bbox_losses.append(avg_train_bbox_loss)
                    train_class_losses.clear()
                    train_bbox_losses.clear()
                    print("Epoch %d, lr is %.7f, iteration %d : total loss is %.7f, conf loss is %0.7f, locs loss is %0.7f" % (i,
                                                                                                                               optimizer.param_groups[0]['lr'],
                                                                                                                               batch_idx,
                                                                                                                               avg_total_loss,
                                                                                                                               avg_train_class_loss,
                                                                                                                               avg_train_bbox_loss))
                    # Calculate accuracy and avg losses against the validation set
                    accuracy, valid_class_loss, valid_bbox_loss = self.evaluate_model(model, loss_criteria, self.valid_loader)
                    avg_valid_class_loss = np.mean(valid_class_loss)
                    avg_valid_bbox_loss = np.mean(valid_bbox_loss)
                    avg_valid_total_loss = avg_valid_class_loss + avg_valid_bbox_loss
                    avg_valid_class_losses += avg_valid_class_loss
                    avg_valid_bbox_losses += avg_valid_bbox_loss
                    print("Validation: Accuracy = {}, total loss = {}, conf loss = {}, locs loss = {}".format(accuracy,
                                                                                                              avg_valid_total_loss,
                                                                                                              avg_valid_class_loss,
                                                                                                              avg_valid_bbox_loss))
            # Save the model every epoch
            net_state = model.state_dict()
            torch.save(net_state, 'vehicle_detection/ssd_net')
            print('--------')
        return avg_train_class_losses, avg_train_bbox_losses, avg_valid_class_losses, avg_valid_bbox_losses
