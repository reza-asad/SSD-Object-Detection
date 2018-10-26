import numpy as np
import json
import os
import random


class CityScapeDatalist():
    '''
    This creates a data list that will be fed to the Dataset class.
    Here is the format for an element in the datalist:
    {''}
    '''
    def __init__(self, base_img_dir, base_label_dir, label_dict, img_size=(300, 300),
                 dtype=np.float32):
        # Image and labels
        self.base_img_dir = base_img_dir
        self.base_label_dir = base_label_dir
        self.label_dict = label_dict

        # Data type and size
        self.img_size = img_size
        self.dtype = dtype

        # Various data lists
        self.data_list = None
        self.train_list = None
        self.valid_list = None
        self.test_list = None

    def prepare_data_list(self):
        self.data_list = []
        # Iterate through the directories and subdirectories to make the data list
        for cur_dir, sub_dirs, files in os.walk(self.base_img_dir):
            # Proceed once you have reached the leaf of the tree that is there
            # are no subdirectories
            if len(sub_dirs) != 0:
                continue
            for img_file in files:
                tokens = img_file.split('_')
                # Ensure the image name has consistent format
                assert len(tokens) == 4
                # Take the city name
                city_name = tokens[0]
                # Take the image number
                img_number = tokens[1] + '_' + tokens[2]
                # Take the image path
                img_path = os.path.join(cur_dir, img_file)
                # Take the label paths
                labels_path = os.path.join(self.base_label_dir, city_name,
                                           city_name + '_' + img_number + "_gtCoarse_polygons.json")
                labels = []
                bounding_boxes = []
                with open(labels_path, 'r') as data:
                    background_list_label = []
                    background_list_bbox = []
                    positive_list_label = []
                    positive_list_bbox = []
                    for item in json.load(data)['objects']:
                        if item['label'] in self.label_dict:
                            # save label
                            positive_list_label.append(self.label_dict[item['label']])
                            # Find top left and bottom right coordinates
                            top_left = np.min(item['polygon'], axis=0)
                            bottom_right = np.max(item['polygon'], axis=0)
                            # Concatenate the top left and bottom right to form a bounding box
                            bbox = np.concatenate((top_left, bottom_right))
                            # save bounding box
                            positive_list_bbox.append(bbox)
                        else:
                            background_list_label.append(0)
                            background_list_bbox.append(np.asarray([0, 0, 0, 0]))
                    if len(positive_list_label) != 0:
                        labels += positive_list_label
                        labels += background_list_label
                        bounding_boxes += positive_list_bbox
                        bounding_boxes += background_list_bbox
                        # For each image add all the accepted labels and their corresponding bounding boxes.
                        self.data_list.append({"img_path": img_path, "labels": np.asarray(labels),
                                               'bounding_boxes':np.asarray(bounding_boxes, dtype=self.dtype)})

    def split_data(self, train_portion=0.8, valid_portion=0.2):
        random.shuffle(self.data_list)
        total_data_size = len(self.data_list)

        train_data_size = int(train_portion * total_data_size)
        valid_data_size = int(valid_portion * total_data_size)

        self.train_list = self.data_list[:train_data_size]
        self.valid_list = self.data_list[train_data_size: ]
        # self.test_list = self.data_list[train_data_size + valid_data_size:]
