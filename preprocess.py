from vehicle_detection.cityscape_datalist import *

# Set image and label directory
imgs_dir = "../cityscapes_dataset/cityscapes_samples"
labels_dir = "../cityscapes_dataset/cityscapes_samples_labels"

# Set accepted labels for this project
# label_dict = {
#     "person": 1,
#     "persongroup": 2,
#     "rider": 2,
#     "bicycle": 3,
#     "bicyclegroup": 5,
#     "car": 6,
#     "cargroup": 7,
#     "bus": 8,
#     "truck": 9,
#     "traffic sign": 10,
#     "traffic light": 11
# }

label_dict = {
    "person": 1,
    "persongroup": 1,
    "rider": 1,
    "bicycle": 2,
    "car": 3,
    "cargroup":3,
    "bus": 3,
    "truck": 3,
}


def main():
    # Create a data list of images
    cs_data_list = CityScapeDatalist(imgs_dir, labels_dir, label_dict)
    cs_data_list.prepare_data_list()

    # Split data into train test and validation
    cs_data_list.split_data()

    print("Training set size:", len(cs_data_list.train_list))
    print("Validation set size:", len(cs_data_list.valid_list))
    # print("Test set size:", len(cs_data_list.test_list))

    np.save('train', cs_data_list.train_list)
    np.save('valid', cs_data_list.valid_list)
    # np.save('test', cs_data_list.test_list)


if __name__ == "__main__":
    main()
