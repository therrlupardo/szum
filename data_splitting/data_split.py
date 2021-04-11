import json
import os

import numpy as np
from tensorflow import keras

from settings import PROCESSED_LABELS_FILEPATH, PROCESSED_DATASET_PATH
from utils.utils import Utils


class DataSplit:
    log_name = '[DATA PREPARATION][FRAME ATTRIBUTES]'

    merged_labels_filename = PROCESSED_LABELS_FILEPATH
    destination_dataset_path = os.path.join(PROCESSED_DATASET_PATH, 'labels')

    train_set_size = 0.8
    val_set_size = 0.1
    test_set_size = train_set_size - val_set_size

    def __init__(self, dataset):
        self.split1(dataset)
        self.split2(dataset)
        self.split3(dataset)

    def split1(self, entries_list, shuffle_datasets=True, merge_datasets=False):
        dataset_size = len(entries_list)
        train_set_size = int(dataset_size * self.train_set_size)
        val_set_size = int(dataset_size * self.val_set_size)

        train_set = self.__get_list_subset(entries_list, 0, train_set_size)
        val_set = self.__get_list_subset(entries_list, train_set_size, train_set_size + val_set_size)
        test_set = self.__get_list_subset(entries_list, train_set_size + val_set_size, dataset_size)

        if merge_datasets:
            train_set.extend(val_set)

        if shuffle_datasets:
            np.random.shuffle(train_set)
            np.random.shuffle(val_set)
            np.random.shuffle(test_set)

        x_train_set, y_train_set = self.__split_dataset_into_xy(train_set)
        x_validate_set, y_validate_set = self.__split_dataset_into_xy(val_set)
        x_test_set, y_test_set = self.__split_dataset_into_xy(test_set)

        return x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set, y_test_set

    def split2(self, entries_list, shuffle_datasets=True, merge_datasets=False):
        dict_entries_list = self.__balance_dataset(entries_list)

        x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set, y_test_set = self.split1(
            dict_entries_list, shuffle_datasets, merge_datasets)

        x_train_set = self.normalize(x_train_set)
        x_validate_set = self.normalize(x_validate_set)
        x_test_set = self.normalize(x_test_set)

        data_generator = self.__create_image_data_generator()
        self.__augment_dataset(data_generator, x_train_set)
        self.__augment_dataset(data_generator, x_validate_set)
        self.__augment_dataset(data_generator, x_test_set)

        return data_generator, x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set, y_test_set

    def split3(self, entries_list, shuffle_datasets=True):
        return self.split2(entries_list, shuffle_datasets, merge_datasets=True)

    @staticmethod
    def __augment_dataset(data_generator, dataset):
        data_generator.fit(dataset)

    @staticmethod
    def __create_image_data_generator():
        image_shift = 0.1
        image_rotation = 20

        data_generator = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
                                                                      featurewise_std_normalization=False,
                                                                      height_shift_range=image_shift,
                                                                      horizontal_flip=True,
                                                                      rotation_range=image_rotation,
                                                                      width_shift_range=image_shift,
                                                                      zca_whitening=False)
        return data_generator

    def __balance_dataset(self, dataset):
        crosswalk_entries = []
        no_crosswalk_entries = []

        for entry in dataset:
            if Utils.label_contains_crosswalk_category(entry):
                crosswalk_entries.append(entry)
            else:
                no_crosswalk_entries.append(entry)

        if len(crosswalk_entries) < len(no_crosswalk_entries):
            records_count = len(crosswalk_entries)
            no_crosswalk_entries = self.__get_list_subset(no_crosswalk_entries, 0, records_count)
        else:
            records_count = len(no_crosswalk_entries)
            crosswalk_entries = self.__get_list_subset(crosswalk_entries, 0, records_count)

        balanced_dataset = crosswalk_entries + no_crosswalk_entries
        return balanced_dataset

    @staticmethod
    def __split_dataset_into_xy(dataset):
        x_set = [elem[0] for elem in dataset]
        y_set = [elem[1] for elem in dataset]

        return np.array(x_set), np.array(y_set)

    @staticmethod
    def __get_list_subset(entries_list, start, end):
        return entries_list[start:end]

    def __get_labels_as_list_of_dicts(self):
        path = os.path.join(self.destination_dataset_path, self.merged_labels_filename)

        with open(path) as file:
            dict_entries_list = json.load(file)

        return dict_entries_list

    @staticmethod
    def normalize(dataset):
        mean = np.mean(dataset)
        std_dev = np.std(dataset)
        return (dataset - mean) / std_dev
