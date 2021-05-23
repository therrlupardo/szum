import os

import numpy as np
from tensorflow import keras
from PIL import Image

from settings import PROCESSED_LABELS_FILEPATH, PROCESSED_DATASET_PATH


class DiskDatasetSplitter:
    log_name = '[DATA SPLITTING][DISK DATASET SPLITTER]'

    merged_labels_filename = PROCESSED_LABELS_FILEPATH

    destination_dataset_path = os.path.join(PROCESSED_DATASET_PATH, 'split_dataset')
    destination_splits_directories = ['split1', 'split2', 'split3']
    destination_datasets_directories = ['train', 'val', 'test']

    train_set_size = 0.8
    val_set_size = 0.1
    test_set_size = train_set_size - val_set_size

    def create_split1_data_generators(self, mock_generator, batch_size=16):
        train_data_generator, val_data_generator, test_data_generator = self.__create_split_data_generators(
            self.destination_splits_directories[0], batch_size, mock_generator=mock_generator)

        return train_data_generator, val_data_generator, test_data_generator

    def create_split2_data_generators(self, mock_generator, batch_size=16):
        train_data_generator, val_data_generator, test_data_generator = self.__create_split_data_generators(
            self.destination_splits_directories[1], batch_size, mock_generator=mock_generator)

        return train_data_generator, val_data_generator, test_data_generator

    def create_split3_data_generators(self, mock_generator, batch_size=16):
        train_data_generator, val_data_generator, test_data_generator = self.__create_split_data_generators(
            self.destination_splits_directories[2], batch_size, mock_generator=mock_generator)

        return train_data_generator, val_data_generator, test_data_generator

    def __create_split_data_generators(self, destination_subdirectory, batch_size, mock_generator):
        destination = os.path.join(self.destination_dataset_path, destination_subdirectory)

        destination_path = os.path.join(destination, self.destination_datasets_directories[0])
        train_datagen = self.__create_image_data_generator(mock_generator=mock_generator)
        train_data_generator = train_datagen.flow_from_directory(destination_path, batch_size=batch_size,
                                                                 target_size=(72, 128),
                                                                 shuffle=False,
                                                                 classes=['0', '1'],
                                                                 class_mode='binary')

        destination_path = os.path.join(destination, self.destination_datasets_directories[1])
        val_datagen = self.__create_image_data_generator(mock_generator=True)
        val_data_generator = val_datagen.flow_from_directory(destination_path, batch_size=batch_size,
                                                             target_size=(72, 128),
                                                             shuffle=False,
                                                             classes=['0', '1'],
                                                             class_mode='binary')

        destination_path = os.path.join(destination, self.destination_datasets_directories[2])
        test_datagen = self.__create_image_data_generator(mock_generator=True)
        test_data_generator = test_datagen.flow_from_directory(destination_path, batch_size=batch_size,
                                                               target_size=(72, 128),
                                                               shuffle=False,
                                                               classes=['0', '1'],
                                                               class_mode='binary')

        return train_data_generator, val_data_generator, test_data_generator

    def create_and_save_split1(self, entries_list):
        split_dataset = self._split1(entries_list)
        self.__save_split(self.destination_splits_directories[0], *split_dataset)

    def create_and_save_split2(self, entries_list):
        split_dataset = self._split2(entries_list)
        self.__save_split(self.destination_splits_directories[1], *split_dataset)

    def create_and_save_split3(self, entries_list):
        split_dataset = self._split3(entries_list)
        self.__save_split(self.destination_splits_directories[2], *split_dataset)

    def __save_split(self, destination_subdirectory, train_filenames, x_train_set, y_train_set, validate_filenames,
                     x_validate_set, y_validate_set, test_filenames, x_test_set, y_test_set):
        destination_subpath = os.path.join(destination_subdirectory,
                                           self.destination_datasets_directories[0])
        self.__save_dataset_to_disk(train_filenames, x_train_set, y_train_set, destination_subpath)

        destination_subpath = os.path.join(destination_subdirectory,
                                           self.destination_datasets_directories[1])
        self.__save_dataset_to_disk(validate_filenames, x_validate_set, y_validate_set, destination_subpath)

        destination_subpath = os.path.join(destination_subdirectory,
                                           self.destination_datasets_directories[2])
        self.__save_dataset_to_disk(test_filenames, x_test_set, y_test_set, destination_subpath)

    def __save_dataset_to_disk(self, filenames, x_set, y_set, destination_directory):
        destination_path = os.path.join(self.destination_dataset_path, destination_directory)
        self.__save_images_to_directories_by_label(filenames, x_set, y_set, destination_path)

    @staticmethod
    def __save_images_to_directories_by_label(filenames, images, labels, destination_path):
        unique_labels = list(set(labels))

        for unique_label in unique_labels:
            path = os.path.join(destination_path, str(unique_label))
            if not os.path.isdir(path):
                os.makedirs(path, exist_ok=True)

        for i in range(len(labels)):
            image_path = os.path.join(destination_path, str(labels[i]), filenames[i])
            image = Image.fromarray(images[i])
            image.save(image_path)

    def _split1(self, entries_list, shuffle_datasets=True, merge_datasets=False):
        print(f'{self.log_name} Splitting dataset into split1')

        dataset_size = len(entries_list)
        train_set_size = int(dataset_size * self.train_set_size)
        val_set_size = int(dataset_size * self.val_set_size)

        train_set = self.__get_list_subset(entries_list, 0, train_set_size)
        val_set = self.__get_list_subset(entries_list, train_set_size, train_set_size + val_set_size)
        test_set = self.__get_list_subset(entries_list, train_set_size + val_set_size, dataset_size)

        if merge_datasets:
            train_set.extend(val_set)

        if shuffle_datasets:
            np.random.seed(0)
            np.random.shuffle(train_set)
            np.random.shuffle(val_set)
            np.random.shuffle(test_set)

        train_filenames, x_train_set, y_train_set = self.__split_dataset_into_filenames_xy(train_set)
        validate_filenames, x_validate_set, y_validate_set = self.__split_dataset_into_filenames_xy(val_set)
        test_filenames, x_test_set, y_test_set = self.__split_dataset_into_filenames_xy(test_set)

        print(f'{self.log_name} Finished splitting dataset into split1')

        return train_filenames, x_train_set, y_train_set, validate_filenames, x_validate_set, y_validate_set, test_filenames, x_test_set, y_test_set

    def _split2(self, entries_list, shuffle_datasets=True, merge_datasets=False):
        print(f'{self.log_name} Splitting dataset into split2')

        dict_entries_list = self.__balance_dataset(entries_list)

        train_filenames, x_train_set, y_train_set, validate_filenames, x_validate_set, y_validate_set, test_filenames, x_test_set, y_test_set = self._split1(
            dict_entries_list, shuffle_datasets, merge_datasets)

        x_train_set = self.__standardize(x_train_set)
        x_validate_set = self.__standardize(x_validate_set)
        x_test_set = self.__standardize(x_test_set)

        x_train_set = self.__normalize(x_train_set)
        x_validate_set = self.__normalize(x_validate_set)
        x_test_set = self.__normalize(x_test_set)

        x_train_set = self.__scale(x_train_set)
        x_validate_set = self.__scale(x_validate_set)
        x_test_set = self.__scale(x_test_set)

        print(f'{self.log_name} Finished splitting dataset into split2')

        return train_filenames, x_train_set, y_train_set, validate_filenames, x_validate_set, y_validate_set, test_filenames, x_test_set, y_test_set

    def _split3(self, entries_list, shuffle_datasets=True):
        print(f'{self.log_name} Splitting dataset into split3')

        dataset = self._split2(entries_list, shuffle_datasets, merge_datasets=True)

        print(f'{self.log_name} Finished splitting dataset into split3')
        return dataset

    @staticmethod
    def __get_list_subset(entries_list, start, end):
        return entries_list[start:end]

    @staticmethod
    def __split_dataset_into_filenames_xy(dataset):
        filenames_set = [elem[0] for elem in dataset]
        x_set = [elem[1] for elem in dataset]
        y_set = [elem[2] for elem in dataset]

        return np.array(filenames_set), np.array(x_set), np.array(y_set)

    def __balance_dataset(self, dataset):
        crosswalk_entries = []
        no_crosswalk_entries = []

        for entry in dataset:
            if entry[2] == 'True' or entry[2] == 1:
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
        np.random.shuffle(balanced_dataset)
        return balanced_dataset

    @staticmethod
    def __create_image_data_generator(mock_generator=False):
        image_shift = 0.1
        image_rotation = 20

        if mock_generator:
            data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        else:
            data_generator = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
                                                                          featurewise_std_normalization=False,
                                                                          height_shift_range=image_shift,
                                                                          horizontal_flip=True,
                                                                          rescale=1. / 255,
                                                                          rotation_range=image_rotation,
                                                                          width_shift_range=image_shift,
                                                                          zca_whitening=False)
        return data_generator

    @staticmethod
    def __standardize(dataset, shift=False):
        dataset_max = np.amax(dataset)
        dataset_min = np.amin(dataset)

        dataset = dataset - dataset_min
        standardized_dataset = dataset / (dataset_max - dataset_min)

        if shift:
            standardized_dataset = standardized_dataset - 0.5

        return standardized_dataset

    @staticmethod
    def __normalize(dataset):
        mean = np.mean(dataset)
        std_dev = np.std(dataset)
        normalized_dataset = (dataset - mean) / std_dev

        return normalized_dataset

    @staticmethod
    def __scale(dataset):
        dataset = (dataset * 255).astype(np.uint8)
        return dataset
