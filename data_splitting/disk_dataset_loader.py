import os

from settings import PROCESSED_DATASET_PATH, PROCESSED_LABELS_FILEPATH
from utils.utils import Utils

import numpy as np
from PIL import Image


class DiskDatasetLoader:
    log_name = '[DATA SPLITTING][DISK DATASET LOADER]'

    destination_dataset_path = PROCESSED_DATASET_PATH
    destination_directories = ['images', 'labels']
    merged_labels_filename = PROCESSED_LABELS_FILEPATH

    def get_dataset(self):
        dataset_data = self.__load_dataset_data()
        # dataset_data = dataset_data[:200]
        dataset = self.__load_dataset(dataset_data)

        return dataset

    def __load_dataset_data(self):
        labels_path = os.path.join(self.destination_dataset_path, self.destination_directories[1])

        print(f'{self.log_name} Loading dataset from file: ({self.merged_labels_filename}) on path: ({labels_path})')

        dataset_data = Utils.get_simplified_images_data_from_file(labels_path, self.merged_labels_filename)
        dataset_data = [(elem['name'], elem['has_crosswalks']) for elem in dataset_data]

        print(f'{self.log_name} Finished loading dataset')

        return np.array(dataset_data)

    def __load_dataset(self, dataset_data):
        dataset = []
        counter = 0
        for data in dataset_data:
            filename = data[0]
            category = 0 if data[1] == 'False' else 1
            print(f'{self.log_name} Loading #{counter} image: ({filename})')

            image = self.__load_image(filename)
            dataset.append((filename, image, category))
            counter += 1

        return dataset

    def __load_image(self, filename):
        image_path = os.path.join(self.destination_dataset_path, self.destination_directories[0], filename)

        image = np.array(Image.open(image_path))
        return image
