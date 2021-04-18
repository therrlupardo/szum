import json
import os

import numpy as np
from PIL import Image

from settings import PROCESSED_DATASET_PATH, PROCESSED_LABELS_FILEPATH, DATA_REPRESENTATION_DIR
from simplify import simplify_images_from_file


class DataRepresent:
    log_name = '[DATA REPRESENTATION]'
    destination_dataset_path = PROCESSED_DATASET_PATH
    destination_directories = ['images', 'labels']
    merged_labels_filename = PROCESSED_LABELS_FILEPATH

    def __init__(self):
        self.create_directory()
        self.get_data_representation()

    def get_data_representation(self):
        print(f'{self.log_name} get data representation files')
        destination_path = os.path.join(self.destination_dataset_path,
                                        self.destination_directories[1],
                                        self.merged_labels_filename)
        images = simplify_images_from_file(destination_path)
        print(f'{self.log_name} loaded json file')
        # whole_set = []

        for image in images:
            image_path = os.path.join(self.destination_dataset_path, self.destination_directories[0])
            img = np.array(Image.open(os.path.join(image_path, image['name'])))
            crosswalk_label = 0
            if image['has_crosswalks']:
                crosswalk_label = 1
            image_data = [img.tolist(), crosswalk_label]
            self.save_represented_file(image_data, image['name']+'.json')
            # whole_set.append([img.tolist(), crosswalk_label])

        print(f'{self.log_name} saved representation data')
        # return whole_set

    def create_directory(self):
        dir_path = os.path.join(self.destination_dataset_path, DATA_REPRESENTATION_DIR)
        os.makedirs(dir_path, exist_ok=True)

    def save_represented_file(self, data, filename):
        destination_path = os.path.join(self.destination_dataset_path, DATA_REPRESENTATION_DIR)
        file_path = destination_path + '/' + filename
        with open(file_path, 'w') as destination:
            json.dump(data, destination)

