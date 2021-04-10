import json
import os

from constants import FRAME_ATTRIBUTES


class FrameAttributes:
    log_name = '[DATA PREPARATION]'
    subdirectory = 'frame_attributes'

    destination_dataset_path = '/home/dominika/Documents/sem.1 mgr/SzUM/projekt/dataset/merged_dataset/labels'
    destination_directories = ['grouped_by_frame_attributes']

    images_list_filename = 'bdd100k_images_list.txt'
    merged_labels_filename = 'bdd100k_labels_images.json'

    def __init__(self):
        self.group_by_frame_attributes()

    def group_by_frame_attributes(self):
        self.__create_destination_directories()
        frame_data_dict = self.__import_json_as_dict()

    def __create_destination_directories(self):
        path = os.path.join(self.destination_dataset_path, self.destination_dataset_path[0])
        print(f'{self.log_name} Creating destination directories in: ({path})')

        for subdirectory in FRAME_ATTRIBUTES.keys():
            path = os.path.join(path, subdirectory)
            os.makedirs(path, exist_ok=True)

    def __import_json_as_dict(self):
        path = os.path.join(self.destination_dataset_path, self.merged_labels_filename)

        with open(path) as file:
            return json.load(file)

    def __split_frame_data_by_attributes_values(self):
        pass
