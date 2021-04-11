import fnmatch
import json
import os
import numpy as np
import shutil
from PIL import Image

from settings import RAW_DATASET_PATH, PROCESSED_DATASET_PATH, IMAGES_FILENAMES_FILEPATH, PROCESSED_LABELS_FILEPATH, \
    UNIQUE_IMAGES_FILENAMES_FILEPATH
from simplify import simplify_images_from_file, has_crosswalk


class Filesystem:
    log_name = '[DATA PREPARATION][FILESYSTEM]'

    source_dataset_path = RAW_DATASET_PATH
    source_directories = ['bdd100k', 'bdd100k_labels_release']

    destination_dataset_path = PROCESSED_DATASET_PATH
    destination_directories = ['images', 'labels']

    images_list_filename = IMAGES_FILENAMES_FILEPATH
    unique_images_list_filename = UNIQUE_IMAGES_FILENAMES_FILEPATH
    merged_labels_filename = PROCESSED_LABELS_FILEPATH
    file_extensions = ['*.jpg', '*.json']

    def __init__(self):
        self.merge_split_dataset()

    def merge_split_dataset(self):
        print(f'{self.log_name} Merging dataset')

        self.__create_destination_directories()

        images = self.__search_files_in_source_directory(self.file_extensions[0])
        self.__copy_files_to_destination_directory(images, self.destination_directories[0], True)
        self.__write_filenames_to_file_in_destination_directory(images)

        labels = self.__search_files_in_source_directory(self.file_extensions[1])
        self.__merge_labels_files_in_destination_directory(labels, self.destination_directories[1])

        images_with_labels = self.__load_merge_and_split_dataset(self.destination_directories[1], self.destination_directories[0])
        print(images_with_labels)
        print(f'{self.log_name} Finished merging dataset')

    def __create_destination_directories(self):
        print(f'{self.log_name} Creating destination directories in: ({self.destination_dataset_path})')

        for subdirectory in self.destination_directories:
            path = os.path.join(self.destination_dataset_path, subdirectory)
            os.makedirs(path, exist_ok=True)

    def __search_files_in_source_directory(self, file_extension):
        print(f'{self.log_name} Looking for {file_extension} files in source directory: ({self.source_dataset_path})')

        matches = []

        for directory in self.source_directories:
            path = os.path.join(self.source_dataset_path, directory)
            for root, _, filenames in os.walk(path):
                matched_filenames = fnmatch.filter(filenames, file_extension)
                for filename in matched_filenames:
                    filepath = os.path.join(root, filename)
                    matches.append(filepath)

        return matches

    def __copy_files_to_destination_directory(self, files, subdirectory, data_normalization_enabled):
        path = os.path.join(self.destination_dataset_path, subdirectory)
        print(f'{self.log_name} Copying files to destination directory: ({path})')

        if data_normalization_enabled:
            self.__save_normalized_data(files, path, 600, 65)
        else:
            for file in files:
                shutil.copy2(file, path)

    def __write_filenames_to_file_in_destination_directory(self, files):
        path = os.path.join(self.destination_dataset_path, self.images_list_filename)
        print(f'{self.log_name} Writing images filenames to file in destination: ({path})')

        with open(path, 'w+') as f:
            for file in files:
                filename = os.path.split(file)[-1]
                f.write(f'{filename}\n')

        path = os.path.join(self.destination_dataset_path, self.unique_images_list_filename)
        print(f'{self.log_name} Writing unique images filenames to file in destination: ({path})')

        files = list(set(files))

        with open(path, 'w+') as f:
            for file in files:
                filename = os.path.split(file)[-1]
                f.write(f'{filename}\n')

    def __merge_labels_files_in_destination_directory(self, files, subdirectory):
        destination_path = os.path.join(self.destination_dataset_path, subdirectory, self.merged_labels_filename)
        print(f'{self.log_name} Merging labels files content to file in destination: ({destination_path})')
        entries_list = []

        for file in files:
            with open(file, 'rb') as source:
                file_content = json.load(source)
                entries_list.extend(file_content)

        with open(destination_path, 'w+') as destination:
            json.dump(entries_list, destination)

    def __load_merge_and_split_dataset(self, subdirectory_label, subdirectory_images):
        destination_path = os.path.join(self.destination_dataset_path, subdirectory_label, self.merged_labels_filename)
        images = simplify_images_from_file(destination_path)
        whole_set = []

        for image in images:
            image_path = os.path.join(self.destination_dataset_path, subdirectory_images)
            img = np.array(Image.open(image_path + "/" + image['name']))
            crosswalk_label = 0
            if image['has_crosswalks']:
                crosswalk_label = 1
            whole_set.append([img, crosswalk_label])
        return whole_set

    def __save_normalized_data(self, images, path, resolution, quality):
        for image in images:
            file_path = path + "/" + os.path.split(image)[-1]
            img = Image.open(image)
            img.thumbnail((resolution, resolution))
            img.save(file_path, optimize=True, quality=quality)
