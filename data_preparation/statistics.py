import json
import os

from constants import LABEL_CATEGORIES
from settings import PROCESSED_DATASET_PATH, IMAGES_FILENAMES_FILEPATH, PROCESSED_LABELS_FILEPATH, STATISTICS_FILEPATH
from utils.utils import Utils


class Statistics:
    log_name = '[DATA PREPARATION][STATISTICS]'

    destination_dataset_path = PROCESSED_DATASET_PATH
    destination_directories = ['labels']

    images_list_filename = IMAGES_FILENAMES_FILEPATH
    merged_labels_filename = PROCESSED_LABELS_FILEPATH
    statistics_filename = STATISTICS_FILEPATH

    def __init__(self):
        self.get_statistics()

    def get_statistics(self):
        print(f'{self.log_name} Getting dataset statistics')

        path = os.path.join(self.destination_dataset_path, self.statistics_filename)

        number_of_images = self.__get_number_of_images()
        Utils.write_line_to_file(number_of_images, path)

        number_of_crosswalks = self.__get_number_of_crosswalks()
        Utils.write_line_to_file(number_of_crosswalks, path)

        categories_statistics = self.__get_categories_statistics()
        Utils.write_line_to_file(categories_statistics, path)

        print(f'{self.log_name} Finished getting dataset statistics')

    def __get_number_of_images(self):
        path = os.path.join(self.destination_dataset_path, self.images_list_filename)
        print(f'{self.log_name} Getting number of images from file: ({path})')

        with open(path) as file:
            lines = file.readlines()

        number_of_files = len(lines)
        number_of_distinct_files = len(set(lines))

        number_of_files_string = f'Number of images:\t{number_of_files}\n'
        number_of_files_string += f'Number of distinct images:\t{number_of_distinct_files}\n'

        return number_of_files_string

    def __get_categories_statistics(self):
        print(f'{self.log_name} Getting categories statistics')
        dict_entries_list = self.__get_labels_dict_entries_list()

        records_in_categories = dict((category, 0) for category in LABEL_CATEGORIES)
        objects_in_categories = dict((category, 0) for category in LABEL_CATEGORIES)

        for entry in dict_entries_list:
            for label in entry['labels']:
                category = label['category']
                objects_in_categories[category] += 1
            for category in LABEL_CATEGORIES:
                if Utils.check_if_value_in_record(entry['labels'], 'category', category):
                    records_in_categories[category] += 1

        statistics = f'Numbers of records in each category:\n{json.dumps(records_in_categories, indent=4)}\n'
        statistics += f'Number of objects in each category:\n{json.dumps(objects_in_categories, indent=4)}\n'

        return statistics

    def __get_number_of_crosswalks(self):
        print(f'{self.log_name} Getting number of crosswalks statistics')

        dict_entries_list = self.__get_labels_dict_entries_list()

        number_of_records_with_crosswalks, number_of_crosswalks = Utils.count_crosswalks_in_records_list(
            dict_entries_list)

        number_of_crosswalks_string = f'Number of records (images) with crosswalks:\t{number_of_records_with_crosswalks}\n'
        number_of_crosswalks_string += f'Number of crosswalks in images:\t{number_of_crosswalks}\n'

        return number_of_crosswalks_string

    def __get_labels_dict_entries_list(self):
        path = os.path.join(self.destination_dataset_path, self.destination_directories[0], self.merged_labels_filename)

        with open(path) as file:
            dict_entries_list = json.load(file)

        return dict_entries_list
