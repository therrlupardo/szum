import os

from utils.constants import FRAME_ATTRIBUTES
from settings import PROCESSED_DATASET_PATH, PROCESSED_LABELS_FILEPATH
from utils.utils import Utils


class FrameAttributes:
    log_name = '[DATA PREPARATION][FRAME ATTRIBUTES]'

    merged_labels_filename = PROCESSED_LABELS_FILEPATH

    destination_dataset_path = os.path.join(PROCESSED_DATASET_PATH, 'labels')
    destination_directories = ['grouped_by_frame_attributes']

    def __init__(self):
        self.group_by_frame_attributes()

    def group_by_frame_attributes(self):
        print(f'{self.log_name} Grouping dataset by frame attributes values')

        self.__create_destination_directories()
        frame_data_list = Utils.import_json_as_dict(self.destination_dataset_path, self.merged_labels_filename)
        self.__split_frame_data_by_attributes_values(frame_data_list)

        print(f'{self.log_name} Finished grouping dataset by frame attributes values')

    def __create_destination_directories(self):
        path = os.path.join(self.destination_dataset_path, self.destination_directories[0])
        print(f'{self.log_name} Creating destination directories in: ({path})')

        for subdirectory in FRAME_ATTRIBUTES.keys():
            final_path = os.path.join(path, subdirectory)
            os.makedirs(final_path, exist_ok=True)

    def __split_frame_data_by_attributes_values(self, frame_data_dict_list):
        print(f'{self.log_name} Splitting data by frame attributes values')

        for attribute, attributes_values in FRAME_ATTRIBUTES.items():
            for attribute_value in attributes_values:
                dict_entries_list = []
                for entry in frame_data_dict_list:
                    frame_attributes = entry['attributes']
                    if frame_attributes[attribute] == attribute_value:
                        dict_entries_list.append(entry)

                attribute_value = attribute_value.replace('/', '-') if '/' in attribute_value else attribute_value
                self.__write_entries_list_by_frame_attribute_to_file(dict_entries_list, attribute, attribute_value)
                self.__write_entries_list_statistics_to_file(dict_entries_list, attribute, attribute_value)

    def __write_entries_list_by_frame_attribute_to_file(self, entries_list, attribute, attribute_value):
        print(
            f'{self.log_name} Writing filtered data to file for attribute [{attribute}] and value [{attribute_value}]')

        path = os.path.join(self.destination_dataset_path, self.destination_directories[0])

        filepath = os.path.join(attribute, f'{attribute_value}.json')
        final_path = os.path.join(path, filepath)

        Utils.export_dict_as_json(final_path, entries_list)

    def __write_entries_list_statistics_to_file(self, entries_list, attribute, attribute_value):
        print(
            f'{self.log_name} Writing filtered data statistics to file for attribute [{attribute}] and value [{attribute_value}]')

        path = os.path.join(self.destination_dataset_path, self.destination_directories[0])

        filepath = os.path.join(attribute, f'{attribute_value}.txt')
        final_path = os.path.join(path, filepath)

        number_of_records = len(entries_list)
        number_of_records_with_crosswalks, number_of_crosswalks = Utils.count_crosswalks_in_records_list(entries_list)

        with open(final_path, 'w+') as file:
            file.write(f'Number of records:\t{number_of_records}\n')
            file.write(f'Number of records (images) with crosswalks:\t{number_of_records_with_crosswalks}\n')
            file.write(f'Number of crosswalks in images:\t{number_of_crosswalks}\n')
