class Utils:

    @staticmethod
    def count_crosswalks_in_record(dict_entry):
        labels_data = dict_entry['labels']

        crosswalk_count = 0
        for label in labels_data:
            if label['category'] == 'lane' and label['attributes']['laneType'] == 'crosswalk':
                crosswalk_count += 1
        return crosswalk_count

    @staticmethod
    def count_crosswalks_in_records_list(dict_entries_list):
        crosswalks_in_records = list(map(lambda x: Utils.count_crosswalks_in_record(x), dict_entries_list))
        number_of_records_with_crosswalks = sum(
            number_of_crosswalks > 0 for number_of_crosswalks in crosswalks_in_records)
        number_of_crosswalks = sum(crosswalks_in_records)

        return number_of_records_with_crosswalks, number_of_crosswalks

    @staticmethod
    def check_if_value_in_record(dict_entry, key, value):
        for elem in dict_entry:
            if elem[key] == value:
                return True
        return False

    @staticmethod
    def write_line_to_file(line, path):
        with open(path, 'a+') as file:
            file.write(line)
