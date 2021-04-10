import fnmatch
import os
import shutil


class Filesystem:
    log_name = '[DATA PREPARATION]'
    source_dataset_path = '/home/dominika/Documents/sem.1 mgr/SzUM/projekt/dataset'
    source_directories = ['bdd100k', 'bdd100k_labels_release']
    destination_dataset_path = '/home/dominika/Documents/sem.1 mgr/SzUM/projekt/dataset/merged_dataset'
    destination_directories = ['images', 'labels']

    images_list_filename = 'bdd100k_images_list.txt'
    merged_labels_filename = 'bdd100k_labels_images.json'
    file_extensions = ['*.jpg', '*.json']

    def __init__(self):
        self.merge_split_dataset()

    def merge_split_dataset(self):
        print(f'{self.log_name} Merging dataset')

        self.__create_destination_directories()

        images = self.__search_files_in_source_directory(self.file_extensions[0])
        self.__copy_files_to_destination_directory(images, self.destination_directories[0])
        self.__write_filenames_to_destination_directory(images)

        labels = self.__search_files_in_source_directory(self.file_extensions[1])
        self.__merge_files_in_destination_directory(labels, self.destination_directories[1])

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

    def __copy_files_to_destination_directory(self, files, subdirectory):
        path = os.path.join(self.destination_dataset_path, subdirectory)
        print(f'{self.log_name} Copying files to destination directory: ({path})')

        for file in files:
            shutil.copy2(file, path)

    def __write_filenames_to_destination_directory(self, files):
        path = os.path.join(self.destination_dataset_path, self.images_list_filename)
        print(f'{self.log_name} Writing filenames to destination: ({path})')

        with open(path, 'w') as f:
            for file in files:
                filename = os.path.split(file)[-1]
                f.write(f'{filename}\n')

    def __merge_files_in_destination_directory(self, files, subdirectory):
        destination_path = os.path.join(self.destination_dataset_path, subdirectory, self.merged_labels_filename)
        print(f'{self.log_name} Merging files content to destination: ({destination_path})')

        with open(destination_path, 'wb') as destination:
            for file in files:
                with open(file, 'rb') as source:
                    shutil.copyfileobj(source, destination)
