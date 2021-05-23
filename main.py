from data_preparation.filesystem import Filesystem
from data_preparation.frame_attributes import FrameAttributes
from data_preparation.statistics import Statistics
from data_splitting.dataset_splitter import DatasetSplitter
from data_splitting.dataset_loader import DatasetLoader
from data_splitting.disk_dataset_loader import DiskDatasetLoader
from data_splitting.disk_dataset_splitter import DiskDatasetSplitter
from data_training.disk_images_classifier import DiskImagesClassifier
from data_training.images_classifier import ImagesClassifier


def data_preparation():
    Filesystem()
    # FrameAttributes()
    # Statistics()


# DATA ON DISK
def disk_data_splitting():
    disk_dataset_loader = DiskDatasetLoader()
    disk_dataset = disk_dataset_loader.get_dataset()

    disk_dataset_splitter = DiskDatasetSplitter()
    disk_dataset_splitter.create_and_save_split1(disk_dataset)
    # disk_dataset_splitter.create_and_save_split2(disk_dataset)
    # disk_dataset_splitter.create_and_save_split3(disk_dataset)


def disk_data_training():
    disk_dataset_splitter = DiskDatasetSplitter()

    disk_images_classifier = DiskImagesClassifier(
        '/home/dominika/Documents/sem.1 mgr/SzUM/projekt/szum_crosswalk_detection/model1-80k-epoch5')

    batch_size = 16

    use_disk_split1(disk_dataset_splitter, disk_images_classifier, batch_size=batch_size, create_model=True)
    # use_disk_split2(disk_dataset_splitter, disk_images_classifier, batch_size=batch_size, create_model=True)
    # use_disk_split3(disk_dataset_splitter, disk_images_classifier, batch_size=batch_size, create_model=False)


def use_disk_split1(disk_dataset_splitter: DiskDatasetSplitter, disk_images_classifier: DiskImagesClassifier,
                    batch_size=32, create_model=True):
    data_generators = disk_dataset_splitter.create_split1_data_generators(not create_model, batch_size=batch_size)
    __use_disk_images_classifier(disk_images_classifier, data_generators, create_model)


def use_disk_split2(disk_dataset_splitter: DiskDatasetSplitter, disk_images_classifier: DiskImagesClassifier,
                    batch_size=32, create_model=True):
    data_generators = disk_dataset_splitter.create_split2_data_generators(not create_model, batch_size=batch_size)
    __use_disk_images_classifier(disk_images_classifier, data_generators, create_model)


def use_disk_split3(disk_dataset_splitter: DiskDatasetSplitter, disk_images_classifier: DiskImagesClassifier,
                    batch_size=32, create_model=True):
    data_generators = disk_dataset_splitter.create_split3_data_generators(not create_model, batch_size=batch_size)
    __use_disk_images_classifier(disk_images_classifier, data_generators, create_model)


def __use_disk_images_classifier(disk_images_classifier: DiskImagesClassifier, data_generators, create_model):
    if create_model:
        disk_images_classifier.create_model(*data_generators)
    else:
        train_data_generator, val_data_generator, test_data_generator = data_generators
        disk_images_classifier.use_model(train_data_generator)


# DATA IN MEMORY
def data_splitting():
    dataset_loader = DatasetLoader()
    dataset = dataset_loader.get_dataset()

    DatasetSplitter(dataset)


def data_training():
    dataset_loader = DatasetLoader()
    dataset = dataset_loader.get_dataset()

    dataset_splitter = DatasetSplitter()

    images_classifier = ImagesClassifier(
        '/home/dominika/Documents/sem.1 mgr/SzUM/projekt/szum_crosswalk_detection/model1-80k-epoch5')

    # use_split1(dataset, dataset_splitter, images_classifier, create_model=False)
    # use_split2(dataset, dataset_splitter, images_classifier)
    use_split3(dataset, dataset_splitter, images_classifier)


def use_split1(dataset, dataset_splitter: DatasetSplitter, images_classifier: ImagesClassifier, create_model=True):
    split1_dataset = dataset_splitter.split1(dataset)

    if create_model:
        images_classifier.create_model(*split1_dataset)
    else:
        x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set, y_test_set = split1_dataset
        images_classifier.use_model(x_test_set, y_test_set)


def use_split2(dataset, dataset_splitter: DatasetSplitter, images_classifier: ImagesClassifier, create_model=True):
    split2_dataset = dataset_splitter.split2(dataset)

    if create_model:
        data_generator, x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set, y_test_set = split2_dataset
        images_classifier.create_model(x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set, y_test_set,
                                       generator=data_generator)
    else:
        x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set, y_test_set, _ = split2_dataset
        images_classifier.use_model(x_test_set, y_test_set)


def use_split3(dataset, dataset_splitter: DatasetSplitter, images_classifier: ImagesClassifier, create_model=True):
    split3_dataset = dataset_splitter.split3(dataset)

    if create_model:
        data_generator, x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set, y_test_set = split3_dataset
        images_classifier.create_model(x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set, y_test_set,
                                       generator=data_generator)
    else:
        x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set, y_test_set, _ = split3_dataset
        images_classifier.use_model(x_test_set, y_test_set)


def main():
    # data_preparation()

    # data_splitting()
    # data_training()

    # disk_data_splitting()
    disk_data_training()


if __name__ == '__main__':
    main()
