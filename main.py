from data_preparation.filesystem import Filesystem
from data_preparation.frame_attributes import FrameAttributes
from data_preparation.statistics import Statistics
from data_splitting.dataset_splitter import DatasetSplitter
from data_splitting.dataset_loader import DatasetLoader
from data_training.images_classifier import ImagesClassifier


def data_preparation():
    Filesystem()
    # FrameAttributes()
    # Statistics()


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

    use_split1(dataset, dataset_splitter, images_classifier, create_model=False)
    # use_split2(dataset, dataset_splitter, images_classifier)
    # use_split3(dataset, dataset_splitter, images_classifier)


def use_split1(dataset, dataset_splitter: DatasetSplitter, images_classifier: ImagesClassifier, create_model=True):
    split1_dataset = dataset_splitter.split1(dataset)

    if create_model:
        images_classifier.create_model(*split1_dataset)
    else:
        _, _, _, _, x_test_set, y_test_set = split1_dataset
        images_classifier.use_model(x_test_set, y_test_set)


def use_split2(dataset, dataset_splitter: DatasetSplitter, images_classifier: ImagesClassifier, create_model=True):
    split2_dataset = dataset_splitter.split2(dataset)

    if create_model:
        data_generator, x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set, y_test_set = split2_dataset
        images_classifier.create_model(x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set, y_test_set,
                                       generator=data_generator)
    else:
        _, _, _, _, x_test_set, y_test_set, _ = split2_dataset
        images_classifier.use_model(x_test_set, y_test_set)


def use_split3(dataset, dataset_splitter: DatasetSplitter, images_classifier: ImagesClassifier, create_model=True):
    split3_dataset = dataset_splitter.split3(dataset)

    if create_model:
        data_generator, x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set, y_test_set = split3_dataset
        images_classifier.create_model(x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set, y_test_set,
                                       generator=data_generator)
    else:
        _, _, _, _, x_test_set, y_test_set, _ = split3_dataset
        images_classifier.use_model(x_test_set, y_test_set)


def main():
    # data_preparation()
    # data_splitting()
    data_training()


if __name__ == '__main__':
    main()
