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

    # for split1
    split1_dataset = dataset_splitter.split1(dataset)
    images_classifier = ImagesClassifier(*split1_dataset)

    # for split2
    # split2_dataset = dataset_splitter.split2(dataset)
    # data_generator, x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set, y_test_set = split2_dataset
    # images_classifier = ImagesClassifier(x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set,
    #                                      y_test_set, generator=data_generator)

    # for split3
    # split3_dataset = dataset_splitter.split3(dataset)
    # data_generator, x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set, y_test_set = split3_dataset
    # images_classifier = ImagesClassifier(x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set,
    #                                      y_test_set, generator=data_generator)


def main():
    # data_preparation()
    # data_splitting()
    data_training()


if __name__ == '__main__':
    main()
