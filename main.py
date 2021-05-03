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
    split1_dataset = dataset_splitter.split1(dataset)

    images_classifier = ImagesClassifier(*split1_dataset)

    # split2_dataset = dataset_splitter.split2(dataset)
    # split3_dataset = dataset_splitter.split3(dataset)


def main():
    # data_preparation()
    # data_splitting()
    data_training()


if __name__ == '__main__':
    main()
