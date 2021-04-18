from data_preparation.filesystem import Filesystem
from data_preparation.frame_attributes import FrameAttributes
from data_preparation.statistics import Statistics
from data_splitting.dataset_splitter import DatasetSplitter
from data_splitting.dataset_loader import DatasetLoader


def data_preparation():
    Filesystem()
    # FrameAttributes()
    # Statistics()


def data_splitting():
    dataset_loader = DatasetLoader()
    dataset = dataset_loader.get_dataset()

    DatasetSplitter(dataset)


def main():
    # data_preparation()
    data_splitting()


if __name__ == '__main__':
    main()
