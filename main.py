from data_preparation.filesystem import Filesystem
from data_preparation.frame_attributes import FrameAttributes
from data_preparation.statistics import Statistics
from data_splitting.data_split import DataSplit


def data_preparation():
    filesystem = Filesystem()
    dataset = filesystem.merge_split_dataset()
    DataSplit(dataset)
    # FrameAttributes()
    # Statistics()


def data_visualization():
    pass


def main():
    data_preparation()
    # data_visualization()


if __name__ == '__main__':
    main()
