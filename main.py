from data_preparation.filesystem import Filesystem
from data_preparation.frame_attributes import FrameAttributes
from data_preparation.statistics import Statistics


def data_preparation():
    Filesystem()
    FrameAttributes()
    Statistics()


def data_visualization():
    pass


def main():
    data_preparation()
    # data_visualization()


if __name__ == '__main__':
    main()
