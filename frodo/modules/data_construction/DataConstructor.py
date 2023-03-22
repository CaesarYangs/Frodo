import os
import numpy as np
import random
import yaml
import frodo
from shutil import copyfile
import xml.etree.ElementTree as ET
from frodo.utilities.utils import get_classes_info


class DataConstructor:
    """DataConstructor parent class
    """

    def __init__(self, dataset_properties=None):
        if dataset_properties:
            self.dataset_properties = dataset_properties
        else:
            self.dataset_properties = dict()

    def set_target_position_dir(self, dataset_properties=None):
        pass

    def move_dataset(self, dataset_properties=None):
        pass

    def check_origin_data(self, dataset_properties=None):
        pass

    def peek_dataset(self, dataset_properties=None):
        pass

    def set_construct_param_default(self, origin_data_path='', target_data_path='', classes_path=''):
        pass

    def set_construct_param_from_config(self, dataset_config_hyp):
        pass

    def dataset_constructor(self, dataset_properties=None):
        pass


def main():
    return


if __name__ == '__main__':
    main()
