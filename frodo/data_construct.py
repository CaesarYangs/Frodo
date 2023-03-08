import frodo
import shutil
from frodo.utilities.utils import recursive_find_python_class


def set_construct_param_from_config(config_path):
    dataset_properties = []
    return dataset_properties


def set_construct_param_default(origin_data_path, target_data_path, label_file_path):
    dataset_properties = []
    return dataset_properties


def main():
    import argparse
    parser = argparse.ArgumentParser()

    # set use configuration file
    parser.add_argument('-c', '--config', type=str, nargs='+',
                        help='Determain whether to use a detailed configuration file to set construct process')
    # ===if not using config file
    parser.add_argument("-cons", "--constructor", type=str, default="DataConstructorCV_T1",
                        help="Name of the construct class for dataset construction.Default for DataConstructorCV_T1.")
    # origin_data_path
    parser.add_argument('-odp', "--origin_data_path", type=str, nargs="+",
                        help="String to locate the dataset you want to construct.NOTICE:must pre classified as Annotations and JPEGImages two folders.")
    # target_data_path
    parser.add_argument('-tdp', "--target_data_path", type=str, nargs='+',
                        help='String to store the target position where you want to store the dataset after construct.NOTICE: if -tdp is none, it means using the origin_data_path as target path.')
    # label_file_path
    parser.add_argument('-lfp', "--label_file_path", type=str, nargs='+',
                        help='String to locate the label or classes file.')

    args = parser.parse_args()

    if args.config:
        dataset_properties = set_construct_param_from_config(args.config)
    else:
        dataset_properties = set_construct_param_default(
            args.origin_data_path, args.target_data_path, args.label_file_path)

    constructor_class = recursive_find_python_class()

    constructor = constructor_class(dataset_properties)
    if constructor.dataset_constructor():
        print('done')
    else:
        print('constructor error')
