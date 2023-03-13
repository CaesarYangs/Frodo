import frodo
import shutil
import yaml
import os
from frodo.utilities.utils import recursive_find_python_class


def set_construct_param_from_config(constructor, config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        data_hyp_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
    dataset_properties = constructor.set_construct_param_from_config(
        data_hyp_dict)
    return dataset_properties


def set_construct_param_default(constructor, origin_data_path, target_data_path, label_file_path):
    dataset_properties = constructor.set_construct_param_default(
        origin_data_path, target_data_path, label_file_path)
    return dataset_properties


def main():
    import argparse
    parser = argparse.ArgumentParser()

    # set use configuration file
    parser.add_argument('-c', '--config', type=str,
                        help='Determain whether to use a detailed configuration file to set construct process')
    # ===if not using config file
    parser.add_argument("-cons", "--constructor", type=str, default="DataConstructorCV_T1",
                        help="Name of the construct class for dataset construction.Default for DataConstructorCV_T1.")
    # origin_data_path
    parser.add_argument('-odp', "--origin_data_path", type=str,
                        help="String to locate the dataset you want to construct.NOTICE:must pre classified as Annotations and JPEGImages two folders.")
    # target_data_path
    parser.add_argument('-tdp', "--target_data_path", type=str,
                        help='String to store the target position where you want to store the dataset after construct.NOTICE: if -tdp is none, it means using the origin_data_path as target path.')
    # label_file_path
    parser.add_argument('-lfp', "--label_file_path", type=str,
                        help='String to locate the label or classes file.')

    args = parser.parse_args()
    print(args.constructor)

    config = args.config
    constructor = args.constructor
    origin_data_path = args.origin_data_path
    target_data_path = args.target_data_path
    label_file_path = args.label_file_path

    search_in = os.path.join(frodo.__path__[0], "modules/data_construction")
    constructor_class = recursive_find_python_class(
        search_in, args.constructor, 'frodo.modules.data_construction')

    constructor_obj = constructor_class()

    if args.config:
        set_construct_param_from_config(
            constructor_obj, config)
    else:
        set_construct_param_default(
            constructor_obj, origin_data_path, target_data_path, label_file_path)

    if constructor_obj.dataset_constructor():
        print('done')
    else:
        print('constructor error')
