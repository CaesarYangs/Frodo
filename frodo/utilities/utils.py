import numpy as np
import importlib
import pkgutil
import os
import logging


def get_classes_info(classes_path):
    """return user-setting classes name and count

    Args:
        classes_path (string): path to class.txt

    Returns:
        list: classes name
        int: classes count
    """
    with open(classes_path, encoding="utf-8") as f:
        classes_name = f.readlines()
    classes_name = [c.strip() for c in classes_name]
    return classes_name, len(classes_name)


def recursive_find_python_class(folder, trainer_name, current_module):
    class_name = None
    modules = os.listdir(folder)

    try:
        for mod_name in modules:
            mod_name = os.path.splitext(mod_name)[0]
            if trainer_name == mod_name:
                m = importlib.import_module(current_module+'.'+mod_name)
                if hasattr(m, trainer_name):
                    class_name = getattr(m, trainer_name)
                    break
        return class_name
    except Exception as e:
        logging.error(e)
        return False


def calculate_time_cost(instance_number,
                        comm_size,
                        comp_speed=None,
                        comm_bandwidth=None,
                        augmentation_factor=3.0):
    # Served as an example, this cost model is adapted from FedScale at
    # https://github.com/SymbioticLab/FedScale/blob/master/fedscale/core/
    # internal/client.py#L35 (Apache License Version 2.0)
    # Users can modify this function according to customized cost model
    if comp_speed is not None and comm_bandwidth is not None:
        comp_cost = augmentation_factor * instance_number * comp_speed
        comm_cost = 2.0 * comm_size / comm_bandwidth
    else:
        comp_cost = 0
        comm_cost = 0

    return comp_cost, comm_cost
