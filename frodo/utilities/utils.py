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
