import numpy as np
import importlib
import pkgutil
import os


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
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, trainer_name):
                tr = getattr(m, trainer_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(
                    [join(folder[0], modname)], trainer_name, current_module=next_current_module)
            if tr is not None:
                break

    return tr
