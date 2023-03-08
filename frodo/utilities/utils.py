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
    class_name = None
    modules = os.listdir(folder)
    for mod_name in modules:
        mod_name = os.path.splitext(mod_name)[0]
        if trainer_name == mod_name:
            m = importlib.import_module(current_module+'.'+mod_name)
            if hasattr(m, trainer_name):
                class_name = getattr(m, trainer_name)
                break

    # for _, modname, ispkg in pkgutil.iter_modules(folder):
    #     print('====', modname, ispkg)
    #     if not ispkg:
    #         m = importlib.import_module(current_module + "." + modname)
    #         if hasattr(m, trainer_name):
    #             tr = getattr(m, trainer_name)
    #             break

    # if tr is None:
    #     for _, modname, ispkg in pkgutil.iter_modules(folder):
    #         print("==--:", modname, ispkg)
    #         if ispkg:
    #             next_current_module = current_module + "." + modname
    #             next_folder = os.path.join(folder, modname)

    #             tr = recursive_find_python_class(
    #                 next_folder, trainer_name, current_module=next_current_module)
    #         if tr is not None:
    #             break

    return class_name
