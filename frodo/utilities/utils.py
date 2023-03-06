import numpy as np


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
