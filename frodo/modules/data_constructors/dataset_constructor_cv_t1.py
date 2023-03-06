import os
import numpy as np
import random
import xml.etree.ElementTree as ET
from frodo.utilities.utils import get_classes_info

# 原始数据和标记存放目录
origin_data_path = '/Users/caesaryang/Developer/Frodo/data'
# 目标数据和标记存放目录，若无需改变位置，设空即可
target_data_path = ''
# 训练时需要的文件索引文件存放目录，置空则为与原始数据同级目录
target_list_path = ''
# 存放类别数据的文件，置空则为与原始数据同级目录
classes_path = 'data/voc_classes.txt'
os_type = ''

# ---模型数据集所需的基本参数
construct_mode = 0
train_val_percent = 0.9
train_percent = 0.9


# ---一些动态计算参数
data_nums = np.zeros(len(origin_data_path))
classes, _ = get_classes_info(classes_path)
classes_nums = np.zeros(len(classes))


class DataConstructorCV_T1(object):
    def __init__(self, dataset_properties):
        if dataset_properties:
            self.dataset_properties = dataset_properties

    def set_target_position(self, dataset_properties=None):
        """自动化设置目标目录结构

        Args:
            dataset_properties (_type_, optional): _description_. Defaults to None.
        """
        if self.dataset_properties:
            target_data_path = self.dataset_properties['target_data_path']
        else:
            target_data_path = dataset_properties['target_data_path']

        if not target_data_path:
            target_data_path = origin_data_path

        if not os.path.exists(target_data_path):
            os.makedirs(target_data_path)

        target_detail_path = list()

        target_detail_path.append(os.path.join(
            target_data_path, 'Annotations'))
        target_detail_path.append(os.path.join(
            target_data_path, 'ImageSets/Main'))
        target_detail_path.append(os.path.join(target_data_path, 'JPEGImages'))

        for i in range(3):
            if not os.path.exists(target_detail_path[i]):
                os.makedirs(target_detail_path[i])

        return target_detail_path

    def get_dataset(self):
        pass

    def convert_annotation(self, image_id, datalist_file):
        """处理标记数据 以xml标记数据为基准

        Args:
            image_id (_type_): _description_
            datalist_file (_type_): _description_
        """
        origin_data_path = self.dataset_properties['origin_data_path']
        classes = self.dataset_properties['classes']

        assert os.path.exists('data')

        file = open(os.path.join(origin_data_path,
                    'Annotations/%s.xml' % (image_id)), encoding='utf-8')
        tree = ET.parse(file)
        root = tree.getroot()

        for obj in root.iter('object'):
            difficult = 0
            if obj.find('difficult') != None:
                difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xml_box = obj.find('bndbox')
            b = (int(float(xml_box.find('xmin').text)), int(float(xml_box.find('ymin').text)), int(
                float(xml_box.find('xmax').text)), int(float(xml_box.find('ymax').text)))

            classes_nums[classes.index(cls)] += 1

            assert os.path.exists(datalist_file)
            datalist_file.write(" " + ",".join([str(a)
                                                for a in b]) + ',' + str(cls_id))

    def check_origin_data(self, annotation='Annotation', image='JPEGImages'):
        """检查原始数据文件是否被正确分为两类，且不为空

        Args:
            annotation (str, optional): _description_. Defaults to 'Annotation'.
            image (str, optional): _description_. Defaults to 'JPEGImages'.

        Returns:
            _type_: _description_
        """

        annotation_path = os.path.join(origin_data_path, annotation)
        image_path = os.path.join(origin_data_path, image)

        if not os.path.exists(annotation_path) and not os.path.exists(image_path):
            print(ValueError("origin data not gather correctly"))
            return False

        return True

    def peek_set():
        pass

    def dataset_constructor(self, dataset_properties=None):
        if not self.check_origin_data():
            raise ValueError("origin data checking error")

        if not dataset_properties:
            dataset_properties = self.dataset_properties

        target_path = self.set_target_position()

        annotation_path = target_path[0]
        imagesets_main_path = target_path[1]

        if construct_mode == 0 or construct_mode == 1:
            print("Generating txt in ImageSets")
            temp_xml = os.listdir(annotation_path)
            total_xml = []
            for file in temp_xml:
                if file.endswith('xml'):
                    total_xml.append(file)


def main():

    dataset_properties = dict()
    dataset_properties['origin_data_path'] = origin_data_path
    dataset_properties['target_list_path'] = target_list_path
    dataset_properties['target_data_path'] = target_data_path

    dataset_properties['os'] = os_type
    dataset_properties['construct_mode'] = construct_mode
    dataset_properties['train_val_percent'] = train_val_percent
    dataset_properties['train_percent'] = train_percent
    dataset_properties['data_nums'] = data_nums
    dataset_properties['classes'] = classes
    dataset_properties['classes_nums'] = classes_nums

    test1 = DataConstructorCV_T1(dataset_properties)
    test1.dataset_constructor()


if __name__ == "__main__":
    main()
