import os
import numpy as np
import random
from shutil import copyfile
import xml.etree.ElementTree as ET
from frodo.utilities.utils import get_classes_info

# 原始数据和标记存放目录
origin_data_path = '/Users/caesaryang/Downloads/test_data'
# 目标数据和标记存放目录，若无需改变位置，设空即可
target_data_path = '/Users/caesaryang/Downloads/test_data2'
# 训练时需要的文件索引文件存放目录，置空则为与原始数据同级目录
target_data_set = ('', 'train.txt', 'val.txt')
# 存放类别数据的文件，置空则为与原始数据同级目录
classes_path = 'data/voc_classes.txt'

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
        # if self.dataset_properties:
        #     target_data_path = self.dataset_properties['target_data_path']
        # else:
        #     target_data_path = dataset_properties['target_data_path']

        if not dataset_properties:
            dataset_properties = self.dataset_properties
        target_data_path = dataset_properties['target_data_path']

        if not target_data_path:
            if self.dataset_properties:
                self.dataset_properties['target_data_path'] = dataset_properties['origin_data_path']
            target_data_path = dataset_properties['origin_data_path']

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

        return target_detail_path, target_data_path

    def move_dataset(self):
        if not self.dataset_properties['target_data_path']:
            return

        target_detail_path, target_data_path = self.set_target_position()
        origin_annotation_path = os.path.join(
            self.dataset_properties['origin_data_path'], 'Annotations')
        origin_image_path = os.path.join(
            self.dataset_properties['origin_data_path'], 'JPEGImages')

        temp_xml = os.listdir(origin_annotation_path)
        temp_img = os.listdir(origin_image_path)

        for xml in temp_xml:
            if xml.endswith('xml'):
                copyfile(os.path.join(origin_annotation_path, xml),
                         os.path.join(target_detail_path[0], xml))
        for img in temp_img:
            copyfile(os.path.join(origin_image_path, img),
                     os.path.join(target_detail_path[2], img))

    def convert_annotation(self, image_id, datalist_file):
        """处理标记数据 以xml标记数据为基准

        Args:
            image_id (_type_): _description_
            datalist_file (_type_): _description_
        """
        target_data_path = self.dataset_properties['target_data_path']
        classes = self.dataset_properties['classes']

        assert os.path.exists('data')

        file = open(os.path.join(target_data_path,
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
        random.seed(0)
        if not self.check_origin_data():
            raise ValueError("origin data checking error")

        if not dataset_properties:
            dataset_properties = self.dataset_properties

        target_path, target_data_path = self.set_target_position()

        self.move_dataset()

        annotation_path = target_path[0]
        imagesets_main_path = target_path[1]

        if construct_mode == 0 or construct_mode == 1:
            print("Generating txt in ImageSets")
            temp_xml = os.listdir(annotation_path)
            total_xml = []
            for file in temp_xml:
                if file.endswith('xml'):
                    total_xml.append(file)

            xml_file_num = len(total_xml)
            list_range = range(xml_file_num)
            tv = int(xml_file_num*dataset_properties['train_val_percent'])
            tr = int(tv*dataset_properties['train_percent'])

            train_val_list = random.sample(list_range, tv)
            train_list = random.sample(train_val_list, tr)

            print("train and val data size: ", tv)
            print("train data size: ", tr)

            print("Generating txt in ImageSets/Main")
            f_trainval = open(os.path.join(
                imagesets_main_path, 'trainval.txt'), 'w')
            f_test = open(os.path.join(imagesets_main_path, 'test.txt'), 'w')
            f_train = open(os.path.join(imagesets_main_path, 'train.txt'), 'w')
            f_val = open(os.path.join(imagesets_main_path, 'val.txt'), 'w')

            for i in list_range:
                name = total_xml[i][:-4]+'\n'
                if i in train_val_list:
                    f_trainval.write(name)
                    if i in train_list:
                        f_train.write(name)
                    else:
                        f_val.write(name)
                else:
                    f_test.write(name)

            f_trainval.close()
            f_train.close()
            f_val.close()
            f_test.close()
            print("Generatd ImageSets/Main done")

        if construct_mode == 0 or construct_mode == 2:
            print("generating txt for training")
            target_data_set = dataset_properties['target_data_set']
            for i in range(1, 3):
                if not target_data_set[0]:
                    target_txt_path = os.path.join(
                        target_data_path, target_data_set[i])
                else:
                    target_txt_path = os.path.join(
                        target_data_set[0], target_data_set[i])

                image_ids = open(os.path.join(target_data_path,
                                              'ImageSets/Main/%s' % (target_data_set[i]))).read().strip().split()
                list_file = open(target_txt_path, 'w', encoding='utf-8')
                for image_id in image_ids:
                    list_file.write('%s/JPEGImages/%s.jpg' %
                                    (os.path.abspath(target_data_path), image_id))
                    self.convert_annotation(image_id, list_file)
                    list_file.write('\n')
                list_file.close()
            print("done---generate txt for training done")
        return True


def main():
    dataset_properties = dict()
    dataset_properties['origin_data_path'] = origin_data_path
    dataset_properties['target_data_path'] = target_data_path
    dataset_properties['target_data_set'] = target_data_set
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
