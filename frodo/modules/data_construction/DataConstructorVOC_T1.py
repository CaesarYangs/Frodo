import os
import numpy as np
import random
import yaml
from frodo.modules.data_construction.DataConstructor import DataConstructor
from shutil import copyfile
import xml.etree.ElementTree as ET
from frodo.utilities.utils import get_classes_info
import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


class DataConstructorVOC_T1(DataConstructor):
    def __init__(self, dataset_properties=None):
        super().__init__(dataset_properties)

    def set_construct_param_from_config(self, dataset_config_hyp):
        """return self.dataset_properties for VOC dataset properties

        Args:
            dataset_config_hyp (_type_): _description_

        Returns:
            dict: dataset_properties
        """
        self.dataset_properties['origin_data_path'] = dataset_config_hyp['origin_data_path']
        self.dataset_properties['target_data_path'] = dataset_config_hyp['target_data_path']
        self.dataset_properties['target_data_set'] = dataset_config_hyp['target_data_set']
        self.dataset_properties['construct_mode'] = dataset_config_hyp['construct_mode']
        self.dataset_properties['train_val_percent'] = dataset_config_hyp['train_val_percent']
        self.dataset_properties['train_percent'] = dataset_config_hyp['train_percent']
        classes = self.dataset_properties['classes'] = dataset_config_hyp['classes']
        self.dataset_properties['classes_nums'] = np.zeros(
            len(classes))

        logging.info('Load config from file successfully')
        return self.dataset_properties

    def set_target_position_dir(self):
        """_summary_

        Args:
            dataset_properties (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        dataset_properties = self.dataset_properties

        target_data_path = dataset_properties['target_data_path']

        if not target_data_path:
            target_data_path = self.dataset_properties['target_data_path'] = dataset_properties['origin_data_path']

        if not os.path.exists(target_data_path):
            os.makedirs(target_data_path)

        target_data_path_folder_detail = list()
        target_data_path_folder_detail.append(os.path.join(
            target_data_path, 'Annotations'))
        target_data_path_folder_detail.append(os.path.join(
            target_data_path, 'ImageSets/Main'))
        target_data_path_folder_detail.append(
            os.path.join(target_data_path, 'images'))
        target_data_path_folder_detail.append(
            os.path.join(target_data_path, 'labels'))

        logging.info('Making target directory')

        for i in range(4):
            if not os.path.exists(target_data_path_folder_detail[i]):
                os.makedirs(target_data_path_folder_detail[i])

        return target_data_path_folder_detail, target_data_path

    def move_dataset(self):
        """copy all datasets to the target directory

        Args:
            dataset_properties (_type_, optional): _description_. Defaults to None.

        Returns:
            list: target_data_path_folder_detail
            str:target_data_set
        """
        dataset_properties = self.dataset_properties

        target_data_path_folder_detail, target_data_set = self.set_target_position_dir()
        if dataset_properties['target_data_path'] == dataset_properties['origin_data_path']:
            return target_data_path_folder_detail, target_data_set

        origin_annotation_path = os.path.join(
            dataset_properties['origin_data_path'], 'Annotations')
        origin_image_path = os.path.join(
            dataset_properties['origin_data_path'], 'images')

        temp_xml = os.listdir(origin_annotation_path)
        temp_img = os.listdir(origin_image_path)

        logging.info('Moving dataset to target path...')
        try:
            for xml in temp_xml:
                if xml.endswith('xml'):
                    copyfile(os.path.join(origin_annotation_path, xml),
                             os.path.join(target_data_path_folder_detail[0], xml))
            for img in temp_img:
                if img.endswith('jpg'):
                    copyfile(os.path.join(origin_image_path, img),
                             os.path.join(target_data_path_folder_detail[2], img))
        except Exception as e:
            logging.error('Moving error')
        logging.info('Completed moving dataset')
        return target_data_path_folder_detail, target_data_set

    def convert_annotation(self, image_id):
        """convert annotation to YOLO like object detection requirements

        Args:
            image_id (_type_): _description_
        """
        dataset_path = self.dataset_properties['target_data_path']
        classes = self.dataset_properties['classes']
        in_file = open(os.path.join(dataset_path, 'Annotations/%s.xml' %
                                    (image_id)), encoding='UTF-8')
        out_file = open(os.path.join(
            dataset_path, 'labels/%s.txt' % (image_id)), 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            # difficult = obj.find('difficult').text
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            b1, b2, b3, b4 = b
            # 标注越界修正
            if b2 > w:
                b2 = w
            if b4 > h:
                b4 = h
            b = (b1, b2, b3, b4)
            bb = self.convert((w, h), b)
            out_file.write(str(cls_id) + " " +
                           " ".join([str(a) for a in bb]) + '\n')

    def convert(self, size, box):
        """convert labeled bbox

        Args:
            size (_type_): _description_
            box (_type_): _description_

        Returns:
            _type_: _description_
        """
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = (box[0] + box[1]) / 2.0 - 1
        y = (box[2] + box[3]) / 2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    def dataset_constructor(self):
        """Main function to construct the dataset structure and files

        Args:
            dataset_properties (_type_, optional): _description_. Defaults to None.

        Returns:
            boolean
        """
        dataset_properties = self.dataset_properties

        self.check_origin_data()
        target_data_path_folder_detail, _ = self.move_dataset()

        sets = dataset_properties['target_data_set']
        dataset_path = dataset_properties['target_data_path']
        construction_mode = dataset_properties['construct_mode']

        annotation_path = target_data_path_folder_detail[0]
        imagesets_main_path = target_data_path_folder_detail[1]
        print(dataset_path)
        images_path = target_data_path_folder_detail[2]
        labels_path = target_data_path_folder_detail[3]

        if construction_mode == 0 or construction_mode == 1:
            logging.info(
                "Construction mode: building train,val and trainval lists")
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

            logging.info("Generating txt in ImageSets/Main")
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
            logging.info("Generation to ImageSets/Main done")

        if construction_mode == 0 or construction_mode == 2:
            logging.info(
                "Construction mode: building txt file for training process")
            for image_set in sets:
                image_ids = open(
                    os.path.join(imagesets_main_path, '%s.txt' % (image_set))).read().strip().split()
                list_file = open(
                    os.path.join(dataset_path, '%s.txt' % (image_set)), 'w')
                for image_id in image_ids:
                    list_file.write(os.path.join(
                        images_path, '%s.jpg\n' % (image_id)))
                    self.convert_annotation(image_id)
                list_file.close()
            logging.info("Generation to txt for train and val done")
            logging.info("Generation to labels done")
        return True
