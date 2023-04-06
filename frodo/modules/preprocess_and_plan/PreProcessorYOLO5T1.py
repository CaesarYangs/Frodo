import os
import pickle
import yaml
from frodo.modules.preprocess_and_plan.PreProcessor import PreProcessor
from batchgenerators.utilities.file_and_folder_operations import *


class PreProcessorYOLO5T1(PreProcessor):
    def __init__(self, preprocess_output_folder, model_config_dir, fed_config_dir, check_hyper_status) -> None:
        self.plan_fname = os.path.join(
            preprocess_output_folder, 'P_YOLO5T1.pkl')
        self.preprocess_output_folder = preprocess_output_folder
        self.model_config_dir = model_config_dir
        self.fed_config_dir = fed_config_dir
        self.check_hyper_status = check_hyper_status


def main():
    test1 = PreProcessorYOLO5T1(
        '/Users/caesaryang/Developer/Frodo/config/hypers/hypers2.pkl')
    # test1.load_hypers_from_config_main(
    #     'config/hypers/hyp.yolov5.voc.finetune.yaml', 'config/fed_config/fed_test1.yaml')
    # # test1.plan_fname = '/Users/caesaryang/Developer/Frodo/config/hypers/hypers.pkl'
    # test1.pre_process()
    test1.load_plan()


if __name__ == "__main__":
    main()
