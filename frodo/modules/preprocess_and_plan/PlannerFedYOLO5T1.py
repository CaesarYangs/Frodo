import os
import pickle
import yaml
import random
import numpy as np
import torch
from frodo.modules.preprocess_and_plan.Planner import Planner
from batchgenerators.utilities.file_and_folder_operations import *


class PlannerFedYOLO5T1(Planner):
    # def __init__(self, preprocess_output_folder, model_config_dir, fed_config_dir, check_hyper_status):
    #     self.plan_fname = os.path.join(
    #         preprocess_output_folder, 'P_YOLO5T1.pkl')
    #     self.preprocess_output_folder = preprocess_output_folder
    #     self.model_config_dir = model_config_dir
    #     self.fed_config_dir = fed_config_dir
    #     self.check_hyper_status = check_hyper_status

    def __init__(self, args):
        self.plan_fname = os.path.join(
            args.preprocess_output_folder, 'P_YOLO5T1.pkl')
        self.preprocess_output_folder = args.preprocess_output_folder
        self.model_config_dir = args.model_config
        self.fed_config_dir = args.fed_config
        self.check_hyper_status = args.check_hyper_status
        self.debug = args.debug
        self.simluate = args.simluate
        self.hpo_type = args.hpo_type

    def pre_process(self, model_config_attr_name='model_config', fl_config_attr_name='fl_config'):
        if self.model_config_dir != 'None':
            model_config, fl_config = self.load_hypers_from_config_main(
                self.model_config_dir, self.fed_config_dir, model_config_attr_name, fl_config_attr_name)
            self.initialization(model_config, fl_config)
        else:
            fl_config = self.load_hypers_from_config_main(
                fl_config_path=self.fed_config_dir, fl_config_attr_name=fl_config_attr_name)
            self.initialization(fl_config=fl_config)

        self.display()

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        if self.fed_config_dir != 'None':
            self.fed_plan(self.hpo_type)

        self.concat_final_config(bypass=True)
        self.save_plan()

        return self.plan_fname

    def initialization(self, model_config=None, fl_config=None):
        self.hyper_keys = {}

        if model_config:
            for key in model_config.keys():
                setattr(self, key, model_config[key])
                self.hyper_keys[key] = 0

        if fl_config:
            for key in fl_config.keys():
                setattr(self, key, fl_config[key])
                self.hyper_keys[key] = 0

        if self.debug:
            print("DEBUG MODE")

        return self.hyper_keys

    def aggerate_analysis(self):
        return

    def customization(self, **kwargs):
        return

    def distribution(self):
        return

    def display(self):
        print('==========hypers opt status: 1/0==========')
        print(self.hyper_keys)
        print('==========hypers values==========')
        for key in self.hyper_keys.keys():
            print(key, ':', getattr(self, key))
        print('=================================')
        return

    def check_opt_status(self, bypass=False):
        if bypass:
            return True
        for key in self.hyper_keys.keys():
            if self.hyper_keys[key] == 0:
                return False
        return True

    def concat_final_config(self, bypass=False):
        if not self.check_opt_status(bypass):
            return False

        self.plan = {}

        for key in self.hyper_keys.keys():
            self.plan[key] = getattr(self, key)

        return self.plan

    def model_plan(self):
        return

    def fed_plan(self, hpo_type):
        if hpo_type == 'FedEx':
            print("Using FedEx to HPO")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        print("seeding success")

        fed_hypers_config = {'num_configs': self.num_configs}
        return

    def asymmetrical_fed_plan(self):
        return


def main():
    test1 = PlannerFedYOLO5T1(
        '/Users/caesaryang/Developer/Frodo/config/hypers/hypers2.pkl')
    # test1.load_hypers_from_config_main(
    #     'config/hypers/hyp.yolov5.voc.finetune.yaml', 'config/fed_config/fed_test1.yaml')
    # # test1.plan_fname = '/Users/caesaryang/Developer/Frodo/config/hypers/hypers.pkl'
    # test1.pre_process()
    test1.load_plan()


if __name__ == "__main__":
    main()
