import os
import pickle
import yaml
from batchgenerators.utilities.file_and_folder_operations import *


class PreProcessor(object):
    """ Base class for HPO(Hyper Parameter Optimization) and planing
    """

    def __init__(self, fname=None) -> None:
        return

    def load_hypers_from_config(self, hyp_path, hyp_type):
        try:
            with open(hyp_path, 'r', encoding='utf-8') as f:
                hypers = yaml.safe_load(f)
        except Exception as e:
            print(e)
        setattr(self, hyp_type, hypers)
        return hypers

    def load_hypers_from_config_main(self, model_config_path, fl_config_path, model_config_attr_name='model_config', fl_config_attr_name='fl_config'):
        self.load_hypers_from_config(model_config_path, model_config_attr_name)
        self.load_hypers_from_config(fl_config_path, fl_config_attr_name)
        return [getattr(self, model_config_attr_name), getattr(self, fl_config_attr_name)]

    def pre_process(self):
        return

    def planning(self):
        return

    def save_plan(self, fname=None):
        if fname:
            self.plan_fname = fname
        if not os.path.exists(self.preprocess_output_folder):
            os.makedirs(self.preprocess_output_folder)
        with open(self.plan_fname, 'wb') as f:
            pickle.dump(self.plan, f)

    def load_plan(self, fname=None):
        if fname:
            self.plan_fname = fname
        self.plan = load_pickle(self.plan_fname)
        print(self.plan)
