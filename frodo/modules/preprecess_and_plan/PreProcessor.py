import os
import pickle
import yaml
from batchgenerators.utilities.file_and_folder_operations import *


class PreProcessor(object):
    """数据+参数加载预处理基类
    """

    def __init__(self) -> None:
        pass

    def load_hypers_from_config(self, hyp_path, hyp_type):
        try:
            with open(hyp_path, 'r', encoding='utf-8') as f:
                hypers = yaml.safe_load(f)
        except Exception as e:
            print(e)
        setattr(self, hyp_type, hypers)
        return hypers

    def save_plan(self, fname=None):
        if fname:
            self.plan_fname = fname
        with open(self.plan_fname, 'wb') as f:
            pickle.dump(self.plan, f)

    def load_plan(self, fname=None):
        if fname:
            self.plan_fname = fname
        self.plan = load_pickle(self.plan_fname)


def main():
    test1 = PreProcessor()
    test1.load_hypers_from_config(
        '/Users/caesaryang/Developer/Frodo/config/hypers/hyp.yolov5.voc.finetune.yaml')
    test1.plan_fname = '/Users/caesaryang/Developer/Frodo/config/hypers/hypers.pkl'
    test1.save_plan()


if __name__ == "__main__":
    main()
