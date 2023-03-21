import os
import yaml


class PreProcessor(object):
    """数据+参数加载预处理基类
    """

    def __init__(self) -> None:
        pass

    def load_hypers_from_config(self, hyp_path=None, hyp_type=None):
        """动态加载初始超参

        Args:
            hyp_path (_type_, optional): _description_. Defaults to None.
            hyp_type (_type_, optional): _description_. Defaults to None.

        Returns:
            dict: hypers
        """
        try:
            if hyp_path == None:
                hyp_path = self.hyp_path
            with open(hyp_path, 'r', encoding='utf-8') as f:
                hypers = yaml.safe_load(f)
        except Exception as e:
            print(e)
        setattr(self, hyp_type, hypers)
        return hypers


def main():
    hyp_name = 'main_hyp_dict'
    test1 = PreProcessor()
    test1.load_hypers_from_config(
        '/Users/caesaryang/Developer/Frodo/config/hypers/hyp.yolov5.voc.finetune.yaml', hyp_name)
    print(getattr(test1, hyp_name))


if __name__ == "__main__":
    main()
