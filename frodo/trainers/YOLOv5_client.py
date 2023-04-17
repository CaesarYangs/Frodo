import logging
import os
import frodo
import torch
from frodo.utilities.utils import recursive_find_python_class
from frodo.common.workers.base_server import BaseServer
from frodo.common.workers.client import Client
from frodo.common.workers.worker import Worker
from frodo.utilities.utils import calculate_time_cost


class YOLOv5Client(Client):
    def __init__(self, ID=-1, state=0, config=None, model=None, strategy=None, mode=None, address='0.0.0.0', saving_dir=None, device='cpu', overall_round=10, local_epochs=5, resume_epoch=1, aggregate_method='FedAvg', model_dir=None):
        super(YOLOv5Client, self).__init__(ID, state, config, model, strategy, mode, address, saving_dir,
                                           device, overall_round, local_epochs, resume_epoch, aggregate_method)
        self.model_dir = model_dir

    def client_train_from_ori(self, epoch):
        print("client {} start training...".format(self.ID))
        model_config = self._cfg
        self.batch_size = 16
        self.data_path = '/Users/caesaryang/Developer/1-Graduate/datasets/helmet_test2_car_factory_full'
        self.weights = ''
        if epoch == 1:  # 如果是首轮训练，使用初始权重。
            logging.info('start training process')
            os.system("python {}".format(self.trainer_dir))
            # os.system(
            #     "python {} --epochs {} --batch {} --data {} --device {} --weights {}".format(self.trainer_dir, self._local_epochs, self.data_path, self._device, self.weights))
        else:  # 非首轮训练，使用聚合后的权重。
            os.system(
                "python {} --weights weights/avg_ckpt_E{}.pt".format(self.trainer_dir, epoch-1))
        weights_save_path = "runs/train/exp/weights/" + "last.pt"  # 每个客户端训练完后的last_ckpt
        ckpt_load = torch.load(
            weights_save_path, map_location='cpu')  # 加载权重文件

        exp_name = "runs/train/exp"  # exp改名
        self.exp_new_name = exp_new_name = "runs/train/E{}_C{}".format(
            epoch, self.ID)
        os.rename(exp_name, exp_new_name)

        self.local_model = ckpt_model = ckpt_load["model"]  # 把model部分拿出来
        return ckpt_model, self.exp_new_name

    def clien_train_from_cfg(self, epoch, cfg=None):
        return

    def client_train(self, epoch, from_origin, cfg=None):
        self.trainer_dir = os.path.join(self.model_dir, 'train.py')
        if from_origin:
            return self.client_train_from_ori(epoch)
        else:
            return self.clien_train_from_cfg(epoch=epoch, cfg=cfg)

    def client_val(self):
        self.val_dir = os.path.join(self.model_dir, 'val.py')
        return


def main():
    yolo = YOLOv5Client(model_dir='./frodo/modules/models/YOLOv5_HNet')
    print(yolo.client_train(1, True))


if __name__ == '__main__':
    main()
