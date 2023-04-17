import logging
import os
import frodo
import torch
import copy
from frodo.utilities.utils import recursive_find_python_class
from frodo.common.workers.base_server import BaseServer
from frodo.common.workers.server import Server
from frodo.common.workers.worker import Worker
from frodo.utilities.utils import calculate_time_cost


class YOLOv5Server(Server):
    def __init__(self, state, config, strategy, mode, address, saving_dir, model=None, ID=0, sim=True, fed_switch=True, client_nums=4, device='cpu', overall_round=10, local_epochs=5, resume_epoch=1, aggregate_method='FedAvg'):
        super(YOLOv5Server, self).__init__(state, config, strategy, mode, address, saving_dir, model, ID, sim,
                                           fed_switch, client_nums, device, overall_round, local_epochs, resume_epoch, aggregate_method)

    def server_train(self):
        """sever start main training process
        """
        client_id_list = [x+1 for x in range(self.client_nums)]
        local_epochs = self.local_epochs
        local_models = {}
        for epoch in range(self.resume_epoch, self.overall_round+1):
            print("Start Training Communication Round:{}".format(
                epoch))
            for client_id in client_id_list:
                _, self.exp_new_name = self.clients[client_id-1].client_train()
            self.gather_local()
            self.fed_process(epoch)
            self.start_val('global')
        return

    def update_global(self, local_models=None):
        local_models = self.local_models
        mean_state_dict = {}
        # copy列表里第0个model，因为要学文件的格式
        global_model = copy.deepcopy(local_models[1])
        for name, param in global_model.state_dict().items():  # 对于全局模型参数的每一块
            vs = []
            for client in local_models.keys():  # local_models是字典，key是客户端编号，value是对应客户端的权重
                vs.append(local_models[client].float().state_dict()[
                    name])  # 把每个客户端的model里的这一块的参数拿出来放列表里
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.mean(dim=0)  # 求平均

            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).mean(dim=0).long()
            mean_state_dict[name] = mean_value  # 把算得的各块平均放到模具里

        # print(mean_state_dict)
        global_model.load_state_dict(
            mean_state_dict, strict=False)  # 加载这个聚合模型到global_model
        return global_model

    def fed_process(self, epoch, saving_model_dir='runs/weights'):
        print("FedAvg start(Epochs = {}):".format(epoch))
        global_model = self.update_global()  # 联邦平均聚合权重。返回聚合后的model
        print("FedAvg finished.")
        print("Updating avg “model” to ckpt...")
        # 先深拷贝一份ckpt
        copy_ckpt = copy.deepcopy(self.exp_new_name + '/weights/best.pt')
        # 加载拷贝的ckpt文件
        Avg_ckpt = torch.load(copy_ckpt, map_location='cpu')
        # 把文件里的model改成聚合后的
        Avg_ckpt["model"] = global_model
        # 把改完的ckpt文件存到weights里
        if not os.path.exists(saving_model_dir):
            os.makedirs(saving_model_dir)
        torch.save(Avg_ckpt, "runs/weights/avg_ckpt_E{}.pt".format(epoch))
        print("”Avg_Ckpt“(Rounds: {}) Save successfully.".format(epoch))
        return
