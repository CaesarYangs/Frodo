import copy
import logging
import os
import torch
import numpy as np


class FedAvg:
    def __init__(self, local_models=[]) -> None:
        self._local_models = local_models
        return

    @property
    def local_models(self):
        return self._local_models

    @local_models.setter
    def local_models(self, value):
        self.local_models = value

    def update_global(self):
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
