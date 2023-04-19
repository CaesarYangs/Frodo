import logging
import os
import frodo
from frodo.utilities.utils import recursive_find_python_class
from frodo.common.workers.base_server import BaseServer
from frodo.common.workers.worker import Worker
from frodo.utilities.utils import calculate_time_cost


class Client(Worker):
    def __init__(self, ID=-1, state=0, config=None, model=None, strategy=None, mode=None, address='0.0.0.0', saving_dir=None, device='cpu', overall_round=10, local_epochs=5, resume_epoch=1, aggregate_method='FedAvg'):
        super(Client, self).__init__(ID, state, config,
                                     model, strategy, mode, address)
        self._saving_dir = saving_dir
        self._device = device
        self._overall_round = overall_round
        self._local_epochs = local_epochs
        self._resume_epoch = resume_epoch
        self._aggregate_method = aggregate_method
        self._local_model = None

    def _gen_timestamp(self, init_timestamp, instance_number):
        if init_timestamp is None:
            return None
        comp_cost, comm_cost = calculate_time_cost(
            instance_number=instance_number,
            comm_size=self.model_size,
            comp_speed=self.comp_speed,
            comm_bandwidth=self.comm_bandwidth)
        return init_timestamp + comp_cost + comm_cost

    def client_train(self):
        print("client {} start training...".format(self.ID))
        model_config = self._cfg
        return

    def client_val(self):
        print("client {} start validation".format(self.ID))
        return

    @property
    def local_model(self):
        return self._local_model

    @local_model.setter
    def local_model(self, value):
        self._local_model = value
        return self._local_model
