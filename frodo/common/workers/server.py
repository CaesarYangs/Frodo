import logging
import os
import frodo
import copy
from frodo.utilities.utils import recursive_find_python_class
from frodo.common.workers.base_server import BaseServer
from frodo.common.workers.client import Client


class Server(BaseServer):
    def __init__(self, state, config, strategy, mode, address, saving_dir, model=None, ID=0, sim=True, fed_switch=True, client_nums=4, device='cpu', overall_round=10, local_epochs=5, resume_epoch=1, aggregate_method='FedAvg'):
        super(Server, self).__init__(
            ID, state, config, model, strategy, mode, address)
        self._register_default_handlers()

        if config is None:
            logging.error('Server Config Error')
            return

        self.sim = sim
        self.overall_round = overall_round
        self.fed_switch = fed_switch
        self.saving_dir = saving_dir
        self.client_nums = client_nums
        self.device = device
        self.overall_round = 10
        self.local_epochs = local_epochs
        self.resume_epoch = resume_epoch
        self.aggregate_method = aggregate_method

        self.clients = []
        self.local_models = {}

        # self.init_aggregate_method()
        self.init_clients()

        print("finishing standard server initialize...")

    def init_aggregate_method(self):
        search_in = os.path.join(frodo.__path__[0], "modules/fed_algorithms")
        aggregate_method = recursive_find_python_class(
            search_in, self.aggregate_method, 'frodo.modules.fed_algorithms')
        print("using fed aggregate method:", aggregate_method)
        return aggregate_method

    def init_clients(self):
        num_clients = self.client_nums
        for num in range(num_clients):
            self.clients.append(Client(num+1, 1, strategy=self.strategy, device=self.device, overall_round=self.overall_round,
                                local_epochs=self.local_epochs, resume_epoch=self.resume_epoch, aggregate_method=self.aggregate_method))
        print("clients status: nums={}, clients={}".format(
            num_clients, self.clients))
        return self.clients

    def init_model(self):
        return

    def init_server(self):
        return

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
                self.clients[client_id-1].client_train()
            self.gather_local()

            self.start_val('global')
        return

    def start_val(self, val_mode='global'):
        if val_mode == 'global':
            self.start_global_val()
        if val_mode == 'central':
            self.start_central_val()
        return

    def start_global_val(self):
        client_id_list = [x+1 for x in range(self.client_nums)]
        print("System Start Val Globaly...")
        for client_id in client_id_list:
            self.clients[client_id-1].client_val()
        return

    def start_central_val(self):
        return

    def gather_local(self, erase=True):
        client_id_list = [x+1 for x in range(self.client_nums)]
        temp_models = {}
        if erase == True:
            self.local_models = {}
        for client_id in client_id_list:
            self.local_models[client_id] = copy.deepcopy(
                self.clients[client_id - 1].local_model)
        return self.local_models

    def fed_process(self):
        client_id_list = [x+1 for x in range(self.client_nums)]
        fed_func = self.init_aggregate_method()
        new_state_model = fed_func(self.local_models)
        for client in client_id_list:
            self.broadcast_global(client, new_state_model)
        return new_state_model

    def broadcast_global(self, client_id, model):
        self.clients[client_id].local_model = model
        return


def main():
    server = Server(1, '', '', 0, '0.0.0.0', '')
    server.server_train()


if __name__ == '__main__':
    main()
