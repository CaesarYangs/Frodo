import logging
import os
import frodo
from frodo.utilities.utils import recursive_find_python_class
from frodo.common.workers.base_server import BaseServer
from frodo.common.workers.worker import Worker


class Server(BaseServer):
    def __init__(self, ID, state, config, local_models, strategy, mode, address, saving_dir, sim=True, fed_switch=True, client_nums=4, device='cpu', overall_round=10, local_epochs=5, resume_epoch=1, aggregate_method='FedAvg'):
        super(Server, self).__init__(
            ID, state, config, local_models, strategy, mode, address)
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

        print("finishing standard server initialize...")

    def init_aggregate_method(self):
        search_in = os.path.join(frodo.__path__[0], "modules/fed_algorithms")
        aggregate_method = recursive_find_python_class(
            search_in, self.aggregate_method, 'frodo.modules.fed_algorithms')
        print("using fed aggregate method:", self.aggregate_method)
        return aggregate_method

    def init_clients(self):
        num_clients = self.client_nums
        self.clients = []
        for client in range(num_clients):
            self.clients.append(Worker(client, 0))
        print(self.clients)
        return self.clients

    def init_model(self):
        return

    def init_server(self):
        return

    def start_train(self):
        return

    def start_val(self):
        return

    def start_glo_val(self):
        return

    def start_central_val(self):
        return

    def gather_local(self):
        return

    def broadcast_global(self):
        return


def main():
    server = Server(1, 0, 0, 0, 0, 0, '0.0.0.0', '')
    server.init_aggregate_method()


if __name__ == '__main__':
    main()
