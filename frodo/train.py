import frodo
import os
from frodo.utilities.utils import recursive_find_python_class
import logging


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--trainer', type=str, default="YOLOv5Server",
                        help='Set specific trainer to use.')
    parser.add_argument('--state', type=str, default='',
                        help='Path to the saved state dictionary of the model.')
    parser.add_argument('--config', type=str, default='',
                        help='Path to the JSON configuration file.')
    parser.add_argument('--strategy', type=str, default='',
                        help='Federated learning strategy to use.')
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode of operation (train or test).')
    parser.add_argument('--address', type=str, default='localhost',
                        help='Address of the server to connect to.')
    parser.add_argument('--saving-dir', type=str, default='/Users/caesaryang/Developer/Frodo/runs',
                        help='Directory to save the trained models.')
    parser.add_argument('--model', type=str, default='',
                        help='Path to the model to be loaded.')
    parser.add_argument('--ID', type=int, default=0,
                        help='ID of the client.')
    parser.add_argument('--sim', action='store_true',
                        help='Flag to simulate data.')
    parser.add_argument('--fed-switch', action='store_true',
                        help='Flag to enable federated switching.')
    parser.add_argument('--client-nums', type=int, default=2,
                        help='Number of clients to use for federated learning.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for training (cpu or gpu).')
    parser.add_argument('--overall-round', type=int, default=1,
                        help='Number of overall rounds for federated training.')
    parser.add_argument('--local-epochs', type=int, default=5,
                        help='Number of local epochs for each client in each round.')
    parser.add_argument('--resume-epoch', type=int, default=1,
                        help='Epoch to resume training from (set to 1 to start from scratch).')
    parser.add_argument('--aggregate-method', type=str, default='FedAvg',
                        help='Aggregation method to use for combining client models.')

    args = parser.parse_args()
    print("Using Trainer:", args.trainer)

    search_in = os.path.join(frodo.__path__[0], "modules/trainers")

    try:
        preprocess_class = recursive_find_python_class(
            search_in, args.trainer, 'frodo.modules.trainers')
    except Exception as e:
        logging.error(e)

    train_server = preprocess_class(args.state,
                                    args.config,
                                    args.strategy,
                                    args.mode,
                                    args.address,
                                    args.saving_dir,
                                    args.model,
                                    args.ID,
                                    args.sim,
                                    args.fed_switch,
                                    args.client_nums,
                                    args.device,
                                    args.overall_round,
                                    args.local_epochs,
                                    args.resume_epoch,
                                    args.aggregate_method)
    train_server.start_fed_training()

