import frodo
import os
from frodo.utilities.utils import recursive_find_python_class
import logging


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-pp', '--preprocessor', type=str, default="",
                        help='Set preprocessor to use.')
    parser.add_argument('-pl', '--planner', type=str, default="PlannerFedYOLO5T1",
                        help='Set planner to use.')
    parser.add_argument('-mcfg', '--model_config', default="None", type=str,
                        help='Set necessary network configuration from config file.')
    parser.add_argument('-fcfg', '--fed_config', default="None", type=str,
                        help='Set necessary federated learning configuration from config file.')
    parser.add_argument('--check_hyper_status', type=str, default="False",
                        help='Check hypers whether they are configurated.')
    parser.add_argument('-pof', '--preprocess_output_folder', type=str, default="output/preprocess_and_plan",
                        help='Check hypers whether they are configurated.')
    parser.add_argument('-sim', '--simluate', type=str, default="True",
                        help='Determain whether to simulate fed process on single machine using single/multi threads.')
    parser.add_argument('--hpo_type', type=str, default="FedEx",
                        help='Determain whether to simulate fed process on single machine using single/multi threads.')
    parser.add_argument('--debug', type=str, default="True",
                        help='open for debug mode. Recommend for changing and tunning scenario.')

    args = parser.parse_args()
    print("Using PreProcessor:", args.preprocessor)
    print("Using Planner:", args.planner)

    search_in = os.path.join(frodo.__path__[0], "modules/preprocess_and_plan")

    try:
        preprocess_class = recursive_find_python_class(
            search_in, args.preprocessor, 'frodo.modules.preprocess_and_plan')
        plan_class = recursive_find_python_class(
            search_in, args.planner, 'frodo.modules.preprocess_and_plan')
    except Exception as e:
        logging.error(e)

    if plan_class:
        # planner = plan_class(
        #     args.preprocess_output_folder, args.model_config, args.fed_config, args.check_hyper_status)
        planner = plan_class(args)
        hypers_pkl = planner.pre_process()
        print("save optimized hypers to:", hypers_pkl)
