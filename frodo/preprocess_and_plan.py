import frodo
import os
from frodo.utilities.utils import recursive_find_python_class
import logging


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-pp', '--preprocessor', type=str, default="PreProcessorYOLO5T1",
                        help='Set preprocessor to use.')
    parser.add_argument('-pl', '--planner', type=str, default="Planer",
                        help='Set planner to use.')
    parser.add_argument('-mc', '--modelconfig', type=str,
                        help='Set necessary network configuration from config file.')
    parser.add_argument('-fc', '--fedconfig', type=str,
                        help='Set necessary federated learning configuration from config file.')
    parser.add_argument('--check_hyper_status', type=str, default="False",
                        help='Check hypers whether they are configurated.')
    parser.add_argument('-pof', '--preprocess_output_folder', type=str, default="output/preprocess_and_plan",
                        help='Check hypers whether they are configurated.')

    args = parser.parse_args()
    print("Using PreProcessor:", args.preprocessor)
    print("Using Planner:", args.planner)

    model_config_dir = args.modelconfig
    fed_config_dir = args.fedconfig
    check_hyper_status = args.check_hyper_status
    preprocess_output_folder = args.preprocess_output_folder

    search_in = os.path.join(frodo.__path__[0], "modules/preprocess_and_plan")

    try:
        preprocess_class = recursive_find_python_class(
            search_in, args.preprocessor, 'frodo.modules.preprocess_and_plan')
        plan_class = recursive_find_python_class(
            search_in, args.planner, 'frodo.modules.preprocess_and_plan')
    except Exception as e:
        logging.error(e)

    if preprocess_class:
        preprocessor = preprocess_class(
            preprocess_output_folder, model_config_dir, fed_config_dir, check_hyper_status)
        hypers_pkl = preprocessor.pre_process()
        print("save optimized hypers to:", hypers_pkl)

    if plan_class:
        plan = plan_class()
