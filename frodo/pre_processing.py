import frodo
import os


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str,
                        help='Set necessary network configuration from config file.')
