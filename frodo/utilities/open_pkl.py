import os
import pickle


def load_plan(plan_fname):
    f = open(plan_fname, 'rb')
    plan = pickle.load(f)
    print(plan)


def main():
    path = '/Users/caesaryang/Downloads/fedex_hpo_res/results.pkl'
    load_plan(path)


if __name__ == '__main__':
    main()
