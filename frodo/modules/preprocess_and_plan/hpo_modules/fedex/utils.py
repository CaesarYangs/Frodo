import argparse
import os
import pdb
import pickle
import random
from copy import deepcopy
from glob import glob
from heapq import nsmallest
from itertools import product
from math import ceil
from math import log
from operator import itemgetter
import numpy as np
import torch
from numpy.linalg import norm
from scipy.special import logsumexp
from tensorboardX import SummaryWriter
from torch import optim
from frodo.modules.preprocess_and_plan.hpo_modules.fedex.main_class import FedEx


def frac(p, q):

    return str(p) + '/' + str(q)


def random_search(max_resources=500, total_resources=2000):
    '''returns a random search rate and schedule for use by 'successive_elimination'
    Args:
        max_resources: most resources (steps) assigned to single arm
        total_resources: overall resource limit
    Returns:
        elimination rate as an int, elimination schedule as a list
    '''

    assert max_resources > 0, "max_resources must be positive"
    return int(total_resources / max_resources), [max_resources]


def get_schedule(
    max_resources=500,
    total_resources=2000,
    elim_rate=3,
    num_elim=0,
    num_eval=1,
):
    '''returns rate and schedule for use by 'successive_elimination'
    Args:
        max_resources: most resources (steps) assigned to single arm
        total_resources: overall resource limit
        elim_rate: multiplicative elimination rate
        num_elim: number of elimination rounds; if 0 runs random search
        num_eval: number of evaluation rounds
    Returns:
        elimination rate as an int, elimination schedule as a list, evaluation schedule as a list
    '''

    assert max_resources <= total_resources, "max_resources cannot be greater than total_resources"
    assert elim_rate > 1, "elim_rate must be greater than 1"
    assert num_eval <= total_resources, "num_eval cannot be greater than total_resources"

    if num_elim:
        diff = total_resources - max_resources
        geos = (elim_rate**(num_elim+1) - 1) / (elim_rate-1)
        u = int(diff / (geos-num_elim-1))
        resources = 0
        def v(i): return 1 + \
            ceil((diff+(num_elim-geos+elim_rate**i)*u) / (elim_rate**i-1))
        for opt in product(*(range(u, v(i)) for i in reversed(range(1, num_elim+1)))):
            used = max_resources + sum((elim_rate**i-1)*r
                                       for i, r in zip(reversed(range(1, num_elim+1)), opt))
            if resources <= used <= total_resources:
                best, resources = opt, used
        assert not 0 in best, "invalid: use more resources or fewer eliminations, or increase rate"
        elim_sched = list(np.cumsum(best)) + [max_resources]
    else:
        elim_rate, elim_sched = random_search(max_resources=max_resources,
                                              total_resources=total_resources)

    eval_sched = [int(step) for step in np.linspace(
        0, total_resources, num_eval+1)[1:]]
    return elim_rate, elim_sched, eval_sched


def successive_elimination(
    sampler,
    eval_traces,
    logdir=None,
    val_discount=0.0,
    elim_rate=3,
    elim_sched=[1],
    eval_sched=[-1],
    traces=[],
    last_round=None,
    eval_global=False,
    **kwargs,
):
    '''
    FedEx最核心算法

    runs successive elimination according to provided schedule
    Args:
        sampler: function of n returning an iterable of n objects with methods 'step' and 'trace'
        eval_traces: list of strings of traces measuring performance; element 0 used for elimination
        logdir: directory to store tensorboard logs; if None does not log anything
        val_discount: discount factor when computing score for a trace; 0.0 is most recent, 1.0 is mean
        elim_rate: multiplicative elimination rate
        elim_sched: list of steps at which to run an elimination
        eval_sched: list of steps at which to call 'test' method of the best config
        traces: list of strings of traces to collect
        last_round: str name of function that last config executes before final round
        kwargs: passed to 'test' method of each config
    Returns:
        best config;
        also dumps tensorboard logs and results.pkl to folder 'logdir', if specified
    '''
    assert len(elim_sched) > 0, "'elim_sched' must be a list of positive length"
    assert type(elim_rate) == int, "'elim_rate' must be an int"
    logger = False if logdir is None else SummaryWriter(logdir)
    traces, eval_sched = deepcopy(traces), deepcopy(eval_sched)
    for trace in eval_traces:
        if not trace in traces:
            traces.append(trace)
    # each config is (index, config_settings())
    configs = list(enumerate(sampler(elim_rate ** max(1, len(elim_sched)-1))))
    output = {index: {'settings': config.settings()}
              for index, config in configs}
    for trace in eval_traces:
        output[trace+' val'] = []
        output[trace+' test'] = []
    output['eval step'] = []
    changed = {index: True for index, _ in configs}

    # evaluate fedex object with best score across all time
    best_score = 100.0
    best_config = None
    best_config_idx = None

    start, last_start, used = 0, 0, 0
    for i, stop in enumerate(elim_sched):
        if len(configs) == 1 and not last_round is None:
            getattr(configs[0][1], last_round)()
        scores = []
        for j, (index, config) in enumerate(configs):
            scores.append(float('inf'))
            for k in range(start, stop):
                changed[index] = True
                print('\r\tRound', frac(i+1, len(elim_sched)),
                      'config', frac(j+1, len(configs)),
                      'step', frac(k+1, stop), end=4*' ')
                config.step()
                for trace in traces:
                    output[index][trace] = config.trace(trace)
                    print(
                        trace+':', round(output[index][trace][-1], 4), end=4*' ')
                    if logger:
                        logger.add_scalars(
                            trace, {str(index): output[index][trace][-1]}, k+1)
                # use refine error, if eval_global use global error
                if eval_global:
                    scores[-1] = discounted_mean(output[index]
                                                 [eval_traces[1]][start:], val_discount)
                else:
                    scores[-1] = discounted_mean(output[index]
                                                 [eval_traces[0]][start:], val_discount)
                used += 1
                current_best, score = min(enumerate(scores), key=itemgetter(1))

                # check if val score beats best score so far
                if score < best_score:
                    best_config_idx = configs[current_best][0]
                    best_config = deepcopy(configs[current_best][1])
                    best_score = score

                print('best:', round(best_score, 4), end=8*' ')

                for trace in eval_traces:
                    if len(output[best_config_idx][trace][start:]) == 0:
                        val = discounted_mean(
                            output[best_config_idx][trace][last_start:], val_discount)
                    else:
                        val = discounted_mean(
                            output[best_config_idx][trace][start:], val_discount)
                    output[trace+' val'].append(val)
                    if logger:
                        logger.add_scalar(trace+' val', val, used)
                if used in eval_sched:
                    if changed[best_config_idx]:
                        results = best_config.test(**kwargs)
                        changed[best_config_idx] = False
                    print('\r\tStep', used, 'test error', end='')
                    for trace in reversed(eval_traces):
                        output[trace+' test'].append(results[trace])
                        if logger:
                            logger.add_scalar(
                                trace+' test', results[trace], used)
                        print('    '+trace, round(results[trace], 4), end='')
                    print(64*' ')
                    output['eval step'].append(
                        eval_sched.pop(eval_sched.index(used)))
        if len(configs) == 1:
            break

        # select top n configs, index of each config is preserved
        _, configs = zip(*nsmallest(int(len(configs) / elim_rate),
                                    zip(scores, configs),
                                    key=itemgetter(0)))
        last_start = start
        start = stop

    #best, config = configs[0]
    best, config = best_config_idx, best_config
    output['best'], output[best]['settings'] = best, config.settings()
    if eval_sched:
        if changed[best]:
            results = config.test(**kwargs)
        print('\r\tStep', used, 'test error', end='')
        for trace in reversed(eval_traces):
            output[trace+' test'].append(results[trace])
            if logger:
                logger.add_scalar(trace+' test', results[trace], used)
            print('    '+trace, round(results[trace], 4), end='')
        print(64*' ')
        output['eval step'].append(used)

    if logger:
        with open(os.path.join(logdir, 'results.pkl'), 'wb') as f:
            pickle.dump(output, f)
        try:
            logger.flush()
        except AttributeError:
            pass
    return config


def wrapped_fedex(
    get_server,
    get_client,
    num_configs=1,
    prod=False,
    stepsize_init='auto',
    stepsize_sched='aggressive',
    cutoff=1E-4,
    baseline_discount=-1.0,
    diff=False,
    mle=False,
    logdir=None,
    val_discount=0.0,
    last_stop=False,
    eval_global=False,
    **kwargs,
):
    '''
    FedEx主入口阶段

    evaluates FedEx wrapped with successive elimination algorithm;
       uses FedAvg when num_configs = 1 and prod = False
    Args:
        get_server: function that takes no input and returns an object that can be passed as the 
                    first argument to FedEx.__init__, e.g. a Server object
        get_client: function that takes no input and returns a dict of local training configs, a
                    list of which is passed as the second argument to 'FedEx.__init__'; can also
                    return a dict of (string, list) pairs to be passed directly to 'FedEx.__init__'
        num_configs: determines number of configs in the list passed to 'FedEx.__init__':
                     - >0: use this value directly
                     - =0: value drawn at random from Unif[1, number of arms given by the wrapper]
                     - =-1: use the number of arms given by the wrapper
                     - else: value drawn at random from Unif{1, ..., abs(num_configs)}
        prod: run FedEx over a product set of single-parameter grids; must be 'True' in the case
                  when 'get_client' returns an object to be passed directly to 'FedEx.__init__'
        stepsize_init: passed to 'eta0' kwarg of 'FedEx.__init__'
        stepsize_sched: passed to 'sched' kwarg of 'FedEx.__init__'
        baseline_discount: determines 'baseline' kwarg of 'FedEx.__init__':
                           - >0.0: use this value directly
                           - else: value drawn at random from Unif[0.0, abs(baseline_discount)]
        diff: passed to 'diff' kwarg of 'FedEx.__init__'
        mle: passed to 'mle' kwarg of 'FedEx.test' via the kwargs of 'successive_elimination'
        logdir: passed to 'logdir' kwarg of 'successive_elimination'
        val_discount: passed to 'val_discount' kwarg of 'successive_elimination'
        last_stop: if True sets 'last_round' kwarg of 'successive_elimination' to 'stop'
        kwargs: passed to 'get_schedule'
    Returns:
        FedEx object
    '''

    elim_rate, elim_sched, eval_sched = get_schedule(**kwargs)
    print('Wrapping with', 'random search' if len(
        elim_sched) == 1 else 'successive elimination')

    if num_configs < -1:
        def samples(n): return random.randint(1, -num_configs)
    elif num_configs == -1:
        def samples(n): return n
    elif num_configs == 0:
        def samples(n): return random.randint(1, n)
    else:
        def samples(n): return num_configs

    if baseline_discount < 0.0:
        def baseline(): return random.uniform(0.0, -baseline_discount)
    else:
        def baseline(): return baseline_discount

    def sampler(n):

        for _ in range(n):
            yield FedEx(
                get_server(),
                get_client() if prod else get_client(samples(n)),
                eta0=stepsize_init,
                sched=stepsize_sched,
                cutoff=cutoff,
                baseline=baseline(),
                diff=diff,
            )

    return successive_elimination(
        sampler,
        ['refine', 'global'],
        logdir=logdir,
        val_discount=val_discount,
        elim_rate=elim_rate,
        elim_sched=elim_sched,
        eval_sched=eval_sched,
        traces=['entropy', 'mle', 'global', 'refine'],
        last_round='stop' if last_stop else None,
        mle=mle,
        eval_global=eval_global,
    )

def discounted_mean(trace, factor=1.0):

    weight = factor ** np.flip(np.arange(len(trace)), axis=0)

    return np.inner(trace, weight) / weight.sum()