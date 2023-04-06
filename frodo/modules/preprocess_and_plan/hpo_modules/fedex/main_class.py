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
from torch import optim
from frodo.modules.preprocess_and_plan.hpo_modules.fedex.utils import discounted_mean

class FedEx:
    '''runs hyperparameter optimization given a federated learning server'''

    def entropy(self):

        entropy = 0.0
        for probs in product(*(theta[theta > 0.0] for theta in self._theta)):
            prob = np.prod(probs)
            entropy -= prob * np.log(prob)
        return entropy

    def mle(self):

        return np.prod([theta.max() for theta in self._theta])

    def __init__(
        self,
        server,
        configs,
        eta0='auto',
        sched='auto',
        cutoff=0.0,
        baseline=0.0,
        diff=False,
    ):
        '''
        Args:
            server: Object that implements two methods, 'communication_round' and 'full_evaluation'
                    taking as input a single argument, 'get_config', itself a function that takes 
                    no inputs and outputs an element of the provided list 'configs'. 
                    - 'communication_round' samples a batch of clients, assigns a config to each 
                    using 'get_config', and runs local training using that config. It then 
                    aggregates the local models to to take a training step and returns three lists 
                    or arrays: a list of each client's validation error before local training, a 
                    list of each client's validation error after local training, and a list of each 
                    client's weight (e.g. size of its validation set). 
                    - 'full_evaluation' assigns a config to each client using 'get_config' and runs
                    local training using that config. It then returns three lists or arrays: a list
                    of each client's test error before local training, a list of each client's test
                    error after local training, and a list of each client's weight (e.g. size of 
                    its test set).
            configs: list of configs used for local training and testing by 'server' 
                     OR dict of (string, list) pairs denoting a grid of configs
            eta0: base exponentiated gradient step size; if 'auto' uses sqrt(2*log(len(configs)))
            sched: learning rate schedule for exponentiated gradient:
                    - 'adaptive': uses eta0 / sqrt(sum of squared gradient l-infinity norms)
                    - 'aggressive': uses eta0 / gradient l-infinity norm
                    - 'auto': uses eta0 / sqrt(t) for t the number of rounds
                    - 'constant': uses eta0
                    - 'scale': uses sched * sqrt(2 * log(len(configs)))
            cutoff: entropy level below which to stop updating the config probability and use MLE
            baseline: discount factor when computing baseline; 0.0 is most recent, 1.0 is mean
            diff: if True uses performance difference; otherwise uses absolute performance
        '''

        self._server = server
        self._configs = configs
        self._grid = [] if type(configs) == list else sorted(configs.keys())

        sizes = [len(configs[param])
                 for param in self._grid] if self._grid else [len(configs)]
        self._eta0 = [np.sqrt(2.0 * np.log(size)) if eta0 ==
                      'auto' else eta0 for size in sizes]
        self._sched = sched
        self._cutoff = cutoff
        self._baseline = baseline
        self._diff = diff
        self._z = [np.full(size, -np.log(size)) for size in sizes]
        self._theta = [np.exp(z) for z in self._z]

        self._store = [0.0 for _ in sizes]
        self._stopped = False
        self._trace = {'global': [], 'refine': [], 'entropy': [self.entropy()], 'mle': [
            self.mle()]}

    def stop(self):

        self._stopped = True

    def sample(self, mle=False, _index=[]):
        '''samples from configs using current probability vector'''

        if mle or self._stopped:
            if self._grid:
                return {param: self._configs[param][theta.argmax()]
                        for theta, param in zip(self._theta, self._grid)}
            return self._configs[self._theta[0].argmax()]
        _index.append([np.random.choice(len(theta), p=theta)
                      for theta in self._theta])

        if self._grid:
            return {param: self._configs[param][i] for i, param in zip(_index[-1], self._grid)}
        return self._configs[_index[-1][0]]

    def settings(self):
        '''returns FedEx input settings'''

        output = {'configs': deepcopy(self._configs)}
        output['eta0'], output['sched'] = self._eta0, self._sched
        output['cutoff'], output['baseline'] = self._cutoff, self._baseline
        if self._trace['refine']:
            output['theta'] = self.theta()
        return output

    def step(self):
        '''takes exponentiated gradient step (calls 'communication_round' once)'''

        index = []
        before, after, weight = self._server.communication_round(
            lambda: self.sample(_index=index))
        before, after = np.array(before), np.array(after)
        weight = np.array(weight, dtype=np.float64) / sum(weight)

        if self._trace['refine']:
            trace = self.trace('refine')
            if self._diff:
                trace -= self.trace('global')
            baseline = discounted_mean(trace, self._baseline)
        else:
            baseline = 0.0
        self._trace['global'].append(np.inner(before, weight))
        self._trace['refine'].append(np.inner(after, weight))
        if not index:
            self._trace['entropy'].append(0.0)
            self._trace['mle'].append(1.0)
            return

        for i, (z, theta) in enumerate(zip(self._z, self._theta)):
            grad = np.zeros(len(z))
            for idx, s, w in zip(index, after-before if self._diff else after, weight):
                grad[idx[i]] += w * (s - baseline) / theta[idx[i]]
            if self._sched == 'adaptive':
                self._store[i] += norm(grad, float('inf')) ** 2
                denom = np.sqrt(self._store[i])
            elif self._sched == 'aggressive':
                denom = 1.0 if np.all(grad == 0.0) else norm(
                    grad, float('inf'))
            elif self._sched == 'auto':
                self._store[i] += 1.0
                denom = np.sqrt(self._store[i])
            elif self._sched == 'constant':
                denom = 1.0
            elif self._sched == 'scale':
                denom = 1.0 / np.sqrt(2.0 * np.log(len(grad))
                                      ) if len(grad) > 1 else float('inf')
            else:
                raise NotImplementedError
            eta = self._eta0[i] / denom
            z -= eta * grad
            z -= logsumexp(z)
            self._theta[i] = np.exp(z)

        self._trace['entropy'].append(self.entropy())
        self._trace['mle'].append(self.mle())
        if self._trace['entropy'][-1] < self._cutoff:
            self.stop()

    def test(self, mle=False):
        '''evaluates found config (calls 'full_evaluation' once)
        Args:
            mle: use MLE config instead of sampling
        Returns:
            output of 'full_evaluation'
        '''

        before, after, weight = self._server.full_evaluation(
            lambda: self.sample(mle=mle))
        return {'global': np.inner(before, weight) / weight.sum(),
                'refine': np.inner(after, weight) / weight.sum()}

    def theta(self):
        '''returns copy of config probability vector'''

        return deepcopy(self._theta)

    def trace(self, key):
        '''returns trace of one of three tracked quantities
        Args:
            key: 'entropy', 'global', or 'refine'
        Returns:
            numpy vector with length equal to number of calls to 'step'
        '''

        return np.array(self._trace[key])
