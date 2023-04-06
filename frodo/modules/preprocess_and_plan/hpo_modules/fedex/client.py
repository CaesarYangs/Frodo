import random
from copy import deepcopy
import numpy as np
import torch
from torch import optim
from frodo.modules.preprocess_and_plan.hpo_modules.fedex.utils import frac

class Server:
    '''object for federated training implementing methods required by FedEx'''

    def _set_test_state(self):

        state = (np.random.get_state(), torch.get_rng_state(),
                 torch.cuda.get_rng_state())
        if self._state is None:
            self._state = state
        else:
            np.random.set_state(self._state[0])
            torch.set_rng_state(self._state[1])
            torch.cuda.set_rng_state(self._state[2])
        return state

    def _reset_state(self, state):

        np.random.set_state(state[0])
        torch.set_rng_state(state[1])
        torch.cuda.set_rng_state(state[2])

    def __init__(
        self,
        model,
        clients,
        train,
        test,
        lr=1.0,
        momentum=0.0,
        step=1,
        gamma=1.0,
        batch=10,
        state=None,
    ):
        '''
        Args:
            model: PyTorch model
            clients: list of clients, each a function that takes one or more strings 'train',
                     'val', 'test' and returns, as one tuple, input and output tensors for each
            train: method that takes as argument a PyTorch model, an input tensor, an output
                   tensor, and optional kwargs and returns the same PyTorch model
            test: method that takes as argument a PyTorch model, an input tensor, and an output
                  tensor and returns the model's error
            lr: server learning rate
            momentum: server momentum
            step: server learning rate decay interval
            gamma: server learning rate decay factor
            batch: number of clients to sample per communication round
            state: np.random, torch, torch.cuda random state tuple; if None uses current states
        '''

        self._model = model
        self._clients = clients
        self._train = train
        self._test = test
        self._opt = optim.SGD(self._model.parameters(),
                              lr=lr, momentum=momentum)
        self._sched = optim.lr_scheduler.StepLR(self._opt, step, gamma=gamma)
        self._batch = batch
        self._state = state
        self._reset_state(self._set_test_state())

    def communication_round(self, get_config):
        '''runs one step of local training and model aggregation
        Args:
            get_config: returns kwargs for 'train' as a dict
        Returns:
            np.array objects for global val error, local val error, and val size of each client
        '''

        self._model.cuda()
        before, after, weight = [np.zeros(self._batch) for _ in range(3)]
        total = 0.0

        for i in range(self._batch):
            Xtrain, Ytrain, Xval, Yval = random.choice(
                self._clients)('train', 'val')
            before[i] = self._test(self._model, Xval, Yval)
            model = self._train(deepcopy(self._model),
                                Xtrain, Ytrain, **get_config())
            after[i] = self._test(model, Xval, Yval)
            weight[i] = len(Yval)
            total += len(Ytrain)
            if i:
                for agg, param in zip(aggregate.parameters(), model.parameters()):
                    agg.data += len(Ytrain) * param.data
            else:
                for param in model.parameters():
                    param.data *= len(Ytrain)
                aggregate = model

        self._opt.zero_grad()
        for agg, param in zip(aggregate.parameters(), self._model.parameters()):
            param.grad = param.data - agg / total
        self._opt.step()
        self._opt.zero_grad()
        self._sched.step()
        self._model.cpu()
        return before, after, weight

    def full_evaluation(self, get_config):
        '''evaluates personalization on each client
        Args:
            get_config: returns kwargs for 'train' as a dict
        Returns:
            np.array objects for global test error, local test error, and test size of each client
        '''

        state = self._set_test_state()
        self._model.cuda()
        before, after, weight = [
            np.zeros(len(self._clients)) for _ in range(3)]
        for i, client in enumerate(self._clients):
            Xtrain, Ytrain, Xtest, Ytest = client('train', 'test')
            before[i] = self._test(self._model, Xtest, Ytest)
            after[i] = self._test(self._train(deepcopy(self._model),
                                              Xtrain, Ytrain, **get_config()),
                                  Xtest, Ytest)
            weight[i] = len(Ytest)
            print('\r\tEvaluated client', frac(i+1, len(self._clients)),
                  '    global error:', round(
                      np.inner(before, weight) / weight.sum(), 4),
                  '    refine error:', round(
                      np.inner(after, weight) / weight.sum(), 4),
                  end=32*' ')
        self._model.cpu()
        self._reset_state(state)
        return before, after, weight
