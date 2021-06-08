from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, sampler
import torch.optim as optim

import torch
import torch.nn as nn
import torchvision.datasets as dset
from torch.nn.utils import clip_grad_norm_

import numpy as np
import matplotlib.pyplot as plt

import copy
import os
import random

from models import CNNMnist
from federated_avg import FederatedAveraging, evaluate


class DPFL1(FederatedAveraging):
    """
    Add noise to client models' param
    """
    def __init__(self, num_clients, batch_size, sigma=1, dataset='mnist', root='./', download=False,
                 iid=False, use_gpu=True, add_to_client=True):
        super(DPFL1, self).__init__(num_clients, batch_size)

        self.sigma = sigma

        path = 'dp1' + + str(sigma) + dataset + 'iid' + '.npy' if iid else 'dp1' + str(sigma) + dataset + '.npy'
        self.log_dir = os.path.join(root, 'log/')
        self.log_path = os.path.join(self.log_dir, path)

    def _train_clients(self, ratio, epochs, opt, criterion, lr):
        num = max(1, int(ratio * self.num_clients))
        selected = np.random.choice(self.num_clients, num, replace=False)

        weights = {}
        train_log = []
        val_log = []
        test_acc = 0

        for t, c in enumerate(selected):
            train_loss, val_loss, acc = self._train_client(c, epochs, opt, criterion, lr)
            train_log += [train_loss]
            val_log += [val_loss]

            with torch.no_grad():
                for param in self.clients[c]['model'].parameters():
                    param += torch.randn(param.shape) * self.sigma

            test_acc += evaluate(self.clients[c]['model'], self.clients[c]['test'], self.device)

            weights[t] = self.clients[c]['model'].state_dict()

        test_acc /= num
        print('average local train loss {:.4f}'.format(sum(train_log) / num))
        print('average local val loss {:.4f}'.format(sum(val_log) / num))
        print('average local test acc {:.4f}'.format(test_acc))
        self.log['test_local'] += [test_acc]
        self.log['train'] += [sum(train_log) / num]
        self.log['val'] += [sum(val_log) / num]

        return weights

    def train(self, ratio, epochs, rounds, opt='adam', criterion='cross_entropy', lr=1e-4):
        for r in range(rounds):
            print('round {:d}'.format(r))
            weights = self._train_clients(ratio, epochs, opt, criterion, lr)
            test_acc = self._update_server(weights)
        np.save(self.log_path, self.log)


class DPFL2(FederatedAveraging):
    """
    dp-sgd
    """
    def __init__(self, num_clients, batch_size, sigma=1, max_norm=1, dataset='mnist', root='./', download=False,
                 iid=False, use_gpu=True):
        super(DPFL2, self).__init__(num_clients, batch_size)

        self.sigma = sigma
        self.max_norm = max_norm

        path = 'dp-sgd' + str(sigma) + dataset + 'iid' + '.npy' if iid else 'dp-sgd' + str(sigma) + dataset + '.npy'
        self.log_dir = os.path.join(root, 'log/')
        self.log_path = os.path.join(self.log_dir, path)

    def _train_client(self, client_id, epochs, opt, criterion, lr, retrain=False):
        if retrain is True:
            self.clients[client_id]['model'] = self.architecture
        else:
            self.clients[client_id]['model'] = copy.deepcopy(self.server_model)

        self.clients[client_id]['model'].to(self.device)

        if opt == 'sgd':
            optimizer = optim.SGD(self.clients[client_id]['model'].parameters(), lr=lr, momentum=0.8)
        elif opt == 'adam':
            optimizer = optim.Adam(self.clients[client_id]['model'].parameters(), lr=lr, weight_decay=1e-5)
        else:
            exit('unsupported optimizer!')

        if criterion == 'cross_entropy':
            loss_fn = nn.CrossEntropyLoss()
        elif criterion == 'nll':
            loss_fn = nn.NLLLoss()
        else:
            exit('unsupported loss!')

        train_loss, val_loss = 0, 0

        for epoch in range(epochs):
            # train
            train_loss = 0
            for i, (img, label) in enumerate(self.clients[client_id]['train']):
                self.clients[client_id]['model'].train()
                img = img.to(self.device)
                label = label.to(self.device)

                out = self.clients[client_id]['model'](img)
                optimizer.zero_grad()
                loss = loss_fn(out, label)
                train_loss += loss.item()
                loss.backward()

                clip_grad_norm_(self.clients[client_id]['model'].parameters(), self.max_norm)
                for param in self.clients[client_id]['model'].parameters():
                    param.grad += torch.randn(param.grad.shape) * self.sigma * self.max_norm

                optimizer.step()

            # print('local epoch {:d}, train loss {:.4f}'.format(epoch, train_loss/len(self.clients[c]['train'])))

            # validation
            val_loss = 0
            for i, (img, label) in enumerate(self.clients[client_id]['val']):
                self.clients[client_id]['model'].eval()
                img = img.to(self.device)
                label = label.to(self.device)

                with torch.no_grad():
                    out = self.clients[client_id]['model'](img)
                    loss = loss_fn(out, label)
                    val_loss += loss.item()

            # print('local epoch {:d}, val loss {:.4f}'.format(epoch, val_loss/len(self.clients[c]['val'])))

        train_loss = train_loss / len(self.clients[client_id]['train'])
        val_loss = val_loss / len(self.clients[client_id]['val'])
        test_acc = evaluate(self.clients[client_id]['model'], self.clients[client_id]['test'], self.device)
        return train_loss, val_loss, test_acc


if __name__ == '__main__':
    sim = DPFL2(100, 10, sigma=1, max_norm=1)
    sim.train(0.1, 1, 10, lr=0.05, opt='sgd')


