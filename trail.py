from federated_avg import FederatedAveraging, evaluate, ClientDataset
from sisa import Sisa
from models import CNNMnist
from dp import DPFL3
from backdoor import BackdoorAttack
from dp_tools import analysis

from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, sampler
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR

import torch
import torch.nn as nn
import torchvision.datasets as dset

import numpy as np
import matplotlib.pyplot as plt

import copy
import os
import random
import pickle


class Trail(BackdoorAttack):
    def __init__(self, num_clients, batch_size, sigma, dataset='mnist', root='./', download=False,
                 iid=False, use_gpu=True, pretrain=True, use_dp=True):
        super(BackdoorAttack, self).__init__(num_clients, batch_size, sigma, dataset, root, download,
                                             iid, use_gpu)

        self.log_dir = './log/trail'
        self.log_path = os.path.join(self.log_dir, '1')

        self.restore_model = None
        self.communication_rounds = 0
        self.restore_epsilon = 5
        self.use_dp = use_dp

        self.pretrain = pretrain

        if pretrain:
            self._create_server_training()
            self._train_server()

    def _create_server_training(self):
        # total = len(self.test_data)
        # idxs = np.random.choice(total, total//2)
        #
        # idxs_test = list(set(list(range(total))) - set(idxs))
        #
        # self.test_loader = DataLoader(ClientDataset(self.test_data, idxs_test),
        #                               batch_size=int(len(self.test_data) / 10), shuffle=False)
        #
        # self.server_training = DataLoader(ClientDataset(self.test_data, idxs),
        #                                   batch_size=int(len(self.test_data) / 10), shuffle=False)

        total = len(self.training_data)
        idxs = np.random.choice(total, 10000)

        self.server_training = DataLoader(ClientDataset(self.training_data, idxs), batch_size=128, shuffle=True)

        print("Distributed {:d} non-sensitive data to training on server.".format(len(idxs)))

    def _train_server(self):
        optimizer = optim.SGD(self.server_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1, step_size_up=10000, step_size_down=10000,
                             mode='triangular')
        #
        # scheduler = CosineAnnealingLR(optimizer, T_max=200)
        loss_fn = nn.CrossEntropyLoss()

        train_log, val_log = [], []
        epochs = 1

        print("training server model.")
        for epoch in range(epochs):
            # train
            train_loss = 0
            for i, (img, label) in enumerate(self.server_training):
                self.server_model.train()
                img = img.to(self.device)
                label = label.to(self.device)

                _, out = self.server_model(img)
                optimizer.zero_grad()
                loss = loss_fn(out, label)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            scheduler.step()

            train_loss /= len(self.server_training)
            print('server epoch {:d}, train loss {:.4f}'.format(epoch, train_loss))
            train_log += [train_loss]

            # validation
            val_loss = 0
            for i, (img, label) in enumerate(self.test_loader):
                self.server_model.eval()
                img = img.to(self.device)
                label = label.to(self.device)

                with torch.no_grad():
                    _, out = self.server_model(img)
                    loss = loss_fn(out, label)
                    val_loss += loss.item()

            val_loss /= len(self.test_loader)
            val_log += [val_loss]

            if evaluate(self.server_model, self.test_loader, self.device) > 98:
                break

        self.log['server'] = {'train':train_log, 'val':val_log}
        self.log['achieve'] = (float('inf'), None, None)

        self.restore_model = copy.deepcopy(self.server_model)

    def _train_client(self, client_id, epochs, opt, criterion, lr, retrain=False, gamma=2):
        """
        :param client_id:
        :param ratio:
        :param epochs:
        :param opt:
        :param criterion:
        :param lr:
        :param retrain: retrain this client model from scratch, if true
        :return:
        """
        if retrain is True:
            self.clients[client_id]['model'] = self.restore_model if self.pretrain else self.architecture
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

                _, out = self.clients[client_id]['model'](img)
                optimizer.zero_grad()
                loss = loss_fn(out, label)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

            # print('local epoch {:d}, train loss {:.4f}'.format(epoch, train_loss/len(self.clients[c]['train'])))

            # validation
            val_loss = 0
            for i, (img, label) in enumerate(self.clients[client_id]['val']):
                self.clients[client_id]['model'].eval()
                img = img.to(self.device)
                label = label.to(self.device)

                with torch.no_grad():
                    _, out = self.clients[client_id]['model'](img)
                    loss = loss_fn(out, label)
                    val_loss += loss.item()

            # print('local epoch {:d}, val loss {:.4f}'.format(epoch, val_loss/len(self.clients[c]['val'])))

        train_loss = train_loss / len(self.clients[client_id]['train'])
        val_loss = val_loss / len(self.clients[client_id]['val'])
        test_acc = evaluate(self.clients[client_id]['model'], self.clients[client_id]['test'], self.device)

        # scale up update
        if retrain is True:
            params1 = self.clients[client_id]['model'].state_dict()
            params2 = self.server_model.state_dict()

            for key in params1.keys():
                params1[key] = gamma * (params1[key] - params2[key]) + params2[key]

            self.clients[client_id]['model'].load_state_dict(params1)

        return train_loss, val_loss, test_acc

    def _update_server(self, weights):
        """
        one step global update: aggregate local models
        """
        self.communication_rounds += 1

        max_norms = {}
        updates = {}
        w = {}
        gradients = {}
        l2_norms = {}

        if self.use_dp:
            # init, if use dp, clip norm and add noise
            for i in weights.keys():
                gradients[i] = {}

            for key in weights[0].keys():
                l2_norms[key] = []
                for i in weights.keys():
                    gradients[i][key] = weights[i][key] - self.server_model.state_dict()[key]
                    gradients[i][key] = gradients[i][key].type(torch.FloatTensor)
                    l2_norms[key].append(torch.norm(gradients[i][key], 2).cpu().numpy())

            for key in gradients[0].keys():
                max_norms[key] = np.median(l2_norms[key])
                coeff = l2_norms[key] / max_norms[key] if max_norms[key] != 0 else l2_norms[key] / (max_norms[key]+1e-6)

                for i in range(len(gradients)):
                    if i == 0:
                        updates[key] = torch.div(gradients[i][key], max(1, coeff[i]))
                    else:
                        updates[key] += torch.div(gradients[i][key], max(1, coeff[i]))

            # add noise
                updates[key] += torch.randn_like(updates[key]) * self.sigma * max_norms[key]
                updates[key] = torch.div(updates[key], len(gradients))

                w[key] = self.server_model.state_dict()[key] + updates[key]

        else:
            w = weights[0]
            if len(weights) == 1:
                self.server_model.load_state_dict(w)
            else:
                for key in w.keys():
                    for i in range(1, len(weights)):
                        w[key] += weights[i][key]
                    w[key] = torch.div(w[key], len(weights))

        self.server_model.load_state_dict(w)

        self.server_model.to(self.device)
        self.server_model.eval()
        test_acc = evaluate(self.server_model, self.test_loader, self.device)
        print('global test acc {:.4f}'.format(test_acc))

        if self.use_dp:
            delta = 1e-3
            epsilon = analysis.epsilon(self.num_clients, 0.2*self.num_clients, self.sigma, self.communication_rounds, delta)
            print('Achieves ({}, {})-DP'.format(epsilon, delta))

        if test_acc >= 95:
            if self.use_dp:
                if epsilon < self.restore_epsilon:
                    self.restore_model = copy.deepcopy(self.server_model)
            else:
                self.restore_model = copy.deepcopy(self.server_model)

        self.log['test_global'] += [test_acc]

        if self.use_dp:
            if test_acc >= 93 and self.log['achieve'][0] >= self.communication_rounds:
                self.log['achieve'] = (self.communication_rounds, epsilon, delta)

        return test_acc

