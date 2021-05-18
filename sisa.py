from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, sampler
import torch.optim as optim

import torch
import torch.nn as nn
import torchvision.datasets as dset

import numpy as np
import matplotlib.pyplot as plt

import copy
import os
import random

from models import CNNMnist
from federated_avg import ClientDataset, FederatedAveraging

class Sisa():
    """"
    SISA data deletion in federated learning system.
    """

    def __init__(self, fl, train=False):
        """
        fl: federated learning object
        train: train fl if true
        """
        self.fl = fl
        self.device = self.fl.device
        self.log = {}

        if train:
            self.fl.train(ratio=0.2, epochs=1, rounds=100, opt='sgd', lr=0.05)

    def _deletion_request(self, id, idxs):
        """
        id: the client pull deletion request
        idxs: List, data idxs to be deleted
        """
        dataset = self.fl.training_data

        ori_idxs = list(self.fl.clients_dataset)  # idxs used in previous training

        idxs_train = ori_idxs[:int(0.8 * len(ori_idxs))]

        to_be_deleted = [ori_idxs[i] for i in idxs]

        idxs_train = list(set(idxs_train) - set(to_be_deleted))

        batch_size = self.fl.batch_size

        self.fl.clients[id]['train'] = DataLoader(ClientDataset(dataset, idxs_train),
                                                  batch_size=batch_size, shuffle=True)

        self.fl.clients[id]['deleted'] = DataLoader(ClientDataset(dataset, to_be_deleted),
                                                    batch_size=len(to_be_deleted), shuffle=False)

        print('Request completed! Total {:d} data deleted in training set.'.format(len(to_be_deleted)))

    def _retrain_client_model(self, id, epochs, opt, criterion, lr):
        """
        Retrain the client model
        """

        weights = {}
        train_log = []
        val_log = []
        test_acc = 0

        if criterion == 'cross_entropy':
            loss_fn = nn.CrossEntropyLoss()
        elif criterion == 'nll':
            loss_fn = nn.NLLLoss()
        else:
            exit('unsupported loss!')

        self.fl.clients[id]['model'] = CNNMnist()
        self.fl.clients[id]['model'].to(self.device)
        if opt == 'sgd':
            optimizer = optim.SGD(self.fl.clients[id]['model'].parameters(), lr=lr, momentum=0.8)
        elif opt == 'adam':
            optimizer = optim.Adam(self.fl.clients[id]['model'].parameters(), lr=lr, weight_decay=1e-5)
        else:
            exit('unsupported optimizer!')

        for e in range(epochs):
            print('Deletion epoch {:d}/{:d}.'.format(e + 1, epochs))
            train_loss = 0
            for i, (img, label) in enumerate(self.fl.clients[id]['train']):
                self.fl.clients[id]['model'].train()
                img = img.to(self.device)
                label = label.to(self.device)

                out = self.fl.clients[id]['model'](img)
                optimizer.zero_grad()
                loss = loss_fn(out, label)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

            train_loss = train_loss / len(self.fl.clients[id]['train'])
            train_log += [train_loss]
            print('train loss {:.4f}'.format(train_loss))

            # validation
            val_loss = 0
            for i, (img, label) in enumerate(self.fl.clients[id]['val']):
                self.fl.clients[id]['model'].eval()
                img = img.to(self.device)
                label = label.to(self.device)

                with torch.no_grad():
                    out = self.fl.clients[id]['model'](img)
                    loss = loss_fn(out, label)
                    val_loss += loss.item()

            val_loss = val_loss / len(self.fl.clients[id]['val'])
            val_log += [val_loss]
            print('val loss {:.4f}'.format(val_loss))

        self.log['train'] = train_log
        self.log['val'] = val_log

        weights[0] = self.fl.clients[id]['model'].state_dict()

        return weights

    def delete(self, id, idxs, ratio, epochs1, epochs2, opt='sgd', criterion='cross_entropy', lr=0.01):
        """
        Perform deletion on FL system:
            1. pull deletion request,
            2. deviate other clients from server,
            3. retrain the client that pulled request,
            4. update server model.

        epochs1: other clients update epoch
        epochs2: deletion epoch
        """
        # pull deletion request
        self._deletion_request(id, idxs)

        # deviate other clients from server
        weights = self.fl._train_clients(ratio, epochs1, opt, criterion, lr)

        # retrain the client that pulled request
        w1 = self._retrain_client_model(id, epochs2, opt, criterion, lr)

        avg_w = w1[0]
        # update server model
        for key in avg_w.keys():
            for i in range(len(weights)):
                avg_w[key] += weights[i][key]
            avg_w[key] = torch.div(avg_w[key], len(weights) + 1)
            self.fl.server_model.load_state_dict(avg_w)

        self.fl.server_model.to(self.device)
        self.fl.server_model.eval()
        test_acc = self.fl.eval(self.fl.server_model, self.fl.test_loader)
        print('global test acc {:.4f}'.format(test_acc))

        deleted_acc = self.fl.eval(self.fl.server_model, self.fl.clients[id]['deleted'])
        print('test acc on deleted data {:.4f}'.format(deleted_acc))


if __name__ == '__main__':
    mnist_non_iid = FederatedAveraging(100, 10)
    mnist_non_iid.train(ratio=0.2, epochs=1, rounds=50, opt='sgd', lr=0.05)

    fl = mnist_non_iid
    sim = Sisa(fl, train=False)
    sim.delete(0, [0], 0.2, 4, 50, lr=0.05)

