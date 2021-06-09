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
from federated_avg import ClientDataset, FederatedAveraging, evaluate


class Sisa(FederatedAveraging):
    """"
    SISA data deletion in federated learning system.
    """

    def __init__(self, num_clients, batch_size, dataset='mnist', root='./', download=False,
                 iid=False, use_gpu=True, train=False):
        """
        fl: federated learning object
        train: train fl if true
        """
        super(Sisa, self).__init__(num_clients, batch_size, dataset, root, download,
                 iid, use_gpu)

        if train:
            self.train(ratio=0.2, epochs=1, rounds=100, opt='sgd', lr=0.05)

    def _deletion_request(self, id, idxs):
        """
        id: the client pull deletion request
        idxs: List, data idxs to be deleted
        """
        dataset = self.training_data

        ori_idxs = list(self.clients_dataset)  # idxs used in previous training

        idxs_train = ori_idxs[:int(0.8 * len(ori_idxs))]

        to_be_deleted = [ori_idxs[i] for i in idxs]

        idxs_train = list(set(idxs_train) - set(to_be_deleted))

        batch_size = self.batch_size

        self.clients[id]['train'] = DataLoader(ClientDataset(dataset, idxs_train),
                                               batch_size=batch_size, shuffle=True)

        self.clients[id]['deleted'] = DataLoader(ClientDataset(dataset, to_be_deleted),
                                                 batch_size=len(to_be_deleted), shuffle=False)

        print('Request completed! Total {:d} data deleted in training set.'.format(len(to_be_deleted)))

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
        weights = self._train_clients(ratio, epochs1, opt, criterion, lr)

        # retrain the client that pulled request
        train_loss, val_loss, test_acc = self._train_client(id, epochs2, opt, criterion, lr, retrain=True)
        w1 = self.clients[id]['model'].state_dict()

        avg_w = w1
        # update server model
        for key in avg_w.keys():
            for i in range(len(weights)):
                avg_w[key] += weights[i][key]
            avg_w[key] = torch.div(avg_w[key], len(weights) + 1)
            self.server_model.load_state_dict(avg_w)

        self.server_model.to(self.device)
        self.server_model.eval()
        test_acc = evaluate(self.server_model, self.test_loader, self.device)
        print('global test acc {:.4f}'.format(test_acc))

        deleted_acc = evaluate(self.server_model, self.clients[id]['deleted'], self.device)
        print('test acc on deleted data {:.4f}'.format(deleted_acc))


if __name__ == '__main__':
    # mnist_non_iid = FederatedAveraging(100, 10)
    # mnist_non_iid.train(ratio=0.2, epochs=1, rounds=50, opt='sgd', lr=0.05)

    # fl = mnist_non_iid
    # sim = Sisa(fl, train=False)
    # sim.delete(0, [0], 0.2, 4, 50, lr=0.05)

    sisa = Sisa(100, 10)
    sisa.delete(0, [0], 0.2, 4, 50, lr=0.05)
