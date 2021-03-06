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
                 iid=False, use_gpu=True):
        """
        fl: federated learning object
        """
        super(Sisa, self).__init__(num_clients, batch_size, dataset, root, download,
                 iid, use_gpu)

        self.poison_idxs = {}

        for i in range(num_clients):
            self.poison_idxs[i] = []

    def _deletion_request(self, id, idxs):
        """
        id: the client pull deletion request
        idxs: List, data idxs to be deleted; or 'poison' to delete poison data
        """
        if idxs == 'poison':
            if 'poison' not in self.clients[id]:
                exit('This client has not been poisoned.')
            else:
                # in deletion dataset, use the true label for poison data
                if self.poison_idxs[id]:
                    self.clients[id]['deleted'] = DataLoader(ClientDataset(self.training_data, self.poison_idxs[id]),
                                                             batch_size=self.batch_size, shuffle=False)
                else:
                    self.clients[id]['deleted'] = self.clients[id]['poison']

                print('Poison data has been deleted.')
        else:
            dataset = self.training_data

            ori_idxs = list(self.clients_dataset[id])  # idxs used in previous training

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

        tmp = len(weights)
        weights[tmp] = self.clients[id]['model'].state_dict()

        self._update_server(weights)

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
