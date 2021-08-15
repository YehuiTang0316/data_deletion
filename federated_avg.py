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

from models import CNNMnist, Cifar10CnnModel, Cifar10ResNet


class ClientDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


mnist_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
    ])

cifar10_transforms = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

transform_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class FederatedAveraging():
    """
    Simulate federated averaging process with virtual clients
    """

    def __init__(self, num_clients, batch_size, dataset='mnist', root='./', download=False,
                 iid=False, use_gpu=True):
        """
        num_clients: # of clients to simulate the experiment
        batch_size: batch size for clients
        dataset: choose from ('mnist', 'cifar10')
        data_dir: root path to store data
        download: download dataset if True
        iid: create iid clients datasets if True
        """
        self.num_clients = num_clients
        self.iid = iid
        self.dataset = dataset
        self.batch_size = batch_size
        self.log = {'train': [], 'val': [], 'test_local': [], 'test_global': []}

        data_dir = os.path.join(root, 'dataset')

        path = dataset + 'iid' + '.npy' if iid else dataset + '.npy'
        self.log_dir = os.path.join(root, 'log/')

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        self.log_path = os.path.join(self.log_dir, path)

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
            download = True

        if dataset == 'mnist':
            self.training_data = dset.MNIST(data_dir, train=True, transform=mnist_transforms, download=download)
            self.test_data = dset.MNIST(data_dir, train=False, transform=mnist_transforms, download=download)
            self.test_loader = DataLoader(self.test_data, batch_size=int(len(self.test_data) / 10), shuffle=False)
            self.clients_dataset = self._distribute_mnist_dataset(self.training_data, num_clients)
            self.server_model = CNNMnist()
            self.architecture = CNNMnist()

        elif dataset == 'cifar10':
            self.training_data = dset.CIFAR10(data_dir, train=True, transform=transform_train, download=download)
            self.test_data = dset.CIFAR10(data_dir, train=False, transform=transform_test, download=download)
            self.test_loader = DataLoader(self.test_data, batch_size=int(len(self.test_data) / 10), shuffle=False)
            self.clients_dataset = self._distribute_mnist_dataset(self.training_data, num_clients)
            self.server_model = Cifar10ResNet()
            self.architecture = Cifar10ResNet()

        ## more experiments expected
        else:
            exit('unsupported dataset!')

        self.clients = self._create_clients(num_clients, self.architecture)

        # activate cuda
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(self.device)

        self.server_model.to(self.device)

    def _create_clients(self, num_clients, architecture=CNNMnist()):
        """
        Return dictionary dict_users,
            keys: ('model', 'train', 'val', 'test')
        """
        dict_users = {}
        dataset = self.training_data

        for i in range(num_clients):
            idxs = list(self.clients_dataset[i])
            # print(idxs)

            dict_users[i] = {}
            dict_users[i]['model'] = architecture

            idxs_train = idxs[:int(0.8 * len(idxs))]
            idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
            idxs_test = idxs[int(0.9 * len(idxs)):]

            if self.batch_size == 'inf':
                # federated sgd, use entire local dataset to train agent
                batch_size = len(idxs_train)
            else:
                batch_size = self.batch_size

            dict_users[i]['train'] = DataLoader(ClientDataset(dataset, idxs_train),
                                                batch_size=batch_size, shuffle=True)
            dict_users[i]['val'] = DataLoader(ClientDataset(dataset, idxs_val),
                                                batch_size=int(len(idxs_val) / 10), shuffle=False)
            dict_users[i]['test'] = DataLoader(ClientDataset(dataset, idxs_test),
                                                batch_size=int(len(idxs_test) / 10), shuffle=False)

        return dict_users

    def _distribute_mnist_dataset(self, dataset, num_clients):
        """
        return indexes in dataset for each clients
        """
        # iid data
        if self.iid:
            # num_items = int(len(dataset)/num_clients)
            # dict_users, all_idxs = {}, [i for i in range(len(dataset))]
            # for i in range(num_clients):
            # dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            # all_idxs = list(set(all_idxs) - dict_users[i])

            dict_users = {i: np.array([]) for i in range(num_clients)}
            idxs = np.arange(len(dataset))
            all_idxs = {}
            labels = torch.tensor(dataset.targets).numpy()

            # sort labels
            idxs_labels = np.vstack((idxs, labels))
            idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

            start = 0
            for i in range(10):
                for j in range(start, len(idxs_labels[0])):
                    if idxs_labels[1][j] != i:
                        all_idxs[i] = idxs_labels[0, start:j]
                        start = j
                        break

            all_idxs[9] = idxs_labels[0, start:]
            num_items = {}
            for i in range(10):
                num_items[i] = int(len(all_idxs[i]) / num_clients)

            for i in range(num_clients):
                dict_users[i] = []
                for j in range(10):
                    add_in = set(np.random.choice(all_idxs[j], num_items[j], replace=False))
                    all_idxs[j] = list(set(all_idxs[j]) - add_in)
                    dict_users[i].extend(add_in)
                random.shuffle(dict_users[i])

        # non-iid data
        else:
            if self.dataset == 'mnist':
                num_shards, num_imgs = 200, 300
            else:
                num_shards, num_imgs = 200, 250

            idx_shard = [i for i in range(num_shards)]
            dict_users = {i: np.array([]) for i in range(num_clients)}
            idxs = np.arange(num_shards * num_imgs)
            labels = torch.tensor(dataset.targets).numpy()

            # sort labels
            idxs_labels = np.vstack((idxs, labels))
            idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
            idxs = idxs_labels[0, :]

            # divide and assign 2 shards/client
            for i in range(num_clients):
                rand_set = set(np.random.choice(idx_shard, 2, replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        return dict_users

    def _train_client(self, client_id, epochs, opt, criterion, lr, retrain=False):
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
        return train_loss, val_loss, test_acc

    def _train_clients(self, ratio, epochs, opt, criterion, lr):
        """
        Federated Averaging,
        one step local update: randomly select clients, retrive global model,
             then train e epochs locally

        ratio: ratio of selected clients to do e epochs local update
        """
        # random select clients
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
            test_acc += acc
            weights[t] = self.clients[c]['model'].state_dict()

        test_acc /= num
        print('average local train loss {:.4f}'.format(sum(train_log) / num))
        print('average local val loss {:.4f}'.format(sum(val_log) / num))
        print('average local test acc {:.4f}'.format(test_acc))
        self.log['test_local'] += [test_acc]
        self.log['train'] += [sum(train_log) / num]
        self.log['val'] += [sum(val_log) / num]

        return weights

    def _update_server(self, weights):
        """
        one step global update: aggregate local models
        """
        avg_w = weights[0]
        if len(weights) == 1:
            self.server_model.load_state_dict(avg_w)
        else:
            for key in avg_w.keys():
                for i in range(1, len(weights)):
                    avg_w[key] += weights[i][key]
                avg_w[key] = torch.div(avg_w[key], len(weights))
            self.server_model.load_state_dict(avg_w)

        self.server_model.to(self.device)
        self.server_model.eval()
        test_acc = evaluate(self.server_model, self.test_loader, self.device)
        print('global test acc {:.4f}'.format(test_acc))
        self.log['test_global'] += [test_acc]

        return test_acc

    def train(self, ratio, epochs, rounds, opt='adam', criterion='cross_entropy', lr=1e-4):
        """
        Train FedAveraging model.

        ratio: precentage of clients to update per local epoch
        epochs: local epochs
        rounds: global epochs
        """
        for r in range(rounds):
            print('round {:d}'.format(r))
            weights = self._train_clients(ratio, epochs, opt, criterion, lr)
            test_acc = self._update_server(weights)
        np.save(self.log_path, self.log)


def evaluate(model, dataloader, device):
    """
    Return prediction accuracy.
    """
    total, correct = 0, 0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        _, output = model(inputs)
        max_pred, pred = torch.max(output.data, dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    if total == 0:
        total += 1e-5
    return 100 * correct / total


if __name__ == '__main__':
    torch.manual_seed(42)

    # iid split
    # mnist_iid = FederatedAveraging(100, 10, iid=True, download=True)
    # mnist_iid.train(ratio=0.2, epochs=1, rounds=20, lr=0.01, opt='sgd')

    # non-iid split
    # mnist_non_iid = FederatedAveraging(100, 10)
    # mnist_non_iid.train(ratio=0.2, epochs=1, rounds=100, opt='sgd', lr=0.05)

    cifar10_iid = FederatedAveraging(100, 10, dataset='cifar10', iid=True, download=False)
    cifar10_iid.train(ratio=0.2, epochs=1, rounds=20, lr=0.01, opt='sgd')
