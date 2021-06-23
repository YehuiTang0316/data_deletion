from federated_avg import FederatedAveraging, evaluate, ClientDataset
from sisa import Sisa
from models import CNNMnist

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
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


class PoisonDataset(Dataset):
    def __init__(self, dataset, idxs, shuffle):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

        self.shuffle = shuffle

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        label = self.shuffle[label]
        return torch.tensor(image), torch.tensor(label)


class ConcatDataset(Dataset):
    def __init__(self, datasets1, datasets2):
        def return_all_items(dataset):
            all_items = []
            for i in range(len(dataset)):
                all_items.append(dataset[i])
            return all_items

        list1 = return_all_items(datasets1)
        list2 = return_all_items(datasets2)
        list1.extend(list2)

        self.dataset = list1

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


class BackdoorAttack(Sisa):
    def __init__(self, num_clients, batch_size, dataset='mnist', root='./', download=False,
                 iid=False, use_gpu=True):
        super(BackdoorAttack, self).__init__(num_clients, batch_size, dataset, root, download,
                 iid, use_gpu)

        self.log_dir = './log/attack'
        self.log_path = os.path.join(self.log_dir, 'backdoor')
        self.poison_idxs = {}

        for i in range(num_clients):
            self.poison_idxs[i] = []

    def _train_attacker(self, client_id, epochs, opt, criterion, lr, alpha, epsilon, gamma, retrain=False):
        # train attack model, difference loss function, scale up before submit
        if retrain is True:
            self.clients[client_id]['model'] = self.architecture
        else:
            self.clients[client_id]['model'] = copy.deepcopy(self.server_model)

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

        ano_loss = nn.CosineSimilarity(dim=0)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        train_loss, val_loss = 0, 0

        for epoch in range(epochs):

            # train
            train_loss = 0
            self.clients[client_id]['model'].train()
            for i, (img, label) in enumerate(self.clients[client_id]['poison train']):
                img = img.to(self.device)
                label = label.to(self.device)

                out = self.clients[client_id]['model'](img)
                optimizer.zero_grad()
                loss = alpha * loss_fn(out, label)

                loss += (1-alpha) *(1. - ano_loss(torch.cat([param.view(-1) for param in self.clients[client_id]['model'].parameters()]),
                                torch.cat([param.view(-1) for param in self.server_model.parameters()])))
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            scheduler.step()

            # print('local epoch {:d}, train loss {:.4f}'.format(epoch, train_loss/len(self.clients[c]['train'])))

            # validation
            val_loss = 0
            self.clients[client_id]['model'].eval()
            for i, (img, label) in enumerate(self.clients[client_id]['val']):
                img = img.to(self.device)
                label = label.to(self.device)

                with torch.no_grad():
                    out = self.clients[client_id]['model'](img)
                    loss = alpha * loss_fn(out, label) + \
                           (1 - alpha) * \
                           ano_loss(
                               torch.cat([param.view(-1) for param in self.clients[client_id]['model'].parameters()]),
                               torch.cat([param.view(-1) for param in self.server_model.parameters()]))
                    val_loss += loss.item()

            # print('local epoch {:d}, val loss {:.4f}'.format(epoch, val_loss/len(self.clients[c]['val'])))

            # check if loss on poison data satisfy stop condition
            poison_loss = 0
            for i, (img, label) in enumerate(self.clients[client_id]['poison']):
                img = img.to(self.device)
                label = label.to(self.device)

                with torch.no_grad():
                    out = self.clients[client_id]['model'](img)
                    loss = loss_fn(out, label)
                    poison_loss += loss.item()

            poison_loss /= len(self.clients[client_id]['poison'])
            if poison_loss < epsilon:
                break

            print('loss on poison data {:.3f}'.format(poison_loss))


        # scale up model
        params1 = self.clients[client_id]['model'].state_dict()
        params2 = self.server_model.state_dict()

        for key in params1.keys():
            params1[key] = gamma * (params1[key]-params2[key]) + params2[key]

        self.clients[client_id]['model'].load_state_dict(params1)

        train_loss = train_loss / len(self.clients[client_id]['train'])
        val_loss = val_loss / len(self.clients[client_id]['val'])
        test_acc = evaluate(self.clients[client_id]['model'], self.clients[client_id]['test'], self.device)
        return train_loss, val_loss, test_acc

    def _create_poison_data(self, client_id, size, shuffle):
        """

        :param client_id:
        :param size: portion of poison data
        :param shuffle: shuffle list for label, e.g. list [0,7,2,3,4,5,6,7,8,9] changes label 1 to 7
        :return:
        """
        # idxs_poison = np.random.choice(len(dataset), size, replace=False)

        idxs = self.clients_dataset[client_id]
        idxs_train = idxs[:int(0.8 * len(idxs))]

        size = min(len(idxs_train), max(int(size*len(idxs_train)), 1))

        idxs_poison = np.random.choice(idxs_train, size, replace=False)
        self.poison_idxs[client_id] = idxs_poison

        idxs_train = list(set(idxs_train) - set(idxs_poison))

        self.clients[client_id]['poison'] = DataLoader(PoisonDataset(self.training_data, idxs_poison, shuffle),
                                                       batch_size=self.batch_size, shuffle=True)

        self.clients[client_id]['poison train'] = DataLoader(ConcatDataset(ClientDataset(
            self.training_data, idxs_train), PoisonDataset(self.training_data, idxs_poison, shuffle)),
                                                             batch_size=self.batch_size, shuffle=True)

        self.clients[client_id]['train'] = DataLoader(ClientDataset(self.training_data, idxs_train),
                                                             batch_size=self.batch_size, shuffle=True)

        print('Poison dataset of size {:d} has been created.'.format(size))

    def attack(self, ratio, client_ids, size, shuffle, epochs1, epochs2, opt, criterion, lr1, lr2, alpha, epsilon, gamma):
        """
        Poison attack on federated learning system.
        :param ratio: ratio of clients in communication
        :param client_ids: client that submits poison attack
        :param size: size of poison dataset
        :param shuffle: shuffle list for label, e.g. list [0,7,2,3,4,5,6,7,8,9] changes label 1 to 7
        :param epochs1: epochs for normal clients
        :param epochs2: epochs for attacker
        :param opt: optimizer, ('sgd', 'adam')
        :param criterion: loss function ('cross entropy', 'nll')
        :param lr1: learning rate for normal clients
        :param lr2: learning rate for attacker
        :param alpha: alpha * cls_loss + (1-alpha) * ano_loss
        :param epsilon: target loss to stop training attacker
        :param gamma: scale up factor to scale up attacker update
        :return:
        """
        # deviate other clients from server
        weights = self._train_clients(ratio, epochs1, opt, criterion, lr1)

        for client_id in client_ids:
            # poison
            if 'poison' not in self.clients[client_id]:
                self._create_poison_data(client_id, size, shuffle)

            # train
            self._train_attacker(client_id, epochs2, opt, criterion, lr2, alpha, epsilon, gamma)

        for client_id in client_ids:
            tmp = len(weights)
            weights[tmp] = self.clients[client_id]['model'].state_dict()

        self._update_server(weights)

        # evaluate on poison data
        poison_acc = self.poison_accuracy(client_ids)
        print('test acc on poison data {:.4f}'.format(poison_acc))

    def poison_accuracy(self, client_ids):
        poison_acc = 0
        for client_id in client_ids:
            poison_acc += evaluate(self.server_model, self.clients[client_id]['poison'], self.device)

        poison_acc /= len(client_ids)
        return poison_acc


class MIA(Sisa):
    def __init__(self, num_clients, batch_size, dataset='mnist', root='./', download=False,
                 iid=False, use_gpu=True, train=False):
        super(MIA, self).__init__(num_clients, batch_size, dataset, root, download,
                 iid, use_gpu, train)

        self.shadow_model = self._create_shadow_model()

    def _create_shadow_model(self):
        model = copy.deepcopy(self.server_model)
        return model

    def _get_posterior(self, model, loader, train=True):
        labels = []
        preds = []

        model.eval()
        for i, (x, y) in enumerate(loader):
            x = x.to(device=self.device)
            y = y.to(device=self.device, dtype=torch.long)

            with torch.no_grad():
                out = model(x)
            out = torch.sigmoid(out)

            labels.extend(y.cpu().numpy())
            if len(preds) == 0:
                preds = out.cpu().numpy()
            else:
                preds = np.vstack((preds, out.cpu().numpy())).reshape(-1, 10)

        if train:
            used = [1 for i in range(len(labels))]
        else:
            used = [0 for i in range(len(labels))]

        assert len(used) == len(labels)

        return np.array(labels), preds, np.array(used)

    def verify_deletion(self, client_id, attack_model=RandomForestClassifier):
        if 'deleted' not in self.clients[client_id]:
            exit('Client {:d} has not submitted deletion request.'.format(client_id))

        if attack_model == RandomForestClassifier:
            model = RandomForestClassifier(max_depth=10, random_state=42)
        elif attack_model == LogisticRegression:
            model = LogisticRegression(random_state=42)
        else:
            exit('not suppported classifier')

        self.shadow_model = self._create_shadow_model()

        training = self.clients[client_id]['train']
        num_train = len(training.dataset)
        testing = DataLoader(self.test_data, sampler=sampler.SubsetRandomSampler(
            np.random.choice(len(self.test_data), num_train, replace=False)))

        labels1, preds1, used1 = self._get_posterior(self.shadow_model, training, train=True)
        labels2, preds2, used2 = self._get_posterior(self.shadow_model, testing, train=False)

        test_labels, test_preds, test_used = self._get_posterior(self.shadow_model, self.clients[client_id]['deleted'], train=False)

        train_labels = np.concatenate([labels1, labels2], axis=0)
        train_preds = np.concatenate([preds1, preds2], axis=0)
        train_used = np.concatenate([used1, used2], axis=0)

        model.fit(train_preds, train_used)
        score = model.predict(train_preds)
        print(f1_score(score, train_used))

        score = model.predict(test_preds)
        print(f1_score(score, test_used))


if __name__ == '__main__':
    # sim = MIA(100, 10)
    # sim.delete(0, [0, 1, 2], 0.2, 4, 50, lr=0.05)
    # sim.verify_deletion(0)

    # shuffle = list(range(10))
    # random.shuffle(shuffle)
    # print(shuffle)
    #
    # sim = BackdoorAttack(100, 10)
    # sim.train(ratio=0.2, epochs=1, rounds=50, opt='sgd', lr=0.05)

    # sim.attack(ratio=0.2, client_ids=[0], size=0.4, epochs1=1, epochs2=1, shuffle=shuffle, opt='sgd', criterion='cross_entropy', lr1=0.05, lr2=0.05, alpha=0.85, epsilon=0.03, gamma=8)
    # with open('try.pkl', 'wb') as f:
    #     pickle.dump(sim, f)

    # attack acc vs portion
    # poison_log = []
    # portions = np.arange(0.1, 0.9, 10)
    # for p in portions:
    #     with open('try.pkl', 'rb') as f:
    #         sim = pickle.load(f)
    #     sim.attack(ratio=0.2, client_ids=[0], size=p, epochs1=1, epochs2=1, shuffle=shuffle, opt='sgd', criterion='cross_entropy', lr1=0.05, lr2=0.05, alpha=0.85, epsilon=0.03, gamma=8)
    #     poison_acc = sim.poison_accuracy([0])
    #     poison_log += [poison_acc]

    # attack acc vs number of attacker
    # num_attackers = np.arange(1, 100, 10)
    # for i in range(10):
    #     with open('try.pkl', 'rb') as f:
    #         sim = pickle.load(f)
    #     clients_ids = np.random.choice(100, num_attackers[i], replace=False)
    #     sim.attack(ratio=0.2, client_ids=clients_ids, size=0.8, epochs1=1, epochs2=1, shuffle=shuffle, opt='sgd', criterion='cross_entropy', lr1=0.05, lr2=0.05, alpha=0.85, epsilon=0.03, gamma=8)
    #     poison_acc = sim.poison_accuracy(clients_ids)
    #     poison_log += [poison_acc]

    # attack vs number of communication rounds
    # cr = 50
    # with open('try.pkl', 'rb') as f:
    #     sim = pickle.load(f)
    # sim.attack(ratio=0.2, client_ids=[0], size=0.4, epochs1=1, epochs2=1, shuffle=shuffle, opt='sgd',
    #            criterion='cross_entropy', lr1=0.05, lr2=0.05, alpha=0.85, epsilon=0.03, gamma=8)
    # poison_acc = sim.poison_accuracy([0])
    # poison_log += [poison_acc]
    # for i in range(cr):
    #     sim.train(ratio=0.2, epochs=1, rounds=100, opt='sgd', lr=0.05)
    #     poison_acc = sim.poison_accuracy([0])
    #     poison_log += [poison_acc]
    #

    # poison acc after sisa


    # poison acc in dp-fl

    shuffle = [0, 7, 2, 3, 4, 5, 6, 7, 8, 9]
    poison_log = []
    global_acc = []

    with open('try.pkl', 'rb') as f:
        sim = pickle.load(f)
    sim.attack(ratio=0.2, client_ids=[5], size=0.8, epochs1=1, epochs2=1, shuffle=shuffle, opt='sgd',
               criterion='cross_entropy', lr1=0.05, lr2=0.05, alpha=0.85, epsilon=0.03, gamma=15)
    print(sim.clients[5].keys())
    sim.delete(5, 'poison', 0.2, 1, 20, lr=0.05)
    print(sim.poison_accuracy([5]))








