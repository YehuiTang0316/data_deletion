import os
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, sampler

import torchvision
from torchvision import datasets, models, transforms

from backdoor import BackdoorAttack, ConcatDataset, ClientDataset


class PoisonDataset(Dataset):
    def __init__(self, poison, label):
        self.poison = poison
        self.label = label

    def __len__(self):
        return len(self.poison)

    def __getitem__(self, item):
        return torch.tensor(self.poison[item]), torch.tensor(self.label)


class CleanLabelAttack(BackdoorAttack):
    def __init__(self, num_clients, batch_size, sigma, dataset='mnist', root='./', download=False,
                 iid=False, use_gpu=True):
        super(CleanLabelAttack, self).__init__(num_clients, batch_size, sigma, dataset, root, download,
                 iid, use_gpu)

    def _create_poison_data(self, client_id, size, shuffle=None):
        base_instance = None
        unnormalized_base_instance = None
        perturbed_instance = None
        target_instance = None

        for imgs, labels in self.test_loader:
            for i in range(imgs.shape[0]):
                if labels[i].item() == 1:
                    base_instance = imgs[i].unsqueeze(0).to(self.device)

                if target_instance is None:
                    if labels[i].item() == 7:
                        target_instance = imgs[i].unsqueeze(0).to(self.device)

        mean_tensor = torch.from_numpy(np.array((0.1307,)))
        std_tensor = torch.from_numpy(np.array((0.3081,)))


        unnormalized_base_instance = base_instance.clone()
        unnormalized_base_instance[:, 0, :, :] *= std_tensor[0]
        unnormalized_base_instance[:, 0, :, :] += mean_tensor[0]
        # unnormalized_base_instance[:, 1, :, :] *= std_tensor[1]
        # unnormalized_base_instance[:, 1, :, :] += mean_tensor[1]
        # unnormalized_base_instance[:, 2, :, :] *= std_tensor[2]
        # unnormalized_base_instance[:, 2, :, :] += mean_tensor[2]

        perturbed_instance = unnormalized_base_instance.clone()

        target_features, outputs = self.server_model(target_instance)

        transforms_normalization = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
                        ])

        epsilon = 16 / 255
        alpha = 0.05 / 255

        start_time = time.time()

        for i in range(5000):
            perturbed_instance.requires_grad = True

            poison_instance = transforms_normalization(perturbed_instance)
            poison_features, _ = self.server_model(poison_instance)

            feature_loss = nn.MSELoss()(poison_features, target_features)
            image_loss = nn.MSELoss()(poison_instance, base_instance)
            loss = feature_loss + image_loss / 1e2
            loss.backward()

            signed_gradient = perturbed_instance.grad.sign()

            perturbed_instance = perturbed_instance - alpha * signed_gradient
            eta = torch.clamp(perturbed_instance - unnormalized_base_instance, -epsilon, epsilon)
            perturbed_instance = torch.clamp(unnormalized_base_instance + eta, 0, 1).detach()

            if i == 0 or (i + 1) % 500 == 0:
                print(f'Feature loss: {feature_loss}, Image loss: {image_loss}, Time: {time.time() - start_time}')

            poison_instance = transforms_normalization(perturbed_instance)

        poison_dataset = PoisonDataset([poison_instance], 1)

        idxs = self.clients_dataset[client_id]
        idxs_train = idxs[:int(0.8 * len(idxs))]

        self.clients[client_id]['poison train'] = DataLoader(ConcatDataset(ClientDataset(
            self.training_data, idxs_train), poison_dataset),
            batch_size=self.batch_size, shuffle=True)

        self.clients[client_id]['poison'] = DataLoader(poison_dataset, batch_size=self.batch_size, shuffle=True)

        print('Created poison data of size {:d}, changing images of label 7 into 1'.format(size))


if __name__ == '__main__':
    sim = CleanLabelAttack(100, 10, 0.1)
    sim.attack(ratio=0.2, client_ids=[0], size=0.4, epochs1=1, epochs2=1, shuffle=None, opt='sgd', criterion='cross_entropy', lr1=0.05, lr2=0.05, alpha=0.85, epsilon=0.03, gamma=8)





