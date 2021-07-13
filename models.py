import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Lenet-5 for Mnist dataset
class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()

        self.feature_extractor = FeatureExtractor()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        features = self.feature_extractor(x)
        outputs = self.fc3(features)
        return features, outputs


if __name__ == '__main__':
    # centralized training, provides benchmark
    from torchvision import transforms as T
    from torch.utils.data import DataLoader, Dataset, sampler
    import torch.optim as optim

    import torchvision.datasets as dset

    import numpy as np
    import matplotlib.pyplot as plt

    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
    ])

    data_dir = './dataset'
    batch_size = 128
    num_train = 55000

    mnist_train = dset.MNIST(data_dir, train=True, transform=transforms, download=True)
    mnist_val = dset.MNIST(data_dir, train=True, transform=transforms, download=True)
    mnist_test = dset.MNIST(data_dir, train=False, transform=transforms, download=True)

    loader_train = DataLoader(mnist_train, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(num_train)))
    loader_val = DataLoader(mnist_val, batch_size=batch_size,
                            sampler=sampler.SubsetRandomSampler(range(num_train, 60000)))
    loader_test = DataLoader(mnist_test, batch_size=batch_size)

    use_gpu = True
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)


    def evaluation(dataloader):
        total, correct = 0, 0
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            max_pred, pred = torch.max(output.data, dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        return 100 * correct / total


    model = CNNMnist().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    epoch = 20
    print_every = 50
    train_loss = []
    val_loss = []
    test_loss = []

    for e in range(epoch):
        train_epoch_loss = 0
        for i, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device)
            y = y.to(device=device, dtype=torch.long)

            out = model(x)

            optimizer.zero_grad()
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()

            if i % print_every == 0:
                print('Epoch:{:d}, Iteration:{:d}, loss = {:.4f}'.format(e, i, loss.item()))

        train_epoch_loss /= len(loader_train)
        train_loss += [train_epoch_loss]

        val_epoch_loss = 0
        for i, (x, y) in enumerate(loader_val):
            model.eval()
            x = x.to(device=device)
            y = y.to(device=device, dtype=torch.long)

            with torch.no_grad():
                out = model(x)
                loss = criterion(out, y)

                val_epoch_loss += loss.item()

        val_epoch_loss /= len(loader_val)
        val_loss += [val_epoch_loss]
        print('Epoch: {:d}, validation loss {:.4f}'.format(e, val_epoch_loss))
        print('train accuracy {:.4f}, val accuracy {:.4f}'.format(evaluation(loader_train), evaluation(loader_val)))

        model.eval()
        print('test arrcuracy: {:.4f}'.format(evaluation(loader_test)))

        # plot training log
        plt.figure()
        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.legend(["train loss", "val loss"])
        plt.show()









