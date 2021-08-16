import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


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


class FTCifar10(nn.Module):
    def __init__(self):
        super(FTCifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class Cifar10CnnModel(nn.Module):
    def __init__(self):
        super(Cifar10CnnModel, self).__init__()
        self.feature_extractor = FTCifar10()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.fc3(features)
        return features, output


class Cifar10ResNet(nn.Module):
    def __init__(self):
        super(Cifar10ResNet, self).__init__()

        # load a pre-trained model for the feature extractor
        self.feature_extractor = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        self.fc = nn.Linear(512, 10)

        # fix the pre-trained network
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, images):
        features = self.feature_extractor(images)
        x = torch.flatten(features, 1)
        outputs = self.fc(x)
        return features, outputs


def imshow(input, title):
    # torch.Tensor => numpy
    input = input.numpy().transpose((1, 2, 0))
    # undo image normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # display images
    plt.imshow(input)
    plt.title(title)
    plt.show()


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

    cifar10_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_dir = './dataset'
    batch_size = 128
    num_train = 45000

    mnist_train = dset.CIFAR10(data_dir, train=True, transform=cifar10_transforms, download=True)
    mnist_val = dset.CIFAR10(data_dir, train=True, transform=cifar10_transforms, download=True)
    mnist_test = dset.CIFAR10(data_dir, train=False, transform=cifar10_transforms, download=True)

    loader_train = DataLoader(mnist_train, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(num_train)))
    loader_val = DataLoader(mnist_val, batch_size=batch_size,
                            sampler=sampler.SubsetRandomSampler(range(num_train, 50000)))
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
            _, output = model(inputs)
            max_pred, pred = torch.max(output.data, dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        return 100 * correct / total


    model = Cifar10ResNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

    epoch = 300
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

            _, out = model(x)

            optimizer.zero_grad()

            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()

            if i % print_every == 0:
                print('Epoch:{:d}, Iteration:{:d}, loss = {:.4f}'.format(e, i, loss.item()))

        train_epoch_loss /= len(loader_train)
        train_loss += [train_epoch_loss]
        scheduler.step()

        val_epoch_loss = 0
        for i, (x, y) in enumerate(loader_val):
            model.eval()
            x = x.to(device=device)
            y = y.to(device=device, dtype=torch.long)

            with torch.no_grad():
                _, out = model(x)

                loss = criterion(out, y)

                val_epoch_loss += loss.item()

        val_epoch_loss /= len(loader_val)
        val_loss += [val_epoch_loss]
        print('Epoch: {:d}, validation loss {:.4f}'.format(e, val_epoch_loss))
        print('train accuracy {:.4f}, val accuracy {:.4f}'.format(evaluation(loader_train), evaluation(loader_val)))

        model.eval()
        print('test arrcuracy: {:.4f}'.format(evaluation(loader_test)))

        # plot training log
        # plt.figure()
        # plt.plot(train_loss)
        # plt.plot(val_loss)
        # plt.legend(["train loss", "val loss"])
        # plt.show()









