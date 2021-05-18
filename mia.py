from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, sampler
import torch.optim as optim

import torchvision.datasets as dset
import torch
import torch.nn as nn

import numpy as np

from models import CNNMnist


def distribute_data(k, dataset, size=10000, random=True):
    target_idxs = {"train": [], "test": []}  # idxs for target models
    shadow_idxs = {}  # idxs for shadow models
    for i in range(k):
        shadow_idxs[i] = {"train": [], "test": []}
    all_idxs = [i for i in range(len(dataset))]

    # ensure test/train not overlap
    test_idxs = np.random.choice(all_idxs, len(dataset) // 2, replace=False)
    train_idxs = list(set(all_idxs) - set(test_idxs))

    target_idxs['train'] = np.random.choice(train_idxs, 10000, replace=False)
    target_idxs['test'] = np.random.choice(test_idxs, 10000, replace=False)

    # no overlap
    if not random:
        train_idxs = list(set(train_idxs) - set(target_idxs['train']))
        test_idxs = list(set(test_idxs) - set(target_idxs['test']))

    for i in range(k):
        shadow_idxs[i]['train'] = np.random.choice(train_idxs, size, replace=False)
        shadow_idxs[i]['test'] = np.random.choice(test_idxs, size, replace=False)

    return target_idxs, shadow_idxs


# distribute data for target and shadow models
class SplitData(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


# train target and shadow models
def train_model(model, loader_train, loader_val, epochs=20, lr=1e-4):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    print_every = 50
    train_loss = []
    val_loss = []
    test_loss = []

    for e in range(epochs):
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
        print('train accuracy {:.4f}, val accuracy {:.4f}'.format(evaluation(model, loader_train),
                                                                  evaluation(model, loader_val)))
    return model.state_dict()


# return labels, prediction vectors, and in/out state
def predict(model, loader, train=True):
    labels = []
    preds = []

    model.eval()
    for i, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device, dtype=torch.long)

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


# spilt data based on labels
def split_class(labels):
    idxs = [i for i in range(len(labels))]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

    res = {}

    all_idxs = {}
    start = 0
    for i in range(10):
        for j in range(start, len(idxs_labels[0])):
            if idxs_labels[1][j] != i:
                all_idxs[i] = idxs_labels[0, start:j]
                start = j
                break

    all_idxs[9] = idxs_labels[0, start:]

    return all_idxs


def create_attack_data(shadows, k):
    res = {}

    for i in range(10):
        res[i] = {'labels': np.array([]), 'preds': np.array([]), 'used': np.array([])}

    for i in range(k):
        for mode in ('train', 'test'):
            labels, preds, used = predict(shadows[i]['model'], shadows[i][mode],
                                          train=True if mode == 'train' else False)
            all_idxs = split_class(labels)
            for j in range(10):
                res[j]['labels'] = np.append(res[j]['labels'], labels[all_idxs[j]])
                res[j]['preds'] = np.vstack((res[j]['preds'], preds[all_idxs[j]])) if len(res[j]['preds']) != 0 else \
                preds[all_idxs[j]]
                res[j]['used'] = np.append(res[j]['used'], used[all_idxs[j]])

    return res


def create_test_data(target):
    res = {}

    for i in range(10):
        res[i] = {'labels': np.array([]), 'preds': np.array([]), 'used': np.array([])}

    for mode in ('train', 'test'):
        labels, preds, used = predict(target['model'], target[mode], train=True if mode == 'train' else False)
        all_idxs = split_class(labels)
        for j in range(10):
            res[j]['labels'] = np.append(res[j]['labels'], labels[all_idxs[j]])
            res[j]['preds'] = np.vstack((res[j]['preds'], preds[all_idxs[j]])) if len(res[j]['preds']) != 0 else preds[
                all_idxs[j]]
            res[j]['used'] = np.append(res[j]['used'], used[all_idxs[j]])

    return res


# dataset for attack model
class AttackModelDataset(Dataset):
    def __init__(self, dict):
        self.x = dict['preds']
        self.y = dict['used']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return torch.tensor(self.x[item]), torch.tensor(self.y[item])


# nn attack model
class AttackModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )

    def forward(self, x):
        return self.nn(x)


# train nn attack model
def train_attack_model(loader_train, loader_test, epochs=10, lr=1e-3):
    model = AttackModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for e in range(epochs):
        train_loss = 0
        for i, (x, y) in enumerate(loader_train):
            model.train()
            x, y = x.to(device), y.to(device, dtype=torch.long)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        # print('epoch {:d} training loss {:4f}'.format(e, train_loss/len(loader_train)))

        val_loss = 0
        for i, (x, y) in enumerate(loader_test):
            model.eval()
            x, y = x.to(device), y.to(device, dtype=torch.long)

            with torch.no_grad():
                out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item()
        # print('epoch {:d} test loss {:4f}'.format(e, val_loss/len(loader_test)))

    acc = evaluation(model, loader_test)
    return acc


def evaluation(model, dataloader):
    total , correct = 0,0
    for data in dataloader:
        inputs , labels = data
        inputs , labels = inputs.to(device) , labels.to(device)
        output = model(inputs)
        max_pred,pred = torch.max(output.data,dim=1)
        total += labels.size(0)
        correct +=(pred == labels).sum().item()
    return 100 * correct / total


def mia(k, size, random=True):
    shadows = {}
    target = {}
    target_idxs, shadow_idxs = distribute_data(k, mnist_train, size=size, random=random)

    target['model'] = CNNMnist()
    target['train'] = DataLoader(SplitData(mnist_train, target_idxs['train']), batch_size=batch_size)
    target['test'] = DataLoader(SplitData(mnist_train, target_idxs['test']), batch_size=batch_size)

    # create test data
    print('train target model\n----------')
    target_weights = train_model(model=target['model'], loader_train=target['train'], loader_val=target['test'],
                                 epochs=10, lr=1e-3)
    target['model'].load_state_dict(target_weights)

    test_data = create_test_data(target)

    # create training data
    for i in range(k):
        shadows[i] = {}
        shadows[i]['model'] = CNNMnist()
        shadows[i]['train'] = DataLoader(SplitData(mnist_train, shadow_idxs[i]['train']), batch_size=batch_size)
        shadows[i]['test'] = DataLoader(SplitData(mnist_train, shadow_idxs[i]['test']), batch_size=batch_size)
        print('train shadow model {:d}/{:d}'.format(i + 1, k))
        print('----------------------------')
        shadow_weights = train_model(model=shadows[i]['model'], loader_train=shadows[i]['train'],
                                     loader_val=shadows[i]['test'], epochs=10, lr=1e-3)
        shadows[i]['model'].load_state_dict(shadow_weights)

    train_data = create_attack_data(shadows, k)

    # create an attck model for each class
    lr_acc, rf_acc = 0, 0
    # logistic regression
    for i in range(10):
        lr = LogisticRegression(random_state=42).fit(train_data[i]['preds'], train_data[i]['used'])
        score = lr.score(test_data[i]['preds'], test_data[i]['used'])
        # print('label {:d} acc {:.4f}'.format(i, score))
        lr_acc += score
    print('lr average acc {:.4f}'.format(lr_acc / 10))

    # random forest
    for i in range(10):
        rf = RandomForestClassifier(max_depth=10, random_state=42).fit(train_data[i]['preds'], train_data[i]['used'])
        score = rf.score(test_data[i]['preds'], test_data[i]['used'])
        # print('label {:d} acc {:.4f}'.format(i, score))
        rf_acc += score
    print('rf average acc {:.4f}'.format(rf_acc / 10))

    # nn
    # nn_acc = 0
    # for i in range(10):
    # loader_train = DataLoader(AttackModelDataset(train_data[i]), batch_size=1000)
    # loader_test = DataLoader(AttackModelDataset(test_data[i]), batch_size=1000)
    # acc = train_attack_model(loader_train, loader_test, epochs=10)
    # print('label {:d} accuracy {:.4f}'.format(i, acc))
    # nn_acc += acc

    # print('nn average acc {:.4f}'.format(nn_acc/1000))

    return lr_acc / 10, rf_acc / 10


if __name__ == '__main__':
    torch.manual_seed(42)
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
    ])

    use_gpu = True
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    data_dir = './dataset'
    batch_size = 128

    mnist_train = dset.MNIST(data_dir, train=True, transform=transforms, download=False)
    mnist_test = dset.MNIST(data_dir, train=False, transform=transforms, download=False)

    # mia random split
    candidate_k = [5, 10, 20, 50, 75]
    log = {'lr': [], 'rf': []}
    for k in candidate_k:
        lr_acc, rf_acc = mia(k, 2500)
        log['lr'] += [lr_acc]
        log['rf'] += [rf_acc]

    # mia no overlap split
    candidate_k = [5, 10, 20, 50]
    log = {'lr': [], 'rf': [], 'nn': []}
    for k in candidate_k:
        lr_acc, rf_acc = mia(k, 2500, random=False)
        log['lr'] += [lr_acc]
        log['rf'] += [rf_acc]
