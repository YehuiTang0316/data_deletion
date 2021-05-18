import pytest
from federated_avg import FederatedAveraging
from collections import Counter


@pytest.fixture
def fl_iid():
    return FederatedAveraging(100, 10, iid=True, download=False)


@pytest.fixture
def fl_non_iid():
    return FederatedAveraging(100, 10, iid=False, download=False)


def test_iid_distribute_mnist_dataset(fl_iid):
    split = fl_iid.clients_dataset
    dataset = fl_iid.training_data
    labels = dataset.train_labels.numpy()

    count = {}

    for i in range(fl_iid.num_clients):
        idxs = list(split[i])
        contains = labels[idxs]

        for j in range(10):
            assert j in contains

        count[i] = Counter(contains)

    for i in range(10):
        for j in range(fl_iid.num_clients-1):
            assert count[j][i] == count[j+1][i]


def test_non_iid_distribute_mnist_dataset(fl_non_iid):
    split = fl_non_iid.clients_dataset
    dataset = fl_non_iid.training_data
    labels = dataset.train_labels.numpy()

    count = {}

    for i in range(fl_non_iid.num_clients):
        idxs = list(map(int, split[i]))
        contains = labels[idxs]
        count[i] = Counter(contains)

        tmp = list(count[i].keys())
        assert len(tmp) != 10



