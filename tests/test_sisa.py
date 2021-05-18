import pytest
from sisa import Sisa
from federated_avg import FederatedAveraging


@pytest.fixture
def sisa():
    fl = FederatedAveraging(100, 10)
    return Sisa(fl, train=False)


@pytest.mark.parametrize('client_id, idxs', [(0, [0]), (1, [5,6,9]), (15, [0,2,3,10])])
def test_delete(sisa, client_id, idxs):
    sisa._deletion_request(client_id, idxs)

    for i, data in enumerate(sisa.fl.clients[client_id]['deleted']):
        assert data[0] not in sisa.fl.clients[client_id]['train']
        assert data[0] not in sisa.fl.clients[client_id]['val']
        assert data[0] not in sisa.fl.clients[client_id]['test']


