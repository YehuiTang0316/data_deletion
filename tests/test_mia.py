from mia import distribute_data
import numpy as np
import pytest


@pytest.mark.parametrize("k", [5, 10, 100])
def test_distribute_data(k):
    dataset = np.arange(100000)
    target_idxs, shadow_idxs = distribute_data(k, dataset, random=False)

    for i in range(k):
        assert len(np.append(target_idxs['train'],(shadow_idxs[i]['train']))) == len(set(target_idxs['train'])) + len(set(shadow_idxs[i]['train']))

