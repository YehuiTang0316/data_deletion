# Can You Delete Data From My Machine Learning Models?
## Federated Learning
To use federated learning algorithm, ```fl = FederatedAveraging(num_clients[int], client_batch_size[int])```. Both i.i.d client
dataset and non-i.i.d dataset are supported. Specify it when declaring,
 like ```fl = FederatedAveraging(num_clients[int], client_batch_size[int], iid=False)``` for non-i.i.d data.
## Data Deletion
Currently, SISA deletion is enabled. To use SISA to delete a training data in a trained federated learning system, 
use 
```sisa = Sisa(fl).delete(client_id[int], idxs[list])```.

## Membership Inference Attack
The initial idea is to use membership inference attack for deletion verification, to use it, ```mia(num_shadow_models[int])```.

## Tests
All tests for codes are available under ```./tests```.