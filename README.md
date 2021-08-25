# Can You Delete Data From My Machine Learning Models?
## Federated Learning
In this project, we only use simple Federated Averaging algorithm.
Class ```FederatedAveraging()``` in federated_avg.py is implemented as the parent class for entire project.  

### Pretraining Before Communication
The code for pretraining can be found at ```trail.py```.

### Averaging with Differential Privacy
In ```dp.py```, we trailed three different ways to add differential privacy protocol in the communication
between the server and clients. The final adopted version is class ```DPFL3()```.
And the implementation of DP-SGD and calculation of $\epsilon, \delta$ is based on 
pyvacy(https://github.com/ChrisWaites/pyvacy).



## Data Deletion
SISA deletion under Federated Learning setting is the deletion algorithm we implemented for client-level
data deletion. The implementation can be found under ```sisa.py```.
 The experiment shown that with strong differential privacy guarantee that delete data from local machine and
 then averaging can successfully delete data from server model.


## Verification
Three types of verification algorithm has been tested in this project.
You can find membership inference attack at ```mia.py```,
label flipping backdoor attack at ```backdoor.py``` and clean label backdoor attack at ```clean_label_attack.py```. 


## Models and Dataset
Models we use is Lenet-5 and Resnet18, you can find the implementation of them in ```models.py```.
The datasets we use are MNIST, Fashion MNIST and CIFAR10.


## Reproduction
To reproduce the experiment in the paper, you can start with ```clean_label_attack.py```.
First create a object with ```sim = CleanLabelAttack()```,
then use ```sim.train()``` to train federated learning model.
After fine-tuning your model, you can use ```sim.attack()``` to insert a backdoor into the server model
and later test deletion effectiveness with ```sim.delete()```.


## Tests
All tests for codes are available under ```./tests``` implemented via pytests.