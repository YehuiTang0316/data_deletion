import matplotlib.pyplot as plt
import numpy as np

plt.figure()

sigmas = [0.1, 1, 2]
legends = []
for sigma in sigmas:
    path = './log/dp-sgd' + '--' + str(sigma) + '--' + str(4.0) +'mnist.npy'
    data = np.load(path, allow_pickle=True).item()
    plt.plot(data['test_local'], '-.')
    plt.plot(data['test_global'])
    legends += ['sigma=' + str(sigma) + 'local', 'sigma=' + str(sigma) + 'global']

plt.legend(legends)
plt.title('max norm = 4.0')
plt.xlabel('communication rounds')
plt.ylabel('accuracy')
plt.savefig('./figures/dp-sgd.pdf')
plt.show()

sigmas = [0.1, 0.5, 2]
legends = []
for sigma in sigmas:
    path = './log/dp-v1' + '-' + str(sigma) +'mnist.npy'
    data = np.load(path, allow_pickle=True).item()
    plt.plot(data['test_local'], '-.')
    plt.plot(data['test_global'])
    legends += ['sigma=' + str(sigma) + 'local', 'sigma=' + str(sigma) + 'global']

plt.legend(legends)
plt.xlabel('communication rounds')
plt.ylabel('accuracy')
plt.savefig('./figures/dpv1.pdf')
plt.show()
