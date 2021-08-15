import matplotlib.pyplot as plt
import numpy as np

# plt.figure()
#
# sigmas = [0.1, 1, 2]
# legends = []
# for sigma in sigmas:
#     path = './log/dp-sgd' + '--' + str(sigma) + '--' + str(4.0) +'mnist.npy'
#     data = np.load(path, allow_pickle=True).item()
#     plt.plot(data['test_local'], '-.')
#     plt.plot(data['test_global'])
#     legends += ['sigma=' + str(sigma) + 'local', 'sigma=' + str(sigma) + 'global']
#
# plt.legend(legends)
# plt.title('max norm = 4.0')
# plt.xlabel('communication rounds')
# plt.ylabel('accuracy')
# plt.savefig('./figures/dp-sgd.pdf')
# plt.show()
#
# sigmas = [0.1, 0.5, 2]
# legends = []
# for sigma in sigmas:
#     path = './log/dp-v1' + '-' + str(sigma) +'mnist.npy'
#     data = np.load(path, allow_pickle=True).item()
#     plt.plot(data['test_local'], '-.')
#     plt.plot(data['test_global'])
#     legends += ['sigma=' + str(sigma) + 'local', 'sigma=' + str(sigma) + 'global']
#
# plt.legend(legends)
# plt.xlabel('communication rounds')
# plt.ylabel('accuracy')
# plt.savefig('./figures/dpv1.pdf')
# plt.show()
#
#

# noise vs attack
# plt.figure(figsize=(12,6))
# sigma = np.array([0, 0.1, 1, 2])
#
# size = 4
# x = np.arange(size)
#
# total_width, n = 0.5, 3
# width = total_width / n
# x = x - (total_width - width) / 2
#
# prob_1_before = [5.345234810327781e-10,  2.916969443766959e-11, 1.3646493934871273e-09, 3.793326486256632e-10]
# prob_7_before = [0.9999982118606567, 0.9999998807907104,  0.999971866607666, 0.9999582767486572]
# prob_1_attack = [0.4952135682106018, 0.5494252443313599, 0.9958010315895081, 1.0]
# prob_7_attack = [0.3913908898830414, 0.41817212104797363, 0.00018810172332450747, 2.1578610594236398e-26]
# prob_1_delete = [0.4475477635860443, 0.35580500960350037, 0.2980102002620697, 6.193727447757615e-20]
# prob_7_delete = [0.44749996066093445, 0.6145727634429932,  0.4474729597568512, 5.359709898122844e-10]
#
#
# accuracy_before = np.array([98.3, 98.1, 97.8, 96.2]) * 0.01
# accuracy_atttack = np.array([88.3, 88.2, 85.0, 73.4]) * 0.01
# accuracy_delete = np.array([94.4, 94.8, 84.7, 70.5]) * 0.01
#
# plt.bar(x, prob_1_before, width=width, label='probability of 1 before attack')
# plt.bar(x, prob_7_before, width=width, label='probability of 7 before attack')
#
# plt.bar(x+width, prob_1_attack, width=width, label='probability of 1 after attack')
# plt.bar(x+width, prob_7_attack, width=width, label='probability of 7 after attack')
#
# plt.bar(x+2*width, prob_7_delete, width=width, label='probability of 7 after deletion')
# plt.bar(x+2*width, prob_1_delete, width=width, label='probability of 1 after deletion')
#
#
# plt.plot(x, accuracy_before, label='model accuracy before attack', marker='+',color="deeppink")
# plt.plot(x, accuracy_atttack, label='model accuracy after attack', marker='o', color="darkblue")
# plt.plot(x, accuracy_delete, label='model accuracy after deletion', marker='*',color="goldenrod")
#
# plt.xticks(ticks=x, labels=['0.0001', '0.1', '1', '2'])
# plt.legend()
# plt.ylabel('probability / accuracy')
# plt.xlabel('std of noise')
#
# plt.savefig('./figures/sisa.pdf')
# plt.show()



plt.figure()
size = 5
x = np.arange(size)

before_attack = [9.72261049447809e-10/0.9999955892562866] * size
acc_0001 = [0.2448897659778595/0.002514706226065755, 0.385954350233078/0.013882355764508247, 0.20339980721473694/0.008326400071382523, 0.2608163356781006/0.02318163961172104, 0.23641103506088257/0.015699708834290504]
acc_01 = [0.04336114972829819/0.00820265430957079 ,0.04700343683362007/0.0005715205916203558, 0.04165816307067871/0.003855187678709626, 0.07811002433300018/0.006719020660966635, 0.10485172271728516/0.00520290294662118]
acc_1 = [3.2001744898479956e-07/0.0048220776952803135, 0.006112037226557732/0.06102008372545242, 0.0004796710272785276/0.6414142847061157, 0.0007538009085692465/0.9647216796875, 0.003453413024544716/0.6952047944068909]

plt.plot(x, np.log(np.array(acc_0001)), label='sigma=0.0001')
plt.plot(x, np.log(np.array(acc_01)), label='sigma=0.1')
plt.plot(x, np.log(np.array(acc_1)), label='sigma=1')
plt.plot(x, np.log(np.array(before_attack)), label='before attack')

plt.xticks(ticks=x, labels=['0.1', '0.2', '0.5', '0.6', '0.9'])
plt.legend()
plt.ylabel('log ratio p(y=1)/p(y=7)')
plt.xlabel('percentage of clients participate in communication')

plt.savefig('./figures/sisa_deletion_ratio.pdf')
plt.show()