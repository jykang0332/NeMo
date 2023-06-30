import matplotlib.pyplot as plt

import numpy as np
from scipy import interpolate

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

#t = np.linspace(1,1638,300)
x0= np.loadtxt('./wer/baseCTC.csv', delimiter=',', dtype=float)
x1= np.loadtxt('./wer/RKD_initialized.csv', delimiter=',', dtype=float)
x2 = np.loadtxt('./wer/RKD_Guided.csv', delimiter=',', dtype=float)
x3 = np.loadtxt('./wer/SelfAttn_Guided.csv', delimiter=',', dtype=float)

x0 = x0[:,2][:50]
x1 = x1[:,2][:50]
x2 = x2[:,2][:50]
x3 = x3[:,2][:50]

# plt.plot(x0, marker='o', linewidth=2, color='red', markersize=2, markeredgewidth=1, markerfacecolor='white', markeredgecolor='black')
# plt.plot(x1, marker='o', linewidth=2, color='blue', markersize=2, markeredgewidth=1, markerfacecolor='white', markeredgecolor='black')

# plt.plot(x0, linewidth=2, color='tomato')
plt.plot(x1, linewidth=2, color='lime')
plt.plot(x2, linewidth=2, color='blue')
plt.plot(x3, linewidth=2, color='salmon')
# 'lime'

# plt.plot(x2, marker='o', linewidth=2, color='lime', markersize=3, markeredgewidth=1, markerfacecolor='white', markeredgecolor='lime')
# plt.plot(x5, marker='o', linewidth=2, color='blue', markersize=3, markeredgewidth=1, markerfacecolor='white', markeredgecolor='blue')#, markeredgewidth=1)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('WER [%]', fontsize=15)
plt.xlabel('epoch', fontsize=15)

plt.ylim(0.06, 0.25)

#plt.ylabel('EER [%]')
#plt.xlabel('epoch')
# plt.legend(['$\lambda = 0$', '$\lambda = 1$'], loc='upper right')
# plt.legend(['$BaseCTC$', '$RKD$', '$RKD + Guided CTC$'], loc='upper right')
plt.legend(['$RKD$', '$RKD + Guided CTC$', '$AD + Guided CTC$'], loc='upper right')
plt.grid()
#plt.show()

plt.savefig('./wer.pdf', dpi=200)
