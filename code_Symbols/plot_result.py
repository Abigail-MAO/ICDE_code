import numpy as np
import matplotlib.pyplot as plt

result_baseline = np.load('./baseline/result.npy')[:100]
result_original = np.load('./original_code/result.npy')[:100]
result_patternldp = np.load('./patternLDP/result_12_04.npy')
epsilons = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
print(result_baseline, result_original, result_patternldp)
plt.plot(epsilons, result_baseline, 'o-', color='#fb90a4', label='Baseline Mechanism', markersize=10)
plt.plot(epsilons, result_original, '^-', color='#98ea8a', label='PrivShape', markersize=10)
plt.plot(epsilons, result_patternldp, '*-', color='#8795e2', label='PatternLDP+KShape', markersize=10)
size = 18
plt.ylabel('Adjusted Rand Index', fontsize = size)
plt.xlabel('Privacy Budget $\epsilon$', fontsize = size)
plt.legend(fontsize = size)
plt.xticks(fontsize = size)
plt.yticks(fontsize = size)
plt.grid(linestyle='-.')
plt.tight_layout()
plt.savefig('result_12_04.pdf')