import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 10,})


with open('predusage.csv') as f:
	data = f.readlines()
# algorithms = data[0].strip().split(',')[1:]
ALGS = (5,6,7,8,9,10,12,13,14)
# datasets = [line.split(',')[0] for line in data[1::4]]
# hitrates = [list(map(float, line.strip().split(',')[1:])) for line in data[2::4]]
usages = np.array([list(map(float, line.strip().split(',')[1:])) for line in data[3::4]])[:,ALGS]
all_ratios = np.array([list(map(float, line.strip().split(',')[1:])) for line in data[1::4]])
normalized_ratios = (all_ratios[:,ALGS] - 1) / (all_ratios[:,1] - 1).reshape(-1,1)
predictor_is_better = all_ratios[:,2] > all_ratios[:,11]

rbad = normalized_ratios[np.logical_not(predictor_is_better)].flatten()
ubad = usages[np.logical_not(predictor_is_better)].flatten()
rgood = normalized_ratios[predictor_is_better].flatten()
ugood = usages[predictor_is_better].flatten()

plt.scatter( ubad,  rbad, color='C1', marker='o', label='Marker is better')
plt.scatter(ugood, rgood, color='C2', marker='*', label='Predictor is better')

x = np.array([0, 1])

m, b = np.polyfit( ubad,  rbad, 1)
plt.plot(x, x * m + b, 'C1')
m, b = np.polyfit(ugood, rgood, 1)
plt.plot(x, x * m + b, 'C2')

ax = plt.gca()
ax.set_yticks(np.arange(0,1.8, 0.2), minor=True)
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'] + ['1.2', '1.4', '1.6', '1.8'], minor=True)
ax.set_yticks(np.arange(0,1.8, 1.0), minor=False)
ax.set_yticklabels(['OPT', 'LRU'], minor=False)
ax.yaxis.grid(True, which='major', linewidth=1.5, zorder=0)
ax.yaxis.grid(True, which='minor', linestyle=':', linewidth=1, zorder=0)
ax.set_axisbelow(True)

plt.xlabel('Prediction usage')
plt.ylabel('Normalized competitive ratio')

plt.legend(loc='lower left')

# plt.gcf().set_size_inches(w=2*6.75, h=2*3)
plt.gcf().set_size_inches(w=1.5*3.25, h=1.5*2.5)
plt.savefig('figure-usage.pdf')
# plt.show()
