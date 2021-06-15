import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams['pdf.fonttype'] = 42


MCFBR = [(0, 510), (5936, None)]
MCFBD = [(0, 2), (3, 5), (6, 7), (9, 10), (11, 16), (5936, None)]
MCFRR = [(0, 418), (5936, None)]
MCFRD = [(0, 2), (3, 5), (6, 7), (9, 10), (11, 16), (5934, None)]

ASTARBR = [(0, -1), (132, 2836)]
ASTARBD = [(0, 2), (3, 5), (6, 7), (9, 10), (11, 14), (18, 21), (22, 27), (29, 31), (32, 33), (34, 35), (37, 38), (40, 43), (46, 2836)]

ASTARRR = [(0, -1), (342, 2836)]
ASTARRD = [(0, 2), (3, 5), (6, 7), (9, 10), (11, 14), (18, 21), (22, 27), (29, 31), (32, 33), (34, 35), (37, 38), (40, 43), (46, 2836)]

def plot(ax, data, offset, color):
  offset = 1.25 * offset
  for x, y in data:
    if y is None:
      continue
    ax.plot([x, y], [0.0+offset, 0.0+offset], '.-', color=color, markersize=2, linewidth=1)
  for (x1, y1), (x2, y2) in zip(data, data[1:]):
    ax.plot([y1, x2], [0.8+offset, 0.8+offset], '.-', color=color, markersize=2, linewidth=1)


fig, axs = plt.subplots(2)

plt.setp(axs, yticks=[0.0+0.15, 0.8+0.15], yticklabels=['Predictor', 'Marker'], ylim=[-0.3,1.4])

plot(axs[0], ASTARBR, 0.3, 'C0')
plot(axs[0], ASTARBD, 0.2, 'C1')
plot(axs[0], ASTARRR, 0.1, 'C2')
plot(axs[0], ASTARRD, 0.0, 'C3')
plot(axs[1],   MCFBR, 0.3, 'C0')
plot(axs[1],   MCFBD, 0.2, 'C1')
plot(axs[1],   MCFRR, 0.1, 'C2')
plot(axs[1],   MCFRD, 0.0, 'C3')

axs[0].set_title("astar 100% (predictors are better than Marker)")
axs[1].set_title("mcf 0.01% (Marker is better than predictors)")

fig.subplots_adjust(top=0.8, left=0.15)
plt.figlegend(loc='upper center', handles=[
  mlines.Line2D([], [], color=color, label=label)
  for color, label in zip(('C0', 'C1', 'C2', 'C3'), (
    'BlindOracleR',
    'BlindOracleD',
    'RobustFtPR',
    'RobustFtPD',
    ))], ncol=2, fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.subplots_adjust(hspace=.5)
plt.gcf().set_size_inches(w=1.67*3.5, h=1.67*3)
plt.savefig('figure-switching.pdf')

