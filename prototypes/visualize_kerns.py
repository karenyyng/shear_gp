from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import george
import sys
from matplotlib.ticker import MaxNLocator
sys.path.append("../")
import kern_deriv as kd


coords = np.array([[i, j] for i in np.arange(0.1, 1, .05)
                   for j in np.arange(0.1, 1., 0.05)])

deriv_ker = {"KappaKappa": kd.KappaKappaExpSquareKernel,
             "KappaGamma1": kd.KappaGamma1ExpSquareKernel,
             "KappaGamma2": kd.KappaGamma2ExpSquareKernel,
             "Gamma1Gamma1": kd.Gamma1Gamma1ExpSquareKernel,
             "Gamma1Gamma2": kd.Gamma1Gamma2ExpSquareKernel,
             "Gamma2Gamma2": kd.Gamma2Gamma2ExpSquareKernel}

deriv_gp = {k: george.GP(1.0 * v(1.0, coords)) for
            k, v in deriv_ker.iteritems()}

for v in deriv_gp.values():
    v.compute(coords, 1e-3)

deriv_samples = {k: v.sample(coords) for k, v in deriv_gp.iteritems()}

plotNo = [[0, 0],
          [0, 1],
          [0, 2],
          [1, 1],
          [1, 2],
          [2, 2]]
side = int(np.sqrt(coords.shape[0]))
space = 0.01
figsize = 5
#fig = plt.figure(figsize=(18, 15))

fig, ax = plt.subplots(3, 3, figsize=(figsize, figsize), sharex=True,
                       sharey=True)
fig.subplots_adjust(wspace=space, hspace=space)
for i in range(6):
    no = plotNo[i]
    thisAx = ax[no[0]][no[1]]
    im = thisAx.imshow(deriv_samples.values()[i].reshape(
        side, side), cmap=plt.cm.winter,
        origin='upper', extent=(0, 1, 1, 0),
        vmin=-1.0, vmax=1.0)
    for tick in thisAx.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    for tick in thisAx.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)


#plt.colorbar()
set_ticks = [[0, 0], [1, 0], [2, 0], [2, 2], [2, 1]]
labels = [r'$\kappa$',
          r'$\gamma_1$',
          r'$\gamma_2$']
for t in set_ticks:
    thisAx = ax[t[0], t[1]]
    thisAx.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    thisAx.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))

    if t != [2, 1] and t != [2, 2]:
        thisAx.set_ylabel(labels[t[0]])
    thisAx.set_xlabel(labels[t[1]])

    for tick in thisAx.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)

    for tick in thisAx.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)

cbar_ax = fig.add_axes([.925, 0.15, 0.025, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar_ax.tick_params(labelsize=6)
plt.savefig("../plots/" + "overall" + ".png",
            pad_inches=100)
plt.clf()
