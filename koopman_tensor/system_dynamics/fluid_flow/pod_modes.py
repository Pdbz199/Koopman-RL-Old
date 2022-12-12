import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from matplotlib.animation import FuncAnimation

p = scipy.io.loadmat('data/POD-COEFFS.mat')
alpha = p['alpha'] # (17000, 8)
alphaS = p['alphaS'] # (17000, 1)
p2 = scipy.io.loadmat('data/POD-MODES.mat')
Xavg = p2['Xavg'] # (89351, 1)
Xdelta = p2['Xdelta'] # (89351, 1)
Phi = p2['Phi'] # (89351, 8)

fps = 10
nSeconds = 8

snapshots = []
for k in range(100, alpha.shape[0], 100):
    u = Xavg[:,0] + Phi[:,0] * alpha[k,0] + Phi[:,1] * alpha[k,1] + Xdelta[:,0] * alpha[k,2]
    snapshots.append(u.reshape(449,199).T)

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure( figsize=(8,8) )

a = snapshots[0]
im = plt.imshow(a, cmap='hot', clim=(-1,1))

def animate_func(i):
    if i % fps == 0:
        print( '.', end ='' )

    im.set_array(snapshots[i])
    return [im]

anim = FuncAnimation(
    fig,
    animate_func,
    frames = nSeconds * fps,
    interval = 1000 / fps # in ms
)

plt.show()