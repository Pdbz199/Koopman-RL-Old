import matplotlib.pyplot as plt
import numpy as np

alpha = np.load('data/ALPHA.npy') # (17000, 8)
alpha_s = np.load('data/ALPHA-S.npy') # (17000, 8)
phi = np.load('data/VORT-PHI.npy') # (89351, 41)
avg = np.load('data/VORT-AVG.npy') # (89351, 1)
delta = np.load('data/VORT-DELTA.npy') # (89351, 1)

# for k=100:100:length(alpha(:,1))
for k in range(100, alpha.shape[0], 100):
    k = 0
    # u = Xavg + Phi(:,1)*alpha(k,1) + Phi(:,2)*alpha(k,2) + Xdelta*alphaS(k);
    u = avg + phi[:,0] * alpha[k,0] + phi[:,1] * alpha[k,1] + delta * alpha_s[k]
    # imshow(reshape(u,199,449)); colormap hot; caxis([-1 1])
    plt.imshow(u.reshape(199,499), cmap='hot', caxis=(-1,1))
    plt.show()
    # pause(0.01)
    plt.pause(0.01)