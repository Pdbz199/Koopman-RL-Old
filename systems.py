import numpy as np
# import matplotlib.pyplot as plt

def ornstein_uhlenbeck(start_time, end_time, length, theta=1.1, mu=0.8, sigma=0.3):
    t = np.linspace(start_time, end_time, length)
    dt = np.mean(np.diff(t))

    y = np.empty(length)
    y_0 = np.random.normal(loc=0.0, scale=1.0)

    drift = lambda y, t: theta * (mu-y)
    diffusion = lambda y, t: sigma
    noise = np.random.normal(loc=0.0, scale=1.0, size=length) * np.sqrt(dt)

    for i in range(1, length):
        y[i] = y[i-1] + drift(y[i-1], i*dt)*dt + diffusion(y[i-1], i*dt) * noise[i]

    return np.array(t), np.array(y)

def vec_ornstein_uhlenbeck(start_times, end_times, length, theta=1.1, mu=0.8, sigma=0.3):
    T = []
    Y = []
    for i, start_time in enumerate(start_times):
        t, y = ornstein_uhlenbeck(start_time, end_times[i], length)
        T.append(t)
        Y.append(y)
    return np.array(T), np.array(Y)

# t_0s = [0,0,0]
# t_ends = [2,2,2]
# length = 1000

# T, Y = vec_orenstein_uhlenbeck(t_0s, t_ends, length)
# for i in range(len(T)):
#     plt.plot(T[i], Y[i])
# plt.show()