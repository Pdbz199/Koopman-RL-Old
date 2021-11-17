#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class DoubleWell():
    def __init__(self, beta, c):
        self.beta = beta
        self.c = c

    def b(self, x):
        return -x**3 + 2*x + self.c
    
    def sigma(self, x):
        return np.sqrt(2/self.beta)

class EulerMaruyama(object):
    def __init__(self, h, nSteps):
        self.h = h
        self.nSteps = nSteps
    
    def integrate(self, s, x):
        y = np.zeros(self.nSteps)
        y[0] = x
        for i in range(1, self.nSteps):
            y[i] = y[i-1] + s.b(y[i-1])*self.h + s.sigma(y[i-1])*np.sqrt(self.h)*np.random.randn()
        return y

#%% create double-well system and integrator
s = DoubleWell(beta=2, c=-1)
em = EulerMaruyama(1e-3, 10000)

#%% generate one trajectory
x0 = 5*np.random.rand() - 2.5
y = em.integrate(s, x0)
plt.clf()
plt.plot(y)

#%% generate training data
m = 1000
X = 5*np.random.rand(m) - 2.5
Y = np.zeros(m)
for i in range(m):
    y = em.integrate(s, x0)
    Y[i] = y[-1]

plt.figure()
plt.hist(Y, 50)
