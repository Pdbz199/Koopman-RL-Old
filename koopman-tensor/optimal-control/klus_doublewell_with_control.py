#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class DoubleWell():
    def __init__(self, beta, c):
        self.beta = beta
        self.c = c

    def b(self, x):
        return -x**3 + 2*x + self.c
    
    def sigma(self, x):
        return np.sqrt(2/self.beta)
    
s = DoubleWell(beta=1, c=0)
h = 1e-2
y = 1

x = np.linspace(-2.5, 2.5, 1000)

# The parametrized function to be plotted
def f(x, beta, c):
    return 1/4*x**4 - x**2 - c*x

# Define initial parameters
init_beta = 1
init_c = 0

# Create the figure
fig, ax = plt.subplots()
line, = plt.plot(x, f(x, init_beta, init_c), lw=2)
point, = plt.plot(y, f(y, init_beta, init_c), 'r.', markersize=20)
ax.set_xlabel('x')
plt.ylim([-5, 5])

axcolor = 'lightgoldenrodyellow'
ax.margins(x=0)

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a vertically oriented slider to control the beta.
ax_beta = plt.axes([0.05, 0.25, 0.0225, 0.63], facecolor=axcolor)
beta_slider = Slider(
    ax=ax_beta,
    label="beta",
    valmin=0,
    valmax=5,
    valinit=init_beta,
    orientation="vertical"
)

ax_c = plt.axes([0.15, 0.25, 0.0225, 0.63], facecolor=axcolor)
c_slider = Slider(
    ax=ax_c,
    label="c",
    valmin=-2,
    valmax=2,
    valinit=init_c,
    orientation="vertical"
)

# The function to be called anytime a slider's value changes
def update(val):
    global y, s
    s.beta = beta_slider.val
    s.c    = c_slider.val
    y = y + s.b(y)*h + s.sigma(y)*np.sqrt(h)*np.random.randn()
    
    line.set_ydata(f(x, beta_slider.val, c_slider.val))
    point.set_xdata(y)
    point.set_ydata(f(y, beta_slider.val, c_slider.val))
    
    fig.canvas.draw_idle()

# register the update function with each slider
beta_slider.on_changed(update)
c_slider.on_changed(update)

while(True):
    update(0)
    plt.pause(0.01)
