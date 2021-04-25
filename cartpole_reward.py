#%%
import math
# import gym
import numpy as np
import numba as nb
# env = gym.make('CartPole-v0')
# cart_position, cart_velocity, pole_angle, pole_velocity = env.reset()
theta_threshold_radians = 12 * 2 * math.pi / 360
x_threshold = 2.4
gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = (masspole + masscart)
length = 0.5  # actually half the pole's length
polemass_length = (masspole * length)
force_mag = 10.0
tau = 0.02  # seconds between state updates
kinematics_integrator = 'euler'

# Angle at which to fail the episode
theta_threshold_radians = 12 * 2 * math.pi / 360
x_threshold = 2.4

# Angle limit set to 2 * theta_threshold_radians so failing observation
# is still within bounds.
high = np.array([x_threshold * 2,
                    np.finfo(np.float32).max,
                    theta_threshold_radians * 2,
                    np.finfo(np.float32).max],
                dtype=np.float32)

# @nb.njit(fastmath=True)
def cartpoleReward(state, action):
    x, x_dot, theta, theta_dot = state

    force = force_mag if action >= 0.5 else -force_mag
    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    if kinematics_integrator == 'euler':
        x = x + tau * x_dot
        x_dot = x_dot + tau * xacc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * thetaacc
    else:  # semi-implicit euler
        x_dot = x_dot + tau * xacc
        x = x + tau * x_dot
        theta_dot = theta_dot + tau * thetaacc
        theta = theta + tau * theta_dot

    # done = bool(
    #     x < -x_threshold
    #     or x > x_threshold
    #     or theta < -theta_threshold_radians
    #     or theta > theta_threshold_radians
    # )

    reward = (1 - (x ** 2) / 11.52 - (theta ** 2) / 288)
    return reward

# @nb.njit(fastmath=True)
def defaultCartpoleReward(state, action):
    x, x_dot, theta, theta_dot = state

    force = force_mag if action == 1 else -force_mag
    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    # For the interested reader:
    # https://coneural.org/florian/papers/05_cart_pole.pdf
    temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    if kinematics_integrator == 'euler':
        x = x + tau * x_dot
        x_dot = x_dot + tau * xacc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * thetaacc
    else:  # semi-implicit euler
        x_dot = x_dot + tau * xacc
        x = x + tau * x_dot
        theta_dot = theta_dot + tau * thetaacc
        theta = theta + tau * theta_dot

    done = bool(
        x < -x_threshold
        or x > x_threshold
        or theta < -theta_threshold_radians
        or theta > theta_threshold_radians
    )

    if not done:
        reward = 1.0
    else:
        reward = 0.0

    return reward
# %%
