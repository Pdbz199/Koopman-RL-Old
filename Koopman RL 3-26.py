# Koopman RL Report: 03.26.2021

## OpenAI Gym Koopman Prediction Application: CartPole
'''
To see how well (and quickly) the Koopman operator could predict future states, I used a Q-learning agent that learned a near-optimal policy for the CartPole environment and fit an approximate Koopman operator to it.
'''
'''
X = np.load('state-action-inputs.npy') # 20,000 entries
X = X[:int(X.shape[0]*0.0015)] # 30 points!

# Fit Koopman operator using closed-form solution to DMD
optdmd = OptDMD(svd_rank=15)
model_optdmd = pk.Koopman(regressor=optdmd)
model_optdmd.fit(X)
'''

'''
import math
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple
import gym
env = gym.make('CartPole-v0')
koopEnv = gym.make('CartPole-v0')

Q_table = np.load('Q_table.npy')

n_bins = ( 6, 12 )
lower_bounds = [ env.observation_space.low[2], -math.radians(50) ]
upper_bounds = [ env.observation_space.high[2], math.radians(50) ]

def discretizer( _, __, angle, pole_velocity ) -> Tuple[int,...]:
    """Convert continuous state into a discrete state"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([ lower_bounds, upper_bounds ])
    return tuple( map( int, est.transform([[ angle, pole_velocity ]])[0] ) )

def policy(state: tuple):
    """ Choosing an action on epsilon-greedy policy """
    return np.argmax(Q_table[state])
'''

'''
current_state = discretizer(*env.reset())
current_stateK = discretizer(*koopEnv.reset())
action = policy(current_state)
actionK = policy(current_state)

q_learner_reward = 0
koopman_reward = 0

for i in range(num_steps):
    # environment details
    observation, reward, done, _ = env.step(action)
    observationK, rewardK, doneK, _ = koopEnv.step(actionK)

    # keep track of rewards
    q_learner_reward += reward
    koopman_reward += rewardK

    # discretize state - hoping generator won't have to!
    new_state = discretizer(*observation)
    new_stateK = discretizer(*observationK)

    # get actions
    next_action = policy(new_state)
    prediction = model_optdmd.predict(np.array([*list(current_stateK), actionK]))
    prediction = np.round(np.real(prediction))
    next_actionK = int(prediction[-1])

    # update environments
    action = next_action
    actionK = next_actionK
    current_state = new_state
    current_stateK = new_stateK

print("Q rewards:", q_learner_reward)
print("K rewards:", koopman_reward)
'''

'''
We can see that the rewards are both 200 which means that the Koopman predictor works very well given good data, though there are plenty of papers on the subject. One thing we may want to look into is how well we can learn a controller from the Koopman operator, but since we are focused on the Generator operator, what we really want to see now is the predictive power of the Koopman Generator and how it can be used for control!
'''

## Stochastic Koopman Generator Analysis
'''
We simulated some paths from a standard Brownian Motion (drift coefficient 0 and diffusion coefficient 1) and then tried 3 different methods from two papers: 
+ Klus et al 2020 <https://arxiv.org/pdf/1909.10638.pdf>
+ Li and Duan 2020 <https://arxiv.org/ftp/arxiv/papers/2005/2005.03769.pdf>
'''
### Simlulation of Brownian Data
'''
We simulated 20 paths of standard BM each with 5000 steps in a time interval of size 5000 so that the time step was 1. We took each of these 20 paths to be a state variable in our state vector. Our state vector is thus comprised of 20 iid BMs. Formally, our state variable dynamics have the form
$$
    \text{d}\tilde X_t = b\text{d}t + \Sigma\text{d}W_t
$$
where $b$ is a $n=20$ dimensional vector of 0s and $\Sigma$ is a $n\times n$ identity matrix.
'''
'''
from brownian import brownian

# The Diffusion process parameter.
sigma = 1
# Total time.
T = 20000.0
# Number of steps.
N = 20000
# Time step size
dt = T/N
# Number of realizations to generate.
m = 20
# Create an empty array to store the realizations.
X = np.empty((m, N+1))
# Initial values of x.
X[:, 0] = 50
brownian(X[:, 0], N, dt, sigma, out=X[:, 1:])
Z = np.roll(X,-1)[:, :-1]
X = X[:, :-1]
'''

### Fitting the BM data using Generator EDMD (gEDMD) from Klus et al. 2020
'''
From the learner's point of view, we assume that we are in the class of continuous Markov processes and thus that the generator is of the form
$$
    \mathcal{L}f = b\cdot\nabla_{\tilde x}f + \frac{1}{2}a:\nabla^2_{\tilde x}f 
    %= \sum_{i=1}^n b_i\frac{\partial f}{\partial \tilde{x}_i} + \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n a_{ij} \frac{\partial^2 f}{\partial \tilde{x}_i \partial \tilde{x}_j},
$$
where $a = \sigma \sigma^\top$, $\nabla^2_x$ denotes the Hessian, and $:$ denotes the double dot product. Applying the generator to each dictionary function \psi_k and assuming that we have access to a single ergodic sample with time step $dt= 1$, we can use the following finite difference estimator of $\mathcal{L}\psi_k$:
$$
\widehat{d\psi_k}(\tilde{\mathbf{x}}_l) = \frac{1}{t}(\tilde{\mathbf{x}}_{l+1} - \tilde{\mathbf{x}}_l) \cdot \nabla\psi_k(\tilde{\mathbf{x}}_l) + \frac{1}{2t} \Big[(\tilde{\mathbf{x}}_{l+1} - \tilde{\mathbf{x}}_l)(\tilde{\mathbf{x}}_{l+1} - \tilde{\mathbf{x}}_l)^\top\Big] : \nabla^2 \psi_k(\tilde{\mathbf{x}}_l)
$$
Note that we are adopting Klus's notation here only for reference. The stochastic total differential $d\psi_k$ is a different object that the generator of the Koopman operator they are related in that the drift of the stochastic total differential is the same thing as the generator.

Next, we set up the dictionary and the matrices

The idea behind generator EDMD is that we assume that the genertor applied to the the dictionary functions can be ("approximately") expressed as a linear combination of the dictionary functions and find the coeficients of those linear combinations by minimizing $|| \text{d}\Psi_{\tilde{\mathbf{X}}} - M\Psi_{\tilde{\mathbf{X}}} ||_F$ which leads to the least-squares approximation
$$ 
M = \text{d}\Psi_{\tilde{\mathbf{X}}} \Psi^{+}_{\tilde{\mathbf{X}}} = (\text{d}\Psi_{\tilde{\mathbf{X}}}\Psi_{\tilde{\mathbf{X}}}^\top)(\Psi_{\tilde{\mathbf{X}}}\Psi_{\tilde{\mathbf{X}}}^\top)^+ $$
$$
Thus, we obtain the empirical estimate $L=M^T$ of the Koopman generator $\mathcal{L}$.

For the dictionary space, we chose monomials of up to order 2. We also tried monomials of order 1, which is not sufficient to pick up the diffusion term, but was successful at picking up the drift quicker. We hypothesize that this is because there are fewer terms in the regression over the dictionary functions.

Note: we used Klus's d3 repo to set up the Psi
'''
'''
Put in code for
'''