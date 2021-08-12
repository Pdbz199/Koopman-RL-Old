import numpy as np

# you can pass functions as parameters to jitted functions IF AND ONLY IF they re also jitted

def rho(u, o='unif', a=0 ,b=1):
        if o == 'unif':
            return 1 / ( b - a )

        if o == 'normal':
            return np.exp( -u**2 / 2 ) / ( np.sqrt( 2 * np.pi ) )

def K_u(K, psi_u):
    return np.einsum('ijz,z->ij', K, psi_u)

def algorithm2(X, U, phi, psi, K_hat, cost, learning_rate=0.1, epsilon=1):
    num_lifted_state_features = K_hat.shape[0]
    num_lifted_action_features = K_hat.shape[2]

    w = np.empty(num_lifted_state_features)
    w.fill(1)

    def pi(u, x):
        return np.exp(-1 * (cost(x, u) + w @ K_u(K_hat, psi(u)) @ phi(x)))

    def bellmanError(w, pi):
        total = 0
        for i in range(X.shape[1]):
            phi_x = phi(x1)
            expectation_u = 0
            for u in U: # U is a collection of ALL possible actions as row vectors
                expectation_u += ( cost(x1, u) - np.log( pi(u, x1) ) - w @ K_u(K_hat, psi(u)) @ phi_x ) * pi(u, x1)
            total += np.power(( w @ phi_x - expectation_u ), 2)
        return total

    BE = bellmanError(w, pi)
    while BE > epsilon:
        # These are row vectors
        u1 = U[:, int(np.random.uniform(0, num_lifted_action_features))]
        u2 = U[:, int(np.random.uniform(0, num_lifted_action_features))]
        x1 = X[:, int(np.random.uniform(0, X.shape[1]))]

        phi_x1 = phi(x1)
        nabla_w = ( w @ phi_x1 \
            - ( ( pi(u1, x1) / rho(u1, b=num_lifted_action_features) ) * ( cost(x1, u1) + np.log( pi(u1, x1) ) + w @ K_u(K_hat, psi(u1)) @ phi_x1 ) ) ) \
            * ( phi_x1 - ( pi(u2, x1) / rho(u2, b=num_lifted_action_features) ) * K_u(K_hat, psi(u2)) @ phi_x1 )
        w -= learning_rate * nabla_w

        BE = bellmanError(w, pi)
        print(BE) # not decreasing at all (value is around 19000)

    return pi