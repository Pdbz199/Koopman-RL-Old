import gym
import numpy as np
env = gym.make("Taxi-v3")

episodes = 100
states = []
actions = []
for episode in range(episodes):
    state = env.reset()

    # taxi_coords = (taxi_y, taxi_x)
    # passenger_coords = env.locs[passenger]
    # destination_coords = env.locs[destination]
    # readable_state = [taxi_coords, passenger_coords, destination_coords]

    # print("State:", taxi_y, taxi_x, passenger, destination)
    # print("Taxi coords:", taxi_coords)
    # print("Passenger coords:", passenger_coords)
    # print("Destination coords:", destination_coords)
    # print(readable_state)

    env.render()
    done = False
    while not done:
        # env.render()

        taxi_y,taxi_x,passenger,destination = env.decode(state)
        decoded_state = [taxi_y, taxi_x, passenger, destination]
        states.append(decoded_state)

        action = env.action_space.sample()
        actions.append(action)

        state, reward, done, _ = env.step(action)

env.close()

states = np.array(states)
actions = np.array(actions)

# print(states.shape)
# print(actions.shape)

np.save('random-agent/taxi/states', states)
np.save('random-agent/taxi/actions', actions)