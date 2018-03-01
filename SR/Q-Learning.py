from gridworld import GridworldEnv
import gym
import numpy as np
from collections import defaultdict

env = GridworldEnv()

EPSILON = 0.3
GAMMA = 0.99
LEARNING_RATE = 0.5
NUM_EPISODES = 20

# initialize Q
Q = defaultdict(lambda: np.zeros(env,action_space.n))

def greedy_policy(Q):

    def policy_fn(state):
        A = np.ones(env.action_space.n, dtype=float) * EPSILON / env.action_space.n
        best_action = np.argmax(Q[state])
        A[best_action] += (1.0 - EPSILON)
        return A
    return policy_fn

for i in range(NUM_EPISODES):

    state = env.reset()

    while True:

        action = policy(state)
        next_state, reward, done, _ = env.step(action)

        Q_target = reward + GAMMA * Q[next_state][np.argmax(Q[next_state])]
        Q_error = Q_target - Q[state][action]

        Q[state][action] += LEARNING_RATE * Q_error

        if done:
            break
