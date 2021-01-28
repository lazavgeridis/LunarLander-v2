import numpy as np
import random


def discretize_state(state):
    discrete_state = (min(2, max(-2, int((state[0]) / 0.05))), \
                        min(2, max(-2, int((state[1]) / 0.1))), \
                        min(2, max(-2, int((state[2]) / 0.1))), \
                        min(2, max(-2, int((state[3]) / 0.1))), \
                        min(2, max(-2, int((state[4]) / 0.1))), \
                        min(2, max(-2, int((state[5]) / 0.1))), \
                        int(state[6]), \
                        int(state[7]))

    return discrete_state


def epsilon_greedy(qstates_dict, state, eps, env_actions):
    prob = np.random.random()

    if prob < eps:
        return random.choice(range(env_actions))
    else:
        qvals = [qstates_dict[state + (action, )] for action in range(env_actions)]
        return np.argmax(qvals)


#def epsilon_greedy(q_network, state, eps, env_actions):
#    prob = np.random.random()
#
#    if prob < eps:
#        return random.choice(range(env_actions))
#    else:


def greedy(qstates_dict, state, env_actions):
    qvals = [qstates_dict[state + (action, )] for action in range(env_actions)]
    return max(qvals)


def discounted_return(episode_return, gamma):
    g = 0
    for i, r in enumerate(episode_return):
        g += gamma**i * r

    return g


def decay_epsilon(curr_eps, exploration_final_eps):
    if curr_eps < exploration_final_eps:
        return curr_eps
    
    return curr_eps * 0.996
