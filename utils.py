import numpy as np
import torch
import random
from deepq_network import *


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
        #if not isinstance(qstates_dict, DQN):
        qvals = [qstates_dict[state + (action, )] for action in range(env_actions)]
        return np.argmax(qvals)
        #else:
        #    qvals = qstates_dict(state)
        #    return torch.argmax(qvals)


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


#def build_qnetwork(env_actions):
#    learning_rate = 5e-4
#    qnet = DQN(env_actions)
#    return qnet, torch.optim.RMSprop(qnet.parameters(), lr=learning_rate)
#
#
#def fit(qnet, qnet_optim, qtarget_net, loss_func, 
#        states, actions, rewards, next_states, dones, 
#        gamma, env_actions):
#    # do i have to move network inputs to device???
#    q_t = qnet(states)
#    q_t_selected = q_t * torch.nn.functional.one_hot(actions, env_actions)
#
#    q_tp1 = qtarget_net(next_states)
#    q_tp1_best = torch.max(q_tp1)
#    q_tp1_best = (1.0 - dones) * q_tp1_best
#    q_targets = rewards + gamma * q_tp1_best
#
#    td_error = q_t_selected - q_targets
#
#    loss = loss_func(q_targets, q_t_selected)
#    qnet_optim.zero_grad()
#    loss.backward()
#    # do gradient buffers in qtarget_net parameters accumulate anything???
#    #for param in qtarget_net.parameters():
#    #   print(param.grad)
#
#    # update q-network's weights
#    qnet_optim.step()
#
#    return td_error
#
#
#def update_target_network(qnet, qtarget_net):
#    qtarget_net.load_state_dict(qnet.state_dict())
