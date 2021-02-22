import numpy as np
import gym
import torch
import cv2
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
        qvals = [qstates_dict[state + (action, )] for action in range(env_actions)]
        return np.argmax(qvals)


# epsilon_greedy function used specifically in dqn agent
def epsilon_greedy(qnetwork, frame, eps, env_actions, device):
    prob = np.random.random()

    if prob < eps:
        return torch.tensor([[random.randrange(env_actions)]], device=device, dtype=torch.long)
    else:
        return qnetwork(frame).max(1)[1].view(1, 1)


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


def get_frame(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW) i.e (3, 400, 600)
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    frame = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = np.expand_dims(frame, -1)   # -> shape is (84, 84, 1)
    frame = frame.transpose((2, 0, 1))
 
    ## Convert to float, rescale, convert to torch tensor
    ## (this doesn't require a copy)
    frame = np.ascontiguousarray(screen, dtype=np.float32) / 255
    frame = torch.from_numpy(frame)
    ## Resize, and add a batch dimension (BCHW)
    #return resize(screen).unsqueeze(0).to(device)

    return frame


def build_qnetwork(env_actions, learning_rate):
    qnet = DQN(env_actions)
    return qnet, torch.optim.RMSprop(qnet.parameters(), lr=learning_rate)


def fit(qnet, qnet_optim, qtarget_net, loss_func, 
        frames, actions, rewards, next_frames, dones, 
        gamma, env_actions):
    q_t = qnet(frames)
    q_t_selected = torch.sum(q_t * torch.nn.functional.one_hot(actions, env_actions), 1)

    q_tp1 = qtarget_net(next_frames).detach()
    q_tp1_best = torch.max(q_tp1, 1)
    q_tp1_best = (torch.ones(dones.size(-1)) - dones) * q_tp1_best
    q_targets = rewards + gamma * q_tp1_best

    loss = loss_func(q_targets, q_t_selected)
    qnet_optim.zero_grad()
    loss.backward()

    # update q-network's weights
    qnet_optim.step()


def update_target_network(qnet, qtarget_net):
    qtarget_net.load_state_dict(qnet.state_dict())
