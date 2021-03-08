import gym
import torch
import cv2
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from deepq_network import CNN, LinearMapNet


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


def epsilon_greedy(q_func, state, eps, env_actions):
    prob = np.random.random()

    if prob < eps:
        return random.choice(range(env_actions))
    elif isinstance(q_func, CNN) or isinstance(q_func, LinearMapNet):
        with torch.no_grad():
            return q_func(state).max(1)[1].item()
    else:
        qvals = [q_func[state + (action, )] for action in range(env_actions)]
        return np.argmax(qvals)


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
    # Returned screen requested by gym is 400x600x3, but is sometimes larger such as 800x1200x3
    # in general env.render(mode='rgb_array') returns a numpy.ndarray with shape (x, y, 3)
    screen = env.render(mode='rgb_array')
    frame = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = np.expand_dims(frame, -1)   # convert into shape (84, 84, 1)
    frame = frame.transpose((2, 0, 1))  # convert into torch shape (C, H, W) -> (1, 84, 84)
 
    # Convert to float, rescale, convert to torch tensor (this doesn't require a copy)
    #frame = frame.astype(np.float)
    frame = np.ascontiguousarray(frame, dtype=np.float32) / 255
    frame = torch.from_numpy(frame)

    # Add a batch dimension -> (B, C, H, W)
    return frame.unsqueeze(0)


def lmn_input(obs):
    net_input = np.expand_dims(obs, 0)
    net_input = torch.from_numpy(net_input)

    return net_input


def build_qnetwork(env_actions, learning_rate, input_shape, network, device):
    if network == 'cnn':
        qnet = CNN(env_actions)
    else:
        # model = 'linear'
        qnet = LinearMapNet(input_shape, env_actions)
    return qnet.to(device), torch.optim.RMSprop(qnet.parameters(), lr=learning_rate)


def fit(qnet, qnet_optim, qtarget_net, loss_func, \
        frames, actions, rewards, next_frames, dones, \
        gamma, env_actions, device):

    # compute action-value for frames at timestep t using q-network
    frames_t = torch.cat(frames).to(device)
    actions = torch.tensor(actions, device=device)
    q_t = qnet(frames_t) # q_t tensor has shape (batch, env_actions)
    q_t_selected = torch.sum(q_t * torch.nn.functional.one_hot(actions, env_actions), 1) 

    # compute td targets for frames at timestep t + 1 using q-target network
    dones = torch.tensor(dones, device=device)
    rewards = torch.tensor(rewards, device=device)
    frames_tp1 = torch.cat(next_frames).to(device)
    q_tp1_best = qtarget_net(frames_tp1).max(1)[0].detach() 
    ones = torch.ones(dones.size(-1), device=device)
    q_tp1_best = (ones - dones) * q_tp1_best
    q_targets = rewards + gamma * q_tp1_best

    # td error
    loss = loss_func(q_t_selected, q_targets)
    qnet_optim.zero_grad()
    loss.backward()
    qnet_optim.step()
    #return loss.item()


def update_target_network(qnet, qtarget_net):
    qtarget_net.load_state_dict(qnet.state_dict())


def save_model(qnet, episode, path):
    torch.save(qnet.state_dict(), os.path.join(path, 'qnetwork_{}.pt'.format(episode)))


def plot_rewards(chosen_agents, agents_returns, num_episodes, window):
    num_intervals = int(num_episodes / window)
    for agent, agent_total_returns in zip(chosen_agents, agents_returns):
        print(len(agent_total_returns))
        print("\n{} lander average reward = {}".format(agent, sum(agent_total_returns) / num_episodes))
        l = []
        for j in range(num_intervals):
            l.append(round(np.mean(agent_total_returns[j * 100 : (j + 1) * 100]), 1))
        plt.plot(range(0, num_episodes, window), l)

    plt.xlabel("Episodes")
    plt.ylabel("Reward per {} episodes".format(window))
    plt.title("RL Lander(s)")
    plt.legend(chosen_agents, loc="lower right")
    plt.show()
