"""
Environment: LunarLander-v2
Reward for moving from the top of the screen to the landing pad and zero speed is about 100..140 points.
If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or
comes to rest, receiving an additional -100 or +100 points. Each leg with ground contact is +10 points.
Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame.
Solved is 200 points.

> https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py <
"""

import gym
import torch
import collections
import os
import numpy as np
from utils import *
from exp_replay_memory import *


"""
render_freq: int
    render the environment every 'render_freq' episodes
"""

def random_lander(env, n_episodes, print_freq=20, render_freq=20):
    return_per_ep = [0.0]

    for i in range(n_episodes):
        state = env.reset()
        t = 0
        if (i + 1) % render_freq == 0:
            render = True
        else:
            render = False
    
        while True:
            if render:
                env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            return_per_ep[-1] += reward
    
            if done:
                if (i + 1) % print_freq == 0:
                    print("Episode finished after {} timesteps".format(t + 1))
                    print("Episode {}: Total return {}\n".format(i + 1, return_per_ep[-1]))
                return_per_ep.append(0.0)

                break
    
            state = observation
            t += 1

    return return_per_ep


def mc_lander(env, n_episodes, gamma, min_eps, print_freq=20, render_freq=20):
    q_states = collections.defaultdict(float)   # note that the first insertion of a key initializes its value to 0.0
    n_visits = collections.defaultdict(int)     # note that the first insertion of a key initializes its value to 0
    return_per_ep = [0.0]
    episode_qstates = []
    episode_return = []
    epsilon = 1.0
    num_actions = env.action_space.n

    for i in range(n_episodes):
        total_return = 0
        t = 0
        curr_state = discretize_state(env.reset())
        if (i + 1) % render_freq == 0:
            render = True
        else:
            render = False
        
        while True:
            if render:
                env.render()

            # choose action A using ε-greedy policy
            action = epsilon_greedy(q_states, curr_state, epsilon, num_actions)
    
            # take action A, earn immediate reward and land into next state S'
            observation, reward, done, info = env.step(action)
    
            qstate = curr_state + (action, )
            episode_qstates.append(qstate)
    
            # increment visit count = N(state, action)
            n_visits[qstate] += 1
    
            return_per_ep[-1] += reward
            episode_return.append(reward)
    
            if done:
                if (i + 1) % print_freq == 0:
                    print("Episode finished after {} timesteps".format(t+1))
                    print("Episode {}: Total return {}".format(i + 1, return_per_ep[-1]))
                    print("Total keys in q_states dictionary = {}".format(len(q_states)))
                    print("Total keys in n_visits dictionary = {}\n".format(len(n_visits)))
    
                # improve policy only when episode is completed
                # policy evaluation step
                for step, qstate in enumerate(episode_qstates):
                    q_states[qstate] += (discounted_return(episode_return[step: ], gamma) - q_states[qstate]) / n_visits[qstate]

                epsilon = decay_epsilon(epsilon, min_eps)
                return_per_ep.append(0.0)
                episode_qstates.clear()
                episode_return.clear()
    
                break
    
            curr_state = discretize_state(observation)
            t += 1

    return return_per_ep


def sarsa_lander(env, n_episodes, gamma, lr, min_eps, print_freq=20, render_freq=20):
    q_states = collections.defaultdict(float)   # note that the first insertion of a key initializes its value to 0.0
    return_per_ep = [0.0]
    epsilon = 1.0
    num_actions = env.action_space.n
    
    for i in range(n_episodes):
        t = 0
        if (i + 1) % render_freq == 0:
            render = True
        else:
            render = False

        # Current State: S
        # Choose A using policy π
        curr_state = discretize_state(env.reset())
        action = epsilon_greedy(q_states, curr_state, epsilon, num_actions)
        
        while True:
            if render:
                env.render()

            # Create (S, A) pair
            qstate = curr_state + (action, )

            # Take action A, earn immediate reward R and land into next state S'
            # S --> A --> R --> S'
            observation, reward, done, info = env.step(action)
            next_state = discretize_state(observation)

            # Next State: S'
            # Choose A' using policy π
            next_action = epsilon_greedy(q_states, next_state, epsilon,
                    num_actions)

            # create (S', A') pair
            new_qstate = next_state + (next_action, )

            ###################################################################
            # Policy evaluation step
            if not done:
                q_states[qstate] += lr * (reward + gamma * q_states[new_qstate] - q_states[qstate]) # (S', A') non terminal state
            else:
                q_states[qstate] += lr * (reward - q_states[qstate])    # (S', A') terminal state
            ###################################################################

            return_per_ep[-1] += reward

            if done:
                if (i + 1) % print_freq == 0:
                    print("Episode finished after {} timesteps".format(t + 1))
                    print("Episode {}: Total Return = {}".format(i + 1, return_per_ep[-1]))
                    print("Total keys in q_states dictionary = {}\n".format(len(q_states)))

                epsilon = decay_epsilon(epsilon, min_eps)
                return_per_ep.append(0.0)

                break

            curr_state = next_state
            action = next_action
            t += 1

    return return_per_ep


def qlearning_lander(env, n_episodes, gamma, lr, min_eps, print_freq=20, render_freq=20):
    q_states = collections.defaultdict(float)   # note that the first insertion of a key initializes its value to 0.0
    return_per_ep = [0.0]
    epsilon = 1.0
    num_actions = env.action_space.n
    
    for i in range(n_episodes):
        t = 0
        if (i + 1) % render_freq == 0:
            render = True
        else:
            render = False

        # Current State: S
        curr_state = discretize_state(env.reset())
        
        while True:
            if render:
                env.render()

            # choose action A using behaviour policy -> ε-greedy
            action = epsilon_greedy(q_states, curr_state, epsilon, num_actions)

            # Create (S, A) pair
            qstate = curr_state + (action, )

            # Take action A, earn immediate reward R and land into next state S'
            # S --> A --> R --> S'
            observation, reward, done, info = env.step(action)
            next_state = discretize_state(observation)

            ###################################################################
            # Policy evaluation step
            if not done:
                q_states[qstate] += lr * (reward + gamma * greedy(q_states, next_state, num_actions) - q_states[qstate]) # (S', A') non terminal state
            else:
                q_states[qstate] += lr * (reward - q_states[qstate])    # (S', A') terminal state
            ###################################################################

            return_per_ep[-1] += reward

            if done:
                if (i + 1) % print_freq == 0:
                    print("Episode finished after {} timesteps".format(t + 1))
                    print("Episode {}: Total Return = {}".format(i + 1, return_per_ep[-1]))
                    print("Total keys in q_states dictionary = {}\n".format(len(q_states)))

                epsilon = decay_epsilon(epsilon, min_eps)
                return_per_ep.append(0.0)

                break

            curr_state = next_state
            t += 1

    return return_per_ep


def dqn_lander(env, n_episodes, gamma, lr, min_eps, \
                memory_capacity=50000, train_freq=1, batch_size=32, \
                learning_starts=1000, target_network_update_freq=500, \
                print_freq=20, checkpoint_freq=10000):
    """
    print_freq: int
        how often to print out training progress
        if None disable printing

    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored 
        at the end of the training. If you do not wish to restore the best version at 
        the end of the training, set this variable to None.

    """

    # set device to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_function = torch.nn.MSELoss(reduction='sum') # or Huber loss

    # path to save checkpoints
    PATH = "./checkpoints"
    os.mkdir(PATH)
    PATH = os.path.join(PATH, "ckeckpoint.pt")

    num_actions = env.action_space.n

    # set up the 2 q-networks, their optimizers and replay memory
    # if there exists a saved checkpoint, load it up and resume training 
    qnet, qnet_optim = build_qnetwork(num_actions, lr)
    qtarget_net, _ = build_qnetwork(num_actions, lr)
    qtarget_net.load_state_dict(qnet.state_dict())
    start_episode = start_from_checkpoint(qnet, qnet_optim, qtarget_net, PATH)

    qnet.to(device).train()
    qtarget_net.to(device).eval()
    replay_memory = ReplayMemory(memory_capacity)

    epsilon = 1.0
    return_per_ep = [0.0]
    saved_mean_reward = None
    t = 0

    for i in range(start_episode, n_episodes, 1):
        env.reset()
        curr_frame = get_frame(env, device)

        while True:
            # choose action A using behaviour policy -> ε-greedy; use q-network
            action = epsilon_greedy(qnet, curr_frame, epsilon, num_actions)
            # take action A, earn immediate reward R and land into next state S'
            _, reward, done, _ = env.step(action)

            next_frame = get_frame(env, device)

            # store transition (S, A, R, S', Done) in replay memory
            replay_memory.store(curr_frame, action, float(reward), next_frame, float(done))

            return_per_ep[-1] += reward

            # if replay memory currently stores > 'learning_starts' transitions, sample a random mini-batch and update q_network's parameters
            if t > learning_starts and t % train_freq == 0:
                frames, actions, rewards, next_frames, dones = replay_memory.sample_minibatch(batch_size)

                #loss = 
                fit(qnet, \
                    qnet_optim, \
                    qtarget_net, \
                    loss_function, \
                    frames, \
                    actions, \
                    rewards, \
                    next_frames, \
                    dones, \
                    gamma, \
                    num_actions, 
                    device)

            # periodically update q-target network's parameters
            if t > learning_starts and t % target_network_update_freq == 0:
                update_target_network(qnet, qtarget_net)

            mean_100ep_reward = round(np.mean(return_per_ep[-101:-1]), 1)

            if done and print_freq is not None and (i + 1) % print_freq == 0:
                print("\nTime-steps: ", t)
                print("Episodes: ", i + 1)
                print("Mean 100 episode reward: ", mean_100ep_reward)

            if (checkpoint_freq is not None and t > learning_starts and i + 1 > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        print("\nSaving checkpoint due to mean reward increase: {} -> {}".format(saved_mean_reward, mean_100ep_reward))
                    save_checkpoint(qnet, qnet_optim, qtarget_net, i, PATH)
                    saved_mean_reward = mean_100ep_reward

            t += 1
            if done:
                return_per_ep.append(0.0)
                epsilon = decay_epsilon(epsilon, min_eps)
                break

            curr_frame = next_frame

    return return_per_ep
