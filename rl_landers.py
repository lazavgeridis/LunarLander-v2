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
from utils import *
from exp_replay_memory import *


def random_lander(env, n_episodes):
    returns_per_ep = []

    for i in range(n_episodes):
        state = env.reset()
        total_return = 0
        t = 0
    
        while True:
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            total_return += reward
    
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                print("Episode {}: Total return {}\n".format(i + 1, total_return))
                returns_per_ep.append(total_return)

                break
    
            state = observation
            t += 1

    return returns_per_ep


def mc_lander(env, n_episodes, gamma, min_eps):
    q_states = collections.defaultdict(float)   # note that the first insertion of a key initializes its value to 0.0
    n_visits = collections.defaultdict(int)     # note that the first insertion of a key initializes its value to 0
    returns_per_ep = []
    episode_qstates = []
    episode_return = []
    epsilon = 1.0

    for i in range(n_episodes):
        total_return = 0
        t = 0
        curr_state = discretize_state(env.reset())
        
        while True:
            env.render()
    
            # choose action A using ε-greedy policy
            action = epsilon_greedy(q_states, curr_state, epsilon, env.action_space.n)
    
            # take action A, earn immediate reward and land into next state S'
            observation, reward, done, info = env.step(action)
    
            qstate = curr_state + (action, )
            episode_qstates.append(qstate)
    
            # increment visit count = N(state, action)
            n_visits[qstate] += 1
    
            total_return += reward
            episode_return.append(reward)
    
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                print("Episode {}: Total return {}".format(i + 1, total_return))
                print("Total keys in q_states dictionary = {}".format(len(q_states)))
                print("Total keys in n_visits dictionary = {}\n".format(len(n_visits)))
    
                # improve policy only when episode is completed
                # policy evaluation step
                for step, qstate in enumerate(episode_qstates):
                    q_states[qstate] += (discounted_return(episode_return[step: ], gamma) - q_states[qstate]) / n_visits[qstate]

                epsilon = decay_epsilon(epsilon, min_eps)
                returns_per_ep.append(total_return)
                episode_qstates.clear()
                episode_return.clear()
    
                break
    
            curr_state = discretize_state(observation)
            t += 1

    return returns_per_ep


def sarsa_lander(env, n_episodes, gamma, lr, min_eps):
    q_states = collections.defaultdict(float)   # note that the first insertion of a key initializes its value to 0.0
    returns_per_ep = []
    epsilon = 1.0
    
    for i in range(n_episodes):
        total_return = 0
        t = 0

        # Current State: S
        # Choose A using policy π
        curr_state = discretize_state(env.reset())
        action = epsilon_greedy(q_states, curr_state, epsilon, env.action_space.n)
        
        while True:
            env.render()

            # Create (S, A) pair
            qstate = curr_state + (action, )

            # Take action A, earn immediate reward R and land into next state S'
            # S --> A --> R --> S'
            observation, reward, done, info = env.step(action)
            next_state = discretize_state(observation)

            # Next State: S'
            # Choose A' using policy π
            next_action = epsilon_greedy(q_states, next_state, epsilon, env.action_space.n)

            # create (S', A') pair
            new_qstate = next_state + (next_action, )

            ###################################################################
            # Policy evaluation step
            if not done:
                q_states[qstate] += lr * (reward + gamma * q_states[new_qstate] - q_states[qstate]) # (S', A') non terminal state
            else:
                q_states[qstate] += lr * (reward - q_states[qstate])    # (S', A') terminal state
            ###################################################################

            total_return += reward

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                print("Episode {}: Total Return = {}".format(i + 1, total_return))
                print("Total keys in q_states dictionary = {}\n".format(len(q_states)))

                epsilon = decay_epsilon(epsilon, min_eps)
                returns_per_ep.append(total_return)

                break

            curr_state = next_state
            action = next_action
            t += 1

    return returns_per_ep


def qlearning_lander(env, n_episodes, gamma, lr, min_eps):
    q_states = collections.defaultdict(float)   # note that the first insertion of a key initializes its value to 0.0
    returns_per_ep = []
    epsilon = 1.0
    
    for i in range(n_episodes):
        total_return = 0
        t = 0

        # Current State: S
        curr_state = discretize_state(env.reset())
        
        while True:
            env.render()

            # choose action A using behaviour policy -> ε-greedy
            action = epsilon_greedy(q_states, curr_state, epsilon, env.action_space.n)

            # Create (S, A) pair
            qstate = curr_state + (action, )

            # Take action A, earn immediate reward R and land into next state S'
            # S --> A --> R --> S'
            observation, reward, done, info = env.step(action)
            next_state = discretize_state(observation)

            ###################################################################
            # Policy evaluation step
            if not done:
                q_states[qstate] += lr * (reward + gamma * greedy(q_states, next_state, env.action_space.n) - q_states[qstate]) # (S', A') non terminal state
            else:
                q_states[qstate] += lr * (reward - q_states[qstate])    # (S', A') terminal state
            ###################################################################

            total_return += reward

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                print("Episode {}: Total Return = {}".format(i + 1, total_return))
                print("Total keys in q_states dictionary = {}\n".format(len(q_states)))

                epsilon = decay_epsilon(epsilon, min_eps)
                returns_per_ep.append(total_return)

                break

            curr_state = next_state
            t += 1

    return returns_per_ep


def dqn_lander(env, total_timesteps, gamma, lr, min_eps, memory_capacity=50000, train_freq=1, batch_size=32, learning_starts=1000, target_network_update_freq=500):
    # decide which device we want to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_function = torch.nn.MSELoss(reduction='sum')

    # set up the 2 q-networks, their optimizers and replay memory
    qnet, qnet_optim = build_qnetwork(env.action_space.n)
    qnet.to(device)
    #qnet.train()
    qtarget_net, qtarget_net_optim = build_qnetwork(env.action_space.n)
    qtarget_net.to(device)
    #qtarget_net.eval()
    replay_memory = ReplayMemory(memory_capacity)

    epsilon = 1.0
    return_per_ep = [0.0]

    curr_state = env.reset()

    for t in range(total_timesteps):

        # choose action A using behaviour policy -> ε-greedy
        # using q_network
        action = epsilon_greedy(q_net, curr_state, epsilon, env.action_space.n)
        # take action A, earn immediate reward R and land into next state S'
        next_state, reward, done, _ = env.step(action)
        # store transition (S, A, R, S', Done) in replay memory
        replay_memory.store(curr_state, action, reward, next_state, float(done))

        curr_state = next_state
        return_per_ep[-1] += reward

        if done:
            curr_state = env.reset()
            return_per_ep.append(0.0)
            epsilon = decay_epsilon(epsilon, min_eps)

        # if replay memory currently stores > 'learning_starts' transitions,
        # sample a random mini-batch and update q_network's (probably) parameters
        if t > learning_starts and t % train_freq == 0:
            states, actions, rewards, next_states, dones = replay_memory.sample_minibatch(batch_size)

            td_errors = fit(qnet, \
                            qnet_optim, \
                            qtarget_net, \
                            qtarget_net_optim, \
                            loss_function, \
                            states, \
                            actions, \
                            rewards, \
                            next_states, \
                            dones, \
                            gamma, \
                            env.action_space.n)

        # periodically update q-target network's parameters
        if t > learning_starts and t % target_network_update_freq == 0:
            update_target_network(qnet, qtarget_net)

    return return_per_ep
