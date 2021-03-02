import gym
import sys
import argparse
from rl_landers import *
import numpy as np


AGENTS_LIST = ["random", "monte-carlo", "sarsa", "q-learning", "dqn"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', nargs='+', help='algorithm(s) of choice to train lander', choices=AGENTS_LIST, required=True)
    parser.add_argument('--n_episodes', type=int, help='number of training episodes', default=10000, required=False)                            # default number of episodes is 10000
    parser.add_argument('--lr', type=float, help='step-size (or learning rate) used in sarsa, q-learning, dqn', default=1e-3, required=False)   # default step-size is 0.001
    parser.add_argument('--gamma', type=float, help='discount rate, should be 0 < gamma < 1', default=0.99, required=False)                     # default gamma is 0.99
    parser.add_argument('--final_eps', type=float, help='decay epsilon unti it reaches its \'final_eps\' value', default=1e-2, required=False)  # default final eploration epsilon is 0.01
    args = parser.parse_args()

    environment = gym.make("LunarLander-v2")
    chosen_agents = []
    agents_returns = []

    for agent in args.agents:
        if agent == "random":
            print("\nTraining Random lander ...")
            total_rewards = random_lander(environment, args.n_episodes)
            print("Done!")

        elif agent == "monte-carlo":
            print("\nTraining Monte-Carlo lander with arguments num_episodes={}, gamma={}, final_epsilon={} ..."\
                            .format(args.n_episodes, args.gamma, args.final_eps))
            total_rewards = mc_lander(environment, args.n_episodes, args.gamma, args.final_eps)
            print("Done!")

        elif agent == "sarsa":
            print("\nTraining Sarsa lander with arguments num_episodes={}, step-size={}, gamma={}, final_epsilon={} ..."\
                            .format(args.n_episodes, args.lr, args.gamma, args.final_eps))
            total_rewards  = sarsa_lander(environment, args.n_episodes, args.gamma, args.lr, args.final_eps)
            print("Done!")

        elif agent == "q-learning":
            print("\nTraining Q-learning lander with arguments num_episodes={}, step-size={}, gamma={}, final_epsilon={} ..."\
                            .format(args.n_episodes, args.lr, args.gamma, args.final_eps))
            total_rewards = qlearning_lander(environment, args.n_episodes, args.gamma, args.lr, args.final_eps)
            print("Done!")

        elif agent == "dqn":
            print("\nTraining DQN lander with arguments num_episodes={}, learning rate={}, gamma={}, final_epsilon={} ..."\
                            .format(args.n_episodes, args.lr, args.gamma, args.final_eps))
            total_rewards = dqn_lander(environment, args.n_episodes, args.gamma, args.lr, args.final_eps)
            print("Done!")

        chosen_agents.append(agent)
        agents_returns.append(total_rewards)

    environment.close()

    # plot rewards per 'win' episodes for each agent
    win = 100
    plot_rewards(chosen_agents, agents_returns, args.n_episodes, win)


if __name__ == '__main__':
    main()
