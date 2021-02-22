import gym
import sys
import argparse
from rl_landers import *
import matplotlib.pyplot as plt
import numpy as np


AGENTS_LIST = ["random", "monte-carlo", "sarsa", "q-learning", "dqn"]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', '--list', nargs='+', help='algorithm(s) of choice to train lander', required=True)
    parser.add_argument('--n_episodes', type=int, help='number of training episodes', required=True)
    parser.add_argument('--lr', type=float, help='learning rate used in sarsa, q-learning', required=True)
    parser.add_argument('--gamma', type=float, help='discount factor', required=True)
    args = parser.parse_args()

    environment = gym.make("LunarLander-v2")
    chosen_agents = []
    agents_returns = []
    min_epsilon = 0.01

    for agent in args.agents:
        if agent == "random":
            print("\nTraining Random lander ...")
            total_rewards = random_lander(environment, args.n_episodes)
            print("Done!")

        elif agent == "monte-carlo":
            print("\nTraining Monte-Carlo lander ...")
            total_rewards = mc_lander(environment, args.n_episodes, args.gamma, min_epsilon)
            print("Done!")

        elif agent == "sarsa":
            print("\nTraining Sarsa lander ...")
            total_rewards  = sarsa_lander(environment, args.n_episodes, args.gamma, args.lr, min_epsilon)
            print("Done!")

        elif agent == "q-learning":
            print("\nTraining Q-learning lander ...")
            total_rewards = qlearning_lander(environment, args.n_episodes, args.gamma, args.lr, min_epsilon)
            print("Done!")

        elif agent == "dqn":
            print("\nTraining dqn lander ...")
            total_rewards = dqn_lander(environment, args.n_episodes, args.gamma, args.lr, min_epsilon)
            print("Done!")

        else:
            print("Invalid lander...\nValid options: {}".format(AGENTS_LIST), file=sys.stderr)
            sys.exit(-1)

        chosen_agents.append(agent)
        agents_returns.append(total_rewards)

    environment.close()

    n = int(args.n_episodes / 100)
    for agent, agent_total_returns in zip(chosen_agents, agents_returns):
        print("\n{} lander average reward = {}".format(agent, sum(agent_total_returns) / args.n_episodes))
        l = []
        for j in range(n):
            l.append(round(np.mean(agent_total_returns[j * 100 : (j + 1) * 100]), 1))
        plt.plot(range(int(args.n_episodes / 100)), l)


    plt.xlabel("Episodes")
    plt.ylabel("Reward per episode")
    plt.title("RL Lander(s)")
    plt.legend(chosen_agents, loc="lower right")
    plt.show()


if __name__ == '__main__':
    main()
