import gym
import sys
import numpy as np


def main():
    episodes = sys.argv[-1]
    env = gym.make('LunarLander-v2')

    for episode in range(int(episodes)):
        episode_reward = 0
        _, done = env.reset(), False

        while not done:
            env.render()
            action = env.action_space.sample()
            _, rew, done, _ = env.step(action)
            episode_reward += rew

        print(f"Episode reward: {episode_reward}")


if __name__ == '__main__':
    main()
