# Introduction  
Implementing various rl algorithms for solving the [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/) OpenAI Gym
environment. For the time being, there are implementations for:  
1. Monte-Carlo 
2. Sarsa
3. Q-learning
4. DQN

# Training Clips
Below, you can find some training clips for each agent. These are captured
towards the end of the training phase (~10000 epsiodes). A random agent is also
provided for comparison:  

**Random**

**Monte-Carlo**  
![monte-carlo](data/monte_carlo.gif)  


**Sarsa**  
![sarsa](data/sarsa.gif)  


**Q-learning**  
![q-learning](data/qlearning.gif)


**DQN**


# Execution
The purpose of this project is to compare the effectiveness of each
reinforcement learning algorithm implemented. So, for a complete comparison,
meaning that all of the above agents are trained during program execution, run the
following:

```
python train.py --agents random sarsa q-learning dqn
```
this will train each agent separately with the default values, which are 


# Implementation References  
1. [OpenAI baselines](https://github.com/openai/baselines)
2. [Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
3. [Solving The Lunar Lander Problem under Uncertainty using Reinforcement Learning](https://arxiv.org/abs/2011.11850)
