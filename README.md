# Introduction  
In this project you can find implementation of rl algorithms for solving the [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/) OpenAI Gym
environment. For the time being, there are implementations for:  
1. Monte-Carlo 
2. Sarsa
3. Q-learning
4. DQN

# Training Clips
These training snapshots are captured towards the end of the training phase (~10000 epsiodes). A random agent is also
provided for comparison:  

**Random**  
![random](data/random.gif)

**Monte-Carlo**  
![monte-carlo](data/monte_carlo.gif)  


**Sarsa**  
![sarsa](data/sarsa.gif)  


**Q-learning**  
![q-learning](data/qlearning.gif)


**DQN**
![dqn](data/dqn.gif)


# Experiments
**In progress**


# Execution
The purpose of this project is to compare the performance of each
reinforcement learning algorithm implemented. So, for a complete comparison,
meaning that all of the above agents are trained during program execution, run the
following:

```
python train.py --agents random sarsa q-learning dqn
```
this will train each agent separately using default values, which are
`n_episodes=10000`, `lr=0.001`, `gamma=0.99`, `final_eps=0.01`.  

For a more custom execution, you can explicitly provide the value for each
argument. For example, to only train a dqn agent:

```
python train.py --agents dqn --n_episodes 3000 --lr 0.0001 --gamma 0.99 --final_eps 0.02
```

After training a dqn agent, you can test how well it generalizes using:
```
python autopilot.py <num_episodes> models/*/qnetwork_{*}.pt
```  
To compare dqn to a random agent:
```
python random.py <num_episodes>
```
- Note: [LunarLander-v2/models/pretrained/qnetwork_2000.pt](models/pretrained/qnetwork_2000.pt) lands the lunar lander optimally 95% of the time


# Implementation References  
1. [OpenAI baselines](https://github.com/openai/baselines)
2. [Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
3. [Solving The Lunar Lander Problem under Uncertainty using Reinforcement Learning](https://arxiv.org/abs/2011.11850)
