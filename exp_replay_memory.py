import numpy as np
import random

class ReplayMemory(object):
    def __init__(self, capacity):
        self.transitions = []
        self.max_capacity = capacity
        self.next_transition_index = 0


    def length(self):
        return len(self.transitions)


    def store(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        if self.next_transition_index >= self.length():
            self.transitions.append(transition)
        else:
            self.transitions[self.next_transition_index] = transition   # overwrite old experiences

        self.next_transition_index = (self.next_transition_index + 1) % self.max_capacity


    def sample_minibatch(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for _ in range(batch_size):
            transition_index = random.randint(0, self.length() - 1)
            transition = self.transitions[transition_index]
            state, action, reward, next_state, done = transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return states, actions, rewards, next_states, dones
