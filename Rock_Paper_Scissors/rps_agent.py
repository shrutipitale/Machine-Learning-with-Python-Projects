''' Defining states in a Q-learning problem involves determining the information that 
the agent uses to make decisions. Some state representations are:
- previous agents moves only (3 states)
- previous opponents moves only (3 states)
- previous agents and opponents moves pair (9 states)
- previous n times moves pair of agent and opponent (num_actions**(2*n) states)
'''

import numpy as np
import random
import itertools

class QLearning:
    def __init__(self, actions = ['R', 'P', 'S'], learning_rate = 0.25, discount_factor = 0.8, exploration_rate = 1):
        self.actions = actions
        self.num_actions = len(actions)
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.decay_rate = 0.0001
        self.Q = np.zeros((self.num_actions**2, self.num_actions))
        self.state_map = self.generate_map()
        self.last_action = ''
        self.last_state = None
        self.step = 0

    def generate_map(self):
        # A. in the form of tuples [('R', 'R'), ('R', 'P'), ...]
        all_states = list(itertools.product(self.actions, repeat = 2))
        return {state: i for i, state in enumerate(all_states)}
        # B. in the form of strings ['RR', 'RP', ...]
        # all_states = [''.join(i) for i in itertools.product(self.actions, repeat = 2)]
        # return {state: i for i, state in enumerate(all_states)}
    
    def get_action(self, state):
        # Exploration: choose a random action
        if np.random.rand() < self.epsilon:
            # print('random action')
            return random.choice(self.actions)
        # Exploitation: choose the action with the highest Q-value
        elif state in self.state_map.keys():
            # print(state)
            state = self.state_map[state]
            return self.actions[np.argmax(self.Q[state])]
    
    def update_q_value(self, state, action, reward, next_state):
        state = self.state_map[state]
        next_state = self.state_map[next_state]
        action = self.actions.index(action)
        # Q-value update using the Q-learning formula
        self.Q[state][action] += self.alpha * (
            reward + self.gamma * max(self.Q[next_state]) - self.Q[state][action] )
        
def determine_reward(agent_action, opponent_action):
    global results
    if (agent_action == "R" and opponent_action == "S") or (
        agent_action == "P" and opponent_action == "R") or (
        agent_action == "S" and opponent_action == "P"):
        return 1    # Agent wins
    elif agent_action == opponent_action:
        return 0    # It's a draw    
    else:
        return -1   # Agent loses
    
def set_agent(q_agent):
    global agent
    agent = q_agent
    
def player(prev_play, opponent_history = []):
    global agent
    
    if agent.last_action is None or prev_play == '':
        action = random.choice(agent.actions)
        agent.last_action = action
        agent.step += 1
        return action
        
    # A. in the form of tuples [('R', 'R'), ('R', 'P'), ...]
    state = (agent.last_action, prev_play) 
    # B. in the form of strings ['RR', 'RP', ...]
    # state = f'{agent.last_action}{prev_play}' 

    action = agent.get_action(state)
    agent.step += 1

    if agent.last_state in agent.state_map.keys():
        reward = determine_reward(agent.last_action, prev_play)
        agent.update_q_value(agent.last_state, agent.last_action, reward, state)
    
    agent.last_action = action
    agent.last_state = state

    agent.epsilon *= np.exp(-agent.decay_rate*agent.step)
    agent.epsilon = max(0.01, agent.epsilon)

    return action    