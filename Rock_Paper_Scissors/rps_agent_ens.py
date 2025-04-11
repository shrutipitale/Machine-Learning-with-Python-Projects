import numpy as np
import random
import itertools

class QLearningAgent:
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
        self.history = []
        self.step = 0 # number of times the agent has played (player function has been called)

    # Create all possible states and map each one to an integer 
    def generate_map(self):
        # A. in the form of tuples [('R', 'R'), ('R', 'P'), ...]
        all_states = list(itertools.product(self.actions, repeat = 2))
        return {state: i for i, state in enumerate(all_states)}
        # B. in the form of strings ['RR', 'RP', ...]
        # all_states = [''.join(i) for i in itertools.product(self.actions, repeat = 2)]
        # return {state: i for i, state in enumerate(all_states)}
    
    def state_to_index(self, state):
        return self.state_map[state]
    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # Exploration: choose a random action
            return random.choice(self.actions)
        elif state in self.state_map.keys():
            state = self.state_to_index(state)
            # Exploitation: choose the action with the highest Q-value
            return self.actions[np.argmax(self.Q[state])]
        
    def update_q_value(self, state, action, reward, next_state):
        state = self.state_to_index(state)
        next_state = self.state_to_index(next_state)
        action = self.actions.index(action)
        # Q-value update using the Q-learning formula
        self.Q[state][action] += self.alpha * (
            reward + self.gamma * max(self.Q[next_state]) - self.Q[state][action] )
        
    def train(self, opponent, num_episodes = 1000):
        for episode in range(num_episodes):
            
            prev_play = opponent(self.last_action)
            
            if self.last_action is None or prev_play == '':
                action = random.choice(self.actions)
                self.last_action = action
                self.history.append(self.last_action)
                self.step += 1
                continue
                    
            # A. in the form of tuples [('R', 'R'), ('R', 'P'), ...]
            state = (self.last_action, prev_play) 
            # B. in the form of strings ['RR', 'RP', ...]
            # state = f'{agent.last_action}{prev_play}' 

            action = self.get_action(state)
            self.step += 1
            
            # if self.last_state is not None:
            if self.last_state in self.state_map.keys():
                reward = determine_reward(self.last_action, prev_play)
                self.update_q_value(self.last_state, self.last_action, reward, state)
            
            self.last_action = action
            self.last_state = state

            self.epsilon *= np.exp(-self.decay_rate*self.step)
            self.epsilon = max(0.1, self.epsilon)
    
    def reset(self, only_history = False):
        if only_history:
            self.history = []
            self.step = 0
            self.epsilon = 0.1
        else:
            self.last_state = None
            self.Q = np.zeros((self.num_actions**(2*self.n), self.num_actions))
            self.history = []
            self.step = 0
            self.epsilon = 0.9

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