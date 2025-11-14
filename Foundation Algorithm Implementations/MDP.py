from enum import Enum
import copy
from helper import DPHelper
from typing import List


class Action(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

class Policy:
    def __init__(self, action_mapper):
        self.action_mapper = action_mapper
    
    def act(self, state):
        actions = self.action_mapper[state]
        return actions, [1/len(actions) for _ in actions]
    
    def get_current_policy(self):
        return self.action_mapper
    
    def update_policy(self, state, actions):
        self.action_mapper[state] = actions

    @staticmethod
    def random_policy(action_space, state_space):
        action_mapper = {i:  action_space for i in state_space}
        return Policy(action_mapper)

class MDP:
    def __init__(self,
                 environment,
                 death_reward= -50000,
                 win_reward = 1000000,
                 moving_cost= -1):
        self.environment = copy.copy(environment)
        self.state_space = list(environment.keys())
        self.action_space = [Action.LEFT, Action.DOWN, Action.RIGHT, Action.UP]
        self.winning_states = DPHelper.get_winning_states(environment)
        self.death_states = DPHelper.get_death_states(environment)
        self.possible_states = DPHelper.get_possible_states(environment, self.state_space)
        self.adjust_reward(death_reward,win_reward,moving_cost)

    def adjust_reward(self, death_reward, win_reward, moving_cost):
        for s in self.environment.keys():
            for a in self.environment[s].keys():
                transitions = self.environment[s][a]
                for i in range(len(transitions)):
                    prob, next_s, reward, done = transitions[i]
                    if done and reward != 1.0:
                        reward = death_reward
                    elif not(done):
                        reward = moving_cost
                    else:
                        reward = win_reward
                    transitions[i] = (prob, next_s, reward, done)
                self.environment[s][a] = transitions

    def observe(self, state, action: Action):
        [(_, next_state, reward, is_terminal)] = self.environment[state][action.value]
        return next_state, reward, is_terminal
    
    def is_winning_state(self, state) -> bool:
        return state in self.winning_states
    
    def is_death_state(self, state) -> bool:
        return state in self.death_states
    
    def get_state_space(self) -> List:
        return self.state_space
    
    def get_action_space(self) -> List:
        return self.action_space
    
    def get_possible_states(self) -> List:
        return self.possible_states
    