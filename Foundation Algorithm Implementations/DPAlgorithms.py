from helper import DPHelper
from MDP import Policy, MDP
import numpy as np

class DPAlgorithms():
    def __init__(self, mdp: MDP):
        self.mdp = mdp

    def evaluate_policy(self, policy: Policy, prev_matrix):
        current_matrix = np.zeros((DPHelper.grid_size, DPHelper.grid_size))
        for state in self.mdp.get_possible_states():
            actions, probabilities = policy.act(state)
            total_reward = 0
            for action, prob in zip(actions,probabilities):
                next_s, reward, _ = self.mdp.observe(state, action)
                x, y = DPHelper.to_matrix_index(next_s)
                total_reward += prob * (reward + prev_matrix[x][y])
            
            cur_x, cur_y = DPHelper.to_matrix_index(state)
            current_matrix[cur_x][cur_y] = total_reward
        return current_matrix

    def policy_evaluation(self, policy: Policy, max_iter = 100):
        pre_reward_matrix = np.zeros((DPHelper.grid_size, DPHelper.grid_size))
        for iter in range(max_iter):
            current_reward_matrix = self.evaluate_policy(policy, pre_reward_matrix)
            pre_reward_matrix = current_reward_matrix
            print(f"Iteration {iter}")
            print(pre_reward_matrix)

        return pre_reward_matrix

    def act_greedily(self, state, reward_matrix):
        best_actions = []
        best_reward = float('inf') * -1.0
        for action in self.mdp.get_action_space():
            next_s, _, _ = self.mdp.observe(state=state,action=action)

            if next_s == state or self.mdp.is_death_state(next_s):
                continue
            
            cur_x, cur_y = DPHelper.to_matrix_index(next_s)
            reward = reward_matrix[cur_x][cur_y]
            if self.mdp.is_winning_state(next_s):
                if best_reward == float('inf'):
                    best_actions.append(action)
                else:
                    best_actions = [action]
                best_reward = float('inf')
            elif reward > best_reward:
                best_actions = [action]
                best_reward = reward
            elif reward == best_reward:
                best_actions.append(action)
        return best_actions

    def improve_policy(self, policy: Policy, reward_matrix):
        for state in self.mdp.get_possible_states():
            actions = self.act_greedily(state=state, reward_matrix=reward_matrix)
            policy.update_policy(state, actions)
        return policy

    def policy_iteration(self, policy: Policy, max_iter = 100):
        pre_reward_matrix = np.zeros((DPHelper.grid_size, DPHelper.grid_size))
        for iter in range(max_iter):
            current_reward_matrix = self.evaluate_policy(policy, pre_reward_matrix)
            policy = self.improve_policy(policy=policy,reward_matrix=current_reward_matrix)
            pre_reward_matrix = current_reward_matrix
            print(f"Iteration {iter}")
            print(policy.get_current_policy())
            print(pre_reward_matrix)

        return policy, pre_reward_matrix
