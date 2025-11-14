import gymnasium as gym
from helper import DPHelper
from MDP import Policy, Action, MDP
import numpy as np

# (probability, next_state, reward, is_terminal)

def run():

    while(not terminated and not truncated):
        action = env.action_space.sample() #actions: 0=left, 1=down, 2 = right, 3 = up
        new_state, reward, terminated, truncated,_ = env.step(action)

        print(f"action {action} - reward {reward} - new state {new_state}")
        state = new_state
    
    env.close()

def evaluate_policy(mdp: MDP, policy: Policy, prev_matrix):
    current_matrix = np.zeros((DPHelper.grid_size, DPHelper.grid_size))
    for state in mdp.get_possible_states():
        actions, probabilities = policy.act(state)
        total_reward = 0
        for action, prob in zip(actions,probabilities):
            next_s, reward, _ = mdp.observe(state, action)
            x, y = DPHelper.to_matrix_index(next_s)
            total_reward += prob * (reward + prev_matrix[x][y])
        
        cur_x, cur_y = DPHelper.to_matrix_index(state)
        current_matrix[cur_x][cur_y] = total_reward
    return current_matrix

def policy_evaluation(mdp: MDP, policy: Policy, max_iter = 100):
    pre_reward_matrix = np.zeros((DPHelper.grid_size, DPHelper.grid_size))
    for iter in range(max_iter):
        current_reward_matrix = evaluate_policy(mdp, policy, pre_reward_matrix)
        pre_reward_matrix = current_reward_matrix
        print(f"Iteration {iter}")
        print(pre_reward_matrix)

    return pre_reward_matrix

def act_greedily(mdp: MDP, state, reward_matrix):
    best_actions = []
    best_reward = float('inf') * -1.0
    for action in action_space:
        next_s, _, _ = mdp.observe(state=state,action=action)

        if next_s == state or mdp.is_death_state(next_s):
            continue
        
        cur_x, cur_y = DPHelper.to_matrix_index(next_s)
        reward = reward_matrix[cur_x][cur_y]
        if mdp.is_winning_state(next_s):
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

def improve_policy(mdp: MDP, policy: Policy, reward_matrix):
    for state in mdp.get_possible_states():
        actions = act_greedily(mdp=mdp, state=state, reward_matrix=reward_matrix)
        policy.update_policy(state, actions)
    return policy

def policy_iteration(mdp: MDP, policy: Policy, max_iter = 100):
    pre_reward_matrix = np.zeros((DPHelper.grid_size, DPHelper.grid_size))
    for iter in range(max_iter):
        current_reward_matrix = evaluate_policy(mdp, policy, pre_reward_matrix)
        policy = improve_policy(mdp=mdp, policy=policy,reward_matrix=current_reward_matrix)
        pre_reward_matrix = current_reward_matrix
        print(f"Iteration {iter}")
        print(policy.get_current_policy())
        print(pre_reward_matrix)

    return policy, pre_reward_matrix



if __name__ == "__main__":
    env = gym.make(
        "FrozenLake-v1",
        map_name="8x8",
        is_slippery=False,
        render_mode="human",
        )
    
    env_unwrapped = env.unwrapped
    environment = env_unwrapped.P 
    
    mdp = MDP(environment=environment)
    action_space = [Action.LEFT, Action.DOWN, Action.RIGHT, Action.UP]
    action_mapper = {i:  mdp.get_action_space() for i in mdp.get_state_space()}
    random_policy = Policy(action_mapper)

    ### Policy Evaluation
    reward_matrix = policy_evaluation(mdp=mdp, policy=random_policy, max_iter= 100)
    ### Policy Iteration
    # new_policy, reward_matrix =  policy_iteration(mdp, random_policy, max_iter = 100)



