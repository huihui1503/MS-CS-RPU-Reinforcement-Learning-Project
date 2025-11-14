import gymnasium as gym
from MDP import Policy, Action, MDP
from DPAlgorithms import DPAlgorithms

# (probability, next_state, reward, is_terminal)

def run():

    while(not terminated and not truncated):
        action = env.action_space.sample() #actions: 0=left, 1=down, 2 = right, 3 = up
        new_state, reward, terminated, truncated,_ = env.step(action)

        print(f"action {action} - reward {reward} - new state {new_state}")
        state = new_state
    
    env.close()




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

    dpAlgorithms = DPAlgorithms(mdp=mdp)

    ### Policy Evaluation
    reward_matrix = dpAlgorithms.policy_evaluation(policy=random_policy, max_iter= 100)
    ### Policy Iteration
    new_policy, reward_matrix =  dpAlgorithms.policy_iteration(random_policy, max_iter = 100)



