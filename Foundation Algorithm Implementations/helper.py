import copy

class DPHelper:
    grid_size = 8

    def __init__(self):
        raise RuntimeError("This class cannot be instantiated")
    
    @staticmethod
    def to_list_index(x, y):
        return y * DPHelper.grid_size + x

    @staticmethod
    def to_matrix_index(index):
        x = index // DPHelper.grid_size
        y = index - x * DPHelper.grid_size
        return [x, y]
    
    @staticmethod
    def get_terminal_states(env):
        terminal_states = []

        for s in env.keys():
            for a in env[s].keys():
                transitions = env[s][a]
                for (_, next_s, _, done) in transitions:
                    if done:
                        terminal_states.append(next_s)
                        break

        terminal_states = list(set(terminal_states))
        return terminal_states
    
    @staticmethod
    def get_winning_states(env):
        terminal_states = []

        for s in env.keys():
            for a in env[s].keys():
                transitions = env[s][a]
                for (_, next_s, reward, done) in transitions:
                    if done and reward == 1.0:
                        terminal_states.append(next_s)
                        break

        terminal_states = list(set(terminal_states))
        return terminal_states
    
    @staticmethod
    def get_death_states(env):
        terminal_states = DPHelper.get_terminal_states(env)
        winning_states = DPHelper.get_winning_states(env)
        return list(set(terminal_states) - set(winning_states))
    
    @staticmethod
    def get_possible_states(env, state_space):
        terminal_states = DPHelper.get_terminal_states(env)
        return list(set(state_space) - set(terminal_states))

        