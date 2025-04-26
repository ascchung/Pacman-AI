import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("MsPacmanDeterministic-v0", full_action_space=False)
state_shape = env.observation_space.shape
state_size = state_shape[0]
