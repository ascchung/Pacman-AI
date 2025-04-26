import gymnasium as gym
from src.train import train_dcqn_agent
from src.dcqn_agent import Agent

env = gym.make("MsPacmanDeterministic-v0", full_action_space=False)
state_shape = env.observation_space.shape
state_size = state_shape[0]
number_actions = env.action_space.n
agent = Agent(number_actions)
print("State shape: ", state_shape)
print("State size: ", state_size)
print("Number of actions: ", number_actions)

if __name__ == "__main__":
    train_dcqn_agent()
