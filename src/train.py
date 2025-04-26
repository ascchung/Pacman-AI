import gymnasium as gym
import numpy as np
import torch
from collections import deque

from src.dcqn_agent import Agent
from src.frames import preprocess_frame
from src.env import env


def train_dcqn_agent():
    n_actions = env.action_space.n
    agent = Agent(n_actions)

    number_episodes = 2000
    max_timesteps = 10000
    epsilon = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    scores = deque(maxlen=100)

    for episode in range(1, number_episodes + 1):
        state, _ = env.reset()
        score = 0
        for t in range(max_timesteps):
            action = agent.act(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores.append(score)
        epsilon = max(eps_end, eps_decay * epsilon)
        print(f"\rEpisode {episode}\tAverage Score: {np.mean(scores):.2f}", end="")
        if episode % 100 == 0:
            print(f"\rEpisode {episode}\tAverage Score: {np.mean(scores):.2f}")
        if np.mean(scores) >= 500.0:
            print(
                f"\nEnvironment solved in {episode-100} episodes! Average Score: {np.mean(scores):.2f}"
            )
            torch.save(agent.local_qnetwork.state_dict(), "checkpoint.pth")
            break


if __name__ == "__main__":
    train_dcqn_agent()
