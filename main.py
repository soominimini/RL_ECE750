import numpy as np
import gym
from gym import spaces


class FairnessGridWorld(gym.Env):
    def __init__(self):
        super(FairnessGridWorld, self).__init__()

        # Define a 5x5 grid environment
        self.grid_size = 5
        self.action_space = spaces.Discrete(4)  # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.float32)

        # Rewards for each type of cell
        self.monetary_reward = 10
        self.fairness_penalty = -5
        self.empty_penalty = -0.1

        self.reset()

    def reset(self):
        # Initialize agent position at the bottom-left corner
        self.agent_pos = [4, 0]

        # Create grid state representation
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.grid[4, 0] = 1  # Initial agent position
        self.grid[0, 4] = 2  # Fair offer (Y cell)
        self.grid[2, 2] = 3  # Low offer (N cell)
        return self.grid

    def step(self, action):
        # Update agent position based on the action
        if action == 0 and self.agent_pos[0] > 0:  # Up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.grid_size - 1:  # Down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # Left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.grid_size - 1:  # Right
            self.agent_pos[1] += 1

        # Calculate the reward
        current_cell = self.grid[self.agent_pos[0], self.agent_pos[1]]
        if current_cell == 2:  # Fair offer
            reward = self.monetary_reward
        elif current_cell == 3:  # Low offer
            reward = self.fairness_penalty
        else:  # Empty cell
            reward = self.empty_penalty

        # Update grid and return new state
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 1

        done = True if current_cell in [2, 3] else False
        return self.grid, reward, done, {}

    def render(self):
        print(self.grid)


if __name__ == "__main__":
    env = FairnessGridWorld()
    env.reset()
    done = False

    while not done:
        env.render()
        action = env.action_space.sample()  # Random action for simplicity
        state, reward, done, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward}")