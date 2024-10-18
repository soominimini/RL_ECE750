import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import random


class FairnessGridWorld(gym.Env):
    def __init__(self):
        super(FairnessGridWorld, self).__init__()

        # Define a 6x6 grid environment
        self.grid_size = 6
        self.action_space = spaces.Discrete(4)  # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.float32)

        # Rewards for each type of cell
        self.monetary_reward = 10
        self.moral_reward_fair_50_50 = 5
        self.moral_reward_unfair_90_10 = 10
        self.empty_penalty = -0.1
        self.acceptance_threshold = 4

        self.success_episodes = 0
        self.total_episodes = 0
        self.cumulative_monetary_reward = 0
        self.cumulative_moral_reward = 0
        self.reset()

    def reset(self):
        # Initialize agent position at the bottom-left corner
        self.agent_pos = [5, 0]

        # Create grid state representation
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.grid[5, 0] = 1  # Initial agent position
        self.grid[1, 1] = 5  # M cell (Monetary windfall)
        self.grid[0, 5] = 2  # 50-50 Fair offer (Y cell)
        self.grid[0, 3] = 3  # 90-10 Unfair offer (Y cell)
        self.grid[3, 3] = 4  # 10-90 Unfair offer (N cell)

        # Reset cumulative rewards
        self.cumulative_monetary_reward = 0
        self.cumulative_moral_reward = 0

        return self.get_state()

    def get_state(self):
        # Return a tuple representing the agent's position
        return tuple(self.agent_pos)

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
        monetary_reward = 0
        moral_reward = 0

        if current_cell == 5:  # M cell (Monetary windfall)
            monetary_reward = self.monetary_reward
            moral_reward = 0
        elif current_cell == 2:  # 50-50 Fair offer
            monetary_reward = self.monetary_reward * 0.5
            moral_reward = self.moral_reward_fair_50_50
            self.success_episodes += 1
        elif current_cell == 3:  # 90-10 Unfair offer
            monetary_reward = self.monetary_reward * 0.1
            moral_reward = self.moral_reward_unfair_90_10
        elif current_cell == 4:  # 10-90 Unfair offer
            monetary_reward = self.monetary_reward * 0.9
            moral_reward = 0
        else:  # Empty cell
            monetary_reward = self.empty_penalty
            moral_reward = 0

        # Update cumulative rewards
        self.cumulative_monetary_reward += monetary_reward
        self.cumulative_moral_reward += moral_reward

        reward = monetary_reward + moral_reward

        # Update grid and return new state
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 1

        done = True if current_cell in [2, 3, 4, 5] else False
        if done:
            self.total_episodes += 1
        return self.get_state(), reward, done, {}

    def render(self):
        print(self.grid)
        print(f"Cumulative Monetary Reward: {self.cumulative_monetary_reward}")
        print(f"Cumulative Moral Reward: {self.cumulative_moral_reward}")

    def success_rate(self):
        if self.total_episodes == 0:
            return 0.0
        return self.success_episodes / self.total_episodes


if __name__ == "__main__":
    env = FairnessGridWorld()
    num_episodes = 500  # Run multiple episodes for better evaluation
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.01

    # Initialize Q-table
    q_table = {}
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            q_table[(x, y)] = [0, 0, 0, 0]  # Four actions

    moral_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, _ = env.step(action)

            # Q-learning update
            old_value = q_table[state][action]
            next_max = np.max(q_table[next_state])
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state][action] = new_value

            state = next_state

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        moral_rewards.append(env.cumulative_moral_reward)
        success_rate = env.success_rate()
        print(f"Episode {episode + 1} Success Rate: {success_rate * 100:.2f}%")

    # Plot the agent's learning process
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_episodes + 1), moral_rewards, label='Cumulative Moral Reward')
    plt.xlabel('Episode')
    plt.ylabel('Moral Reward')
    plt.title('Agent Learning Process: Moral Rewards')
    plt.legend()
    plt.grid()
    plt.show()