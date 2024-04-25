import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


def process_table(input_table):
    if len(input_table) != 4 or any(len(row) != 4 for row in input_table):
        raise ValueError("输入表格必须是6x12的二维数组")

    result_table = [row[:] for row in input_table]

    for _ in range(1):
        r, a = perform_random_action(result_table)
    
    max_row_length = max(len(row) for row in result_table)
    result_table = [row + [0] * (max_row_length - len(row)) for row in result_table]
    return result_table, r

def perform_random_action(table):
    action_type = random.choice(["merge", "delete"])
    
    if action_type == "merge":
        indices = [(i, j) for i in range(len(table)) for j in range(len(table[i])-1)]
        if indices:
            i, j = random.choice(indices)
            table[i][j] += table[i][j+1]
            table[i].pop(j + 1)
            record = (i,j)
            
    elif action_type == "delete":
        i = random.randint(0, len(table)-1)
        j = random.randint(0, len(table[i])-1)
        table[i].pop(j)
        record = (i, j)
    return record, action_type

class Environment:
    def __init__(self):
        self.state_space_size = 16
        self.action_space_size = 16
        self.state = [] 
        self.initstate = [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ]

    def init_state(self):
        return self.initstate

    def get_state(self):
        return self.state

    def get_reward_row(self, action_row, record):
        print(action_row ,record)
        if action_row == record[0]:
            reward = 6
        else:
            reward = 2
        return reward
    
    def get_reward_col(self, action_col, record):
        print(action_col ,record)
        if action_col == record[1]:
            reward = 7
        else:
            reward = 3
        return reward
    
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add_experience(self, experience):
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

class QNetwork(nn.Module):
    def __init__(self, action_space_size, state_space_size):
        super(QNetwork, self).__init__()
        self.dense1 = nn.Linear(state_space_size, 64)
        self.dense2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, action_space_size)

    def forward(self, state):
        x = torch.relu(self.dense1(state.view(-1, state.shape[1] * state.shape[2])))
        x = torch.relu(self.dense2(x))
        return self.output_layer(x)
    
class DQNAgent:
    def __init__(self, state_space_size, action_space_size, buffer_size=1000, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, target_update_rate=0.01):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_rate = target_update_rate

        self.q_network = QNetwork(action_space_size, state_space_size)
        self.target_q_network = QNetwork(action_space_size, state_space_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action_row = np.random.randint(0, 4)
            action_col = np.random.randint(0, 4)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action_index = q_values.max(1)[1].item()
            action_row = action_index // 4
            action_col = action_index % 4
        return action_row, action_col

    def add_agent_experience(self, experience):
        self.memory.add_experience((tuple(experience[0]),) + experience[1:])

    def train(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        batch = self.memory.sample_batch(self.batch_size)
        states, action_rows, action_cols, rewards_rows, rewards_cols = zip(*batch)
        states = [list(state) for state in states]
        for i in range(len(states)):
            for j in range(len(states[i])):
                states[i][j] = states[i][j][:4]
        states = torch.tensor(states, dtype=torch.float32)

        predictions = self.q_network(states)

        targets = predictions.clone()
        for i in range(self.batch_size):
            action_row = action_rows[i]
            action_col = action_cols[i]
            reward_row = rewards_rows[i]
            reward_col = rewards_cols[i]
            targets[i, action_row] = reward_row
            targets[i, action_col] = reward_col

        loss = nn.functional.mse_loss(predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        self.update_target_network()

    def update_target_network(self):
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data = (1.0 - self.target_update_rate) * target_param.data + self.target_update_rate * param.data
    
def train_dqn_agent():
    env = Environment()
    agent = DQNAgent(state_space_size=env.state_space_size, action_space_size=env.action_space_size)
    episodes = 100

    for episode in range(episodes):
        state = env.init_state()
        state, record = process_table(env.initstate)
        env.state = state
        print(env.state)
        total_reward = 0
                    
        action_row, action_col = agent.select_action(state)

        reward_row = env.get_reward_row(action_row, record)
        reward_col = env.get_reward_col(action_col, record)

        agent.add_agent_experience((state, action_row, action_col, reward_row, reward_col))

        total_reward = reward_row * reward_col - 13

        agent.train()
        
        print(env.state, action_row, action_col, reward_row, reward_col)   
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
    correct_count = 0
    for _ in range(100):
        state = env.init_state()
        state, record = process_table(env.initstate)
        env.state = state
        print(env.state)
        total_reward = 0
                    
        action_row, action_col = agent.select_action(state)

        reward_row = env.get_reward_row(action_row, record)
        reward_col = env.get_reward_col(action_col, record)

        total_reward = reward_row * reward_col - 13
   
        if total_reward == 29: correct_count += 1
    print(correct_count, '%')

if __name__ == "__main__":
    train_dqn_agent()