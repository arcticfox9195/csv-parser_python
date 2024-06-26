import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

def process_table(input_table):
    if len(input_table) != 6 or any(len(row) != 6 for row in input_table):
        raise ValueError("输入表格必须是6x12的二维数组")

    result_table = [row[:] for row in input_table]

    for _ in range(1):
        r, a = perform_random_action(result_table)
    
    max_row_length = max(len(row) for row in result_table)
    result_table = [row + [0] * (max_row_length - len(row)) for row in result_table]
    return result_table, r, a

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

def test_data_delete(table, index):
    i = int(index / 6)
    j = index % 6
    table[i].pop(j)
    table[i].append(0)
    return table

def test_data_merge(table, index):
    i = int(index / 5)
    j = index % 5
    table[i][j] += table[i][j+1]
    table[i].pop(j + 1)
    table[i].append(0)
    return table

def patternScore(csvList):
    sc = 0
    listLen = [len(i) - next((j for j, x in enumerate(i[::-1]) if x != 0), 0) for i in csvList]
    numExist = list(set(listLen))
    
    for i in listLen:
        for j in numExist:
            if i == j: break
            elif j == numExist[-1]: numExist.append(i)
        sc += ((i-1)/i)

    sc /= len(numExist) if numExist else 1.0
    return sc

def typeScore(csvList):
    typeMatrix = [[1 if j == 3 else 2 if j == 2 else 0 for j in i] for i in csvList]
    typeCell = sum(sum(row) for row in typeMatrix)
    totalCell = sum(len(row) for row in csvList)
    
    return typeCell / totalCell if typeCell != 0 else 10 ** (-10)

class Environment:
    def __init__(self):
        self.state_space_size = 36
        self.action_space_size = 66
        self.state = [] 
        self.initstate = [
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3]
        ]

    def init_state(self):
        return self.initstate

    def get_state(self):
        return self.state

    def perform_split_action(self, row, column):
        overflow = 0
        n = self.state[row][column]
        if n < 2: return overflow
        if column > 4: return overflow
        appear_time = []
        appear_num = []

        for i in range(0,6):
            if i != row :
                try:
                    if self.state[i][column] not in appear_num:
                        appear_num.append(self.state[i][column])
                        appear_time.append(1)
                    else:
                        tmp = appear_num.index(self.state[i][column])
                        appear_time[tmp] += 1
                except:
                    pass
                
        maxappear = max(appear_time)
        max_index = appear_time.index(maxappear)
        num1 = appear_num[max_index]
        num2 = n - num1

        if num1 <= 0 or num2 <= 0: return overflow

        self.state[row][column] = num1
        self.state[row].insert(column, num2)

        #if len(self.state[row]) > 6: 
        #   if 0 in self.state[row]: self.state[row].remove(0)
        if len(self.state[row]) > 6:
                overflow = self.state[row][6]
                self.state[row] = self.state[row][:6]
        #print("after split: ", self.state)
        return overflow
        
    def perform_add_null_action(self, row, column):
        self.state[row].insert(column, 1)
        overflow = 0
        #if len(self.state[row]) > 6: 
        #   if 0 in self.state[row]: self.state[row].remove(0)
        if len(self.state[row]) > 6:
                overflow = self.state[row][6]
                self.state[row] = self.state[row][:6]
                
        return overflow

    def take_action(self, action):
        of = 0
        if action == 0:
            of = self.perform_split_action(action[1], action[2])
        elif action == 1:
            of = self.perform_add_null_action(action[1], action[2])
        return of
    
    def get_reward(self, originQs, new_csv_state):
        ps = patternScore(new_csv_state)
        ts = typeScore(new_csv_state)
        qs = ps * ts
        return qs - originQs

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
    def __init__(self, state_space_size, action_space_size, buffer_size=1000, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_decay=0.9998, epsilon_min=0.0001, target_update_rate=0.01):
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
            act_type = np.random.randint(0,1)
            if act_type == 0:
                row = np.random.randint(0,5)
                column = np.random.randint(0,4)
            else:
                row = np.random.randint(0,5)
                column = np.random.randint(0,5)
            action = [act_type, row, column]
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action_idx = q_values.max(1)[1].item()
            action_type, row, column = self.index_to_action(action_idx)
            action = [action_type, row, column]

        return action

    def index_to_action(self, index):
        action_type = 0 if index < 30 else 1
        if index >= 30: 
            index -= 30 
            row = index // 6
            column = index % 6
        else:
            row = index // 5
            column = index % 5
        return action_type, row, column

    def add_agent_experience(self, experience):
        self.memory.add_experience((tuple(experience[0]),) + experience[1:])

    def train(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        batch = self.memory.sample_batch(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = [list(state) for state in states]
        for i in range(len(states)):
           for j in range(len(states[i])):
               states[i][j] = states[i][j][:6]
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        predictions = self.q_network(states)
        next_q_values = self.target_q_network(next_states)

        targets = np.copy(predictions.detach().numpy())
        for i in range(self.batch_size):
            action = actions[i]
            reward = rewards[i]
            done = dones[i]

            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.gamma * torch.max(next_q_values[i]).item()

        targets = torch.tensor(targets, dtype=torch.float32)
        loss = nn.functional.mse_loss(predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        self.update_target_network()

    def update_target_network(self):
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data = (1.0 - self.target_update_rate) * target_param.data + self.target_update_rate * param.data

# 训练循环
def train_dqn_agent():
    env = Environment()
    agent = DQNAgent(state_space_size=env.state_space_size, action_space_size=env.action_space_size)
    episodes = 500000

    for episode in range(episodes):
        state = env.init_state()
        state, record, act_type = process_table(env.initstate)
        env.state = state
        print(env.state)
        total_reward = 0
        done = False
        
        while not done:
            ps = patternScore(state)
            ts = typeScore(state)
            originQs = ps * ts
            
            action = agent.select_action(state)
            
            action_type, row, column = action[0], action[1] , action[2]
            if act_type == 0:
                action_idx = row*5 + column
            else:
                action_idx = 30 + row * 6 + column
            of = env.take_action(action)
            if action_type == 0:
                print("split", end = " ")
            else:
                print("add null", end = " ")
            print(row, end = ",")
            print(column)
            next_state = env.get_state()
            reward = env.get_reward(originQs, next_state)

            if record[0] != row:
                reward = -0.2  # 分配負獎勵
                done = True
            elif record == (row,column) and ((act_type == "delete" and action == 1) or (act_type == "merge" and action == 0)):
                reward = 1
                done = True
            elif record == (row, column):
                reward = -0.01
                done = True 
            elif record[0] == row and record[1] != column:
                reward = -0.1
                done = True
            agent.add_agent_experience((state, action_idx, reward, next_state, done))

            state = next_state
            total_reward += reward

            agent.train()
            
        print(env.state)
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
    for test1 in range(36):
        state = [
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3]
        ]
        state = test_data_delete(state, test1)
        env.state = state
        action = agent.select_action(state)
        action_type, row, column = action[0], action[1] , action[2]
        of = env.take_action(action)
        if action_type == 0:
            print("split", end = " ")
        else:
            print("add null", end = " ")
        print(row, end = ",")
        print(column)
        next_state = env.get_state()
        reward = env.get_reward(originQs, next_state)
            
        if (row == int(test1/6) and column == test1%6) and ((act_type == "delete" and action == 1) or (act_type == "merge" and action == 0)):
            reward = 1
        else:
            reward = -0.2  # 分配負獎勵
        state = next_state
        print(env.state)
        print(f"Test Episode: {test1 + 1}, Reward: {reward}")
    for test2 in range(30):
        state = [
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3]
        ]
        state = test_data_merge(state, test2)
        env.state = state
        action = agent.select_action(state)
        action_type, row, column = action[0], action[1] , action[2]
        of = env.take_action(action)
        if action_type == 0:
            print("split", end = " ")
        else:
            print("add null", end = " ")
        print(row, end = ",")
        print(column)
        next_state = env.get_state()
        reward = env.get_reward(originQs, next_state)
            
        if (row == int(test2/5) and column == test2%5) and ((act_type == "delete" and action == 1) or (act_type == "merge" and action == 0)):
            reward = 1
        else:
            reward = -0.2  # 分配負獎勵
        state = next_state
        print(env.state)
        print(f"Test Episode: {test2 + 37}, Reward: {reward}")
if __name__ == "__main__":
    train_dqn_agent()
