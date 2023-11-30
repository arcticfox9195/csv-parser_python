import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

def process_table(input_table):
    # 检查输入表格的维度是否正确
    if len(input_table) != 6 or any(len(row) != 6 for row in input_table):
        raise ValueError("输入表格必须是6x6的二维数组")

    # 复制输入表格，以免修改原始数据
    result_table = [row[:] for row in input_table]

    # 在整个表格上执行3次随机操作
    for _ in range(3):
        perform_random_action(result_table)

    return result_table

def perform_random_action(table):
    action_type = random.choice(["merge", "delete"])
    if action_type == "merge":
        indices = [(i, j) for i in range(len(table)) for j in range(len(table[i])-1)]
        if indices:
            i, j = random.choice(indices)
            table[i][j] += table[i][j+1]
            table[i].pop( j + 1)
            
    elif action_type == "delete":
        i = random.randint(0, len(table)-1)
        j = random.randint(0, len(table[i])-1)
        table[i].pop( j)

def patternScore(csvList):
    sc = 0
    listLen = []
    for i in csvList: listLen.append(len(i))    # 記錄每一列有幾個

    numExist = []    # 記錄有幾種長度
    numExist.append(listLen[0])
    
    for i in listLen:
        for j in numExist:
            if i == j: break
            elif j == numExist[len(numExist)-1]: numExist.append(i)
        sc += ((i-1)/i)

    sc /= len(numExist)
    return sc 

def typeScore(csvList):
    typeMatrix = []    # 0, 1 矩陣
    typeArray = []

    for i in csvList: 
        subArray = []

        for j in i:            
            if j == 2: 
                typeMatrix.append(1)
                subArray.append(2)

            elif j == 1: 
                typeMatrix.append(1)
                subArray.append(1)

            else: 
                typeMatrix.append(0)
                subArray.append(0)
        
        typeArray.append(subArray)

    totalCell = len(typeMatrix)
    typeCell = 0    # "1" 個數

    for i in typeMatrix: 
        if i == 1: typeCell += 1

    if typeCell == 0: return 10 ** (-10)
    else: return typeCell / totalCell

class Environment:
    def __init__(self):
        self.state_space_size = 36  # 6x6 表格的大小
        self.action_space_size = 2  # split 和 add 两个动作
        self.state = [] 
        self.initstate = [
                    [1, 1, 2, 2, 3, 3],
                    [1, 1, 2, 2, 3, 3],
                    [1, 1, 2, 2, 3, 3],
                    [1, 1, 2, 2, 3, 3],
                    [1, 1, 2, 2, 3, 3],
                    [1, 1, 2, 2, 3, 3]] 


    def init_state(self):
        return self.initstate

    def get_state(self):
        return self.state

    def perform_split_action(self, row, column):
        n = self.state[row][column]
        if n < 2: return

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

        if num1 == 0 or num2 == 0: return

        self.state[row][column] = num1
        self.state[row].insert(column, num2)
            
        
    def perform_add_null_action(self, row, column):
        self.state[row].insert(column, 1)

    def take_action(self, action, row, column):
        if action == 0:  # Split action
            self.perform_split_action(row, column)
        elif action == 1:  # Add Null action
            self.perform_add_null_action(row, column)

    def get_reward(self,originQs, new_csv_state):

        ps = patternScore(new_csv_state)
        ts = typeScore(new_csv_state)
        qs = ps * ts
        return qs - originQs

# 定义经验回放缓存
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add_experience(self, experience):
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

# QNetwork 类，使用 PyTorch
class QNetwork(nn.Module):
    def __init__(self, action_space_size, state_space_size):
        super(QNetwork, self).__init__()
        self.dense1 = nn.Linear(state_space_size, 64)
        self.dense2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, action_space_size)

    def forward(self, state):
        x = torch.relu(self.dense1(state))
        x = torch.relu(self.dense2(x))
        return self.output_layer(x)

# DQNAgent 类，使用 PyTorch
class DQNAgent:
    def __init__(self, state_space_size, action_space_size, buffer_size=1000, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        # ...[保留原有代码的大部分，但需要对网络实例化和优化器进行更改]...
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_network = QNetwork(action_space_size, state_space_size)
        self.target_q_network = QNetwork(action_space_size, state_space_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

    def select_action(self, state):
        # 选择一个动作（例如，移动、抓取等）
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = q_values.max(1)[1].item()

        # 根据状态或其他逻辑确定行和列
        # 例如，可以随机选择或根据特定的策略选择
        row = np.random.randint(0, num_rows)
        column = np.random.randint(0, num_columns)

        return action, row, column

    def train(self):
        # 确保缓冲区中有足够的经验可以进行训练
        if len(self.memory.buffer) < self.batch_size:
            return

        # 从缓冲区中随机采样一批数据
        batch = self.memory.sample_batch(self.batch_size)

        # 解包批数据
        states, actions, rewards, next_states, dones = zip(*batch)

        # 将列表的状态转换为 PyTorch 张量
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        # 使用代理的 Q 网络获得当前状态的预测 Q 值
        predictions = self.q_network(states)

        # 使用目标 Q 网络获得下一个状态的 Q 值
        next_q_values = self.target_q_network(next_states)

        # 计算目标 Q 值
        targets = np.copy(predictions.detach().numpy())
        for i in range(self.batch_size):
            action = actions[i]
            reward = rewards[i]
            done = dones[i]

            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.gamma * np.max(next_q_values[i])

        # 将目标 Q 值转换为 PyTorch 张量
        targets = torch.tensor(targets, dtype=torch.float32)

        # 计算损失函数
        loss = nn.functional.mse_loss(predictions, targets)

        # 执行反向传播和优化步骤
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # 更新目标网络
        if len(self.memory.buffer) % 10 == 0:
            self.target_q_network.set_weights(self.q_network.get_weights())

# 训练循环
def train_dqn_agent():
    # ...[保留您原有的训练逻辑，但确保所有与张量相关的操作都使用 PyTorch]...
    env = Environment()
    agent = DQNAgent(state_space_size=env.state_space_size, action_space_size=env.action_space_size)
    episodes = 1000

    for episode in range(episodes):
        state = env.init_state()  
        
        state = process_table(env.initstate)    # 执行随机操作
        env.state = state
        #print(state)
        
        total_reward = 0
        done = False
        
        while not done:
            ps = patternScore(state)
            ts = typeScore(state)
            originQs = ps * ts
            
            action, row , column = agent.select_action(state)
            #print(state)
            #print(action, row , column)
                    
            env.take_action(action, row , column)

            next_state = env.get_state()
            reward = env.get_reward(originQs, next_state)
            
            for st in state:
                if len(st) > 6:
                    done = True
                    break
            
            sum = 0
            for st in state:
                sum += (len(st)-6) ** 2
                   
            if sum == 0:
                done = True
            
            agent.memory.add_experience((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            agent.train()
            
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

if __name__ == "__main__":
    train_dqn_agent()
