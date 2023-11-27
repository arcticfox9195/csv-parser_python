import numpy as np
import tensorflow as tf
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

# 定义神经网络模型
class QNetwork(tf.keras.Model):
    def __init__(self, action_space_size, state_space_size):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_space_size, activation='linear')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)

# 定义DQN Agent
class DQNAgent:
    def __init__(self, state_space_size, action_space_size, buffer_size=1000, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_network = QNetwork(action_space_size, state_space_size)  # Pass both action_space_size and state_space_size

        # Create Q networks and optimizer
        self.target_q_network = QNetwork(action_space_size, state_space_size)
        self.target_q_network.set_weights(self.q_network.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            # Randomly choose between split and add_null
            action = np.random.choice(self.action_space_size)
            row_to_repair = np.random.randint(0, len(state))
            column_to_repair = np.random.randint(0, len(state[row_to_repair]))
            return action, row_to_repair, column_to_repair

        # Pad the shorter rows with zeros to make them equal in length
        max_len = max(len(row) for row in state)
        state_padded = [row + [0] * (max_len - len(row)) for row in state]

        # Convert state to a valid input tensor
        state_tensor = tf.convert_to_tensor(state_padded, dtype=tf.float32)

        # Flatten the state tensor
        state_tensor_flat = tf.reshape(state_tensor, shape=(1, -1))  # Flatten without specifying the size
        print(state_tensor_flat)

        # Q-values for the current state
        q_values = self.q_network(state_tensor_flat)

        # Split Q-values into action and index predictions
        action_values, index_values = tf.split(q_values, [self.action_space_size, self.state_space_size])

        # Choose action with the highest Q-value
        chosen_action = tf.argmax(action_values)

        # Choose index based on the highest Q-value in the index values
        index_to_repair = tf.argmax(index_values)

        return chosen_action.numpy(), index_to_repair // len(state[0]), index_to_repair % len(state[0])

    def train(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        # 从经验回放中采样一批数据
        batch = self.memory.sample_batch(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        # Pad the shorter lists with zeros to make them equal in length
        max_len = max(len(row) for state_row in states for row in state_row)
        states = np.array([row + [0] * (max_len - len(row)) for state_row in states for row in state_row])
        next_states = np.array([row + [0] * (max_len - len(row)) for state_row in next_states for row in state_row])

        q_values = self.q_network(states)
        next_q_values = self.target_q_network(next_states)

        targets = np.copy(q_values)

        for i in range(self.batch_size):
            action = actions[i]
            reward = rewards[i]
            done = dones[i]

            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.gamma * np.max(next_q_values[i])

        with tf.GradientTape() as tape:
            predictions = self.q_network(states)
            loss = tf.keras.losses.mean_squared_error(targets, predictions)
        
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        # 更新epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # 更新目标网络
        if len(self.memory.buffer) % 10 == 0:
            self.target_q_network.set_weights(self.q_network.get_weights())

# 训练DQN Agent
def train_dqn_agent():
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
            print(state)
            print(action, row , column)
                    
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
