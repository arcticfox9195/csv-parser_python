import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from csv_parsing import csvParsing, count, takeType, makeType
from repair import repair

class Environment:
    def __init__(self):
        self.action_row_space_size = row_space
        self.action_col_space_size = column_space
        self.state_space_size = row_space * column_space
        self.action_choice_space_size = 2
        self.action_finish_space_size = 2

    def get_state(self):
        return self.state

    def get_reward_row(self, action_row, record):
        index = -1
        for row in record:
            index += 1
            if row[0] == action_row: 
                print('row', action_row)
                return 5, index
        return -2, -1
    
    def get_reward_col(self, action_col, record, index):
        if index == -1: return -1
        if action_col == record[index][1]: 
            print('col', action_col)
            return 6
        return -1
    
    def get_reward_choice(self, action_choice, record, index):
        if index == -1: return -8
        if action_choice == record[index][2]: return 5
        return -8
    
    def get_reward_finish(self, action_finish, finish):
        if action_finish == finish: return 4
        else: return -4
    
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add_experience(self, experience):
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

class QNetwork(nn.Module):
    def __init__(self, state_space_size, action_row_space_size, action_col_space_size, action_choice_space_size, action_finish_space_size):
        super(QNetwork, self).__init__()
        self.dense1 = nn.Linear(state_space_size, 64)
        self.dense2 = nn.Linear(64, 64)
        self.output_layer_1 = nn.Linear(64, action_row_space_size)
        self.output_layer_2 = nn.Linear(64, action_col_space_size)
        self.output_layer_3 = nn.Linear(64, action_choice_space_size)
        self.output_layer_4 = nn.Linear(64, action_finish_space_size)

    def forward(self, state):
        x = torch.relu(self.dense1(state.view(-1, state.shape[1] * state.shape[2])))
        x = torch.relu(self.dense2(x))
        action_row_q_values = self.output_layer_1(x)
        action_col_q_values = self.output_layer_2(x)
        action_choice_q_values = self.output_layer_3(x)
        action_finish_q_values = self.output_layer_4(x)
        return action_row_q_values, action_col_q_values, action_choice_q_values, action_finish_q_values
    
class DQNAgent:
    def __init__(self, state_space_size, action_row_space_size, action_col_space_size, action_choice_space_size, action_finish_space_size, buffer_size=1000, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, target_update_rate=0.01):
        self.state_space_size = state_space_size
        self.action_row_space_size = action_row_space_size
        self.action_col_space_size = action_col_space_size
        self.action_choice_space_size = action_choice_space_size
        self.action_finish_space_size = action_finish_space_size
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_rate = target_update_rate

        self.q_network = QNetwork(state_space_size, action_row_space_size, action_col_space_size, action_choice_space_size, action_finish_space_size)
        self.target_q_network = QNetwork(state_space_size, action_row_space_size, action_col_space_size, action_choice_space_size, action_finish_space_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action_row = np.random.randint(0, row_space)
            action_col = np.random.randint(0, column_space)
            action_choice = np.random.randint(0, 2)
            action_finish = np.random.randint(0, 2)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_row_q_values, action_col_q_values, action_choice_q_values, action_finish_q_values = self.q_network(state_tensor)

            # 选择行动
            row_action = torch.argmax(action_row_q_values, dim=1).item()
            col_action = torch.argmax(action_col_q_values, dim=1).item()
            choice_action = torch.argmax(action_choice_q_values, dim=1).item()
            finish_action = torch.argmax(action_finish_q_values, dim=1).item()

            action_row = row_action
            action_col = col_action
            action_choice = choice_action
            action_finish = finish_action

        return action_row, action_col, action_choice, action_finish

    def add_agent_experience(self, experience):
        self.memory.add_experience((tuple(experience[0]),) + experience[1:])

    def train(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        batch = self.memory.sample_batch(self.batch_size)
        states, action_rows, action_cols, action_choices, action_finishes, rewards_rows, rewards_cols, rewards_choices, rewards_finishes = zip(*batch)
        states = [list(state) for state in states]
        for i in range(len(states)):
            for j in range(len(states[i])):
                states[i][j] = states[i][j][:column_space]
        #print(states)
        states = torch.tensor(states, dtype=torch.float32)

        predictions_row, predictions_col, predictions_choice, predictions_finish = self.q_network(states)

        targets_row = predictions_row.clone()
        targets_col = predictions_col.clone()
        targets_choice = predictions_choice.clone()
        targets_finish = predictions_finish.clone()

        for i in range(self.batch_size):
            action_row = action_rows[i]
            action_col = action_cols[i]
            action_choice = action_choices[i]
            action_finish = action_finishes[i]

            reward_row = rewards_rows[i]
            reward_col = rewards_cols[i]
            reward_choice = rewards_choices[i]
            reward_finish = rewards_finishes[i]

            targets_row[i, action_row] = reward_row
            targets_col[i, action_col] = reward_col
            targets_choice[i, action_choice] = reward_choice
            targets_finish[i, action_finish] = reward_finish

        loss_row = nn.functional.mse_loss(predictions_row, targets_row)
        loss_col = nn.functional.mse_loss(predictions_col, targets_col)
        loss_choice = nn.functional.mse_loss(predictions_choice, targets_choice)
        loss_finish = nn.functional.mse_loss(predictions_finish, targets_finish)
        loss = loss_row + loss_col + loss_choice + loss_finish

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
    agent = DQNAgent(state_space_size=env.state_space_size, action_row_space_size=env.action_row_space_size, action_col_space_size=env.action_col_space_size, action_choice_space_size=env.action_choice_space_size, action_finish_space_size=env.action_finish_space_size)
    episodes = 100

    for episode in range(episodes):
        process_list, state, record_array = csvParsing()
        correct_array = takeType()
        correct_array = correct_array[0]
        env.state = state
        #print(env.state)
        print('before', process_list)
        print('record', record_array)

        finish = 0
        count = -1
        while finish == 0:
            count += 1
            if count >= 3: break
            total_reward = 0        
                        
            action_row, action_col, action_choice, action_finish = agent.select_action(state)
            print('action', action_choice)

            reward_row, index = env.get_reward_row(action_row, record_array)
            reward_col = env.get_reward_col(action_col, record_array, index)
            reward_choice = env.get_reward_choice(action_choice, record_array, index)
            reward_finish = env.get_reward_finish(action_finish, finish)
            print('index', index)

            repair_list = repair(process_list, action_row, action_col, action_choice)
            type_array = makeType(repair_list)
            
            #print(type_array)

            if len(record_array) == 0: finish = 1
            else: finish = 0

            agent.add_agent_experience((state, action_row, action_col, action_choice, action_finish, reward_row, reward_col, reward_choice, reward_finish))

            total_reward = reward_row * reward_col + reward_choice + reward_finish

            agent.train()

            print('after', repair_list)
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
    
    '''correct_all = 0

    for episode in range(100):
        process_list, state, record_array = csvParsing()
        correct_array = takeType()
        correct_array = correct_array[0]
        env.state = state
        print('before', process_list)
        print('record', record_array)

        finish = 0
        count = -1
        correct_in_one = 0

        while finish == 0:
            if correct_in_one == 3: correct_all += 1

            count += 1

            if count >= 3: break

            total_reward = 0
                        
            action_row, action_col, action_choice, action_finish = agent.select_action(state)
            print('action', action_choice)

            reward_row, index = env.get_reward_row(action_row, record_array)
            reward_col = env.get_reward_col(action_col, record_array, index)
            reward_choice = env.get_reward_choice(action_choice, record_array, index)
            reward_finish = env.get_reward_finish(action_finish, finish)
            print('index', index)

            if reward_row == 7 and reward_col == 6 and reward_choice == 6: 
                record_array.pop(index)
                print('record', record_array)

            repair_list = repair(process_list, action_row, action_col, action_choice)

            if len(record_array) == 0: finish = 1
            else: finish = 0

            total_reward = reward_row * reward_col + reward_choice + reward_finish

            print('after', repair_list)
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
    
            if total_reward == 52: correct_in_one += 1

    print(correct_all, '%')'''

if __name__ == "__main__":
    global row_space
    global column_space
    row_space, column_space = count()
    train_dqn_agent()