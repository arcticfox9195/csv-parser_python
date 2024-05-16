import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

def random_init_state():
    l = []
    for i in range (0,4):
        rand_int = random.randint(0,13)
        l.append(rand_int)
    rand_state = []
    for j in range(0,4):
        rand_state.append(l)
    return rand_state
def process_table(input_table):
    if len(input_table) != 4 or any(len(row) != 4 for row in input_table):
        raise ValueError("wrong format")

    result_table = [row[:] for row in input_table]
    rand = random.randint(0,1)
    if rand == 1:
        for _ in range(1):
            r, a = perform_random_action(result_table)
            rand_add = random.randint(0,13)
            result_table[r[0]].append(rand_add)
    
    return result_table, rand

def perform_random_action(table):
    action_type = random.choice(["merge", "delete"])
    
    if action_type == "merge":
        indices = [(i, j) for i in range(len(table)) for j in range(len(table[i])-1)]
        if indices:
            i, j = random.choice(indices)
            table[i][j] = random.randint(0,13)
            table[i].pop(j + 1)
            record = (i,j)
            a = 1
    elif action_type == "delete":
        i = random.randint(0, len(table)-1)
        j = random.randint(0, len(table[i])-1)
        table[i].pop(j)
        record = (i, j)
        a = 0
    return record, a

class Environment:
    def __init__(self):
        self.state_space_size = 16
        self.action_choice_space_size = 2
        self.state = [] 
        self.initstate = random_init_state()

    def init_state(self):
        return self.initstate

    def get_state(self):
        return self.state
    
    def get_reward(self, act, rand):
        if act == rand:
            return 1
        else:
            return -1
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add_experience(self, experience):
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

class QNetwork(nn.Module):
    def __init__(self, state_space_size,  action_choice_space_size):
        super(QNetwork, self).__init__()
        self.dense1 = nn.Linear(state_space_size, 64 )
        self.dense2 = nn.Linear(64, 64)
        self.output_layer_1 = nn.Linear(64, action_choice_space_size)



    def forward(self, state):
        x = torch.relu(self.dense1(state.view(-1, state.shape[1] * state.shape[2])))
        x = torch.relu(self.dense2(x))
        action_choice_q_values = self.output_layer_1(x)
        return  action_choice_q_values
    
class DQNAgent:
    def __init__(self, state_space_size, action_choice_space_size, buffer_size=1000, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.001, target_update_rate=0.01):
        self.state_space_size = state_space_size

        self.action_choice_space_size = action_choice_space_size
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_rate = target_update_rate

        self.q_network = QNetwork(state_space_size, action_choice_space_size)
        self.target_q_network = QNetwork(state_space_size,  action_choice_space_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action_choice = np.random.randint(0, 2)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                 action_choice_q_values = self.q_network(state_tensor)

            # 选择行动

            choice_action = torch.argmax(action_choice_q_values, dim=1).item()

            action_choice = choice_action
            

        return  action_choice

    def add_agent_experience(self, experience):
        self.memory.add_experience((tuple(experience[0]),) + experience[1:])

    def train(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        batch = self.memory.sample_batch(self.batch_size)
        states, action_choices, rewards_choices = zip(*batch)
        states = [list(state) for state in states]
        for i in range(len(states)):
            for j in range(len(states[i])):
                states[i][j] = states[i][j][:4]
        #print(states)
        states = torch.tensor(states, dtype=torch.float32)

        predictions_choice = self.q_network(states)

        targets_choice = predictions_choice.clone()
       

        for i in range(self.batch_size):

            action_choice = action_choices[i]
     


            reward_choice = rewards_choices[i]
   

            targets_choice[i, action_choice] = reward_choice
      

        loss_choice = nn.functional.mse_loss(predictions_choice, targets_choice)
        loss = loss_choice 
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
    agent = DQNAgent(state_space_size=env.state_space_size, action_choice_space_size=env.action_choice_space_size)
    episodes = 20000

    for episode in range(episodes):
        state = random_init_state()
        state, rand = process_table(state)
        env.state = state
        print(env.state)
        total_reward = 0
                    
        action_choice = agent.select_action(state)

        reward_choice = env.get_reward(action_choice, rand)

        agent.add_agent_experience((state, action_choice, reward_choice))

        total_reward = reward_choice

        agent.train()
        
        print(env.state, action_choice,  reward_choice)   
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
    
    correct_count = 0
    for _ in range(100):
        state = random_init_state()
        state, rand = process_table(state)
        env.state = state
        print(env.state)
        total_reward = 0
                    
        action_choice = agent.select_action(state)

        reward_choice = env.get_reward(action_choice, rand)
        total_reward =  reward_choice
   
        if total_reward == 1: correct_count += 1
    print(correct_count , '%')
    torch.save(agent.q_network.state_dict(), 'q_network.pth')
if __name__ == "__main__":
    train_dqn_agent()