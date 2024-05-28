import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_space_size, action_choice_space_size):
        super(QNetwork, self).__init__()
        self.dense1 = nn.Linear(state_space_size, 64)
        self.dense2 = nn.Linear(64, 64)
        self.output_layer_1 = nn.Linear(64, action_choice_space_size)


    def forward(self, state):
        x = torch.relu(self.dense1(state.view(-1, state.shape[1] * state.shape[2])))
        x = torch.relu(self.dense2(x))
        action_choice_q_values = self.output_layer_1(x)
        return  action_choice_q_values



state_space_size = 16  

action_choice_space_size = 2  


model = QNetwork(state_space_size,  action_choice_space_size)


model.load_state_dict(torch.load('error_detect.pth'))
model.eval()  


def detect(table):
    for t in table:
        if len(t) < 4:
            t += [0] * (4 - len(t))
    new_state_tensor = torch.tensor(table, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
         action_choice_q_values = model(new_state_tensor)


    action_choice = torch.argmax(action_choice_q_values, dim=1).item()

    print("Selected actions:",  action_choice)
    return action_choice