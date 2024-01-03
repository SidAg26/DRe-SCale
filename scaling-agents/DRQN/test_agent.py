import torch
import torch.nn as nn
import random
from collections import namedtuple, deque
from test_env import Environment as env


# Define the Deep Recurrent Q Network (DRQN) model
class DRQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DRQN, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x, hidden=None):
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, hidden)
        # Take the output from the last time step (assuming batch_first=True)
        out = out[:, -1, :]
        # Fully connected layer 1
        out = self.fc1(out)
        out = torch.relu(out)
        # Fully connected layer 2
        out = self.fc2(out)

        return out, (hn, cn)

# Experience replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Hyperparameters
en = env()
input_size = en.observation_space.shape[0]  # environment state
hidden_size = 256
output_size = 5  # possible actions
capacity = 1000
batch_size = 128
gamma = 0.99  # Discount factor


drqn_model = DRQN(input_size, hidden_size, output_size)
drqn_model.load_state_dict(torch.load('./models/${MODEL_NAME}/${MODEL}.pt')) # Load the DRQN model
target_drqn_model = DRQN(input_size, hidden_size, output_size)
target_drqn_model.load_state_dict(drqn_model.state_dict())  # Initialize target network with the same weights
replay_buffer = ReplayBuffer(capacity) # Initialize the replay buffer

# Test the agent
for episode in range(50):
    state, info = en.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and time dimensions

    total_reward = 0
    hidden = None  # Initialize the hidden state

    while True:
        # Forward pass to get Q-values and hidden state
        q_values, hidden = drqn_model(state, hidden)
        # Choose an action based on Q-values and hidden state (epsilon-greedy policy)
        action = torch.argmax(q_values, dim=1).item()
        # Take the chosen action and observe the next state and reward
        next_state, reward, done, terminated, info = en.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and time dimensions
        
        total_reward += reward
        state = next_state
        if done:
            break

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

env.close()
