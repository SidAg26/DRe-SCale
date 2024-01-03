import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque
from env import Environment as env


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

# Hyperparameters setup and model initialization
en = env()
input_size = en.observation_space.shape[0]  # environment state
hidden_size = 256
output_size = 5  # possible actions
capacity = 1000
batch_size = 128
gamma = 0.99  # Discount factor

# Instantiate the DRQN model, target network, and replay buffer
drqn_model = DRQN(input_size, hidden_size, output_size)
target_drqn_model = DRQN(input_size, hidden_size, output_size)
target_drqn_model.load_state_dict(drqn_model.state_dict())  # Initialize target network with the same weights
replay_buffer = ReplayBuffer(capacity)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(drqn_model.parameters(), lr=0.001)


# Training loop
for episode in range(100000):
    state, info = en.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and time dimensions
    total_reward = 0
    hidden = None  # Initialize the hidden state of the LSTM
    while True:
        # Forward pass to get Q-values and hidden state
        q_values, hidden = drqn_model(state, hidden)

        # Choose an action based on Q-values and hidden state (epsilon-greedy policy)
        epsilon = 0.1
        if random.random() < epsilon:
            action = en.action_space.sample()
        else:
            action = torch.argmax(q_values, dim=1).item()

        
        next_state, reward, done, terminated, info = en.step(action) # Take the chosen action and observe the next state and reward
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and time dimensions

        # Store the transition in the replay buffer
        transition = Transition(state, action, next_state, reward)
        replay_buffer.push(transition)

        # Sample a random batch from the replay buffer
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            batch = Transition(*zip(*batch))

            # batch tensors
            batch_state = torch.cat(batch.state)
            batch_next_state = torch.cat(batch.next_state)

            # Forward pass for the current state and next state
            q_values, _ = drqn_model(batch_state)
            next_q_values, _ = target_drqn_model(batch_next_state)

            # get Q-values corresponding to the selected actions
            q_values = q_values.gather(dim=1, index=torch.tensor(batch.action).unsqueeze(1))

            # Calculate target Q-values using the Q-learning update rule
            target_q_values = torch.max(next_q_values, dim=1)[0].detach()
            target_q_values = target_q_values * gamma + torch.tensor(batch.reward, dtype=torch.float32)

            # Calculate the loss
            loss = criterion(q_values.squeeze(), target_q_values)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_reward += reward
        state = next_state

        # Update the target network periodically
        if episode % 128 == 0:
            target_drqn_model.load_state_dict(drqn_model.state_dict())

        if done:
            break
        # Save the model every 100 episodes - Checkpointing
        if episode % 100 == 0:
            torch.save(drqn_model.state_dict(), f'./models/${MODEL_NAME}/${MODEL}.pt')
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

env.close()
