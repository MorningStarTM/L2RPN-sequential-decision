import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """
        building model for Deep Q Network

        Args:
            state_size (int): Dimension of state/observation
            actions_size (int): Dimension of action

        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 364)
        self.fc2 = nn.Linear(364, 364)
        self.fc3 = nn.Linear(364, 182)
        self.fc4 = nn.Linear(182, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return self.fc4(x)
    
class ReplayBuffer:
    def __init__(self, MEM_SIZE, BATCH_SIZE):
        self.mem_count = 0
        self.MEM_SIZE = MEM_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        
        self.states = np.zeros((self.MEM_SIZE, 211),dtype=np.float32)
        self.actions = np.zeros(self.MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(self.MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((self.MEM_SIZE, 211),dtype=np.float32)
        self.dones = np.zeros(self.MEM_SIZE, dtype=np.bool_)
    
    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % self.MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, self.MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, self.BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

class DQNAgent:
    def __init__(self, env, HP:dict):
        self.HP = HP
        self.exploration_rate = HP['EXPLORATION_MAX']
        self.network = QNetwork(182, 132, 1000)
        self.env = env
        self.losses = []
        self.replay_buffer = ReplayBuffer(50000, 128)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def choose_action(self, observation):
        if random.random() < self.exploration_rate:
            return self.env.action_space.sample()
        
        state = torch.tensor(observation).float().detach()
        state = state.to(self.device)
        state = state.unsqueeze(0)
        q_values = self.network(state)
        return torch.argmax(q_values).item()
    
    def learn(self):
        if self.memory.mem_count < self.HP['BATCH_SIZE']:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states , dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        states_ = torch.tensor(states_, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        batch_indices = np.arange(self.HP['BATCH_SIZE'], dtype=np.int64)

        q_values = self.network(states)
        next_q_values = self.network(states_)
        
        predicted_value_of_now = q_values[batch_indices, actions]
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]
        
        q_target = rewards + self.HP['GAMMA'] * predicted_value_of_future * dones

        loss = self.network.loss(q_target, predicted_value_of_now)
        self.losses.append(loss)
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

        self.exploration_rate *= self.HP['EXPLORATION_DECAY']
        self.exploration_rate = max(self.HP['EXPLORATION_MIN'], self.exploration_rate)

    def returning_epsilon(self):
        return self.exploration_rate
    
    def add_experience(self, state:np.ndarray, action:np.ndarray, reward:np.ndarray, done:bool, new_state:np.ndarray):
        state = torch.tensor(state).float()
        new_state = torch.tensor(new_state).float()
        self.replay_buffer.add(state, action, reward, new_state, done)
    
    def save_model(self, path):
        torch.save(self.network.state_dict(), path)

    def load_model(self, path):
        self.network.load_state_dict(path)