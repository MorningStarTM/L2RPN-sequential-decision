import torch
import torch.backends
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from EpisodeMemory import EpisodeMemory, EpisodeBuffer

class QNet(nn.Module):
    def __init__(self, state_space=None, action_space=None):
        super(QNet, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be selected."
        assert action_space is not None, "None action_space input: action_space should be selected."

        self.hidden_space = 64
        self.state_space = state_space
        self.action_space = action_space

        self.linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.lstm = nn.LSTM(self.hidden_space, self.hidden_space, batch_first=True)
        self.linear2 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, x, h, c):
        x = F.relu(self.linear1(x))
        x, (new_h, new_c) = self.lstm(x, (h,c))
        x = self.linear2(x)
        return x, new_h, new_c
    

    def sample_action(self, obs, h, c, epsilon):
        output = self.forward(obs, h, c)

        if random.random() < epsilon:
            return random.randint(0,1), output[1], output[2]
        else:
            return output[0].argmax().item(), output[1] , output[2]
        

    def init_hidden_state(self, batch_size, training=None):

        assert training is not None, "training step parameter should be dtermined"

        if training is True:
            return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])
        else:
            return torch.zeros([1, 1, self.hidden_space]), torch.zeros([1, 1, self.hidden_space])



class DRQN:
    """ Deep Recurrent Q Network"""
    def __init__(self, HP):
        self.HP = HP
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNet(self.HP['state_space'], self.HP['action_space']).to(self.device)
        self.target_net = QNet(self.HP['state_space'], self.HP['action_space']).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.HP['lr'])
        self.episode_memory = EpisodeMemory(self.HP['random_update'], self.HP['max_epi_num'], self.HP['max_epi_len'], self.HP['batch_size'], self.HP['looup_step'])
        self.episode_record = EpisodeBuffer()


    def get_action(self, observation):
        action = self.q_net.sample_action(torch.tensor(observation).float().to(self.device))
        return action
    
    def train(self):
        samples, seq_len = self.episode_memory.sample()

        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for i in range(self.HP['batch_size']):
            observations.append(samples[i]['obs'])
            actions.append(samples[i]['acts'])
            next_observations.append(samples[i]['next_obs'])
            rewards.append(samples[i]['rew'])
            dones.append(samples[i]['done'])

        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)
        dones = np.array(dones)

        observations = torch.FloatTensor(observations.reshape(self.HP['batch_size'],seq_len,-1)).to(self.device)
        actions = torch.LongTensor(actions.reshape(self.HP['batch_size'],seq_len,-1)).to(self.device)
        rewards = torch.FloatTensor(rewards.reshape(self.HP['batch_size'],seq_len,-1)).to(self.device)
        next_observations = torch.FloatTensor(next_observations.reshape(self.HP['batch_size'],seq_len,-1)).to(self.device)
        dones = torch.FloatTensor(dones.reshape(self.HP['batch_size'],seq_len,-1)).to(self.device)

        h_target, c_target = self.target_q_net.init_hidden_state(batch_size=self.HP['batch_size'], training=True)
        q_target, _, _ = self.target_q_net(next_observations, h_target.to(self.device), c_target.to(self.device))

        q_target_max = q_target.max(2)[0].view(self.HP['batch_size'],seq_len,-1).detach()
        targets = rewards + self.HP['gamma'] * q_target_max * dones

        h, c = self.q_net.init_hidden_state(batch_size=self.HP['batch_size'], training=True)
        q_out, _, _ = self.q_net(observations, h.to(self.device), c.to(self.device))
        q_a = q_out.gather(2, actions)

        loss = F.smooth_l1_loss(q_a, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def seed_torch(self, seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def save_model(self, path):
        torch.save(self.q_net.state_dict())

        



