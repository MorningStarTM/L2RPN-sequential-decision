import torch
import yaml
import grid2op
import numpy as np
from collections import namedtuple, deque
import random
import pickle
import os
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from grid2op.Action import TopologyChangeAction
from stable_baselines3 import PPO
from converter import Converter
from DQN import DQNAgent
from tqdm import tqdm
import time

class RLTrajectory(Dataset):
    def __init__(self, dataset_path, context_len, rtg_scale):
        self.context_len = context_len

        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        min_len = 10**6
        states = []
        for trajectory in self.trajectories:
            trajectory_len = trajectory['observations'].shape[0]
            min_len = min(min_len, trajectory_len)
            states.append(trajectory['observations'])

        #input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        #normalize the states
        for trajectory in self.trajectories:
            trajectory['observation'] = (trajectory['observations'] - self.state_mean) / self.state_std


    def get_state_stats(self):
        return self.state_mean, self.state_std
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, index):
        traj = self.trajectories[index]
        traj_len = traj['observations'].shape[0]

        if traj_len >= self.context_len:
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            return_to_go = torch.from_numpy(traj['return_to_go'][si : si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations'])
            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)

            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions,
                                torch.zeros(([padding_len] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                               dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go,
                                torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                dtype=returns_to_go.dtype)],
                               dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)

        return  timesteps, states, actions, returns_to_go, traj_mask
    


class EnvDataset(Dataset):
    """
    PyTorch Dataset class for trajectories in vector form.
    """

    def __init__(self, root: str):
        """
        Loads dataset to memory and transforms it to tensor.
        :param root: Directory where data files are located
        """
        self.root = root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(self.root + '/state.pkl', 'rb') as file:
            self.states = pickle.load(file)
        with open(self.root + '/action.pkl', 'rb') as file:
            self.actions = pickle.load(file)
        with open(self.root + '/reward.pkl', 'rb') as file:
            self.rewards = pickle.load(file)
        with open(self.root + '/done.pkl', 'rb') as file:
            self.dones = pickle.load(file)

        self.states = torch.tensor(self.states).float()
        self.actions = torch.tensor(self.actions)
        self.rewards = torch.tensor(self.rewards).float()
        self.dones = torch.tensor(self.dones).bool()

    def __len__(self) -> int:
        """
        Returns number of samples in dataset.
        :return: number of samples in dataset
        """
        return len(self.rewards) - 1

    def __getitem__(self, idx: int) -> dict:
        """
        Given an index, return a dictionary with the matching tuples.
        :param idx: Index of entry in dataset
        :return: Dict with state, action, reward, done and new state at
        index position
        """
        sample = {
            'state': self.states[idx, :],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'done': self.dones[idx],
            'new_state': self.states[idx + 1, :]
        }
        return sample


class DataSaver(object):
    """
    Saves environment data to disc.
    """
    def __init__(self, directory: str):
        """
        Initializes lists to be saved.
        :param directory: Saving destination
        """
        self.directory = directory
        self.init_dirs()
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def init_dirs(self):
        """
        Create saving directory if it is not existent
        """
        os.makedirs(self.directory, exist_ok=True)

    def save(self,
             state: np.ndarray,
             action: np.ndarray,
             reward: np.ndarray,
             done: np.ndarray):
        """
        Appends passed data to saving list.
        :param state: State
        :param action: Action
        :param reward: Reward
        :param done: Done
        """

        self.states.append(np.expand_dims(state, axis=0))
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def close(self):
        """
        Converts lists to numpy format and dumps it as binary file in the specified
        directory
        """
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        with open(self.directory + '/state.pkl', 'wb') as output:
            pickle.dump(states, output)
        with open(self.directory + '/action.pkl', 'wb') as output:
            pickle.dump(actions, output)
        with open(self.directory + '/reward.pkl', 'wb') as output:
            pickle.dump(rewards, output)
        with open(self.directory + '/done.pkl', 'wb') as output:
            pickle.dump(dones, output)


class Generator:
    def __init__(self):
        super().__init__()
        self.env = grid2op.make("rte_case5_example", test=True, action_class=TopologyChangeAction)
        self.converter = Converter(self.env)

    def main(self):
        with open('E:\\github_clone\\L2RPN-sequential-decision\\config.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.run(cfg)

    def run(self,cfg):
        print("Gathering data") 
        temp_obs = self.env.reset()
        self.observation_space = len(temp_obs.to_vect())
        self.action_space = 132
        state = np.zeros(self.observation_space)

        agent = DQNAgent(self.env, cfg)
        saver = DataSaver(cfg['GEN_DATA_PATH'])

        for _ in tqdm(range(cfg['WARM_UP_STEPS'])):
            action = np.asarray(random.randint(0, self.action_space-1))
            new_state, reward, done, info = self.env.step(int(action))
            agent.add_experience(state, self.converter.convert_env_act_to_one_hot_encoding_act(action), reward, done, new_state)
            state = new_state
            if done:
                self.env.reset()
             
            print('Starting training with {} steps.'.format(cfg['STEPS']))
            mean_step_reward = []
            reward_cur_episode = []
            reward_last_episode = 0
            episode = 1
            start_time = time.time()
            start_time_episode = time.time()

            for steps in range(1, cfg['STEPS'] + 1):
                action = agent.choose_action(state)
                new_state, reward, done, info = self.env.step(self.converter.convert_one_hot_encoding_act_to_env_act(action))
                agent.learn()
                saver.save(state, (action), reward, done)

                mean_step_reward.append(reward)
                reward_cur_episode.append(reward)

                if steps % cfg['VERBOSE_STEPS'] == 0:
                    elapsed_time = time.time() - start_time
                    print('Ep. {0:>4} with {1:>7} steps total; {2:8.2f} last ep. reward; {3:+.3f} step reward; {4}h elapsed' \
                        .format(episode, steps, reward_last_episode, np.mean(mean_step_reward), self.format_timedelta(elapsed_time)))
                    mean_step_reward = []

                if done:
                    self.env.reset()
                state = new_state

            print("Closing environment")

            agent.save_model()
            self.env.close()

    def format_timedelta(self, timedelta):
        total_seconds = int(timedelta)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return '{:0>2d}:{:0>2d}:{:0>2d}'.format(hours, minutes, seconds)
    
if __name__ == "__main__":
    gen = Generator()
    gen.main()
