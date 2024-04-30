import numpy as np
from itertools import product

class Converter:
  def __init__(self, env):
    self.env = env
    self.n_powerlines = self.env.n_line
    self.n_substations = self.env.n_sub
    self.total_bus_actions = 21

  def create_one_hot_converter(self):
    """
    Creates two 2-d np.arrays used for conversion between grid2op action to one hot encoding action vector used by a neural network
    """
    one_hot_encoding_act_conv = []
    env_act_dict_list = []
    zero_act = np.zeros((self.n_powerlines+self.total_bus_actions,1))

    ## Add do nothing action vector (all zeroes)
    one_hot_encoding_act_conv.append(zero_act)
    env_act_dict_list.append({}) ## {} is the do nothing dictonary for actions in grid2op

    ## Powerline change actions
    for idx in range(self.n_powerlines):
      one_hot_encoding_act_conv_pwline = zero_act.copy()
      one_hot_encoding_act_conv_pwline[self.total_bus_actions+idx] = 1
      one_hot_encoding_act_conv.append(one_hot_encoding_act_conv_pwline)
      env_act_dict_list.append({'change_line_status': [idx]}) ## {'change_line_status': [idx]} set an action of changing line status for lineid with id idx


    ## Bus change actions
    start_slice = 0
    for sub_station_id, nb_el in enumerate(self.env.action_space.sub_info):
      one_hot_encoding_act_conv_substation = zero_act.copy()

      possible_bus_actions = np.array(list(product('01', repeat=nb_el))).astype(int)
      for possible_bus_action in possible_bus_actions:
        if possible_bus_action.sum()>0: # Do not include no change action vector
          one_hot_encoding_act_conv_substation[start_slice:(start_slice+nb_el)] = possible_bus_action.reshape(-1,1)
          one_hot_encoding_act_conv.append(one_hot_encoding_act_conv_substation.copy())
          env_act_dict_list.append({"change_bus": {"substations_id": [(sub_station_id, possible_bus_action.astype(bool))]}})
      start_slice += nb_el

    one_hot_encoding_act_conv = np.array(one_hot_encoding_act_conv).reshape(len(one_hot_encoding_act_conv),self.n_powerlines+self.total_bus_actions)

    return one_hot_encoding_act_conv,env_act_dict_list

  def convert_env_act_to_one_hot_encoding_act(self,env_act):
    """
    Converts an grid2op action (in numpy format) to a one hot encoding vector
    """
    one_hot_encoding_act_conv,env_act_dict_list = self.create_one_hot_converter()
    one_hot_encoding_act = np.zeros(len(one_hot_encoding_act_conv))
    env_act = env_act.reshape(-1,)
    action_idx = (one_hot_encoding_act_conv[:, None] == env_act).all(-1).any(-1)
    one_hot_encoding_act[action_idx] = 1
    return one_hot_encoding_act

  def convert_one_hot_encoding_act_to_env_act(self, env_act_dict_list,one_hot_encoding_act):
    """
    Converts a one hot encoding action to a grid2op action
    """
    return self.env.action_space(env_act_dict_list[one_hot_encoding_act.argmax().item()])