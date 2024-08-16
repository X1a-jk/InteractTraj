import os
import random
from re import T

import numpy as np
import json
import torch
from srcs.utils.typedef import *
from srcs.utils.agent_process import WaymoAgent

from srcs.utils.utils import get_degree_accel
from .typedef import Action

def degree_to_range(degree):
    result = (degree - 360) / 2
    if result < -180:
        result += 180
    elif result > 180:
        result -= 180
    return result

class InitDescription:
  PARKING_SPEED_LIMIT = 0.3
  SPARSE_LIMIT = 8
  MEDIUM_LIMIT = 15

  FEW_LIMIT = 3
  SOME_LIMIT = 8

  def __init__(self, data, cfg):
    if type(data['agent']) is list:
      agents = data['agent']
      agents = [WaymoAgent(agent.feature[0]) for agent in data['agent']]
      agent_lanes = data['center'][data['agent_vec_index']]
      if type(data['agent_vec_index']) is torch.Tensor:
        agent_vec_index = data['agent_vec_index'].cpu().numpy().tolist()
      elif type(data['agent_vec_index']) is np.ndarray:
        agent_vec_index = data['agent_vec_index'].tolist()
      else:
        agent_vec_index = data['agent_vec_index']
      file = data['file']
    else:
      masked_agents = data['agent'][data['agent_mask']]
      agents = [WaymoAgent(masked_agents[i]) for i in range(masked_agents.shape[0])]
      agent_vec_index = data['agent_vec_index'][data['agent_mask']]
      agent_lanes = data['center'][agent_vec_index]
      file = data['file']
    
    self.data = data
    self.agent_types = [int(agent.type) for agent in agents]
    self.agent_num = len(agents)
    self.cfg = cfg
    
    self._compute_property(agents, agent_lanes, agent_vec_index, file)
    self._generate_text()
  
  def _lane2heading(self, lane, deg=True):
    rad = np.arctan2(lane[3] -lane[1], lane[2] - lane[0])
    if deg:
      return np.rad2deg(rad)
    else:
      return rad

  def _relative_pos_cnt(self, agents, agent_lanes, *args):
    # count the number of agents in each quadrant relative to ego agent
    pos_cnt = [[] for _ in range(4)]
    
    for aidx, agent in enumerate(agents[1:]):
      rel_pos = agent.position - agents[0].position
      idx_x = 0 if rel_pos[0] > 0 else 1
      idx_y = 0 if rel_pos[1] > 0 else 1
      idx = idx_x + 2 * idx_y
      pos_cnt[idx].append(aidx+1)
    
    self.pos_cnt = [pos_cnt[0], pos_cnt[1], pos_cnt[3], pos_cnt[2]]
  
  def _vertical_horizontal_cnt(self, agents, agent_lanes, *args):
    # count the number of agents going horizontal or parallel to ego agent
    
    # idx = 0: horizontal, idx = 1: parallel
    self.vertical_horizontal_cnt = [[] for _ in range(2)]

    # idx = 0: same direction, idx = 1: opposite direction
    self.paral_direction_cnt = [[] for _ in range(2)]

    ego_heading = self._lane2heading(agent_lanes[0], deg=True)
    for aidx, agent in enumerate(agents[1:]):
      lane = agent_lanes[aidx+1]
      heading = self._lane2heading(lane, deg=True)

      diff = abs(heading - ego_heading)
      idx = 1 if diff < 45 or abs(diff - 180) < 45 else 0
      self.vertical_horizontal_cnt[idx].append(aidx+1)


      if idx == 1:
        pidx = 0 if diff < 45 else 1
        self.paral_direction_cnt[pidx].append(aidx+1)
      
  def _parking_cnt(self, agents, agent_lanes, *args):
    self.parking_cnt = []

    for aidx, agent in enumerate(agents):
      speed = np.linalg.norm(agent.velocity)
      if speed < self.PARKING_SPEED_LIMIT:
        self.parking_cnt.append(aidx)

  def _compute_property(self, *data):
    self._relative_pos_cnt(*data)
    self._vertical_horizontal_cnt(*data)
    self._parking_cnt(*data)
  
  def _generate_general_text(self):
    self.texts['general'] = []
    self.texts['general'].append(f"{self.agent_num} cars in total in the scene")

    self.texts['crowdedness'] = []
    if self.agent_num == 1:
      self.texts['crowdedness'].append("the scene is nearly empty")
    elif self.agent_num < self.SPARSE_LIMIT:
      self.texts['crowdedness'].append("the scene is very sparse")
    elif self.agent_num < self.MEDIUM_LIMIT:
      self.texts['crowdedness'].append("the scene is medium crowded")
    else:
      self.texts['crowdedness'].append("the scene is very crowded")
    
    self.texts['ego'] = []
    if 0 in self.parking_cnt:
      self.texts['ego'].append("the ego car is parking")
    else:
      self.texts['ego'].append("the ego car is moving")

  def _generate_pos_text(self):
    quad_text = ['front left', 'back left', 'back right', 'front right']
    self.texts['pos'] = []
    self.texts['pos_binary'] = []
    self.texts['pos_crowd'] = []

    for idx, cnt in enumerate(self.pos_cnt):
      if len(cnt) == 0:
        self.texts['pos'].append(f"no car on the {quad_text[idx]} of ego car")
        self.texts['pos_binary'].append(f"no car on the {quad_text[idx]} of ego car")
        self.texts['pos_crowd'].append(f"no car on the {quad_text[idx]} of ego car")
      else:
        self.texts['pos'].append(f"{len(cnt)} cars on the {quad_text[idx]} of ego car")
        self.texts['pos_binary'].append(f"exist cars on the {quad_text[idx]} of ego car")

        if len(cnt) < self.FEW_LIMIT:
          self.texts['pos_crowd'].append(f"a few cars on the {quad_text[idx]} of ego car")
        elif len(cnt) < self.SOME_LIMIT:
          self.texts['pos_crowd'].append(f"some cars on the {quad_text[idx]} of ego car")
        else:
          self.texts['pos_crowd'].append(f"many cars on the {quad_text[idx]} of ego car")
    
    self.dir_cnt = {}
    self.dir_cnt['left'] = len(self.pos_cnt[0]) + len(self.pos_cnt[1])
    self.dir_cnt['right'] = self.agent_num - self.dir_cnt['left'] - 1
    self.dir_cnt['front'] = len(self.pos_cnt[0]) + len(self.pos_cnt[3])
    self.dir_cnt['behind'] = self.agent_num - self.dir_cnt['front'] - 1
    for key, val in self.dir_cnt.items():
      if val > 0:
        self.texts['pos'].append(f"{val} cars on the {key} of ego car")
      else:
        self.texts['pos'].append(f"no car on the {key} of ego car")
    
  def _generate_vertical_horizontal_text(self):
    self.texts['vertical_horizontal'] = []
    for idx, cnt in enumerate(self.vertical_horizontal_cnt):
      if len(cnt) > 0:
        self.texts['vertical_horizontal'].append(f"{len(cnt)} cars are {'horizontal' if idx == 0 else 'parallel'} to ego car")
      else:
        self.texts['vertical_horizontal'].append(f"no car is {'horizontal' if idx == 0 else 'parallel'} to ego car")
    
    self.texts['paral_direction'] = []
    for idx, cnt in enumerate(self.paral_direction_cnt):
      if len(cnt) > 0:
        self.texts['paral_direction'].append(f"{len(cnt)} cars going the {'same' if idx == 0 else 'opposite'} direction to ego car")
      else:
        self.texts['paral_direction'].append(f"no car going the {'same' if idx == 0 else 'opposite'} direction to ego car")

  def _generate_text(self):
    self.texts = {}
    self._generate_general_text()
    self._generate_pos_text()
    self._generate_vertical_horizontal_text()

    self.texts['all'] = self._get_all_text()
    self.texts['full'] = [self.get_all_text(shuffle=False)]
  
  def _get_all_text(self):
    all_text = []
    txt_classes = ['general', 'ego', 'pos', 'vertical_horizontal', 'paral_direction', 'pos_binary']
    for cls in txt_classes:
      all_text.extend(self.texts[cls])
    return all_text

  def _process_text_output(self, texts):
    result = "traffic scene of "
    for idx, text in enumerate(texts):
      result += text
      if idx != len(texts) - 1:
        result += ", "
    result += "."
    return result

  def get_all_text_sample(self, sample_nums=1.0, rand_sample=True):
    if rand_sample:
      texts = rand_sample.sample(self.texts['all'], sample_nums)
    else:
      texts = self.texts['all'][:sample_nums]
    return self._process_text_output(texts)
  
  def get_text_from_index(self, index_list):
    index_list = index_list.split(',')
    text_list = [self.texts[idx.split('-')[0]][int(idx.split('-')[1])] for idx in index_list]
    return text_list
  
  def get_category_text(self, categories=['general'], sample_nums=None, rand_sample=True, return_idx=False):
    # categories: general, pos, ego, vertical_horizontal, paral_direction, all

    index_list = []
    for idx, category in enumerate(categories):
      class_all_idx = np.arange(0, len(self.texts[category]))
      if sample_nums is not None:
        class_idx = class_all_idx[:sample_nums[idx]]
      else:
        class_idx = class_all_idx
      index_list += ['{}-{}'.format(category, idx) for idx in class_idx]
    
    if rand_sample:
      random.shuffle(index_list)
    
    index_list = ','.join(index_list)
    
    text_list = self.get_text_from_index(index_list)

    if 'full' not in categories:
      result = self._process_text_output(text_list)
    else:
      result = text_list[0]
    
    if return_idx:
      return result, index_list
    else:
      return result
  
  def get_all_text(self, shuffle=True):
    all_text_list = self.texts['all']
    if shuffle:
      random.shuffle(all_text_list)

    return self._process_text_output(all_text_list)
class AttrCntDescription(InitDescription):
  TXT_CLASSES = ['cnt', 'pos', 'distance', 'paral_direction', 'vertical_horizontal', 'speed', 'ego_speed', 'cnt_bins', 'cnt_norm', 'lane_cnt']
  MAX_DISTANCE = 72
  MAX_SPEED = 50
  MAX_CNT = 32
  CNT_BIN_BASE = 1

  def __init__(self, data, cfg):
    self.cnt_base = cfg.COUNT_BASE
    super().__init__(data, cfg)
  
  def _distance_cnt(self, agents, agent_lanes, *args):
    # count the number of cars in different distance ranges
    dist_base = self.cfg.DISTANCE_BASE
    self.distance_cnt = [0] * int(self.MAX_DISTANCE // dist_base + 1)

    for agent in agents[1:]:
      rel_pos = agent.position - agents[0].position
      dist = np.linalg.norm(rel_pos)
      if dist >= self.MAX_DISTANCE:
        dist = self.MAX_DISTANCE - 1e-3
      self.distance_cnt[int(dist // dist_base)] += 1
  
  def _speed_cnt(self, agents, agent_lanes, *args):
    # count the number of cars in different speed ranges
    speed_base = self.cfg.SPEED_BASE
    self.speed_cnt = [0] * int(self.MAX_SPEED // speed_base + 1)

    for agent in agents[1:]:
      speed = np.linalg.norm(agent.velocity)
      speed_idx = int(speed // speed_base)
      if speed_idx >= len(self.speed_cnt):
        speed_idx = len(self.speed_cnt) - 1
      self.speed_cnt[speed_idx] += 1
    
    self.ego_speed = int(np.linalg.norm(agents[0].velocity) // speed_base)
  
  def _agent_on_lane_cnt(self, agent_seg_ids, lanes, *args):
    cnt = 0
    indices = []
    if type(lanes[0]) is list:
      lane_seg_ids = []
      for lane in lanes:
        lane_seg_ids += lane
    else:
      lane_seg_ids = lanes

    for idx, seg_id in enumerate(agent_seg_ids):
      if seg_id in lane_seg_ids:
        cnt += 1
        indices.append(idx)

    return cnt, indices

  def _lane_cnt(self, agents, agent_lanes, agent_vec_index, file, *args):
    map_id = file.split('/')[-1].split('.')[0]
    map_desc_file = os.path.join(self.cfg.MAP_DESC_PATH, '{}.json'.format(map_id))

    # dim 0-1: count the number of cars in left and right neighbor lanes
    # dim 2-3: count the number of cars in the front and back of ego car's lane
    self.lane_cnt = [0, 0, 0, 0]
    
    if not os.path.exists(map_desc_file):
      return
    
    with open(map_desc_file, 'r') as f:
      map_desc = json.load(f)
    agent_seg_ids = self.data['center_id'][agent_vec_index[1:]].squeeze().tolist()

    if type(agent_seg_ids) is not list:
      agent_seg_ids = [agent_seg_ids]
    
    right_lanes = map_desc['same_right_dir_lanes']
    if len(right_lanes) > 0:
      right_neighbor_lanes = right_lanes[0]
      self.lane_cnt[0], _ = self._agent_on_lane_cnt(agent_seg_ids, right_neighbor_lanes)
    
    left_lanes = map_desc['same_left_dir_lanes']
    opposite_lanes = map_desc['all_opposite_lanes']

    if len(left_lanes) > 0:
      left_neighbor_lanes = left_lanes[0]
      self.lane_cnt[1], _ = self._agent_on_lane_cnt(agent_seg_ids, left_neighbor_lanes)
    elif len(opposite_lanes) > 0:
      left_neighbor_lanes = opposite_lanes[0]
      self.lane_cnt[1], _ = self._agent_on_lane_cnt(agent_seg_ids, left_neighbor_lanes)
    
    if len(map_desc['all_same_dir_lanes']) == 0:
      return
    
    ego_lane = map_desc['all_same_dir_lanes'][0]
    # print(agent_indices, len(agents))
    _, agent_indices = self._agent_on_lane_cnt(agent_seg_ids, [ego_lane])
    for idx in agent_indices:
      if agents[idx + 1].position[0] > agents[0].position[0]:
        self.lane_cnt[2] += 1
      else:
        self.lane_cnt[3] += 1

  def _compute_property(self, *data):
    super()._compute_property(*data)
    self._distance_cnt(*data)
    self._speed_cnt(*data)
    self._lane_cnt(*data)
  
  def _generate_text(self):
    self.texts = {}
    self._generate_cnt_text()
    self._generate_pos_text()
    self._generate_dist_text()
    self._generate_vertical_horizontal_text()
    self._generate_speed_text()

    self.texts['all'] = self._get_all_text()
    # self.texts['full'] = [self.get_all_text(shuffle=False)]
  
  def _get_cnt_base_value(self, value):
    return np.ceil(value / self.cnt_base)

  def _generate_cnt_text(self):
    self.texts['cnt'] = self._get_cnt_base_value(np.array([self.agent_num]))

    cnt_bins = np.zeros(int(self.MAX_CNT // self.CNT_BIN_BASE + 1))
    idx = int(self.agent_num // self.CNT_BIN_BASE)
    if idx >= len(cnt_bins):
      idx = -1
    cnt_bins[idx] = 1
    self.texts['cnt_bins'] = cnt_bins

    self.texts['cnt_norm'] = np.clip(self.texts['cnt'] / self.MAX_CNT, 0, 1)

    self.texts['lane_cnt'] = self._get_cnt_base_value(np.array(self.lane_cnt))

  def _generate_pos_text(self):
    self.texts['pos'] = [len(cnt) for cnt in self.pos_cnt]
    self.texts['pos'] = self._get_cnt_base_value(np.array(self.texts['pos']))
  
  def _generate_dist_text(self):
    self.texts['distance'] = self._get_cnt_base_value(np.array(self.distance_cnt))
  
  def _generate_vertical_horizontal_text(self):
    self.texts['paral_direction'] = self._get_cnt_base_value(np.array([len(cnt) for cnt in self.paral_direction_cnt]))
    self.texts['vertical_horizontal'] = self._get_cnt_base_value(np.array([len(cnt) for cnt in self.vertical_horizontal_cnt]))
  
  def _generate_speed_text(self):
    self.texts['speed'] = self._get_cnt_base_value(np.array(self.speed_cnt))
    self.texts['ego_speed'] = np.array([self.ego_speed])

  def _get_all_text(self):
    all_text = []
    for cls in self.TXT_CLASSES:
      all_text.append(self.texts[cls])
    all_text = np.concatenate(all_text, axis=0)
    return np.float32(all_text)
  
  def get_category_text(self, categories=['general']):
    result = []
    for categry in categories:
      result.append(self.texts[categry])
    result = np.concatenate(result, axis=0)
    return np.float32(result)

  def get_text_dict(self):
    result = {}
    for cls in self.TXT_CLASSES:
      result[cls] = np.float32(self.texts[cls])
    return result

class AttrIndDescription(InitDescription):
  TXT_CLASSES = ['pos', 'distance', 'direction', 'speed', 'action', 'traj_type'] # ['lane']
  MAX_DISTANCE = 72
  MAX_SPEED = 50
  MAX_CNT = 32
  VALID_LIMIT = 100
  SOFT_TURN_DEG = 3
  HARD_TURN_DEG = 12
  SPEEDUP_ACCEL = 0.5
  SPEEDDOWN_ACCEL = -1.0
  STOP_SPEED = 1.0

  def __init__(self, data, cfg):
    self.use_padding = cfg.USE_PADDING
    self.padding_num = cfg.PADDING
    self.flatten = cfg.FLATTEN
    self.use_traj = cfg.USE_TRAJ

    super().__init__(data, cfg)

  def _sort_agents(self, agents, agent_lanes, agent_vec_index, file, max_agent):
    # sort agents by their distance to ego
    dists = np.array([np.linalg.norm(agent.position) for agent in agents[1:]])
    sorted_idx = np.argsort(dists)[:max_agent-1]

    agents = agents[:1] + [agents[idx+1] for idx in sorted_idx]
    agent_lanes = np.concatenate([agent_lanes[:1], agent_lanes[sorted_idx+1]], axis=0)
    agent_vec_index = agent_vec_index[:1] + [agent_vec_index[idx+1] for idx in sorted_idx]

    output_idx = np.concatenate([np.array([0]), sorted_idx+1])

    data = [agents, agent_lanes, agent_vec_index, file, output_idx]
    return data

  def _compute_property(self, agents, agent_lanes, agent_vec_index, file):
    self.actor_dict = {}
    max_agent = self.MAX_CNT if self.use_padding else len(agents)

    for aid in range(max_agent):
      self.actor_dict[aid] = {}
      for attr in self.TXT_CLASSES:
        if attr == 'action':
          self.actor_dict[aid][attr] = [self.padding_num] * 5 #self.cfg.ACTION_STEP * self.cfg.ACTION_DIM
        elif attr == 'traj_type':
          self.actor_dict[aid][attr] = -2
        else:
          self.actor_dict[aid][attr] = self.padding_num
      
    
    data = [agents, agent_lanes, agent_vec_index, file]
    data = self._sort_agents(*data, max_agent)

    self._pos_cnt(*data)
    self._direction_cnt(*data)
    self._distance_cnt(*data)
    self._speed_cnt(*data)
    if self.use_traj:
      self._action_cnt(*data)
    # self._lane_cnt(*data)

  def _get_action_id_1_dim(self, traj_mask, degree, speed, accel):
    if not traj_mask:
      return [-1]

    degree_abs = abs(degree)
    direction = 'left' if degree > 0 else 'right'

    if speed <= self.STOP_SPEED:
      action = 'stop'

    elif degree_abs >= self.SOFT_TURN_DEG:
      if degree_abs >= self.HARD_TURN_DEG:
          action = 'turn_{}'.format(direction)
      else:
          action = '{}_lane_change'.format(direction)

    elif accel >= self.SPEEDUP_ACCEL:
      action = 'accelerate'
    elif accel <= self.SPEEDDOWN_ACCEL:
      action = 'decelerate'
    else:
      action = 'keep_speed'

    return [Action[action].value]

  def _get_action_id_2_dim(self, traj_mask, degree, speed, accel):
    if not traj_mask:
      return [-1, -1]

    degree_abs = abs(degree)
    direction = 'left' if degree > 0 else 'right'

    if speed <= self.STOP_SPEED:
      dir_act = 'stop'
      accl_act = 'stop'
      return [Action[dir_act].value, Action[accl_act].value]

    if degree_abs >= self.SOFT_TURN_DEG:
      if degree_abs >= self.HARD_TURN_DEG:
          dir_act = 'turn_{}'.format(direction)
      else:
          dir_act = '{}_lane_change'.format(direction)
    else:
      dir_act = 'straight'

    if accel >= self.SPEEDUP_ACCEL:
      accl_act = 'accelerate'
    elif accel <= self.SPEEDDOWN_ACCEL:
      accl_act = 'decelerate'
    else:
      accl_act = 'keep_speed'

    return [Action[dir_act].value, Action[accl_act].value]
  
  def _get_action_from_traj(self, traj, sample_rate, init_speed):
    TIME_STEP = 5
    MAX_STEPS = 5
    actions = [1 for i in range(5)] 
    #stop, straigt, left-turn, right-turn, left-change-lane, right-change-lane
    speed_base = self.cfg.SPEED_BASE

    stop_lim = 1.0
    lane_width = 4.0
    ang_lim = 15
    accl_lim = 5
    keep_speed_lim = 1

    # print(self.data['file'])

    valid_traj, _ = traj.shape
    if valid_traj<=2:
      traj_type = -1
      for i in range(len(actions)):
        actions[i] = int(init_speed // speed_base)
      return actions, traj_type
    
    pos_init = traj[0]
    pos_final = traj[-1]
    shift_final = traj[-1]-traj[-2]
    x_final = pos_final[0]
    y_final = pos_final[1]
    deg_final = np.rad2deg(np.arctan2(shift_final[1], shift_final[0]))
    vel_init = traj[1]-traj[0]
    
    speed_traj = [np.linalg.norm(traj[i+1]-traj[i])/0.1 for i in range(len(traj)-1)]
    pivots = speed_traj[::sample_rate]
    # phase_vel = [np.linalg.norm(pivots[i+1]-pivots[i])/1 for i in range(len(pivots)-1)]    
    phase_vel = [init_speed] + pivots

    if np.linalg.norm(pos_final)<stop_lim: # stop during the process
      traj_type = 0
      for i in range(len(actions)):
        actions[i] = 0
      return actions, traj_type

    if np.abs(y_final)<lane_width:
      traj_type = 1 # straight
    elif y_final >= lane_width:
        if deg_final < ang_lim or y_final < 2 * lane_width:
            traj_type = 4 # left lc
        else:
            traj_type = 2 # left turn
    else:
        if deg_final > -1* ang_lim or y_final > -2 * lane_width:
            traj_type = 5 # right lc
        else:
            traj_type = 3 # right turn
    

    for j in range(len(actions)):
        if j >= len(phase_vel):
            actions[j] = int(phase_vel[-1] / speed_base)
        else:
            actions[j] = int(phase_vel[j] / speed_base)

    return actions, traj_type

  def _action_cnt(self, agents, agent_lanes, agent_vec_index, file, sorted_idx):
    if len(self.data['agent_mask']) == 1:
      all_trajs = self.data['traj']
    else:
      all_trajs = self.data['traj'][:, self.data['agent_mask']]
    trajs = all_trajs[:, sorted_idx]
    if 'all_agent_mask' not in self.data:
      traj_masks = np.ones_like(trajs[:, :, 0]) == True
    else:
      traj_masks = self.data['all_agent_mask'][:, self.data['agent_mask']][:, sorted_idx]
    
    action_step = self.cfg.ACTION_STEP
    action_dim = self.cfg.ACTION_DIM
    traj_step = trajs.shape[0]
    sample_rate = traj_step // (action_step+1)
    valid_data_lim = 100

    if action_dim == 1:
      action_func = self._get_action_id_1_dim
    else:
      action_func = self._get_action_id_2_dim

    for adix, agent in enumerate(agents):
      try:
        init_speed = np.linalg.norm(agent.velocity)
        traj = trajs[:, adix, :]
        x = traj[:, 0]
        y = traj[:, 1]
        valid_mask_x = np.abs(x)<valid_data_lim
        valid_mask_y = np.abs(y)<valid_data_lim
        valid_mask = [a and b for (a,b) in zip(valid_mask_x, valid_mask_y)]
        traj = traj[valid_mask]
        degrees, accels, speeds = get_degree_accel(trajs[::sample_rate, adix], init_speed)
        step = len(degrees)
        actions = []
        traj_mask = traj_masks[:, adix]
        speed_interval, type_traj = self._get_action_from_traj(traj, sample_rate, init_speed)
        for i in range(step):
          actions += action_func(traj_mask[i+1], degrees[i], speeds[i], accels[i])
        self.actor_dict[adix]['action'] = speed_interval #actions
        self.actor_dict[adix]['traj_type'] = type_traj
      except:
        self.actor_dict[adix]['action'] = [self.padding_num] * 5 #action_step * action_dim
        self.actor_dict[adix]['traj_type'] = -2

  def _pos_cnt(self, agents, agent_lanes, *args):
    # compute the pos value for each agent
    self.actor_dict[0]['pos'] = -1
      

    for aidx, agent in enumerate(agents[1:]):
      rel_pos = agent.position - agents[0].position
      idx_x = 0 if rel_pos[0] > 0 else 1
      idx_y = 0 if rel_pos[1] > 0 else 1

      deg_rel = np.arctan2(rel_pos[1], rel_pos[0])

      if deg_rel > np.pi:
        deg_rel -= 2 * np.pi
      elif deg_rel < -1 * np.pi:
        deg_rel += 2*np.pi
        
      if deg_rel < np.pi/9 and deg_rel > -1 * np.pi / 9:
        ang = 0
      elif deg_rel <= -1 * np.pi / 9 and deg_rel > -1 * np.pi / 2:
        ang = 1
      elif  deg_rel <= -1 * np.pi / 2 and deg_rel > -8 * np.pi/9:
        ang = 2
      elif deg_rel >= np.pi/9 and deg_rel < np.pi/2:
        ang = 5
      elif deg_rel >= np.pi/2 and deg_rel < 8*np.pi/9:
        ang = 4
      else:
        ang = 3

      self.actor_dict[aidx+1]['pos'] = ang
      
  def _direction_cnt(self, agents, agent_lanes, *args):
    # compute the direction value for each agent
    self.actor_dict[0]['direction'] = 0
    # ego_heading = self._lane2heading(agent_lanes[0], deg=True)

    for aidx, agent in enumerate(agents[1:]):
      diff = np.rad2deg(agent.heading)
      diff = np.mod(diff + 180, 360) - 180

      # horizontal same
      if -45 < diff < 45:
        label = 0
      elif 135 < diff or diff < -135:
      # horizontal opposite
        label = 1
      elif 45 < diff < 135:
      # vertical up
        label = 2
      else:
      # vertical down
        label = 3

      self.actor_dict[aidx+1]['direction'] = label

  def _distance_cnt(self, agents, agent_lanes, *args):
    # count the number of cars in different distance ranges
    dist_base = self.cfg.DISTANCE_BASE
    
    for aidx, agent in enumerate(agents):
      rel_pos = agent.position - agents[0].position
      dist = np.linalg.norm(rel_pos)
      if dist >= self.MAX_DISTANCE:
        dist = self.MAX_DISTANCE - 1e-3
      self.actor_dict[aidx]['distance'] = int(dist // dist_base)

  def _speed_cnt(self, agents, agent_lanes, *args):
    # count the number of cars in different speed ranges
    speed_base = self.cfg.SPEED_BASE

    for aidx, agent in enumerate(agents):
      speed = np.linalg.norm(agent.velocity)
      if speed >= self.MAX_SPEED:
        speed = self.MAX_SPEED - 1e-3
      self.actor_dict[aidx]['speed'] = int(speed // speed_base)

  def _get_all_text(self):
    return self.get_category_text(self.TXT_CLASSES)
  
  def _generate_text(self):
    pass
  
  def get_category_text(self, categories=['pos'], padding=True):
    results = []
    traj_type = []
    for aidx in range(len(self.actor_dict)):
      attr_dict = self.actor_dict[aidx]
      act_text = []
      # for categry in categories:
      for category in self.TXT_CLASSES:
        if category == 'traj_type':
          continue
        attr_value = attr_dict[category]
        if type(attr_value) is not list:
          attr_value = [attr_value]
        if category in categories:
          act_text += attr_value
        elif padding:
          act_text += [self.padding_num for _ in range(len(attr_value))]
      results.append(np.array(act_text)[None, :])
      type_id = [attr_dict['traj_type']]
      traj_type.append(np.array(type_id)[None, :])

    results = np.float32(np.concatenate(results, axis=0))
    traj_type = np.int64(np.concatenate(traj_type, axis=0))
    if self.flatten:
      results = np.reshape(results, [-1])
    return results, traj_type

  def get_text_dict(self):
    result = {}
    for cls in self.TXT_CLASSES:
      result[cls] = self.get_category_text([cls], padding=False)
    return result

class AttrCntDescriptionManual(AttrCntDescription):
    def __init__(self, cfg, kv_dict):
      self.cnt_base = cfg.COUNT_BASE
      self.cfg = cfg
      self._compute_property_manual(kv_dict)
      self._generate_text()
    
    def _compute_property_manual(self, kv_dict):
      self.agent_num = kv_dict['agent_num']
      
      self.pos_cnt = kv_dict['pos_cnt']
      self.pos_cnt = [[1] * cnt for cnt in self.pos_cnt]

      self.vertical_horizontal_cnt = kv_dict['vertical_horizontal_cnt']
      self.vertical_horizontal_cnt = [[1] * cnt for cnt in self.vertical_horizontal_cnt]

      self.paral_direction_cnt = kv_dict['paral_direction_cnt']
      self.paral_direction_cnt = [[1] * cnt for cnt in self.paral_direction_cnt]

      self.distance_cnt = kv_dict['distance_cnt']
      self.speed_cnt = kv_dict['speed_cnt']
      self.ego_speed = kv_dict['ego_speed']

class NeighborCarsDescription(AttrIndDescription):
    MAX_DISTANCE = 72
    MAX_SPEED = 50
    MAX_CNT = 32
    SOFT_TURN_DEG = 3
    HARD_TURN_DEG = 12
    SPEEDUP_ACCEL = 0.5
    SPEEDDOWN_ACCEL = -1.0
    STOP_SPEED = 1.0
    VALID_LIMIT = 100
    DIS_INTERVAL = 2.5

    def __init__(self, data, cfg):
        self.use_padding = cfg.USE_PADDING
        self.padding_num = cfg.PADDING
        self.flatten = cfg.FLATTEN
        self.use_traj = cfg.USE_TRAJ
        self.max_cnt = 32
        super().__init__(data, cfg)

    def _sort_agents(self, agents, agent_lanes, agent_vec_index, file, max_agent):
        # sort agents by their distance to ego
        dists = np.array([np.linalg.norm(agent.position) for agent in agents[1:]])
        sorted_idx = np.argsort(dists)[:max_agent-1]
        self.sorted_idx = sorted_idx
        agents = agents[:1] + [agents[idx+1] for idx in sorted_idx]
        agent_lanes = np.concatenate([agent_lanes[:1], agent_lanes[sorted_idx+1]], axis=0)
        agent_vec_index = agent_vec_index[:1] + [agent_vec_index[idx+1] for idx in sorted_idx]

        output_idx = np.concatenate([np.array([0]), sorted_idx+1])

        data = [agents, agent_lanes, agent_vec_index, file, output_idx]
        self.packed_data = data
        return data

    def _compute_property(self, agents, agent_lanes, agent_vec_index, file):
        self.actor_dict = {}
        max_agent = self.MAX_CNT if self.use_padding else len(agents)
        for aid in range(max_agent):
            self.actor_dict[aid] = [self.padding_num] * self.cfg.ACTION_STEP * self.cfg.ACTION_DIM
                                                                                          
                                                                                              
        data = [agents, agent_lanes, agent_vec_index, file]
        data = self._sort_agents(*data, max_agent)

    def _get_all_traj(self, agents, action_step, s_rate=None, sample_num=None):
      trajs = self.data['gt_pos']
      all_heading = self.data["future_heading"][:, self.data['agent_mask']]
      traj_each_agent = {}
      heading_each_agent = {}
      for aix in range(trajs[:, self.data['agent_mask'], :].shape[1]):
        pos_agent = trajs[:, self.data['agent_mask'], :][:, aix, :]
        heading_agent = all_heading[:, aix]
        valid_mask = (abs(pos_agent[:, 0])<self.VALID_LIMIT) * (abs(pos_agent[:, 1])<self.VALID_LIMIT)
        pos_agent = pos_agent[valid_mask]
        pos_step = pos_agent.shape[0]
        if s_rate == None:
          sample_rate = pos_step // (action_step+1)
        else:
          sample_rate = s_rate 
        if sample_num == None:
          sample_num = -1
        
        # sample_rate = 1 if sample_rate <= 0 else sample_rate
        # print(sample_num)
        # print(f"sample_num: {sample_num}")
        # print(f"sample_rate: {sample_rate}")


        pos_agent = pos_agent[::sample_rate][:sample_num]
        traj_each_agent.update({aix: pos_agent})
        heading_agent = heading_agent[valid_mask]
        heading_agent = heading_agent[::sample_rate][:sample_num].reshape((-1,1))
        heading_each_agent.update({aix: heading_agent})

      return traj_each_agent, heading_each_agent
    
    def _get_neighbor_rel_pos(self, agents, agent_lanes, agent_vec_index, file, output_idx):
      traj_each_agent, heading_each_agent = self._get_all_traj(agents, self.cfg.ACTION_STEP, s_rate=1, sample_num=None)
      ego_heading = heading_each_agent[0]
      init_pos = []
      final_pos = []
      max_agent = self.MAX_CNT if self.use_padding else len(agents)
      init_pos = self.padding_num * np.ones((max_agent, 1))
      final_pos = self.padding_num * np.ones((max_agent, 1))

      if len(traj_each_agent)<1:
        return init_pos, final_pos
      init_pos_0 = traj_each_agent[0][0]
      final_pos_0 = traj_each_agent[0][-1]
      for i in range(len(traj_each_agent)):
        init_pos_i = traj_each_agent[i][0]
        final_pos_i = traj_each_agent[i][-1]
        '''
        init_pos[i, 0] = np.sqrt((init_pos_0[0]-init_pos_i[0])**2+(init_pos_0[1]-init_pos_i[1])**2)
        final_pos[i, 0] = np.sqrt((final_pos_0[0]-final_pos_i[0])**2+(final_pos_0[1]-final_pos_i[1])**2)
        '''
        init_pos[i, 0] = self.pos_rel(ego_heading[0], init_pos_0, init_pos_i)[1]
        final_pos[i, 0] = self.pos_rel(ego_heading[-1], final_pos_0, final_pos_i)[1]
      return init_pos, final_pos

    def _get_neighbor_text(self, agents, agent_lanes, agent_vec_index, file, output_idx):
        # print(self.data['file'])
        SAMPLE_NUM = 5
        '''
        if len(self.data['agent_mask']) == 1:
            all_trajs = self.data['traj']
        else:
            all_trajs = self.data['traj'][:, self.data['agent_mask']]
        trajs = all_trajs#[:, self.sorted_idx]
        if 'all_agent_mask' not in self.data:
            traj_masks = np.ones_like(trajs[:, :, 0]) == True
        else:
            traj_masks = self.data['all_agent_mask'][:, self.data['agent_mask']][:, self.sorted_idx]
        '''
        action_step = self.cfg.ACTION_STEP
        action_dim = self.cfg.ACTION_DIM
        max_agent = self.MAX_CNT if self.use_padding else len(agents)
        # future_angles = np.cumsum(self.data["future_heading"], axis=0)
        trajs = self.data['gt_pos']
        all_heading = self.data["future_heading"][:, self.data['agent_mask']]
        traj_each_agent = {}
        heading_each_agent = {}
        traj_each_agent, heading_each_agent = self._get_all_traj(agents, action_step, s_rate=None, sample_num=SAMPLE_NUM)

        default = self.padding_num * torch.ones((max_agent, SAMPLE_NUM, 2))
        default[0, :] = torch.zeros((1, SAMPLE_NUM, 2)) # both dis, pos
        # default[0, :, 0] = 0  # w/o dis
        # default[0, :, 1] = 0  # w/o pos
        ego_heading = heading_each_agent[0]
        
        if len(traj_each_agent) <= 1:
            default = default.view((max_agent, -1))
            return [False, default]    
        neighbor_trajs_tensor = self.padding_num * torch.ones((max_agent, SAMPLE_NUM * 2))

        ego_traj = traj_each_agent[0]
        neighbor_trajs_tensor[0, :] = torch.zeros((1, SAMPLE_NUM * 2)) # both dis, pos
        # neighbor_trajs_tensor[0, SAMPLE_NUM:] = torch.zeros((1, SAMPLE_NUM)) # w/o dis
        # neighbor_trajs_tensor[0, :SAMPLE_NUM] = torch.zeros((1, SAMPLE_NUM)) # w/o pos
        
        for aidx in range(1,len(traj_each_agent)):
            traj_temp = traj_each_agent[aidx]
            lst_temp = []
            for time_step in range(traj_temp.shape[0]):
                ego_pos = ego_traj[time_step]
                current_pos = traj_temp[time_step]
                current_pos_rel = self.pos_rel(ego_heading[time_step], ego_pos, current_pos)
                # neighbor_trajs_tensor[aidx][time_step] = current_pos_rel
                lst_temp.append(current_pos_rel)
            
            ll = []
            for j in lst_temp:
                ll.append(j[0])
            for k in lst_temp:
                ll.append(k[1])

            neighbor_trajs_tensor[aidx] = torch.tensor(ll)
        neighbor_trajs_tensor = neighbor_trajs_tensor.view((max_agent, -1))
        return [True, neighbor_trajs_tensor]
    
    def pos_rel(self, ego_heading, ego_pos, other_pos):
        angle_init = ego_heading[0]
        pos_rel = other_pos - ego_pos
        dis_rel = np.linalg.norm(pos_rel)
        degree_pos_rel = int(np.clip(dis_rel/self.DIS_INTERVAL, a_min=0, a_max=8))
        deg_other = np.arctan2(pos_rel[1], pos_rel[0])
        deg_rel = deg_other - ego_heading
        if deg_rel > np.pi:
            deg_rel -= 2 * np.pi
        elif deg_rel < -1 * np.pi:
            deg_rel += 2*np.pi
        
        if deg_rel < np.pi/9 and deg_rel > -1 * np.pi / 9:
            ang = 0
        elif deg_rel <= -1 * np.pi / 9 and deg_rel > -1 * np.pi / 2:
            ang = 1
        elif  deg_rel <= -1 * np.pi / 2 and deg_rel > -8 * np.pi/9:
            ang = 2
        elif deg_rel >= np.pi/9 and deg_rel < np.pi/2:
            ang = 5
        elif deg_rel >= np.pi/2 and deg_rel < 8*np.pi/9:
            ang = 4
        else:
            ang = 3
        # degree_pos_rel = -1 # w/o relstive distance
        # ang = -1 # w/o relative pos
        return [degree_pos_rel, ang]

    def get_neighbor_text(self):
        return self._get_neighbor_text(*self.packed_data)
    def get_neighbor_rel_pos(self):
        return self._get_neighbor_rel_pos(*self.packed_data)



descriptions = {'static': InitDescription, 'attr_cnt': AttrCntDescription, 'attr_cnt_manual': AttrCntDescriptionManual, 'attr_ind': AttrIndDescription,}
