import torch
import os
import json
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import copy
from srcs.core.registry import registry
from .typedef import *
from .agent_process import WaymoAgent
from .visual_init import get_heatmap, draw, draw_seq, draw_seq_map
from shapely.geometry import Polygon
from torch import Tensor
import pickle

def get_degree_accel(traj, init_speed=0):
  last_speed = init_speed
  last_deg = 0
  
  accels = []
  degrees = []
  speeds = []

  step = traj.shape[0]
  for i in range(step-1):
    shift = traj[i+1] - traj[i]
    degree = np.rad2deg(np.arctan2(shift[1], shift[0]))
    degrees.append(degree-last_deg)
    last_deg = degree

    speed = np.linalg.norm(shift)
    accels.append(speed-last_speed)
    last_speed = speed
    speeds.append(speed)
  
  return degrees, accels, speeds

def map_dict_to_vec(map_data):
  DIST_INTERVAL = 5

  map_vector = np.zeros(6)
  map_vector[0] = map_data['same_direction_lane_cnt']
  map_vector[1] = map_data['opposite_direction_lane_cnt']
  map_vector[2] = map_data['vertical_up_lane_cnt']
  map_vector[3] = map_data['vertical_down_lane_cnt']
  map_vector[4] = map_data['dist_to_intersection'] // DIST_INTERVAL
  map_vector[5] = 1 + len(map_data['same_right_dir_lanes'])

  return map_vector

def map_vec_distance(query, map_vec):
  weight = np.array([1, 1, 1, 1, 2, 1])
  '''
  if query[2] + query[3] == 0:
    weight[4] = 0
  '''
  result = np.abs(np.array(query)-map_vec)
  result = result * weight
  return np.sum(result, axis=1)

def load_map_data(map_id, data_root):
  map_path = os.path.join(data_root, map_id + '.json')
  with open(map_path, 'r') as f:
    map_data = json.load(f)

  return map_data

def map_retrival(target_vec, map_vecs):
  map_dist = map_vec_distance(target_vec, map_vecs)
  map_dist_idx = np.argsort(map_dist)
  return map_dist_idx

def load_all_map_vectors(map_file):
  map_data = np.load(map_file, allow_pickle=True).item()
  map_vectors = map_data['vectors']
  map_ids = map_data['ids']

  data_list = []
  for map_id in map_ids:
    data_list.append('_'.join(map_id.split('_')[:2]) + '.pkl' + ' ' + map_id.split('_')[-1])

  return map_vectors, data_list

def get_map_data_batch(map_id, cfg):
  from srcs.datasets.utils import fc_collate_fn
  dataset_type = cfg.DATASET.TYPE
  cfg['DATASET']['CACHE'] = False
  dataset = registry.get_dataset(dataset_type)(cfg, 'train')
  dataset.data_list = [map_id]
  collate_fn = fc_collate_fn
  loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory = False,
                  drop_last=False, num_workers=1, collate_fn=collate_fn)
  
  for idx, batch in enumerate(loader):
    if idx == 1:
      break

  return batch

def load_inference_model(cfg):
  model_cls = registry.get_model(cfg.MODEL.TYPE)
  lightning_model = model_cls.load_from_checkpoint(cfg.LOAD_CHECKPOINT_PATH, config=cfg, metrics=[], strict=False)

  return lightning_model

def vis_decode(batch, ae_output):
  img = visualize_input_seq(batch, agents=ae_output[0]['agent'], traj=ae_output[0]['traj'])
  return Image.fromarray(img)

def output_formating_cot(result):
  lines = result.split('\n')
  agent_vectors = []
  event_vectors = []
  vector_idx = [idx for idx, line in enumerate(lines) if 'Actor Vector' in line]
  event_idx = [idx for idx, line in enumerate(lines) if 'Event Vector' in line]
  if len(vector_idx) == 0:
    return [], []
  vector_idx = vector_idx[0]
  event_idx = event_idx[0]

  for line in lines[vector_idx+1:]:
    if 'V' in line or 'Map' in line:
      if 'Vector' in line:
        continue
      data_line = line.split(':')[-1].strip()
      data_vec = eval(data_line)
      
      if 'Map' in line:
        map_vector = data_vec
      else:
        agent_vectors.append(data_vec)

  for line in lines[event_idx+1:]:
    if 'E' in line or 'Map' in line:
      if 'Vector'in line or 'Map' in line:
        continue
        
      data_line = line.split(':')[-1].strip()
      d_l = data_line.split("|")
      data_vec = eval(d_l[0])+eval(d_l[1])
      event_vectors.append(data_vec)

  print('Agent vectors:', agent_vectors)
  print('Map vector:', map_vector)
  print('Event vectors: ', event_vectors)

  return agent_vectors, map_vector, event_vectors

def transform_dist_base(agent_vector, cfg):
  text_distance_base = 5
  distance_base = cfg.DISTANCE_BASE
  ratio = text_distance_base / distance_base

  if ratio == 1:
    return agent_vector

  for idx in range(len(agent_vector)):
    agent_vector[idx][1] = int(agent_vector[idx][1] * ratio)
  
  print('Transformed agent vector:', agent_vector)
  
  return agent_vector

def visualize_decoder(data, decode_probs):

  center = data['center'][0].cpu().numpy()
  rest = data['rest'][0].cpu().numpy()
  bound = data['bound'][0].cpu().numpy()
  center_mask = data['center_mask'][0].cpu().numpy()

  prob = decode_probs['prob'][0].cpu().numpy()
  pos = torch.clip(decode_probs['pos'].sample(), min=-0.5, max=0.5)[0].cpu().numpy()
  #  debug: remove position prediction from heatmap visualization
  pos = pos * 0

  coords = get_agent_coord_from_vec(center, pos).numpy()

  heatmap = get_heatmap(coords[:, 0][center_mask], coords[:, 1][center_mask], prob[center_mask], 20)

  return draw(center, [], other=rest, edge=bound, save_np=True, showup=False, heat_map=heatmap)


def get_agent_coord_from_vec(vec, long_lat):
  vec = torch.tensor(vec)
  x1, y1, x2, y2 = vec[:, 0], vec[:, 1], vec[:, 2], vec[:, 3]
  x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2

  vec_len = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

  vec_dir = torch.atan2(y2 - y1, x2 - x1)

  long_pos = vec_len * long_lat[..., 0]
  lat_pos = vec_len * long_lat[..., 1]

  coord = rotate(lat_pos, long_pos, -np.pi / 2 + vec_dir)

  coord[:, 0] += x_center
  coord[:, 1] += y_center

  return coord

def rotate(x, y, angle):
  if isinstance(x, torch.Tensor):
      other_x_trans = torch.cos(angle) * x - torch.sin(angle) * y
      other_y_trans = torch.cos(angle) * y + torch.sin(angle) * x
      output_coords = torch.stack((other_x_trans, other_y_trans), axis=-1)

  else:
      other_x_trans = np.cos(angle) * x - np.sin(angle) * y
      other_y_trans = np.cos(angle) * y + np.sin(angle) * x
      output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
  return output_coords

def visualize_input(data, agents = None):
  center = data["center"][0].cpu().numpy()
  rest = data["rest"][0].cpu().numpy()
  bound = data["bound"][0].cpu().numpy()
  agent_mask = data["agent_mask"][0].cpu().numpy()
  
  if agents is None:
    agent = data["agent"][0].cpu().numpy()
    agents = [WaymoAgent(agent[i:i+1]) for i in range(agent.shape[0]) if agent_mask[i]]

  return draw(center, agents, other=rest, edge=bound, save_np=True, showup=False)

def visualize_map(data, save=False, path=None):
  center = data["center"][0].cpu().numpy()
  rest = data["rest"][0].cpu().numpy()
  bound = data["bound"][0].cpu().numpy()
  #bound = data["boundaries"][0].cpu().numpy()

  return draw_seq_map(center, other=rest, edge=bound, save_np=True, save=save, path=path)

def visualize_input_seq(data, agents = None, traj=None, sort_agent=True, clip_size=True, save=False, filename=None):
  MIN_LENGTH = 0.2
  MIN_WIDTH = 0.1

  center = data["center"][0].cpu().numpy()
  rest = data["rest"][0].cpu().numpy()
  bound = data["bound"][0].cpu().numpy()
  # bound = data["boundaries"][0].cpu().numpy()

  agent_mask = data["agent_mask"][0].cpu().numpy()
  
  if agents is None:
    agent = data["agent"][0].cpu().numpy()
    agents = [WaymoAgent(agent[i:i+1]) for i in range(agent.shape[0]) if agent_mask[i]]
  if traj is None:
    traj = data['gt_pos'][0][:, agent_mask].cpu().numpy()

  if sort_agent:
    agent_dists = [np.linalg.norm(agent.position) for agent in agents]
    agent_idx = np.argsort(agent_dists)
    agents = [agents[i] for i in agent_idx]
    traj = traj[:, agent_idx]
  
  if clip_size:
    for i in range(len(agents)):
      agents[i].length_width = np.clip(agents[i].length_width, [MIN_LENGTH, MIN_WIDTH], [10.0, 5.0])
    
  return draw_seq(0, center, agents, traj=traj, other=rest, edge=bound, save_np=True, save=save, path=filename)

def transform_traj_output_to_waymo_agent(output, fps=10):
  STOP_SPEED = 1.0
  init_agent = output['agent']
  traj = output['traj']
  pred_future_heading = 'future_heading' in output
  pred_future_vel = 'future_vel' in output
  
  T = len(traj)

  pred_agents = [init_agent]

  if not pred_future_heading:
    traj_headings = []
    traj_speeds = []
    for i in range(len(traj)-1):
      start = traj[i]
      end = traj[i+1]
      traj_headings.append(np.arctan2(end[:, 1]-start[:, 1], end[:, 0]-start[:, 0]))
      traj_speeds.append(np.linalg.norm(end-start, axis=1)*fps)

    init_speeds = np.array([np.linalg.norm(np.linalg.norm(agent.velocity[0])) for agent in init_agent])
    traj_speeds = [init_speeds] + traj_speeds
    traj_headings.append(traj_headings[-1])

  for t in range(T-1):
    agents_t = copy.deepcopy(init_agent)
    for aidx, agent in enumerate(agents_t):
      agent.position = np.array([traj[t+1, aidx]])

      if not pred_future_heading:
        if traj_speeds[t][aidx] < STOP_SPEED:
          agent.heading = init_agent[aidx].heading
        else:
          agent.heading = np.array([traj_headings[t+1][aidx]])
      else:
        agent.heading = output['future_heading'][t][aidx]
      
      if pred_future_vel:
        agent.velocity = output['future_vel'][t][aidx]
    
    pred_agents.append(agents_t)
  
  return pred_agents

def draw_frame(t, output_scene, pred_agents, data):
  img = draw_seq(t, output_scene['center'].cpu(), pred_agents[t], traj=output_scene['traj'], \
                    other=data['rest'][0].cpu(), edge=data['bound'][0].cpu(),save_np=True)
  frame = Image.fromarray(img)
  return frame

def visualize_output_seq(data, output, fps=10, pool_num=16):
  pred_agents = transform_traj_output_to_waymo_agent(output, fps=fps)
  T = len(pred_agents)
  image_list = []

  for i in range(T):
    frame = draw_frame(i, output_scene=output, pred_agents=pred_agents, data=data)
    image_list.append(frame)

  return image_list

def MDdata_to_initdata(MDdata):
  ret = {}
  tracks = MDdata['tracks']

  ret['context_num']=1
  all_agent= np.zeros([128,7])
  agent_mask = np.zeros(128)

  sdc = tracks[MDdata['sdc_index']]['state']
  all_agent[0,:2] = sdc[0,:2]
  all_agent[0,2:4] =sdc[0,7:9]
  all_agent[0,4] = sdc[0,6]
  all_agent[0,5:7] = sdc[0,3:5]

  cnt=1
  for id, track in tracks.items():
    if id == MDdata['sdc_index']:continue
    if not track['type'] == AgentType.VEHICLE: continue
    if track['state'][0,-1]==0:continue
    state = track['state']
    all_agent[cnt, :2] = state[0, :2]
    all_agent[cnt, 2:4] = state[0, 7:9]
    all_agent[cnt, 4] = state[0, 6]
    all_agent[cnt, 5:7] = state[0, 3:5]
    cnt+=1

  all_agent = all_agent[:32]
  agent_num = min(32,cnt)
  agent_mask[:agent_num]=1
  agent_mask=agent_mask.astype(bool)

  lanes = []
  for k, lane in input['map'].items():
    a_lane = np.zeros([20, 4])
    tp = 0
    try:
        lane_type = lane['type']
    except:
        lane_type = lane['sign']
        poly_line = lane['polygon']
        if lane_type == 'cross_walk':
            tp = 18
        elif lane_type == 'speed_bump':
            tp = 19

    if lane_type == 'center_lane':
        poly_line = lane['polyline']
        tp = 1

    elif lane_type == RoadEdgeType.BOUNDARY or lane_type == RoadEdgeType.MEDIAN:
        tp = 15 if lane_type == RoadEdgeType.BOUNDARY else 16
        poly_line = lane['polyline']
    elif 'polyline' in lane:
        tp = 7
        poly_line = lane['polyline']
    if tp == 0:
        continue

    a_lane[:, 2] = tp
    a_lane[:, :2] = poly_line

    lanes.append(a_lane)
  lanes = np.stack(lanes)
  return

def get_polygon(center, yaw, L, W):
  l, w = L / 2, W / 2
  yaw += torch.pi / 2
  theta = torch.atan(w / l)
  s1 = torch.sqrt(l ** 2 + w ** 2)
  x1 = abs(torch.cos(theta + yaw) * s1)
  y1 = abs(torch.sin(theta + yaw) * s1)
  x2 = abs(torch.cos(theta - yaw) * s1)
  y2 = abs(torch.sin(theta - yaw) * s1)

  p1 = [center[0] + x1, center[1] + y1]
  p2 = [center[0] + x2, center[1] - y2]
  p3 = [center[0] - x1, center[1] - y1]
  p4 = [center[0] - x2, center[1] + y2]
  return Polygon([p1, p3, p2, p4])

def get_agent_pos_from_vec(vec, long_lat, speed, vel_heading, heading, bbox, use_rel_heading=True):
  x1, y1, x2, y2 = vec[:, 0], vec[:, 1], vec[:, 2], vec[:, 3]
  x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2

  vec_len = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

  vec_dir = torch.atan2(y2 - y1, x2 - x1)

  long_pos = vec_len * long_lat[..., 0]
  lat_pos = vec_len * long_lat[..., 1]

  coord = rotate(lat_pos, long_pos, -np.pi / 2 + vec_dir)

  coord[:, 0] += x_center
  coord[:, 1] += y_center

  if use_rel_heading:
      agent_dir = vec_dir + heading
  else:
      agent_dir = heading
      
  v_dir = vel_heading + agent_dir

  vel = torch.stack([torch.cos(v_dir) * speed, torch.sin(v_dir) * speed], axis=-1)
  agent_num, _ = vel.shape

  type = Tensor([[1]]).repeat(agent_num, 1).to(coord.device)
  agent = torch.cat([coord, vel, agent_dir.unsqueeze(1), bbox, type], dim=-1).detach().cpu().numpy()

  vec_based_rep = torch.cat([long_lat, speed.unsqueeze(-1), vel_heading.unsqueeze(-1), heading.unsqueeze(-1), vec],
                            dim=-1).detach().cpu().numpy()

  agent = WaymoAgent(agent, vec_based_rep)

  return agent


def process_lane(lane, lane_range, max_vec = None, offset=-40):
  vec_dim = 6
  lane_point_mask = (abs(lane[..., 0] + offset) < lane_range) * (abs(lane[..., 1]) < lane_range)
  lane_id = np.unique(lane[..., -2]).astype(int)

  vec_list = []
  vec_mask_list = []
  vec_id_list = []
  b_s, _, lane_dim = lane.shape

  for id in lane_id:
    id_set = lane[..., -2] == id
    points = lane[id_set].reshape(b_s, -1, lane_dim)
    masks = lane_point_mask[id_set].reshape(b_s, -1)

    vec_ids = np.ones([b_s, points.shape[1] - 1, 1]) * id
    vector = np.zeros([b_s, points.shape[1] - 1, vec_dim])
    vector[..., 0:2] = points[:, :-1, :2]
    vector[..., 2:4] = points[:, 1:, :2]
    vector[..., 4] = points[:, 1:, 2]

    vector[..., 5] = points[:, 1:, 4]
    vec_mask = masks[:, :-1] * masks[:, 1:]
    vector[vec_mask == 0] = 0
    vec_list.append(vector)
    vec_mask_list.append(vec_mask)
    vec_id_list.append(vec_ids)

  vector = np.concatenate(vec_list, axis=1) if vec_list else np.zeros([b_s, 0, vec_dim])
  vector_mask = np.concatenate(vec_mask_list, axis=1) if vec_mask_list else np.zeros([b_s, 0], dtype=bool)
  vec_id = np.concatenate(vec_id_list, axis=1) if vec_id_list else np.zeros([b_s, 0, 1])

  num_vec = vector.shape[0]
  if max_vec is None:
      max_vec = num_vec

  all_vec = np.zeros([b_s, max_vec, vec_dim])
  all_mask = np.zeros([b_s, max_vec])
  all_id = np.zeros([b_s, max_vec, 1])

  for t in range(b_s):
    mask_t = vector_mask[t]
    vector_t = vector[t][mask_t]
    vec_id_t = vec_id[t][mask_t]

    dist = vector_t[..., 0] ** 2 + vector_t[..., 1] ** 2
    idx = np.argsort(dist)
    vector_t = vector_t[idx]
    mask_t = np.ones(vector_t.shape[0])
    vec_id_t = vec_id_t[idx]

    vector_t = vector_t[:max_vec]
    mask_t = mask_t[:max_vec]
    vec_id_t = vec_id_t[:max_vec]

    vector_t = np.pad(vector_t, ([0, max_vec - vector_t.shape[0]], [0, 0]))
    mask_t = np.pad(mask_t, ([0, max_vec - mask_t.shape[0]]))
    vec_id_t = np.pad(vec_id_t, ([0, max_vec - vec_id_t.shape[0]], [0, 0]))

    all_vec[t] = vector_t
    all_mask[t] = mask_t
    all_id[t] = vec_id_t

  return all_vec, all_mask.astype(bool), all_id.astype(int)


def process_map(lane, traf=None, center_num=384, edge_num=128, lane_range=60, offest=-40, rest_num=192):
  lane_with_traf = np.zeros([*lane.shape[:-1], 5])

  lane_with_traf[:, :, :4] = lane

  lane_id = lane[..., -1]
  b_s = lane_id.shape[0]

  
  if (traf is not None) and len(traf) > 0:
    for i in range(b_s):
      traf_t = traf[i]
      lane_id_t = lane_id[i]
      for a_traf in traf_t:
        control_lane_id = a_traf[0]
        state = a_traf[-2]
        lane_idx = np.where(lane_id_t == control_lane_id)
        lane_with_traf[i, lane_idx, -1] = state
  lane = lane_with_traf

  lane_type = lane[0, :, 2]
  
  tps = set()
  for it in lane_type:
      tps.add(it)

  center_0 = lane_type == 0.0
  center_1 = lane_type == 1.0
  center_2 = lane_type == 2.0
  center_3 = lane_type == 3.0
  center_ind = center_0 + center_1 + center_2 + center_3

  boundary_1 = lane_type == 5.0
  boundary_2 = lane_type == 6.0
  boundary_3 = lane_type == 7.0
  boundary_4 = lane_type == 8.0
  bound_ind = boundary_1 + boundary_2 + boundary_3 + boundary_4

  cross_walk = lane_type == 18.0
  speed_bump = lane_type == 19.0
  cross_ind = cross_walk + speed_bump

  rest = ~(center_ind + bound_ind + cross_walk + speed_bump + cross_ind)

  
  cent, cent_mask, cent_id = process_lane(lane = lane[:, center_ind], max_vec = center_num, lane_range = lane_range, offset = offest)
  bound, bound_mask, _ = process_lane(lane = lane[:, bound_ind], max_vec = edge_num, lane_range = lane_range, offset = offest)
  cross, cross_mask, _ = process_lane(lane = lane[:, cross_ind], max_vec = 32, lane_range = lane_range, offset = offest)
  rest, rest_mask, _ = process_lane(lane = lane[:, rest], max_vec = rest_num, lane_range = lane_range, offset = offest)
  

  return cent, cent_mask, cent_id, bound, bound_mask, cross, cross_mask, rest, rest_mask


def normalize_angle(angle):
  if isinstance(angle, torch.Tensor):
    while not torch.all(angle >= 0):
      angle[angle < 0] += np.pi * 2
    while not torch.all(angle < np.pi * 2):
      angle[angle >= np.pi * 2] -= np.pi * 2
    return angle

  else:
    while not np.all(angle >= 0):
      angle[angle < 0] += np.pi * 2
    while not np.all(angle < np.pi * 2):
      angle[angle >= np.pi * 2] -= np.pi * 2

  return angle


def cal_rel_dir(dir1, dir2):
  dist = dir1 - dir2

  while not np.all(dist >= 0):
    dist[dist < 0] += np.pi * 2
  while not np.all(dist < np.pi * 2):
    dist[dist >= np.pi * 2] -= np.pi * 2

  dist[dist > np.pi] -= np.pi * 2
  return dist

def from_list_to_batch(inp_list):
  keys = inp_list[0].keys()

  batch = {}
  for key in keys:
    one_item = [item[key] for item in inp_list]
    batch[key] = Tensor(np.stack(one_item))

  return batch


def transform_to_agent(agent_i, agent, lane):
  all_ = copy.deepcopy(agent)

  center = copy.deepcopy(agent_i[:2])
  center_yaw = copy.deepcopy(agent_i[4])

  all_[..., :2] -= center
  coord = rotate(all_[..., 0], all_[..., 1], -center_yaw)
  vel = rotate(all_[..., 2], all_[..., 3], -center_yaw)

  all_[..., :2] = coord
  all_[..., 2:4] = vel
  all_[..., 4] = all_[..., 4] - center_yaw
  # then recover lane's position
  lane = copy.deepcopy(lane)
  lane[..., :2] -= center
  output_coords = rotate(lane[..., 0], lane[..., 1], -center_yaw)
  if isinstance(lane, Tensor):
    output_coords = Tensor(output_coords)
  lane[..., :2] = output_coords

  return all_, lane


def get_type_class(line_type):
  if line_type == 0 or line_type == 1:
      return 'center_lane'
  elif line_type == 2:
      return RoadLineType.BROKEN_SINGLE_WHITE
  elif line_type == 3:
      return RoadLineType.SOLID_SINGLE_WHITE
  elif line_type == 4:
      return RoadEdgeType.BOUNDARY
  elif line_type == 5:
      return RoadEdgeType.MEDIAN
  else:
      return 'other'

def transform_to_metadrive_data(pred_i, other):
  output_temp = {}
  output_temp['id'] = 'fake'
  output_temp['ts'] = [x / 10 for x in range(190)]
  output_temp['dynamic_map_states'] = [{}]
  output_temp['sdc_index'] = 0

  center_info = other['center_info']
  output = copy.deepcopy(output_temp)
  output['tracks'] = {}
  output['map'] = {}
  # extract agents
  agent = pred_i

  for i in range(agent.shape[1]):
    track = {}
    agent_i = agent[:, i]
    track['type'] = AgentType.VEHICLE
    state = np.zeros([agent_i.shape[0], 10])
    state[:, :2] = agent_i[:, :2]
    state[:, 3] = 5.286
    state[:, 4] = 2.332
    state[:, 7:9] = agent_i[:, 2:4]
    state[:, -1] = 1
    state[:, 6] = agent_i[:, 4]  # + np.pi / 2
    track['state'] = state
    output['tracks'][i] = track

  # extract maps
  lane = other['unsampled_lane']
  lane_id = np.unique(lane[..., -1]).astype(int)
  for id in lane_id:
    a_lane = {}
    id_set = lane[..., -1] == id
    points = lane[id_set]
    polyline = np.zeros([points.shape[0], 3])
    line_type = points[0, -2]
    polyline[:, :2] = points[:, :2]
    a_lane['type'] = get_type_class(line_type)
    a_lane['polyline'] = polyline
    if id in center_info.keys():
        a_lane.update(center_info[id])
    output['map'][id] = a_lane

  return output

def save_as_metadrive_data(pred_i, other, save_path):
  output = transform_to_metadrive_data(pred_i, other)
    
  with open(save_path, 'wb') as f:
    pickle.dump(output, f)
