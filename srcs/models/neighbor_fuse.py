import torch
import numpy as np


def _get_all_traj(data, action_step, s_rate=None, sample_num=None):
    VALID_LIMIT = 100
    trajs = data['gt_pos']
    all_heading = data["future_heading"][:, data['agent_mask']]
    
    traj_each_agent = {}
    heading_each_agent = {}

    for aix in range(trajs[:, data['agent_mask'], :].shape[1]):
        pos_agent = trajs[:, data['agent_mask'], :][:, aix, :]
        heading_agent = all_heading[:, aix]
        valid_mask = (abs(pos_agent[:, 0])<VALID_LIMIT) * (abs(pos_agent[:, 1])<VALID_LIMIT)
        pos_agent = pos_agent[valid_mask]
        pos_step = pos_agent.shape[0]
        if s_rate == None:
          sample_rate = pos_step // (action_step+1)
        else:
          sample_rate = s_rate 
        if sample_num == None:
          sample_num = -1
        
        final_pos = pos_agent[-1]
        pos_agent = pos_agent[::sample_rate][:sample_num]
        pos_agent[-1] = final_pos
        
        traj_each_agent.update({aix: pos_agent})
        heading_agent = heading_agent[valid_mask]
        heading_agent = heading_agent[::sample_rate][:sample_num].reshape((-1,1))

        for i in range(heading_agent.shape[0]):
            if pos_agent[i, 1] == 0.0:
                continue
            elif pos_agent[i, 0] == 0.0:
                heading_agent[i, 0] += np.pi / 2
            else:
                heading_agent[i, 0] += np.arctan(pos_agent[i, 1] / pos_agent[i, 0])
        
        heading_each_agent.update({aix: heading_agent})

    return traj_each_agent, heading_each_agent


def _get_neighbor_text(data, default, max_agents):
    # print(self.data['file'])
    SAMPLE_NUM = 5

    action_step = 4
    action_dim = 1
    trajs = data['gt_pos']
    traj_each_agent = {}
    heading_each_agent = {}
    traj_each_agent, heading_each_agent = _get_all_traj(data, action_step, s_rate=None, sample_num=SAMPLE_NUM)
    default_mask = np.zeros((max_agents, max_agents), dtype=bool)
    # print(heading_each_agent)
    if len(traj_each_agent) <= 1:
        default_mask[0] = 1
        return default_mask

    num_veh = len(heading_each_agent)

    for i in range(num_veh):
        ego_heading = heading_each_agent[i]      
        ego_traj = traj_each_agent[i]   
        for aidx in range(len(traj_each_agent)):
            if i == aidx:

                continue
            traj_temp = traj_each_agent[aidx]
            lst_temp = []
            for time_step in range(traj_temp.shape[0]):
                ego_pos = ego_traj[time_step]
                current_pos = traj_temp[time_step]
                current_pos_rel = pos_rel(ego_heading[time_step], ego_pos, current_pos)
                # neighbor_trajs_tensor[aidx][time_step] = current_pos_rel
                lst_temp.append(current_pos_rel)
            ll = []
            for j in lst_temp:
                ll.append(j[0])
            for k in lst_temp:
                ll.append(k[1])
            ll.append(-1)
            default[i, aidx] = torch.tensor(ll)

        # neighbor_trajs_tensor[aidx] = torch.tensor(ll)
    # neighbor_trajs_tensor = neighbor_trajs_tensor.view((max_agents, -1))
        default_mask[i, aidx] = 1
    return default_mask
    
def pos_rel(ego_heading, ego_pos, other_pos):
    angle_init = ego_heading[0]
    pos_rel = other_pos - ego_pos
    dis_rel = np.linalg.norm(pos_rel)
    degree_pos_rel = int(np.clip(dis_rel/2.5, a_min=0, a_max=8))
    if pos_rel[1] == 0:
        deg_other = 0
    elif pos_rel[0] == 0:
        deg_other = np.pi/2
    else:
        deg_other = np.arctan2(pos_rel[1], pos_rel[0])
    deg_rel = deg_other - ego_heading
    if deg_rel > np.pi:
        deg_rel -= 2 * np.pi
    elif deg_rel < -1 * np.pi:
        deg_rel += 2*np.pi
        
    if deg_rel < np.pi/6 and deg_rel > -1 * np.pi / 6:
        ang = 0
    elif deg_rel <= -1 * np.pi / 6 and deg_rel > -1 * np.pi / 2:
        ang = 1
    elif  deg_rel <= -1 * np.pi / 2 and deg_rel > -5 * np.pi/6:
        ang = 2
    elif deg_rel >= np.pi/6 and deg_rel < np.pi/2:
        ang = 5
    elif deg_rel >= np.pi/2 and deg_rel < 5*np.pi/6:
        ang = 4
    else:
        ang = 3
    return [degree_pos_rel, ang]




def get_type_interactions(data, max_agents = 32):
    inter_type = _get_inter_type(data, max_agents)
    return inter_type

def type_traj(traj):
    stop_lim = 1.0
    lane_width = 4.0
    ang_lim = 15
    valid_traj, _ = traj.shape
    if valid_traj<=2:
        traj_type = -1
        return traj_type
    
    pos_init = traj[0]
    pos_final = traj[-1]

    
    shift_final = traj[-1]-traj[-2]
    x_final = pos_final[0] - pos_init[0]
    y_final = pos_final[1] - pos_init[1]
    deg_final = np.rad2deg(np.arctan2(shift_final[1], shift_final[0]))
    vel_init = traj[1]-traj[0]
    
    speed_traj = [np.linalg.norm(traj[i+1]-traj[i])/0.1 for i in range(len(traj)-1)]

    # print(np.linalg.norm([x_final, y_final]))
    if np.linalg.norm(pos_final - pos_init)<stop_lim: # stop during the process
      traj_type = 0
      return traj_type

    
    if np.abs(y_final) < 0.7 * lane_width:
      traj_type = 1 # straight
    
    elif y_final >= 0.7 * lane_width:
        
        if deg_final < ang_lim or y_final < 2 * lane_width:
            
            traj_type = 4 # left lc
        else:
            traj_type = 2 # left turn
    else:
        if deg_final > -1* ang_lim or y_final > -2 * lane_width:
            traj_type = 5 # right lc
        else:
            traj_type = 3 # right turn
    
    return traj_type


def _get_inter_type(data, max_agents = 32):
    inter_type = {"overtake" : -1, "follow" : -1, "merge" : -1, "yield" : -1, "surround" : -1, "jam" : -1}
    SAMPLE_NUM = 5
    action_step = 4
    action_dim = 1
    # future_angles = np.cumsum(self.data["future_heading"], axis=0)
    trajs = data['gt_pos']
    all_heading = data["future_heading"][:, data['agent_mask']]
    traj_each_agent = {}
    heading_each_agent = {}
    traj_each_agent, heading_each_agent = _get_all_traj(data, action_step, s_rate=None, sample_num=SAMPLE_NUM)
    if len(traj_each_agent) <= 1:
        return inter_type
    
    num_veh = len(heading_each_agent)
    num_jam = 0
    for i in range(num_veh):
        ego_heading = heading_each_agent[i]      
        ego_traj = traj_each_agent[i]        
        ego_type = type_traj(ego_traj)
        if ego_type == 0:
            num_jam += 1
        for aidx in range(len(traj_each_agent)):
            if i == aidx:
                continue
            other_traj = traj_each_agent[aidx]
            other_heading = heading_each_agent[aidx]
            other_type = type_traj(other_traj)
            is_overtake = type_overtake(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type)
            is_follow = type_follow(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type)
            is_merge = type_merge(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type)
            is_yield = type_yield(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type)
            is_surround = type_surround(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type)


            if is_overtake:
                inter_type["overtake"] = 1
            if is_follow:
                inter_type["follow"] = 1
            if is_merge:
                inter_type["merge"] = 1
            if is_yield:
                inter_type["yield"] = 1
            if is_surround:
                inter_type["surround"] = 1
           
    is_jam = type_jam(num_jam, num_veh, traj_each_agent)
    if is_jam:
        inter_type["jam"] = 1
    
    return inter_type

def type_overtake(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type):
    LANE_WIDTH = 4.0
    
    
    if other_type != 5 and other_type !=4:
        return False
    lane_width = 4.0
    rel_dis_init, rel_pos_init = pos_rel([0.], ego_traj[0], other_traj[0])
    
    rel_dis_final, rel_pos_final = pos_rel(ego_heading[-1], ego_traj[-1], other_traj[-1])
    if rel_pos_init != 3:
        return False
    
    pos_front = [0, 1, 5]
    if not (rel_pos_final in pos_front) and abs(other_traj[-1,0] - ego_traj[-1,0]) > 2.0:
        return False
    
    if abs(ego_traj[0, 1] - other_traj[0, 1]) > 0.8 * LANE_WIDTH:
        return False
    
    return True

def type_follow(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type):
    # only straight
    LANE_WIDTH = 4.0
    HEAD_LIMIT = 20.0 / 180.0 * np.pi
    if other_type != 1 or ego_type !=1:
        return False
    for i in range(len(ego_heading)):
        rel_dis_init, rel_pos_init = pos_rel(ego_heading[i], ego_traj[i], other_traj[i])
        if rel_pos_init != 3:
            return False
        if abs(ego_heading[i] - other_heading[i]) > HEAD_LIMIT:
            return False 
    return True

def type_merge(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type):
    LANE_WIDTH = 4.0
    rel_dis_init, rel_pos_init = pos_rel(ego_heading[-1], ego_traj[-1], other_traj[-1])
    if abs(ego_traj[0,1] - other_traj[0,1]) < 0.8 * LANE_WIDTH:
        return False
    if rel_pos_init != 0 and rel_pos_init != 3:
        return False
    if other_type != 4 and other_type != 5:
        return False
    
    return True 

def type_yield(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type):
    if ego_type != 0:
        return False
    rel_dis_init, rel_pos_init = pos_rel(ego_heading[-1], ego_traj[-1], other_traj[-1])
    if rel_pos_init != 3:
        return False
    
    max_dis = 100
    for i in range(len(ego_heading)):
        rel_dis_init, rel_pos_init = pos_rel(ego_heading[i], ego_traj[i], other_traj[i])
        if rel_dis_init >= max_dis:
            return False
        max_dis = rel_dis_init

    return True

def type_surround(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type):
    if ego_type != 0:
        return False
    rel_dis_init, rel_pos_init = pos_rel(ego_heading[-1], ego_traj[-1], other_traj[-1])
    rel_dis_final, rel_pos_final = pos_rel(ego_heading[-1], ego_traj[-1], other_traj[-1])
    if not rel_pos_init in [2, 3, 4]:
        return False

    if not rel_pos_final in [0, 1, 5]:
        return False
    
    if not other_type in [4, 5]:
        return False

    return True

def type_jam(num_jam, num_veh, traj_each_agent):
    prob_jam = 0.8
    lim_speed = 1.0
    mean_scene = num_jam / num_veh
    if len(traj_each_agent) < 10:
        return False

    if mean_scene < prob_jam:
        return False
    
    return True

