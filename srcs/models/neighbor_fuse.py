import torch
import numpy as np


def _get_all_traj_batch(data, action_step, s_rate=None, sample_num=None):
    VALID_LIMIT = 100
    
    traj_each_sample = []
    heading_each_sample = []

    for j in range(len(data["future_heading"])):
        if isinstance(data["future_heading"], torch.Tensor):
            future_heading = data["future_heading"][j].cpu().numpy()
            agent_mask = data['agent_mask'][j].cpu().numpy()
            gt_pos = data['gt_pos'][j].cpu().numpy()
        else:
            future_heading = data["future_heading"][j]
            agent_mask = data['agent_mask'][j]
            gt_pos = data['gt_pos'][j]

        all_heading = future_heading[:, agent_mask]
        trajs = gt_pos
        
        traj_each_agent = {}
        heading_each_agent = {}

        for aix in range(trajs[:, agent_mask, :].shape[1]):
            pos_agent = trajs[:, agent_mask, :][:, aix, :]
            heading_agent = all_heading[:, aix]
            valid_mask = (abs(pos_agent[:, 0]) < VALID_LIMIT) & (abs(pos_agent[:, 1]) < VALID_LIMIT)
            pos_agent = pos_agent[valid_mask]
            pos_step = pos_agent.shape[0]
            
            if s_rate is None:
                sample_rate = pos_step // (action_step + 1)
            else:
                sample_rate = s_rate 
            if sample_num is None:
                sample_num = -1
            
            final_pos = pos_agent[-1]
            pos_agent = pos_agent[::sample_rate][:sample_num]
            pos_agent[-1] = final_pos
            
            traj_each_agent.update({aix: pos_agent})
            heading_agent = heading_agent[valid_mask]
            heading_agent = heading_agent[::sample_rate][:sample_num].reshape((-1, 1))

            for i in range(heading_agent.shape[0]):
                if pos_agent[i, 1] == 0.0:
                    continue
                elif pos_agent[i, 0] == 0.0:
                    heading_agent[i, 0] += np.pi / 2
                else:
                    heading_agent[i, 0] += np.arctan(pos_agent[i, 1] / pos_agent[i, 0])
            
            heading_each_agent.update({aix: heading_agent})

        traj_each_sample.append(traj_each_agent)
        heading_each_sample.append(heading_each_agent)

    return traj_each_sample, heading_each_sample


def _get_all_traj(data, action_step, s_rate=None, sample_num=None):
    VALID_LIMIT = 100
    
    # all_heading = data["future_heading"][:, data['agent_mask']]
    if type(data["future_heading"])==torch.Tensor:
        future_heading =  data["future_heading"].cpu().numpy()[0]
        agent_mask =  data['agent_mask'].cpu().numpy()[0]
        gt_pos = data['gt_pos'].cpu().numpy()[0]
    else:
        future_heading = data["future_heading"]
        agent_mask = data['agent_mask']
        gt_pos = data['gt_pos']

    all_heading = future_heading[:, agent_mask]
    trajs = gt_pos
    
    traj_each_agent = {}
    heading_each_agent = {}

    for aix in range(trajs[:, agent_mask, :].shape[1]):
        pos_agent = trajs[:, agent_mask, :][:, aix, :]
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

def _get_type_traj(data, action_step, s_rate=None, sample_num=None):
    VALID_LIMIT = 100
    trajs = data['gt_pos']
    all_type = data["traj_type"][data['agent_mask']]

    return all_type

def _get_neighbor_text(data, default, max_agents):
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
    action_step = 4
    action_dim = 1
    # future_angles = np.cumsum(self.data["future_heading"], axis=0)
    trajs = data['gt_pos']
    # all_heading = data["future_heading"].swapaxe(0,1)[:, data['agent_mask']]
    traj_each_agent = {}
    heading_each_agent = {}
    traj_each_agent, heading_each_agent = _get_all_traj(data, action_step, s_rate=None, sample_num=SAMPLE_NUM)
    default_mask = np.zeros((max_agents, max_agents), dtype=bool)
    # print(heading_each_agent)
    if len(traj_each_agent) <= 1:
        default_mask[0] = 1
        return default_mask

    # default = -1 * torch.ones((max_agents, SAMPLE_NUM, 2))
    # default[0, :] = torch.zeros((1, SAMPLE_NUM, 2)) # both dis, pos
    # default[0, :, 0] = 0  # w/o dis
    # default[0, :, 1] = 0  # w/o pos
    num_veh = len(heading_each_agent)

    for i in range(num_veh):
        
        ego_heading = heading_each_agent[i]      
        # neighbor_trajs_tensor = -1 * torch.ones((max_agents, SAMPLE_NUM * 2))
        ego_traj = traj_each_agent[i]
    # neighbor_trajs_tensor[0, :] = torch.zeros((1, SAMPLE_NUM * 2)) # both dis, pos
    # neighbor_trajs_tensor[0, SAMPLE_NUM:] = torch.zeros((1, SAMPLE_NUM)) # w/o dis
    # neighbor_trajs_tensor[0, :SAMPLE_NUM] = torch.zeros((1, SAMPLE_NUM)) # w/o pos
        
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
    # degree_pos_rel = -1 # w/o relstive distance
    # ang = -1 # w/o relative pos
    return [degree_pos_rel, ang]



def kmeans_fuse(data, k, max_time = 10, max_agents = 32, dimension = 5):
    default = -1 * torch.ones((max_agents, dimension))
    try:
        label, init_pos, init_heading, init_vel, init_type = kmeans_label(data, k, max_time)
        km_feat = attr_fuse(data, k, label, max_agents, init_pos, init_heading, init_vel, init_type, dimension)
        return km_feat
    except Exception as e:
        print(e)
        return default
    
def binary_fuse(data, max_agents = 32, dimension = 6):
    default = -1 * torch.ones((max_agents**2, dimension))

    try:
        init_pos, init_heading, init_vel, init_type = collect_data(data)
        binary_mask = binary_attr_fuse(data, default, max_agents, init_pos, init_heading, init_vel, init_type, dimension)
        return default, binary_mask
    except Exception as e:
        raise(e)
        return default
    
def star_fuse(data, max_agents = 32, dimension = 5*2+1):
    default = -1 * torch.ones((max_agents, max_agents, dimension))
    try:
        star_mask = _get_neighbor_text(data, default, max_agents)
        return default, star_mask
    except Exception as e:
        raise(e)
        return default
    
def get_type_interactions(data, max_agents = 32):
    inter_type = _get_inter_type(data, max_agents)[1]
    return inter_type

def calculate_angle_between_vectors(v1, v2):
    # 计算向量v1的方向角
    angle_v1 = np.arctan2(v1[1], v1[0])
    
    # 计算向量v2的方向角
    angle_v2 = np.arctan2(v2[1], v2[0])
    
    # 计算夹角差值
    angle_diff = np.rad2deg(angle_v2 - angle_v1)
    
    # 将角度限制在[-180, 180]范围内
    if angle_diff > 180:
        angle_diff -= 360
    elif angle_diff < -180:
        angle_diff += 360
    
    return angle_diff

    


def type_traj(traj):
    stop_lim = 1.0  # 停止的距离阈值
    ang_straight_lim = 10  # 直行的最大夹角阈值
    ang_lane_change_lim = 20  # 变道的最大夹角阈值
    valid_traj, _ = traj.shape
    # print(traj,"nftypetraj")
    
    # 轨迹过短，无法判断
    if valid_traj <= 2:
        return -1
    
    # 初始位置和最终位置
    pos_init = traj[0]
    pos_final = traj[-1]
    
    # 判断是否为停止状态
    if np.linalg.norm(pos_final - pos_init) < stop_lim:
        return 0  # 停止状态
        
    valid_cnt = 0
    # 寻找第一个有效的向量
    init_vector = None
    for i in range(1, valid_traj):
        candidate_vector = traj[i] - traj[i-1]
        if np.any(np.abs(candidate_vector) > 0.5):  # 至少有一个维度的改变大于0.5
            init_vector = candidate_vector
            valid_cnt = i
            break
    
    # 如果找不到有效的初始向量，则无法判断
    if init_vector is None:
        return -1
    if valid_cnt == (valid_traj-1):
        return 1
    # 记录每一段轨迹的夹角与初始方向的夹角
    angle_changes = []

    # 计算每一段向量与初始向量之间的夹角
    for i in range(valid_cnt, valid_traj - 1):
        current_vector = traj[i+1] - traj[i]
        angle_diff = calculate_angle_between_vectors(init_vector, current_vector)
        angle_changes.append(angle_diff)

    # 获取最大和最小的夹角变化
    max_angle = np.max(angle_changes)
    min_angle = np.min(angle_changes)
    # print(angle_changes,"nftypetraj")

    # 判断轨迹类型
    if all(abs(angle) < ang_straight_lim for angle in angle_changes):
        return 1  # 直行
    
    elif max_angle > 0:  # 左侧的轨迹变化
        if max_angle <= ang_lane_change_lim:#有修改余地
            return 4  # 左变道
        elif max_angle > ang_lane_change_lim:
            return 2  # 左转弯
    
    elif min_angle < 0:  # 右侧的轨迹变化
        if -ang_lane_change_lim <= min_angle :
            return 5  # 右变道
        elif min_angle < -ang_lane_change_lim:
            return 3  # 右转弯
    
    return -1  # 无法判断
    
def _get_inter_type_batch(data, max_agents=32):
    inter_type_batch = []
    traj_type_batch = []
    SAMPLE_NUM = 5
    action_step = 4

    traj_each_sample, heading_each_sample = _get_all_traj_batch(data, action_step, s_rate=None, sample_num=SAMPLE_NUM)

    for j in range(len(traj_each_sample)):
        inter_type = {"overtake": [], "follow": [], "yield": [], "jam": []}
        traj_type = []

        traj_each_agent = traj_each_sample[j]
        heading_each_agent = heading_each_sample[j]

        if len(traj_each_agent) <= 1:
            inter_type_batch.append(inter_type)
            traj_type_batch.append(traj_type)
            continue
        
        num_veh = len(heading_each_agent)
        num_jam = 0

        for i in range(num_veh):
            ego_heading = heading_each_agent[i] 
    
            ego_traj = traj_each_agent[i]        
            ego_type = type_traj(ego_traj)
            traj_type.append(ego_type)

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
                is_yield = type_yield(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type)

                if is_overtake:
                    inter_type["overtake"].append([i, aidx])
                if is_follow:
                    inter_type["follow"].append([i, aidx])
                if is_yield:
                    inter_type["yield"].append([i, aidx])

        is_jam = type_jam(num_jam, num_veh, traj_each_agent)
        if is_jam:
            inter_type["jam"].append(-1)

        inter_type_batch.append(inter_type)
        traj_type_batch.append(traj_type)
    
    return traj_type_batch, inter_type_batch



def _get_inter_type(data, max_agents = 32):
    # print(Interaction.overtake)
    inter_type = {"overtake" : [], "follow" : [], "yield" : [],  "jam" : []}
    traj_type =  []
    SAMPLE_NUM = 10
    action_step = 4
    action_dim = 1
    # future_angles = np.cumsum(self.data["future_heading"], axis=0)
    # trajs = data['gt_pos']
    # all_heading = data["future_heading"][:, data['agent_mask']]
    traj_each_agent = {}
    heading_each_agent = {}
    traj_each_agent, heading_each_agent = _get_all_traj(data, action_step, s_rate=None, sample_num=SAMPLE_NUM)
    # type_each_agent = _get_type_traj(data, action_step, s_rate=None, sample_num=SAMPLE_NUM)
    if len(traj_each_agent) <= 1:
        # inter_type.append(-1)
        return inter_type
    
    num_veh = len(heading_each_agent)
    num_jam = 0
    # print(num_veh)
    for i in range(num_veh):
        # print(i,"nf_getintertype") #当前车辆编号，为了trajtype里debug
        ego_heading = heading_each_agent[i] 
        # print(i,ego_heading,"nf_getintertype")     
        ego_traj = traj_each_agent[i]        
        ego_type = type_traj(ego_traj)
        traj_type.append(ego_type)
        #print(i,ego_traj,"neighborfuse")
        # "VEHICLE": 1, "PEDESTRIAN": 2, "CYCLIST": 3
        # print(ego_type)
        if ego_type == 0:
            num_jam += 1
        for aidx in range(len(traj_each_agent)):
            if i == aidx:
                continue
            other_traj = traj_each_agent[aidx]
            other_heading = heading_each_agent[aidx]
            # print(aidx)
            other_type = type_traj(other_traj)
            #print(i,aidx)
            # print(other_type)
            is_overtake = type_overtake(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type)
            is_follow = type_follow(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type)
            #is_merge = type_merge(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type)
            is_yield = type_yield(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type)
            #is_surround = type_surround(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type)


            if is_overtake:
                inter_type["overtake"].append([i, aidx])
            if is_follow:
                inter_type["follow"].append([i, aidx])
            #if is_merge:
                #inter_type["merge"].append([i, aidx])
            if is_yield:
                inter_type["yield"].append([i, aidx])
            #if is_surround:
                #inter_type["surround"].append([i, aidx])
           
    is_jam = type_jam(num_jam, num_veh, traj_each_agent)
    if is_jam:
        inter_type["jam"].append(-1)
    
    return traj_type,inter_type


def compute_speed_vector(traj, i):
    return traj[i] - traj[i - 1]

def vector_angle(v1, v2, magnitude_threshold=0.5):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # 如果向量模小于阈值，返回None
    if norm_v1 <= magnitude_threshold or norm_v2 <= magnitude_threshold:
        return None

    cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return angle

def check_parallel_when_stuck(ego_traj, other_traj, valid_step, min_valid_parallel):
    # 针对stuck状态下的有效并行计数
    valid_parallel_count = 0

    for i in range(1, valid_step):
        start_distance = np.linalg.norm(ego_traj[i - 1] - other_traj[i - 1])
        end_distance = np.linalg.norm(ego_traj[i] - other_traj[i])

        # 在stuck状态下，直接判断距离是否在范围内，而不计算角度
        if start_distance <= 16.0 and end_distance <= 16.0:
            valid_parallel_count += 1

    return valid_parallel_count >= min_valid_parallel

def check_parallel_when_moving(ego_traj, other_traj, angle_threshold, valid_step, min_valid_parallel):
    # 针对正常行驶状态下的有效并行计数
    valid_parallel_count = 0

    for i in range(1, valid_step):
        # print(valid_parallel_count)
        ego_vector = compute_speed_vector(ego_traj, i)
        other_vector = compute_speed_vector(other_traj, i)

        angle = vector_angle(ego_vector, other_vector)
        # 如果无法计算夹角或夹角大于阈值，跳过
        if angle is None or angle > angle_threshold:
            continue

        start_distance = np.linalg.norm(ego_traj[i - 1] - other_traj[i - 1])
        end_distance = np.linalg.norm(ego_traj[i] - other_traj[i])
        #print(start_distance,end_distance)

        # 如果起始点和结束点的距离都小于 LANE_WIDTH，则计数一次有效并行
        if start_distance <= 16.0 and end_distance <= 16.0:
            valid_parallel_count += 1

    return valid_parallel_count >= min_valid_parallel

def type_overtake(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type):
    LANE_WIDTH = 4.0
    ANGLE_THRESHOLD = 15.0 / 180.0 * np.pi  # 设定角度阈值
    valid_step = min(len(ego_traj),len(ego_heading),len(other_traj))
    MIN_VALID_PARALLEL = 0.2*valid_step#0.2  # 最小有效并行次数

    # 确保其他车的类型符合条件
    if other_type not in [1, 4, 5]:
        return False

    # 初始和最终位置确认
    rel_dis_init, rel_pos_init = pos_rel(ego_heading[0], ego_traj[0], other_traj[0])
    rel_dis_final, rel_pos_final = pos_rel(ego_heading[-1], ego_traj[-1], other_traj[-1])
    #print(ego_traj[0],other_traj[0])
    #print(ego_traj[-1],other_traj[-1])
    #print(ego_heading)
    #print(other_heading)
    #print("initpos",rel_pos_init)
    #print("finalpos",rel_pos_final)

    # 确保初始位置在other车后方
    if rel_pos_init != 3:
        return False

    # 确保最终位置在other车前方
    pos_front = [0, 1, 5]
    if rel_pos_final not in pos_front:
        return False

    # 判断是否为stuck状态
    if ego_type == 0:
        return check_parallel_when_stuck(ego_traj, other_traj,valid_step, MIN_VALID_PARALLEL)
    else:
        return check_parallel_when_moving(ego_traj, other_traj, ANGLE_THRESHOLD, valid_step, MIN_VALID_PARALLEL)



def type_follow(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type):
    LANE_WIDTH = 4.0
    ANGLE_THRESHOLD_TIGHT = 20.0 / 180.0 * np.pi  # 紧的角度阈值
    ANGLE_THRESHOLD_LOOSE = 60.0 / 180.0 * np.pi  # 宽松的角度阈值
    INTERACTION_THRESHOLD = 16.0  # 设定一个阈值判断交互距离

    # 初始化变量
    consistent_steps = 0
    following_detected = False
    interaction_exists = False
    pos_rear = [2, 3, 4]
    #print("1",len(ego_heading))
    #print("2",len(ego_traj))
    valid_step = min(len(ego_traj),len(ego_heading),len(other_traj))
    MIN_CONSISTENT_STEPS = 0.4*valid_step

    for i in range(1, valid_step):
        #print(consistent_steps)
        # print(consistent_steps)
        _, rel_pos = pos_rel(ego_heading[i], ego_traj[i], other_traj[i])
        rel_dis = np.linalg.norm(other_traj[i] - ego_traj[i])
        #print(rel_dis)

        # 检查是否在交互范围内
        if rel_dis <= INTERACTION_THRESHOLD:
            interaction_exists = True

        ego_vector = compute_speed_vector(ego_traj, i)
        other_vector = compute_speed_vector(other_traj, i)

        angle = vector_angle(ego_vector, other_vector)
        # 如果角度无法计算，跳过当前循环
        if angle is None:
            continue

        if following_detected:
            # 在跟随的情况下，判断相对位置或角度超出允许范围
            if (rel_pos not in pos_rear) or angle > ANGLE_THRESHOLD_LOOSE:
                #print(rel_pos,angle)
                following_detected = False
                break
        else:
            # 初步判断是否符合跟随的条件
            if rel_pos != 3 or angle > ANGLE_THRESHOLD_TIGHT:
                consistent_steps = 0
                continue
            else:
                #print(-1)
                consistent_steps += 1

            # 当连续符合条件的步数达到最小要求时，标记为检测到跟随
            if consistent_steps >= MIN_CONSISTENT_STEPS:
                following_detected = True


    # 最后返回时，确保只有在检测到跟随且存在交互的情况下返回True
    return following_detected and interaction_exists

def type_yield(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type):
    # 常量定义
    HEAD_LIMIT_INF = 15.0 / 180.0 * np.pi  # 判断小型偏航的下限
    HEAD_LIMIT_SUP = 60.0 / 180.0 * np.pi  # 判断小型偏航的上限
    LANE_WIDTH = 4.0
    valid_step = min(len(ego_traj),len(ego_heading),len(other_traj))
    # 判断基于向量的让行行为
    def vector_yield():
        INTERACTION_THRESHOLD = 16.0
        _, rel_pos_init = pos_rel(ego_heading[0], ego_traj[0], other_traj[0])
        interaction_exists = False
        significant_heading_change = False

        # 确保初始状态下other车在ego车的前方
        if rel_pos_init not in [1, 0, 5]:
            return False
            
        if rel_pos_init == 0:

            for i in range(1, valid_step):
                ego_vector = compute_speed_vector(ego_traj, i)
                other_vector = compute_speed_vector(other_traj, i)

                angle = vector_angle(ego_vector, other_vector)
                # 如果角度无法计算，继续下一个循环
                if angle is None:
                    continue

            # 检查是否存在交互
                if np.linalg.norm(other_traj[i] - ego_traj[i]) <= INTERACTION_THRESHOLD:
                    interaction_exists = True

            # 判断是否存在适度的速度方向变化
                if HEAD_LIMIT_INF < angle < HEAD_LIMIT_SUP:
                    significant_heading_change = True
        elif rel_pos_init == 1:
        
            for i in range(1, valid_step):
                ego_vector = compute_speed_vector(ego_traj, i)
                other_vector = compute_speed_vector(other_traj, i)

                angle = vector_angle(ego_vector, other_vector)
                
                cross_product = ego_vector[0] * other_vector[1] - ego_vector[1] * other_vector[0]
    
                # 判断方向并调整角度
                # 如果角度无法计算，继续下一个循环
                if angle is None:
                    continue

            # 检查是否存在交互
                if np.linalg.norm(other_traj[i] - ego_traj[i]) <= INTERACTION_THRESHOLD:
                    interaction_exists = True

            # 判断是否存在适度的速度方向变化
                if HEAD_LIMIT_INF < angle < HEAD_LIMIT_SUP:
                    if cross_product > 0:
                        significant_heading_change = False
                    significant_heading_change = True
        else:
        
            for i in range(1, valid_step):
                ego_vector = compute_speed_vector(ego_traj, i)
                other_vector = compute_speed_vector(other_traj, i)

                angle = vector_angle(ego_vector, other_vector)
                
                cross_product = ego_vector[0] * other_vector[1] - ego_vector[1] * other_vector[0]
    
                # 判断方向并调整角度
                # 如果角度无法计算，继续下一个循环
                if angle is None:
                    continue

            # 检查是否存在交互
                if np.linalg.norm(other_traj[i] - ego_traj[i]) <= INTERACTION_THRESHOLD:
                    interaction_exists = True

            # 判断是否存在适度的速度方向变化
                if HEAD_LIMIT_INF < angle < HEAD_LIMIT_SUP:
                    if cross_product < 0:
                        significant_heading_change = False
                    significant_heading_change = True
        
            

        # 当有显著方向变化且存在交互时，返回True
        return significant_heading_change and interaction_exists
    def speed_yield():
        _, rel_pos_init = pos_rel(ego_heading[0], ego_traj[0], other_traj[0])
        rel_dis_init = np.linalg.norm(other_traj[0] - ego_traj[0])
        speed_change = False
        Interaction_existence = False
        decreasing_distance_count = 0
        Threshold = 16.0 #控制记录interaction的距离
        otherspeed = []
        validcount = 0.4 *valid_step

        if rel_pos_init not in [1, 0, 5]:
            return False

        for i in range(2, len(ego_traj)):
            _, rel_pos = pos_rel(ego_heading[i], ego_traj[i], other_traj[i])
            rel_dis = np.linalg.norm(other_traj[i] - ego_traj[i])

            if rel_dis <= Threshold and not Interaction_existence:
                Interaction_existence = True

            other_speed = np.linalg.norm(other_traj[i] - other_traj[i-1])

            if rel_dis < 0.9*rel_dis_init and (other_speed <= 0.7*np.linalg.norm(other_traj[i-1] - other_traj[i-2]) or other_speed <= 1):#避免数值异常，严格一点确保提取正确
                decreasing_distance_count += 1
            else:
                decreasing_distance_count = 0

            if decreasing_distance_count >= validcount and not speed_change:
                speed_change = True

            rel_dis_init = rel_dis

        
        return speed_change and Interaction_existence

    val1 = vector_yield()
    val2 = speed_yield()
    return val1 or val2

def type_jam(num_jam, num_veh, traj_each_agent):
    prob_jam = 0.8
    lim_speed = 1.0
    mean_scene = num_jam / num_veh
    if len(traj_each_agent) < 10:
        return False

    if mean_scene < prob_jam:
        return False
    
    return True

