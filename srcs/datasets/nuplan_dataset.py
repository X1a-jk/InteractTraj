import copy
import os
import pickle

import numpy as np
from torch.utils.data import Dataset

from srcs.utils.agent_process import WaymoAgent
from srcs.utils.utils import process_map, rotate, cal_rel_dir
from srcs.core.registry import registry
from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.type import MetaDriveType

from .description import descriptions
from .description import NeighborCarsDescription

from metadrive.scenario.utils import read_scenario_data, read_dataset_summary

from metadrive.component.block.base_block import BaseBlock
from metadrive.component.lane.scenario_lane import ScenarioLane
from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork
from metadrive.constants import PGDrivableAreaProperty
from metadrive.constants import PGLineType, PGLineColor
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.type import MetaDriveType
from metadrive.utils.interpolating_line import InterpolatingLine
from metadrive.utils.math import resample_polyline, get_polyline_length

from metadrive.scenario.utils import read_dataset_summary, read_scenario_data

from scenarionet.converter.nuplan.type import get_traffic_obj_type, set_light_status
import time
import random

@registry.register_dataset(name='Nuplan')
class NuplanDataset(Dataset):
    def __init__(self, cfg, mode="", prefix=""):
        
        model_cfg = copy.deepcopy(cfg.MODEL)
        data_cfg = copy.deepcopy(cfg.DATASET)
        
        self.data_list_file = os.path.join(data_cfg.DATA_LIST.ROOT, data_cfg.DATA_LIST[mode.upper()])
        self.mode = mode
        cities = ["pittsburgh", "boston", "singapore"]
        if self.mode == 'train':
            dt_path = []
            for city in cities:
                dt_path.append(data_cfg.DATA_PATH + f'processed/{city}/')
            
            # self.data_path = data_cfg.DATA_PATH + 'processed/pittsburgh/'
            self.data_path = dt_path
        else:
            self.data_path = [data_cfg.DATA_PATH + 'processed/test/']

        self.summary_dict = {}
        self.summary_list = []
        self.mapping = {}
        for data_path in self.data_path:
            a, b, c = read_dataset_summary(data_path)
            self.summary_dict.update(a)
            self.summary_list.extend(b)
            self.mapping.update(c)
        
        #self.summary_dict, self.summary_list, self.mapping = read_dataset_summary(self.data_path)

        self.RANGE = data_cfg.RANGE
        self.MAX_AGENT_NUM = data_cfg.MAX_AGENT_NUM
        self.THRES = data_cfg.THRES
        self.mt = data_cfg.MAX_TIME
        self.k = data_cfg.CLUSTER_NUM
        self.kd = data_cfg.CLUSTER_DIM

        self.data_len = len(self.summary_list)
        self.data_loaded = {}
        self.text_lib = set()
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.prefix = prefix

        self.use_cache = False #data_cfg.CACHE
        self.enable_out_range_traj = data_cfg.ENABLE_OUT_RANGE_TRAJ
        self._spawned_objects = dict()
        self.raw_index = 0

        if self.use_cache:
            self._load_cache_data()

    def __len__(self):
        # return sum([len(_) for _ in self.summary_list])
        return len(self.summary_list)
    
    def __getitem__(self, index):
        init_time = time.time()
        # try:
        # index = self.raw_index
        if self.use_cache:
            data = self._cached_data[index]
            data, _ = self._add_text_attr(data)
        else:

            use_cache = True
            
            data = self._get_item_helper(index, use_cache)
            # except:
            #     print(f"{index=}")
            #     try:
            #         index_list =  [i for i in range(200)]
            #         id_temp = random.choice(index_list)
            #         data = self._get_item_helper(id_temp, use_cache)
            #     except:
            #         print(f"{id_temp=}")
            #         try:
            #             index_list =  [i for i in range(200, 400)]
            #             id_temp = random.choice(index_list)
            #             data = self._get_item_helper(id_temp, use_cache)
            #         except:
            #             print(f"{id_temp=}")
            #             try:
            #                 index_list =  [i for i in range(400, 600)]
            #                 id_temp = random.choice(index_list)
            #                 data = self._get_item_helper(id_temp, use_cache)
            #             except:
            #                 index_list =  [i for i in range(600, 700)]
            #                 id_temp = random.choice(index_list)
            #                 data = self._get_item_helper(id_temp, use_cache)


            data, _ = self._add_text_attr(data)
            

        get_time = time.time()
        # print(f"get_data_time: {get_time-init_time}")
        self.raw_index += 1
        return data

    def _load_cache_data(self):
        print('Loading cached data...')
        data_name = 'train' if self.mode == 'train' else 'val'

        if self.enable_out_range_traj:
            file_template = 'cached_{}_data_out_range_mask.npy'
        else:
            file_template = 'cached_{}_data.npy'

        cache_data_path = os.path.join(self.data_path, 'cache', file_template.format(data_name))
        self._cached_data = np.load(cache_data_path, allow_pickle=True).item()
        print('Loading cached data finished.')

        assert len(self._cached_data) == len(self.data_list)

    def _add_text_attr(self, data):
        txt_result = self._get_text(data)
        data['text'] = txt_result['text']
        data['token'] = txt_result['token']
        data['text_index'] = txt_result['index']
        data['traj_type'] = txt_result['traj_type']
        data['nei_text'] = txt_result['nei_text']
        data['nei_pos_i'] = txt_result['nei_pos_i']
        data['nei_pos_f'] = txt_result['nei_pos_f']
        data['type_pos'] = data['text'][:, 0]
        # cluster_input = kmeans_fuse(data, self.k, self.mt, self.MAX_AGENT_NUM, self.kd)
        # data['cluster_info'] = cluster_input
        # binary_input, binary_mask = binary_fuse(data, self.MAX_AGENT_NUM, dimension=6)
        # data['binary_info'] = binary_input
        # data['binary_mask'] = binary_mask
        # star_input, star_mask = star_fuse(data, self.MAX_AGENT_NUM, dimension=11)
        # data['star_info'] = star_input
        # data['star_mask'] = star_mask
        # inter_type = get_type_interactions(data, self.MAX_AGENT_NUM)
        # data['inter_type'] = inter_type

        return data, txt_result
    
    

    def _get_text(self, data):
        txt_cfg = self.data_cfg.TEXT
        description = descriptions[txt_cfg.TYPE](data, txt_cfg)
        result = {}
        neighbor_description = NeighborCarsDescription(data, txt_cfg)
        text, type_traj = description.get_category_text(txt_cfg.CLASS)
        token = text
        index = []
        neighbor_txt = neighbor_description.get_neighbor_text()
        neighbor_pos_i, neighbor_pos_f = neighbor_description.get_neighbor_rel_pos()
        result['text'] = text
        result['token'] = token
        result['index'] = index
        result['traj_type'] = type_traj 
        result['nei_text'] = neighbor_txt
        result['nei_pos_i'] = neighbor_pos_i
        result['nei_pos_f'] = neighbor_pos_f
        result['type_pos'] = result['text'][:, 0]
        return result

    def _get_item_helper(self, index, use_cache = False):
        # print(file)
        # if len(file.split(' ')) > 1:
        #     file, num = file.split(' ')
        #     index = int(num)
        # else:
        #     index = -1
        # print(index)
        if not use_cache:
            
            summary = self.summary_list[index]
            mapping = self.mapping[summary]
            if "boston" in mapping:
                _f = mapping.split("/")
                mapping = _f[0] + "/" + _f[1] + "/" + "boston" + _f[2]
            
            for city_path in self.data_path:
                if summary in os.listdir(city_path + mapping):
                    self.city = city_path.split("/")[-1]
                    file_path = city_path + mapping + "/" + summary
                    break


            # file_path = self.data_path + mapping + "/" + summary
            self.summary = self.summary_dict[summary]
            init = time.time()

            datas = read_scenario_data(file_path, centralize=True)
        
            index = -1
            self.file = summary


            data = self.nuplan_process(datas, index)
                  

            extra_list = ['label', 'auxiliary_label', \
                          'label_mask', "other_label_mask", "map_fea", "scene_id", "obejct_id_lis", "other_label"]

            data['file'] = summary
            data['index'] = index
            
            for extra_item in extra_list:
                data[extra_item] = datas[extra_item]

            # print(f"{len(data['map_fea'][0]) + len(data['map_fea'][1])=}")
            # print(f"{data['lane_inp'].shape=}")

            data, txt_result = self._add_text_attr(data)
            data = self._add_veh_type(data)

            data["pred_num"] = data["num_veh"]

            if self.mode == 'train':
                root_path = f"/home/ubuntu/DATA2/nuplan/processed/{self.city}_0/"
            else:
                root_path = f"/home/ubuntu/DATA2/nuplan/processed/test_0/"

            file_path = root_path + str(data['file'])

            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
                f.close()
            

            '''
            print(f"{data['lane_inp'].shape=}")
            print(f"{data['lane_mask'].shape=}")
            '''

            # print(f"{data['label'].shape=}")
            # print(f"{data['label_mask'].shape=}")

            return data
        
        else:
            summary = self.summary_list[index]
            mapping = self.mapping[summary]


            if self.mode == 'train':
                for city in ["pittsburgh", "boston", "singapore"]:
                    cached_file_path = f"/home/ubuntu/DATA2/nuplan/processed/{city}_0/"
                    if summary in os.listdir(cached_file_path):
                        file_path = cached_file_path + summary
                        self.city = city
                        self.cached_data = os.listdir(cached_file_path)
                        
                        break
            else:
                cached_file_path = "/home/ubuntu/DATA2/nuplan/processed/test_0/"
            
                self.cached_data = os.listdir(cached_file_path)

            self.file = summary

            # if summary not in self.cached_data:
            #     summary = self.summary_list[0]
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                f.close()
            # data, txt_result = self._add_text_attr(data)  
            # data = self.hdgt_process(data)

            data = self._add_veh_type(data)    
            data = self.wash(data)

            

            # data = self.reconstruct_lane(data)

            # print(f"{len(data['map_fea'][0])=}")
            # print(f"{len(data['map_fea'][1])=}")
            # print(f"{data['lane_inp'].shape=}")

            # if not cache_graph:
            #     data = self.construct_graph(data)
         
            return data
        
    def reconstruct_lane(self, data):
        # lane_inp: 736*6
        # lane_mask: 736, 
        center_num = 384
        edge_num = 736 - 384

        center_len = len(data['map_fea'][0])   # list of dict
        edge_len = len(data['map_fea'][1])

        map_0 = data['map_fea'][0][:center_num]
        map_1 = data['map_fea'][1][:edge_num]

        default_0 = {'xyz': np.zeros((data['map_fea'][0][0]['xyz'].shape)), 'speed_limit': 10000, 'type': 0, 'stop': [], 'signal': [], 'yaw': 0, 'prev': [], 'follow': []}
        default_1 = [7, np.zeros((data['map_fea'][1][0][1].shape))]
        
        padding_0 = [default_0 for _ in range(center_num - center_len)]
        padding_1 = [default_1 for _ in range(edge_num - edge_len)]

        map_0 += padding_0
        map_1 += padding_1

        data['map_fea'][0] = map_0
        data['map_fea'][1] = map_1

        return data

        


    def wash(self, data):
        try:
            data.pop("gt_agent")
            data.pop("gt_agent_mask")
            data.pop("all_agent")
            return data
        except:
            return data
    
    def hdgt_process(self, data):
        agents = data['agent_feature']
        b_s, agent_num, agent_dim = agents.shape

        max_agent_num = self.MAX_AGENT_NUM

        agent_ = np.zeros([b_s, max_agent_num, agent_dim])
        agent_mask_ = np.zeros([b_s, max_agent_num]).astype(bool)

        for i in range(agents.shape[0]):
            agent_i = agents[i]
            agent_i = agent_i[:max_agent_num]

            agent_i = np.pad(agent_i, [[0, max_agent_num - agent_i.shape[0]], [0, 0]])

            agent_[i] = agent_i

        data['agent_feature'] = agent_
        return data


    def _add_veh_type(self, data):
        valid_info = data['agent'][data['agent_mask'], :]
        
        veh_type = -1 * np.ones((data['agent'].shape[0], 1))
        for i in range(valid_info.shape[0]):
            # {"VEHICLE": 0, "TRAFFIC_CONE": 1, "PEDESTRIAN": 2, "CYCLIST": 3, "TRAFFIC_BARRIER": 4}
            if valid_info[i, 7] in [1, 2, 3]:
                veh_type[i, 0] = valid_info[i, 7]
    
            else:
                print("not cleared")
                veh_type[i, 0] = -1
        
        
       
        data['veh_type'] = veh_type
        data['num_veh'] = valid_info.shape[0]

        return data

    
    
    def nuplan_process(self, data, index):
        # objects = self.engine.get_objects(lambda obj: not is_map_related_instance(obj))
        # this_frame_objects = self._append_frame_objects(objects)
        # self.history_objects.append(this_frame_objects)
        self.block_network = EdgeRoadNetwork()
        self.crosswalks = {}
        self.sidewalks = {}

        processed_data = copy.deepcopy(data)

        self.map_data = data["map_fea"]
        self.traffics = data["dynamic_map_states"]

        self.tracks = data["tracks"]
        self.id = data["id"]

        # ego_car_id = data[SD.METADATA][SD.SDC_ID]
        
        case_info = {}
        other = {}

        max_time_step = self.data_cfg.MAX_TIME_STEP
 
        gap = self.data_cfg.TIME_SAMPLE_GAP

        all_agents, agent_feature, agent_type= self.extract_agents(data, self.tracks, max_time_step)
        
        traffics = self.extract_dynamics(self.traffics)
        
        other['traf'] = traffics
        max_time_step = self.data_cfg.MAX_TIME_STEP
        gap = self.data_cfg.TIME_SAMPLE_GAP
        
        if index == -1:
            processed_data['all_agent'] = all_agents[0:max_time_step:gap]
            processed_data['traffic_light'] = traffics[0:max_time_step:gap]
        else:
            
            index = min(index, len(all_agents)-1)
            processed_data['all_agent'] = all_agents[index:index+self.data_cfg.MAX_TIME_STEP:gap]
            processed_data['traffic_light'] = traffics[index:index+self.data_cfg.MAX_TIME_STEP:gap]     
            

        processed_data["lane"], processed_data["unsampled_lane"] = self.process_map(processed_data)

        
        processed_data['lane'], other['unsampled_lane'] = self._transform_coordinate_map(processed_data)

        other['lane'] = processed_data['lane']

        agent = copy.deepcopy(processed_data['all_agent'])
        ego = agent[:, 0]
        ego_pos = copy.deepcopy(ego[[0], :2])[:, np.newaxis]
        ego_heading = ego[[0], [4]]
        agent[..., :2] -= ego_pos
        agent[..., :2] = rotate(agent[..., 0], agent[..., 1], -ego_heading)
        agent[..., 2:4] = rotate(agent[..., 2], agent[..., 3], -ego_heading)
        agent[..., 4] -= ego_heading
        agent_mask = agent[..., -1]
        agent_type_mask = agent[..., -2]
        agent_range_mask = (abs(agent[..., 0]) < self.RANGE) * (abs(agent[..., 1]) < self.RANGE)
        mask = agent_mask * agent_type_mask * agent_range_mask

        raw_agent = copy.deepcopy(agent)
        agent = WaymoAgent(agent)
        other['gt_agent'] = agent.get_inp(act=True)
        other['gt_agent_mask'] = mask

        case_info["agent"], case_info["agent_mask"] = self._process_agent(processed_data['all_agent'], False)  

        
        
        case_info['center'], case_info['center_mask'], case_info['center_id'], case_info['bound'], case_info['bound_mask'], \
        case_info['cross'], case_info['cross_mask'], case_info['rest'], case_info['rest_mask'] = process_map(
            processed_data['lane'], processed_data['traffic_light'], lane_range=self.RANGE, offest=0)

        # case_info['boundaries'] = processed_data["boundaries"]

        case_info, agent_feature, agent_type = self._get_vec_based_rep(case_info, agent_feature, agent_type)

        agent = WaymoAgent(case_info['agent'], case_info['vec_based_rep'])

        # for idx in range(processed_data['all_agent'].shape[1]):
        #     ag = agent[0, idx, :]
        #     if ag[7] == 2 or ag[7] == 3:
        #         print(f"length: {ag[5]}, width: {ag[6]}")

        case_info['agent_feat'] = agent.get_inp()
        
        
        case_info, lane_num = self._process_map_inp(case_info)
        case_info = self._get_gt(case_info)

        case_num = case_info['agent'].shape[0]
        case_list = []
        for i in range(case_num):
            dic = {}
            for k, v in case_info.items():
                dic[k] = v[i]
            case_list.append(dic)

        
        
        future_attrs = ['gt_pos', 'gt_vec_index', 'traj', 'future_heading', 'future_vel']
        for attr in future_attrs:
            case_list[0][attr] = case_info[attr]
        case_list[0]['all_agent_mask'] = case_info['agent_mask']
        case_list[0]['agent_feature'] = agent_feature
        case_list[0]['agent_type'] = agent_type
        # case_list[0]['pred_num'] = agent_feature.shape[0]
        # include other info for use in MetaDrive
        if self.data_cfg.INCLUDE_LANE_INFO:
            case_list[0]['lane'] = other['lane']
            case_list[0]['traf'] = data['traffic_light']
            case_list[0]['other'] = other
        
        # case_list[0]['gt_agent'] = other['gt_agent']
        # case_list[0]['gt_agent_mask'] = other['gt_agent_mask']
        # case_list[0]['all_agent'] = raw_agent[mask.astype(bool)]

        return case_list[0]



        # traffic_lights = self.extract_dynamics(data, scenario_center)
        # print(traffic_lights)
    

    def _get_gt(self, case_info):
        # 0: vec_index
        # 1-2 long and lat percent
        # 3-5 speed, angle between velocity and car heading, angle between car heading and lane vector
        # 6-9 lane vector
        # 10-11 lane type and traff state
        center_num = case_info['center'].shape[1]
        lane_inp, agent_vec_index, vec_based_rep, bbox, pos = \
            case_info['lane_inp'][:, :center_num], case_info['agent_vec_index'], case_info['vec_based_rep'], case_info['agent'][..., 5:7], case_info['agent'][..., :2]
        b, lane_num, _ = lane_inp.shape
        gt_distribution = np.zeros([b, lane_num])
        gt_vec_based_coord = np.zeros([b, lane_num, 5])
        gt_bbox = np.zeros([b, lane_num, 2])
        for i in range(b):
            mask = case_info['agent_mask'][i]
            index = agent_vec_index[i][mask].astype(int)
            gt_distribution[i][index] = 1
            gt_vec_based_coord[i, index] = vec_based_rep[i, mask, :5]
            gt_bbox[i, index] = bbox[i, mask]
        case_info['gt_vec_index'] = agent_vec_index
        case_info['gt_pos'] = pos
        case_info['gt_bbox'] = gt_bbox
        case_info['gt_distribution'] = gt_distribution
        case_info['gt_long_lat'] = gt_vec_based_coord[..., :2]      
        case_info['gt_speed'] = gt_vec_based_coord[..., 2]
        case_info['gt_vel_heading'] = gt_vec_based_coord[..., 3]
        case_info['gt_heading'] = gt_vec_based_coord[..., 4]
        case_info['gt_agent_heading'] = case_info['agent'][..., 4]
        case_info['traj'] = self._get_traj(case_info)
        case_info['future_heading'], case_info['future_vel'] = self._get_future_heading_vel(case_info)

        return case_info

    def _get_vec_based_rep(self, case_info, agent_feature, agent_type):

        thres = self.THRES
        max_agent_num = self.MAX_AGENT_NUM
        # _process future agent

        agent = case_info['agent']
        vectors = case_info["center"]

        agent_mask = case_info['agent_mask']

        vec_x = ((vectors[..., 0] + vectors[..., 2]) / 2)
        vec_y = ((vectors[..., 1] + vectors[..., 3]) / 2)

        agent_x = agent[..., 0]
        agent_y = agent[..., 1]

        b, vec_num = vec_y.shape
        _, agent_num = agent_x.shape

        vec_x = np.repeat(vec_x[:, np.newaxis], axis=1, repeats=agent_num)
        vec_y = np.repeat(vec_y[:, np.newaxis], axis=1, repeats=agent_num)

        agent_x = np.repeat(agent_x[:, :, np.newaxis], axis=-1, repeats=vec_num)
        agent_y = np.repeat(agent_y[:, :, np.newaxis], axis=-1, repeats=vec_num)

        dist = np.sqrt((vec_x - agent_x) ** 2 + (vec_y - agent_y) ** 2)

        cent_mask = np.repeat(case_info['center_mask'][:, np.newaxis], axis=1, repeats=agent_num)
        dist[cent_mask == 0] = 10e5
        vec_index = np.argmin(dist, -1)
        min_dist_to_lane = np.min(dist, -1)
        min_dist_mask = min_dist_to_lane < thres

        selected_vec = np.take_along_axis(vectors, vec_index[..., np.newaxis], axis=1)

        vx, vy = agent[..., 2], agent[..., 3]
        v_value = np.sqrt(vx ** 2 + vy ** 2)
        low_vel = v_value < 0.1

        dir_v = np.arctan2(vy, vx)
        x1, y1, x2, y2 = selected_vec[..., 0], selected_vec[..., 1], selected_vec[..., 2], selected_vec[..., 3]
        dir = np.arctan2(y2 - y1, x2 - x1)
        agent_dir = agent[..., 4]

        v_relative_dir = cal_rel_dir(dir_v, agent_dir)
        relative_dir = cal_rel_dir(agent_dir, dir)

        v_relative_dir[low_vel] = 0

        v_dir_mask = abs(v_relative_dir) < np.pi / 6
        dir_mask = abs(relative_dir) < np.pi / 4

        agent_x = agent[..., 0]
        agent_y = agent[..., 1]
        vec_x = (x1 + x2) / 2
        vec_y = (y1 + y2) / 2

        cent_to_agent_x = agent_x - vec_x
        cent_to_agent_y = agent_y - vec_y

        coord = rotate(cent_to_agent_x, cent_to_agent_y, np.pi / 2 - dir)

        vec_len = np.clip(np.sqrt(np.square(y2 - y1) + np.square(x1 - x2)), a_min=4.5, a_max=5.5)

        lat_perc = np.clip(coord[..., 0], a_min=-vec_len / 2, a_max=vec_len / 2) / vec_len
        long_perc = np.clip(coord[..., 1], a_min=-vec_len / 2, a_max=vec_len / 2) / vec_len
        
        if self.enable_out_range_traj:
            # ignore other masks for future agents (to support out-of-range agent prediction)
            total_mask = agent_mask
            # for the first frame, use all masks to filter out off-road agents
            # total_mask[0, :] = (min_dist_mask * agent_mask * v_dir_mask * dir_mask)[0, :]
            total_mask[0, :] = (min_dist_mask * agent_mask )[0, :] #* v_dir_mask
        else:
            total_mask = agent_mask * min_dist_mask * v_dir_mask * dir_mask

        total_mask[:, 0] = 1
        total_mask = total_mask.astype(bool)

        b_s, agent_num, agent_dim = agent.shape # 50 * 16 * dim

        # agent_feature  16 * dim * 10

        agent_ = np.zeros([b_s, max_agent_num, agent_dim])
        agent_fea = np.pad(agent_feature, [[0, max_agent_num-agent_feature.shape[0]], [0, 0], [0, 0]])
        agent_tp = np.pad(agent_type, [[0, max_agent_num-agent_type.shape[0]]])
        agent_mask_ = np.zeros([b_s, max_agent_num]).astype(bool)

        the_vec = np.take_along_axis(vectors, vec_index[..., np.newaxis], 1)
        # 0: vec_index
        # 1-2 long and lat percent
        # 3-5 velocity and direction
        # 6-9 lane vector
        # 10-11 lane type and traff state
        info = np.concatenate(
            [vec_index[..., np.newaxis], long_perc[..., np.newaxis], lat_perc[..., np.newaxis],
            v_value[..., np.newaxis], v_relative_dir[..., np.newaxis], relative_dir[..., np.newaxis], the_vec], -1)

        info_ = np.zeros([b_s, max_agent_num, info.shape[-1]])

        start_mask = total_mask[0]
        for i in range(agent.shape[0]):
            agent_i = agent[i][start_mask]
            info_i = info[i][start_mask]

            step_mask = total_mask[i]
            valid_mask = step_mask[start_mask]

            agent_i = agent_i[:max_agent_num]
            info_i = info_i[:max_agent_num]

            valid_num = agent_i.shape[0]
            agent_i = np.pad(agent_i, [[0, max_agent_num - agent_i.shape[0]], [0, 0]]) # 7 * 9 -> 32 * 9 
            info_i = np.pad(info_i, [[0, max_agent_num - info_i.shape[0]], [0, 0]])

            agent_[i] = agent_i
            info_[i] = info_i
            agent_mask_[i, :valid_num] = valid_mask[:valid_num]            

        case_info['vec_based_rep'] = info_[..., 1:]
        case_info['agent_vec_index'] = info_[..., 0].astype(int)
        case_info['agent_mask'] = agent_mask_       
        case_info["agent"] = agent_

        agent_feature = agent_fea
        agent_type = agent_tp

        return case_info, agent_feature, agent_type
    
    def _process_agent(self, agent, sort_agent):

        ego = agent[:, 0]

        # transform every frame into ego coordinate in the first frame
        ego_pos = copy.deepcopy(ego[[0], :2])[:, np.newaxis]
        ego_heading = ego[[0], [4]]

        agent[..., :2] -= ego_pos
        agent[..., :2] = rotate(agent[..., 0], agent[..., 1], -ego_heading)
        agent[..., 2:4] = rotate(agent[..., 2], agent[..., 3], -ego_heading)
        agent[..., 4] -= ego_heading

        agent_mask = agent[..., -1]
        agent_type_mask = agent[..., -2]
        agent_range_mask = (abs(agent[..., 0]) < self.RANGE) * (abs(agent[..., 1]) < self.RANGE)
        
        if self.enable_out_range_traj:
            mask = agent_mask * agent_type_mask
            # use agent range mask only for the first frame
            # allow agent to be out of range in the future frames
            mask[0, :] *= agent_range_mask[0, :]
        else:
            mask = agent_mask * agent_type_mask * agent_range_mask

        return agent, mask.astype(bool)

    def _transform_coordinate_map(self, data):
        """
        Every frame is different
        """
        timestep = data['all_agent'].shape[0]

        ego = data['all_agent'][:, 0]
        pos = ego[:, [0, 1]][:, np.newaxis]

        lane = data['lane'][np.newaxis]
        lane = np.repeat(lane, timestep, axis=0)
        lane[..., :2] -= pos

        x = lane[..., 0]
        y = lane[..., 1]
        ego_heading = ego[:, [4]]
        lane[..., :2] = rotate(x, y, -ego_heading)

        unsampled_lane = data['unsampled_lane'][np.newaxis]
        unsampled_lane = np.repeat(unsampled_lane, timestep, axis=0)
        unsampled_lane[..., :2] -= pos

        x = unsampled_lane[..., 0]
        y = unsampled_lane[..., 1]
        ego_heading = ego[:, [4]]
        unsampled_lane[..., :2] = rotate(x, y, -ego_heading)
        return lane, unsampled_lane[0]
    



    def process_map(self, datas):   
        boundary_tp = {0 : "LANE_SURFACE_STREET", 1 : "LANE_SURFACE_UNSTRUCTURE", 2 : 'ROAD_LINE_BROKEN_SINGLE_WHITE', 3: 'ROAD_LINE_SOLID_SINGLE_WHITE'}
        center_tp = {0: "LANE_SURFACE_STREET", 1: "LANE_SURFACE_UNSTRUCTURE", 2:'ROAD_LINE_BROKEN_SINGLE_WHITE', 3:'ROAD_LINE_SOLID_SINGLE_WHITE'}
        total_info = {}

         # entry_lanes, exit_lanes

        for idx, info  in enumerate(datas['map_fea'][0]):
            type_ = center_tp[int(info['type'])]
            polyline_ = info['xyz'][:, :2]
            if len(info['prev']):
                prev = [str(i) for i in info['prev'][0]]
            else:
                prev = []
            if len(info['follow']):
                follow = [str(i) for i in info['follow'][0]]
            else:
                follow = []
            total_info[str(idx)] = {"type" : type_, "polyline" : polyline_, "entry_lanes": prev, "exit_lanes": follow}
        
        for idx, info  in enumerate(datas['map_fea'][1]):
            type_ = boundary_tp[int(info[0]) - 5]
            polyline_ = info[1][:, :2]
            prev = []
            follow = []
            total_info[f"boundary_{idx}"] = {"type" : type_, "polyline" : polyline_, "entry_lanes": prev, "exit_lanes": follow}

        datas['map_fea_new'] = total_info
        
        global SAMPLE_NUM
        SAMPLE_NUM = 10
        lane_info = self.extract_map(datas["map_feature"])
        SAMPLE_NUM = 10e9
        unsampled_lane_info = self.extract_map(datas["map_feature"])
        return lane_info, unsampled_lane_info


    def extract_map(self, f):
    
        maps = []
        center_infos = {}
        boundaries = []
        # nearbys = dict()
        for k, v in f.items():
            id = k


            if MetaDriveType.is_lane(v.get("type", None)):            
                line, center_info = self.extract_center(v)
                center_infos[id] = center_info

            elif MetaDriveType.is_road_line(v.get("type", None)): 
                
                line = self.extract_boundaries(v)
                

            elif MetaDriveType.is_crosswalk(v.get("type", None)) or MetaDriveType.is_sidewalk(v.get("type", None)): 
                line = self.extract_crosswalk(v)
        
            else:
                print(f'{v.get("type", None)=}')
                continue

            try:
                line = [np.insert(x, 3, int(id)) for x in line]
            
            except:
                line = [np.insert(x, 3, 18) for x in line] # try

            maps.append(line)

        maps = np.vstack(maps)
        return maps
    
    def extract_poly(self, message):
        x = [i[0] for i in message]
        y = [i[1] for i in message]
        z = [0 for i in message]
        coord = np.stack((x, y, z), axis=1)

        return coord
    
    def extract_boundaries(self, f):
        line_type2int = {"LANE_SURFACE_STREET": 0, "LANE_SURFACE_UNSTRUCTURE": 1, 'ROAD_LINE_BROKEN_SINGLE_WHITE': 2, 'ROAD_LINE_SOLID_SINGLE_WHITE': 3}
        poly = self.down_sampling(self.extract_poly(f['polyline'])[:, :2], 1)
        if type(f['type']) is str:
            tp = line_type2int[f['type']] + 5
        else:
            tp = f['type'] + 5
        poly = [np.insert(x, 2, tp) for x in poly]

        return poly

    def extract_crosswalk(self, f):
        poly = self.down_sampling(self.extract_poly(f['polygon'])[:, :2], 1)
        poly = [np.insert(x, 2, 18) for x in poly]
        return poly
    
    def down_sampling(self, line, type=0):
    # if is center lane
        point_num = len(line)

        ret = []

        if point_num < SAMPLE_NUM or type == 1:
            for i in range(0, point_num):
                ret.append(line[i])
        else:
            for i in range(0, point_num, SAMPLE_NUM):
                ret.append(line[i])

        return ret
    
    def _get_traj(self, case_info):
        traj = case_info['gt_pos']
        
        if self.data_cfg.TRAJ_TYPE == 'xy_relative':
            traj = traj - traj[[0], :]
        elif self.data_cfg.TRAJ_TYPE == 'xy':
            traj = traj
        elif self.data_cfg.TRAJ_TYPE == 'xy_theta_relative':
            # rotate traj of each actor to the direction of the vehicle

            traj = traj - traj[[0], :]
            init_heading = case_info['gt_agent_heading'][0]
            traj = rotate(traj[..., 0], traj[..., 1], -init_heading)
        
        return traj

    def extract_center(self, f):
    # plt.scatter([x[0] for x in f["polyline"]], [y[1] for y in f["polyline"]], s=0.1)
        center = {}
        line_type2int = {"LANE_SURFACE_STREET": 0, "LANE_SURFACE_UNSTRUCTURE": 1, 'ROAD_LINE_BROKEN_SINGLE_WHITE': 2, 'ROAD_LINE_SOLID_SINGLE_WHITE': 3}
        poly = self.down_sampling(self.extract_poly(f['polyline'])[:, :2])
        poly = [np.insert(x, 2, line_type2int[f['type']]) for x in poly]

        center['interpolating'] = [] #f.interpolating

        center['entry'] = [x for x in f['entry_lanes']]

        center['exit'] = [x for x in f['exit_lanes']]

        if "left_boundaries" in f.keys():
            center['left_boundaries'] = [] # extract_boundaries(f["left_boundaries"])
        else:
            center['left_boundaries'] = []

        if "right_boundaries" in f.keys():
            center['right_boundaries'] = [] # extract_boundaries(f["right_boundaries"])
        else:
            center['right_boundaries'] = []
    
        if "left_neighbor" in f.keys():
            center['left_neighbor'] = [] # extract_neighbors(f['left_neighbor'])
        else:
            center['left_neighbor'] = []

        if "right_neighbor" in f.keys():
            center['right_neighbor'] = [] # extract_neighbors(f['right_neighbor'])
        else:
            center['right_neighbor'] = []

        return poly, center

    def extract_agents(self, data, tracks, max_time_step):
        type_vehicle = {"VEHICLE": 1, "TRAFFIC_CONE": 4, "PEDESTRIAN": 2, "CYCLIST": 3, "TRAFFIC_BARRIER": 5}

        extension_rate = 0.5

        num_agents = len(tracks)
        ego_track = tracks["ego"]
        time_steps = max_time_step #ego_track["state"]["position"].shape[0]
        all_tracks = np.zeros((num_agents, time_steps, 9))

        all_tracks[0, :, :2] = ego_track["state"]["position"][:time_steps, :2]
        
        all_tracks[0, :, 2:4] = ego_track["state"]["velocity"][:time_steps, :]
        all_tracks[0, :, 4] = ego_track["state"]["heading"][:time_steps]
        all_tracks[0, :, 5] = ego_track["state"]["length"][:time_steps, 0]
        all_tracks[0, :, 6] = ego_track["state"]["width"][:time_steps, 0]
        all_tracks[0, :, 7] = ego_track[SD.TYPE] # type_vehicle[ego_track[SD.TYPE]]
        all_tracks[0, :, 8] = ego_track["state"]["valid"][:time_steps]

        idx = 0
        rows_to_del = []

        

        for id, track in tracks.items():
            if id == "ego":
                continue
            all_tracks[idx+1, :, :2] = track["state"]["position"][:time_steps, :2]
            
            all_tracks[idx+1, :, 2:4] = track["state"]["velocity"][:time_steps, :]
            all_tracks[idx+1, :, 4] = track["state"]["heading"][:time_steps]
            all_tracks[idx+1, :, 5] = track["state"]["length"][:time_steps, 0]
            all_tracks[idx+1, :, 6] = track["state"]["width"][:time_steps, 0]
            all_tracks[idx+1, :, 7] = track[SD.TYPE] #type_vehicle[track[SD.TYPE]]
            all_tracks[idx+1, :, 8] = track["state"]["valid"][:time_steps]
            idx += 1


            if not track[SD.TYPE] in [1, 2, 3]:
                rows_to_del.append(idx)

            if all_tracks[idx, 0, 0] < 0.5 and all_tracks[idx, 0, 1] < 0.5:
                # print(all_tracks[idx, :, :2])
                rows_to_del.append(idx)
            
            if all_tracks[idx, -1, 0] < 0.5 and all_tracks[idx, -1, 1] < 0.5:
                # print(all_tracks[idx, :, :2])
                rows_to_del.append(idx)
            
            # elif track[SD.TYPE] in ["PEDESTRIAN", "CYCLIST"]:
            #     if all_tracks[idx, -1, 0] < 0.5 and all_tracks[idx, -1, 1] < 0.5:
            #         rows_to_del.append(idx)
            
        agent_feature = data['agent_feature']
        agent_type = data['agent_type']

        if len(rows_to_del):  
            all_tracks = np.delete(all_tracks, rows_to_del, axis = 0)
            agent_feature = np.delete(data['agent_feature'], rows_to_del, axis = 0)
            agent_type = np.delete(data['agent_type'], rows_to_del, axis = 0)
        
        return all_tracks.swapaxes(0, 1), agent_feature, agent_type
        

        # attach_to_world
    def _get_future_heading_vel(self, case_info):
        # get the future heading and velocity of each agent
        # use the relative direction and velocity to the initial direction

        future_heading = copy.deepcopy(case_info['gt_agent_heading'])
        future_vel = copy.deepcopy(case_info['agent'][..., 2:4])

        future_heading = cal_rel_dir(future_heading, case_info['gt_agent_heading'][0])
        future_vel = rotate(future_vel[..., 0], future_vel[..., 1], -case_info['gt_agent_heading'][0])

        return future_heading, future_vel

    def extract_dynamics(self, f):
        time_steps = 0
        for k,v in f.items():
            time_steps = len(v['state']['object_state'])
            break

        traffic_lights = [list() for i in range(time_steps)]

        for k,v in f.items():
            # states = f[i * time_sample].lane_states
            traf_list = np.zeros(6)

            for i, state in enumerate(v['state']['object_state']):
                traf_list[0] = int(v['lane'])
                try:
                    traf_list[1:4] = np.array([[v['start_point'][0], v['stop_point'][1], 0]])
                except:
                    traf_list[1:4] = np.array([[0, v['stop_point'][1], 0]])
                if state == 'TRAFFIC_LIGHT_RED':
                    state_ = 1  # stop
                elif state == 'TRAFFIC_LIGHT_YELLOW':
                    state_ = 2  # caution
                elif state == 'TRAFFIC_LIGHT_GREEN':
                    state_ = 3  # go
                else:
                    state_ = 0  # unknown
        
                traf_list[4] = state_
                traf_list[5] = 1 if v['state'] else 0
            
                traffic_lights[i].append(traf_list)
            
        return traffic_lights

    
        
    


    def get_objects(self, filter = None):
        """
        Return objects spawned, default all objects. Filter_func will be applied on all objects.
        It can be a id list or a function
        Since we don't expect a iterator, and the number of objects is not so large, we don't use built-in filter()
        :param filter: a filter function, only return objects satisfying this condition
        :return: return all objects or objects satisfying the filter_func
        """
        if filter is None:
            return self._spawned_objects
        elif isinstance(filter, (list, tuple)):
            return {id: self._spawned_objects[id] for id in filter}
        elif callable(filter):
            res = dict()
            for id, obj in self._spawned_objects.items():
                if filter(obj):
                    res[id] = obj
            return res
        else:
            raise ValueError("filter should be a list or a function")


    def _get_episode_light_data(self):
        ret = dict()
        for lane_id, light_info in self.current_scenario[SD.DYNAMIC_MAP_STATES].items():
            ret[lane_id] = copy.deepcopy(light_info[SD.STATE])
            ret[lane_id]["metadata"] = copy.deepcopy(light_info[SD.METADATA])

            if SD.TRAFFIC_LIGHT_POSITION in ret[lane_id]:
                # Old data format where position is a 2D array with shape [T, 2]
                traffic_light_position = ret[lane_id][SD.TRAFFIC_LIGHT_POSITION]

                if not np.any(ret[lane_id][SD.TRAFFIC_LIGHT_LANE].astype(bool)):
                    # This traffic light has no effect.
                    first_pos = -1
                else:
                    first_pos = np.argwhere(ret[lane_id][SD.TRAFFIC_LIGHT_LANE] != 0)[0, 0]
                traffic_light_position = traffic_light_position[first_pos]
            else:
                # New data format where position is a [3, ] array.
                traffic_light_position = light_info[SD.TRAFFIC_LIGHT_POSITION][:2]

            ret[lane_id][SD.TRAFFIC_LIGHT_POSITION] = traffic_light_position

            assert light_info[SD.TYPE] == MetaDriveType.TRAFFIC_LIGHT, "Can not handle {}".format(light_info[SD.TYPE])
        return ret

    def _process_map_inp(self, case_info):
        center = copy.deepcopy(case_info['center'])
        center[..., :4] /= self.RANGE
        edge = copy.deepcopy(case_info['bound'])
        edge[..., :4] /= self.RANGE
        cross = copy.deepcopy(case_info['cross'])
        cross[..., :4] /= self.RANGE
        rest = copy.deepcopy(case_info['rest'])
        rest[..., :4] /= self.RANGE


        case_info['lane_inp'] = np.concatenate([center, edge, cross, rest], axis=1)
        case_info['lane_mask'] = np.concatenate(
            [case_info['center_mask'], case_info['bound_mask'], case_info['cross_mask'], case_info['rest_mask']],
            axis=1)

        lane_num = case_info['lane_inp'].shape[0]

        return case_info, lane_num

   
