import numpy
import pickle
import torch
import sys
from torch.utils.data import DataLoader
sys.path.append("/home/ubuntu/xiajunkai/lctgen/")
from srcs.utils.agent_process import WaymoAgent
from PIL import Image
import time
from srcs.datasets.utils import fc_collate_fn
from srcs.configs.default import get_config
from srcs.core.registry import registry
from srcs.utils.utils import visualize_input_seq, visualize_output_seq, visualize_map
from srcs.utils.typedef import *
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
from srcs.models.neighbor_fuse import _get_inter_type


def generate_prompt(data):
    interactions = _get_inter_type(data)
    bz = data['traj_type'].shape[0]
    res = []
    for j in range(bz):
        # print(f"{data['traj_type'].shape=}")
        # print(len(data['traj_type'][:, data['agent_mask'][0], :].cpu().tolist()))
        # print(f"{interactions=}")
        traj_type = data['traj_type'][:, data['agent_mask'][j], :].cpu().tolist()[j]
        veh_type = data['veh_type'][:, data['agent_mask'][j], :].cpu().int().tolist()[j]
        assert len(traj_type) == len(veh_type)
        num_vehicle = len(traj_type)
        trajectories = ["stopping", "driving straight forward", "making a left turn", "making a right turn", "making a left lane change", "making a right lane change"]
        agent_type = ["", "Vehicle", "Pedestrian", "Bicycle"] # -> "black", "red", "yellow"

        template = f"There are {num_vehicle} agents in the scenario. "
        for i in range(num_vehicle):
            try:
                # print(veh_type[i][0])
                template += f"{agent_type[veh_type[i][0]]} {i} is {trajectories[traj_type[i][0]]}."
            except:
                print(traj_type)
                print(len(traj_type))

        for behaviour, participants in interactions.items():
            if behaviour == "jam":
                template += "There is a traffic jam. "
                continue
            if len(participants) == 0:
                continue
            for actor1, actor2 in participants:
                s = f"There is a {behaviour} taking place between {actor1} and {actor2}. "
                template += s

        res.append(template)

    return res

if __name__ == "__main__":

    cfg_file = './cfgs/inference.yaml'
    cfg = get_config(cfg_file)

    dataset_type = cfg.DATASET.TYPE
    cfg.DATASET['CACHE'] = False
    dataset = registry.get_dataset(dataset_type)(cfg, 'train')

    print(len(dataset))

    collate_fn = fc_collate_fn
    loader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory = False,
                    drop_last=False, num_workers=4, collate_fn=collate_fn)

    print(f"total capacity: {len(loader)}")
    type_lst = [0,0,0]
    init_time = time.time()




    pairs = {}
    for i, batch in enumerate(loader):
        data = batch
        veh_type = data['veh_type'][:, data['agent_mask'][0], :].cpu().tolist()[0]

        traj_type = data['traj_type'][:, data['agent_mask'][0], :].cpu().tolist()[0]
        
        file_id = batch['file'][0].split(".")[1]
        
        prompt = generate_prompt(data)

        if data["num_veh"].cpu().int().item() >= 0:
            # print(f'{data["agent_mask"].sum()=}')
            file_name = "./nuplan_vis/" + file_id+'.png'
            # map_name = "./nuplan_vis/" + file_id+'_map.png'
            gif_name = './nuplan_vis/' + file_id+'.gif'

        pairs[file_id] = prompt
        # break
        # demo_fig = visualize_input_seq(data, save=True, filename=file_name)
        # maps = visualize_map(data, save=True, path=map_name)
        
        if i % 100 == 0 and i > 0:
            pre_time = time.time()
            print(f"processed: {i}, total: {len(loader)}")
            print(f"current time: {pre_time-init_time}")
        
            # break

    import json
    with open('prompt_demo.json', 'w') as f:
        json.dump(pairs, f)

    print("over")
