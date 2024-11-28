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
from srcs.models.neighbor_fuse import _get_inter_type, _get_inter_type_batch
import random


def generate_prompt_batch(data):# may need to adapt to evaluate step
    traj_type,interactions = _get_inter_type_batch(data)
    bz = data['traj_type'].shape[0]
    res = []
    #print(data['traj_type'])
    
    behaviour_templates = {
        "overtake": [
            "{actor2} overtakes {actor1} while changing lanes.",
            "{actor2} speeds up and passes {actor1}.",
            "{actor2} moves ahead, overtaking {actor1} swiftly."
        ],
        "yield": [
            "{actor2} slows down to let {actor1} pass.",
            "{actor2} yields the road to {actor1}.",
            "{actor2} gives way to {actor1}, allowing them to proceed first."
        ],
        "follow": [
            "{actor2} is following closely behind {actor1}.",
            "{actor2} keeps a steady distance behind {actor1}.",
            "{actor2} follows {actor1}, maintaining its position in the same lane."
        ],
        "jam": [
            "There is a heavy traffic jam, with vehicles stuck in place.",
            "The road is blocked due to a traffic jam.",
            "Traffic is congested, causing a significant jam."
        ]
    }
    
    trajectory_templates = {
        0: ["comes to a stop.", "halts completely.", "is stuck."],
        1: ["continues driving straight ahead.", "moves forward steadily in a straight path.", "proceeds straight."],
        2: ["executes a left turn, changing direction.", "makes a smooth left turn, adjusting its course.", "turns left."],
        3: ["executes a right turn, changing direction.", "makes a smooth right turn, adjusting its course.", "turns right."],
        4: ["performs a left lane change.", "shifts to the left lane.", "switches to the left lane, altering its position."],
        5: ["performs a right lane change.", "shifts to the right lane.", "switches to the right lane, altering its position."]
    }

    templates = [
        "In this scenario, there are {num_vehicle} agents. {description}",
        "{num_vehicle} agents are present. {description}",
        "{num_vehicle} agents are involved, and {description}"
    ]
    
    for j in range(bz):
        #print(data['traj_type'])
        #print(data['traj_type'].shape)
        #print(data['agent_mask'])
        #print(data['agent_mask'].shape)
        #traj_type = data['traj_type'][data['agent_mask'],j].tolist()
        #print(traj_type)
        #veh_type = data['veh_type'][:, data['agent_mask'][j], :].int().tolist()[j]
        num_vehicle = len(traj_type[j])
        # print(num_vehicle)
        agent_type = ["Agent", "Vehicle", "Pedestrian", "Bicycle"]  
        description = ""
        

        for i in range(num_vehicle):
            traj_idx = traj_type[j][i]
            if traj_idx != -1:
                trajectory_desc = random.choice(trajectory_templates[traj_type[j][i]])
                #print(trajectory_desc)
                description += f"Agent {i} {trajectory_desc}"


        for behaviour, participants in interactions[j].items():
            if behaviour == "jam" and participants != []:
                behaviour_desc = random.choice(behaviour_templates[behaviour])
                description += behaviour_desc
            else:
                if len(participants) > 0:
                    for actor_1, actor_2 in participants:
                        behaviour_desc = random.choice(behaviour_templates[behaviour])
                        description += behaviour_desc.format(actor1=f"Agent {actor_1}", actor2=f"Agent {actor_2}")

        template = random.choice(templates)
        prompt = template.format(num_vehicle=num_vehicle, description=description.strip())
        res.append(prompt)
    return res 

def generate_prompt(data): #capable to generate single prompt not batch
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
        print(len(prompt))
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