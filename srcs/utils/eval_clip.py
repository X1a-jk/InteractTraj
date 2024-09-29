import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

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
from tqdm import tqdm
from srcs.models.clip_model import CLIP
import clip
from transformers import CLIPProcessor, CLIPModel
import torch.distributed as dist_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from multiprocessing import Pool, cpu_count, RLock, freeze_support
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch
import openai
from srcs.utils import load_all_map_vectors
from srcs.utils import output_formating_cot, map_retrival, get_map_data_batch

def get_eval_model(cfg_path):
    cfg = get_config(cfg_path)
    model_cls = registry.get_model(cfg.MODEL.TYPE)
    model = model_cls.load_from_checkpoint(cfg.LOAD_CHECKPOINT_PATH, config=cfg, metrics=[], strict=False)
    model.eval()
    return model

def get_query(query_path):
    queries = []
    labels = []
    return queries, labels

def load_clip_model(ckpt, device):
    model = CLIP(device = device).to(torch.float32) 
    model.load_state_dict(torch.load(ckpt))   # 然后加载模型的state_dict
    model.eval()
    return model


def build_dataset(cfg_path):
    cfg = get_config(cfg_path)
    dataset_type = cfg.DATASET.TYPE
    cfg.DATASET['CACHE'] = False
    dataset = registry.get_dataset(dataset_type)(cfg, 'train')
    return dataset

def gen_scenario_from_gpt_text(llm_text, cfg, map_id=None):
    map_data_file = "/home/ubuntu/xiajunkai/lctgen/data/map.npy"
    map_vecs, map_ids = load_all_map_vectors(map_data_file)
    # format LLM output to Structured Representation (agent and map vectors)
    MAX_AGENT_NUM = 32
    agent_vector, map_vector, event_vector = output_formating_cot(llm_text)
    agent_num = len(agent_vector)
    vector_dim = len(agent_vector[0])
    event_dim = len(event_vector[0])

    type_vector = [it[-1] for it in agent_vector]
    agent_vector = [it[:-1] + [-1] for it in agent_vector]
    
    agent_vector = agent_vector + [[-1]*vector_dim] * (MAX_AGENT_NUM - agent_num)
    event_vector = event_vector + [[-1]*event_dim] * (MAX_AGENT_NUM - agent_num)
    
    # retrive map from map dataset
    if not map_id:
        sorted_idx = map_retrival(map_vector, map_vecs)[:1]
        map_id = map_ids[sorted_idx[0]]
    else:
        map_id = '6_3921.pkl 10'
    #load map data
    batch = get_map_data_batch(map_id, cfg)
    type_len = batch['traj_type'].shape[1]
    for i in range(type_len):
        if i<len(type_vector):
            batch['traj_type'][0, i, 0] = type_vector[i]
        else:
            batch['traj_type'][0, i, 0] = -2
    return batch, agent_vector, event_vector
    
def get_embeddings(batch, query, agent_vector, event_vector, model, clip_model):
    
    MAX_AGENT_NUM = 32
    agent_num = len(agent_vector)
    
    batch['text'] = torch.tensor(agent_vector, dtype=batch['text'].dtype, device=model.device)[None, ...]
    event_tensor = torch.tensor(event_vector, dtype=batch['nei_text'][1].dtype, device=model.device)[None, ...]
    batch['text'] = batch['text'][:, :, :-1]
    batch['nei_text'] = [batch['nei_text'][0], event_tensor]
    b, d, _ = batch['text'].shape
    padding = -1 * torch.ones((b, d, 1), device=model.device)
#    batch['text'] = torch.cat((padding, batch['text']), dim=-1)
    b_2, d_2, _ = batch['nei_text'][1].shape
    
    # batch['nei_text'][1] = torch.cat((batch['nei_text'][1],padding_2), dim=-1)
    batch['agent_mask'] = torch.tensor([1]*agent_num + [0]*(MAX_AGENT_NUM - agent_num), \
            dtype=batch['agent_mask'].dtype, device=model.device)[None, ...]

    model_output = model.forward(batch, 'val')['text_decode_output']
    output_scene = model.process(model_output, batch, num_limit=1, with_attribute=True, pred_ego=True, pred_motion=True)

    traj_infer = output_scene[0]['traj']
    if traj_infer.shape[1]<MAX_AGENT_NUM:
        padding_2 = -1 * torch.ones((50, MAX_AGENT_NUM-traj_infer.shape[1],2), device=model.device)
        traj_infer = torch.cat([traj_infer, padding_2], dim = 1)

    prompt = [query]
    batch['traj'] = traj_infer

    logits_per_image, logits_per_text = clip_model(batch, prompt)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    #     # return "finished"
    #     exceed = output_scene[0]['exceed']
    #     print(exceed)
    #     i+=1
    # time_b = time.time()
    # print(f"{time_a-time_c=}")
    # print(f"pure decoding: {time_b-time_a}")
    # print(f"{time_b-time_c=}")
    return probs



def cosine_sim(cfg, query, llm_model, inter_model, lct_model, clip_model):
    
    llm_result = llm_model.forward(query)
    batch, agent_vector, event_vector = gen_scenario_from_gpt_text(llm_result, cfg, map_id=None)
    lct_embed = get_embeddings(batch, query, agent_vector, event_vector, lct_model, clip_model)
    inter_embed = get_embeddings(batch, query, agent_vector, event_vector, inter_model, clip_model)
    return lct_embed, inter_embed

if __name__ == "__main__":
    lctgen_cfg = "/home/ubuntu/xiajunkai/network/lctgen/cfgs/inference.yaml"
    inter_cfg = "/home/ubuntu/xiajunkai/lctgen/cfgs/inference.yaml"
    clip_ckpt = "/home/ubuntu/xiajunkai/lctgen/best_clip_model.pth"

    openai.api_key = "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"
    openai.base_url = "https://integrate.api.nvidia.com/v1/"

    lctgen_model = get_eval_model(lctgen_cfg)
    inter_model = get_eval_model(inter_cfg)
    clip_model = load_clip_model(clip_ckpt)
    llm_cfg = get_config('./lctgen/gpt/cfgs/attr_ind_motion/new.yaml') # new.yaml')
    llm_model = registry.get_llm('codex')(llm_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = build_dataset(inter_cfg)

    query_path = ""
    queries, labels = get_query(query_path)

    tot = 0
    for i, query in enumerate(queries):
        sim_inter, sim_lct = cosine_sim(inter_cfg, query, llm_model, inter_model, lctgen_model, clip_model)
        if sim_inter >= sim_lct and labels[i]:
            tot += 1

    result = tot / len(queries)

    print(f"Accuracy: {result}")


