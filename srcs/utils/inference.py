import torch
from PIL import Image
import time
import openai
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from srcs.configs.default import get_config
from srcs.core.registry import registry
from srcs.utils.utils import load_all_map_vectors, output_formating_cot, map_retrival, get_map_data_batch, visualize_input_seq, visualize_output_seq
from srcs.typedef import *


def vis_decode(batch, ae_output):
    img = visualize_output_seq(batch, output=ae_output[0], pool_num=1)
    return img

def vis_stat(batch, ae_output):
    img = visualize_input_seq(batch, agents=ae_output[0]['agent'], traj=ae_output[0]['traj'])
    return Image.fromarray(img)

def load_model(cfg_file):
    # cfg_file = './cfgs/inference.yaml'
    cfg = get_config(cfg_file)
    model_cls = registry.get_model(cfg.MODEL.TYPE)
    model = model_cls.load_from_checkpoint(cfg.LOAD_CHECKPOINT_PATH, config=cfg, metrics=[], strict=False)
    model.eval()
    return cfg, model

def gen_scenario_from_gpt_text(llm_text, cfg, model, map_vecs, map_ids):
    MAX_AGENT_NUM = 32
    agent_vector, map_vector, event_vector = output_formating_cot(llm_text)
    agent_num = len(agent_vector)
    vector_dim = len(agent_vector[0])
    event_dim = len(event_vector[0])

    type_vector = [it[-1] for it in agent_vector]
    agent_vector = [it[:-1] + [-1] for it in agent_vector]   
    agent_vector = agent_vector + [[-1]*vector_dim] * (MAX_AGENT_NUM - agent_num)
    event_vector = event_vector + [[-1]*event_dim] * (MAX_AGENT_NUM - agent_num)
    sorted_idx = map_retrival(map_vector, map_vecs)[:1]
    map_id = map_ids[sorted_idx[0]]
    
    batch = get_map_data_batch(map_id, cfg)
    type_len = batch['traj_type'].shape[1]
    for i in range(type_len):
        if i<len(type_vector):
            batch['traj_type'][0, i, 0] = type_vector[i]
        else:
            batch['traj_type'][0, i, 0] = -2

    batch['text'] = torch.tensor(agent_vector, dtype=batch['text'].dtype, device=model.device)[None, ...]
    event_tensor = torch.tensor(event_vector, dtype=batch['nei_text'][1].dtype, device=model.device)[None, ...]
    batch['text'] = batch['text'][:, :, :-1]
    batch['nei_text'] = [batch['nei_text'][0], event_tensor]
    batch['agent_mask'] = torch.tensor([1]*agent_num + [0]*(MAX_AGENT_NUM - agent_num), \
            dtype=batch['agent_mask'].dtype, device=model.device)[None, ...]

    for k in batch.keys():
        if type(batch[k])==torch.Tensor:
            batch[k] = batch[k].to(model.device)

    model_output = model.forward(batch, 'val')['text_decode_output']
    output_scene = model.process(model_output, batch, num_limit=1, with_attribute=True, pred_ego=True, pred_motion=True)

    return vis_decode(batch, output_scene), vis_stat(batch, output_scene)




if __name__ == "__main__":

    init_time = time.time()
    cfg_file = './cfgs/inference.yaml'
    cfg, model = load_model(cfg_file)

    map_data_file = "/home/ubuntu/xiajunkai/lctgen/data/map.npy"
    map_vecs, map_ids = load_all_map_vectors(map_data_file)

    llm_cfg = get_config('./lctgen/gpt/new.yaml')
    llm_model = registry.get_llm('codex')(llm_cfg)

    '''
    openai.api_key = ""
    openai.base_url = "https://api.openai-proxy.com/v1/"
    '''

    openai.api_key = "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"
    openai.base_url = "https://integrate.api.nvidia.com/v1/"

    query = "Generate a scenario with ten vehicles."
    print(f"{query=}")

    llm_result = llm_model.forward(query)

    answer_time = time.time()
    print(f"llm processing time: {answer_time-init_time}")

    gif_list, jpg = gen_scenario_from_gpt_text(llm_result, cfg, model, map_vecs, map_ids)

    process_time = time.time()
    print(f"decode time: {process_time-answer_time}")

    gif_path = "./rebuttal/demo_13.gif"
    jpg_path = "./rebuttal/demo_13.jpg"

    gif_list[0].save(gif_path, save_all=True, append_images=gif_list[1:])
    jpg.save(jpg_path, "JPEG")

