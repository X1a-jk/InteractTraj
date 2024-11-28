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
from srcs.utils.utils import visualize_input_seq, visualize_output_seq, visualize_map, visualize_input_batch
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

from collections import OrderedDict
import numpy as np
import pandas as pd


from new.lctgen.lctgen.models import LCTGen as lct_old

def vis_decode(batch, ae_output):
    img = visualize_output_seq(batch, output=ae_output[0], pool_num=1)
    return img

def vis_stat(batch, ae_output):
    img = visualize_input_seq(batch, agents=ae_output[0]['agent'], traj=ae_output[0]['traj'])
    return Image.fromarray(img)


def get_eval_model(cfg_path, model_type):
    cfg = get_config(cfg_path)
    # model_cls = registry.get_model(cfg.MODEL.TYPE)
    if model_type == "inter":
        model_cls = lctgen.models.lctgen.LCTGen
    else:
        model_cls = lct_old
    model = model_cls.load_from_checkpoint(cfg.LOAD_CHECKPOINT_PATH, config=cfg, metrics=[], strict=False)
    model.eval()
    return model

def get_query(query_path):
    queries = []
    labels = []
    with open(query_path, "rb") as f:
        lines = f.readlines()
        for line in lines:
            line = line.decode("utf-8").strip()
            if len(line):
                try:
                    query, label = line.split(":")
                except Exception as e:
                    print(line.split(":"))
                    raise(e)
                queries.append(query)
                labels.append(label) 
    return queries, labels

def load_clip_model(ckpt, device):
    model = CLIP(device = device).to(torch.float32).to(device)
    state_dict = torch.load(ckpt)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    

    return model


def build_dataset(cfg_path):
    cfg = get_config(cfg_path)
    dataset_type = cfg.DATASET.TYPE
    cfg.DATASET['CACHE'] = False
    dataset = registry.get_dataset(dataset_type)(cfg, 'train')
    return dataset

def gen_scenario_from_gpt_text(llm_text, cfg, map_id=None):
    # map_data_file = "/home/ubuntu/xiajunkai/lctgen/data/nuplan_map.npy"
    map_data_file = "/home/ubuntu/xiajunkai/lctgen/data/waymo_map.npy"
    map_vecs, map_ids = load_all_map_vectors(map_data_file)
    # print(f"{map_ids=}")
    # format LLM output to Structured Representation (agent and map vectors)
    MAX_AGENT_NUM = 32
    agent_vector, map_vector, event_vector = output_formating_cot(llm_text)
    agent_num = len(agent_vector)
    vector_dim = len(agent_vector[0])

    

    if event_vector:
        event_dim = len(event_vector[0])

    type_vector = [it[-1] for it in agent_vector]
    agent_vector = [it[:-1] + [-1] for it in agent_vector]
    
    agent_vector = agent_vector + [[-1]*vector_dim] * (MAX_AGENT_NUM - agent_num)
    if event_vector:
        event_vector = event_vector + [[-1]*event_dim] * (MAX_AGENT_NUM - agent_num)
    
    
    # retrive map from map dataset
    if not map_id:
        sorted_idx = map_retrival(map_vector, map_vecs)[:1]
        map_id = map_ids[sorted_idx[0]]
        print(f"{map_id=}")
    else:
        map_id = map_id
    #load map data
    batch = get_map_data_batch(map_id, cfg)
    type_len = batch['traj_type'].shape[1]
    
    for i in range(type_len):
        if i<len(type_vector):
            batch['traj_type'][0, i, 0] = type_vector[i]
        else:
            batch['traj_type'][0, i, 0] = -2
    
    return batch, agent_vector, event_vector, agent_num, map_id
    
def get_embeddings(batch, query, agent_vector, event_vector=None, agent_num=1, model=None, clip_model=None, file_name = None):
    import numpy
    MAX_AGENT_NUM = 32
    agent_num = agent_num

    print(f"{query=}")

    prev_img = np.array(visualize_input_batch(data=batch), dtype=np.float32)
    
    batch['text'] = torch.tensor(agent_vector, dtype=batch['text'].dtype, device=model.device)[None, ...]
    if event_vector:
        event_tensor = torch.tensor(event_vector, dtype=batch['nei_text'][1].dtype, device=model.device)[None, ...]
    batch['text'] = batch['text'][:, :, :-1]
    if event_vector:
        batch['nei_text'] = [batch['nei_text'][0], event_tensor]
    
    if "lct" in file_name:
        b, d, _ = batch['text'].shape
        padding = -1 * torch.ones((b, d, 1), device=model.device)
        batch['text'] = torch.cat((padding, batch['text']), dim=-1)
#     b_2, d_2, _ = batch['nei_text'][1].shape
    
    # batch['nei_text'][1] = torch.cat((batch['nei_text'][1],padding_2), dim=-1)
    batch['agent_mask'] = torch.tensor([1]*agent_num + [0]*(MAX_AGENT_NUM - agent_num), \
            dtype=batch['agent_mask'].dtype, device=model.device)[None, ...]


    for k,v in batch.items():
        if type(batch[k])==torch.Tensor:
            batch[k] = batch[k].to(model.device)
        if type(batch[k])==numpy.ndarray:
            batch[k] = torch.tensor(batch[k]).to(model.device)
    
    batch['traj'] = batch['traj'].unsqueeze(0)
        

    model_output = model.forward(batch, 'val')['text_decode_output']
    output_scene = model.process(model_output, batch, num_limit=1, with_attribute=True, pred_ego=True, pred_motion=True)

    traj_infer = torch.tensor(output_scene[0]['traj'])
    if traj_infer.shape[1]<MAX_AGENT_NUM:
        padding_2 = -1 * torch.ones((50, MAX_AGENT_NUM-traj_infer.shape[1],2), device=model.device)
        traj_infer = torch.cat([traj_infer, padding_2], dim = 1)

    prompt = [query]
    batch['traj'] = traj_infer

    with open(file_name, 'wb') as file:
        pickle.dump(batch, file)
        file.close()

    # print(clip_model)

    img = np.array(visualize_input_batch(data=batch), dtype=np.float32)
    
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(clip_model.device)
    #logits_per_image, logits_per_text = clip_model(batch, prompt)
    logits_per_image, logits_per_text, img_resized, img_embedding = clip_model(img, prompt)

    # similarity = clip_model(batch, prompt)
    # print(f"{img_embedding.tolist()=}")


    # gif_list = vis_decode(batch, output_scene)
    jpg = vis_stat(batch, output_scene)

    # return logits_per_image, logits_per_text, jpg
    return logits_per_text, jpg, prev_img, img_resized, img_embedding


def batch_saved_embeddings(num_samples, queries, model, clip_model, data_type):           #(batch, query, agent_vector, event_vector=None, model=None, clip_model=None, file_path = None):
    import numpy
    MAX_AGENT_NUM = 32
    # agent_num = len(agent_vector)

    keys = ['veh_type', 'text', 'lane_inp', 'traj', 'center', 'rest', 'bound', 'agent_mask', 'agent', 'gt_pos']
    saved_path = "./llm_results"
    batch_processed = {}

    for key in keys:
        lct_results = []
        # inter_results = []                
        for i in range(num_samples):
            lct_path = saved_path + f"/{data_type}_{i}.pkl"
            # inter_path = saved_path + f"/inter_{i}.pkl"

            # with open(inter_path, "rb") as f:
            #     batch_inter = pickle.load(f)
            #     inter_results.append(batch_inter[key])
            #     f.close()
            with open(lct_path, "rb") as f:
                batch_lct = pickle.load(f)
            if key == 'traj':
                lct_results.append(torch.tensor(batch_lct[key]))    
            else:
                lct_results.append(batch_lct[key])              
        try:    
            info = torch.stack(lct_results, dim=0).to(model.device)
            if info.shape[1] == 1:
                info = info.squeeze(dim=1)
        except Exception as e:
            print(key)
            raise(e)
        batch_processed[key] = info
    
    img = np.array(visualize_input_batch(data=batch_processed), dtype=np.float32)
    print(img)    
    #img1 = torch.tensor(img, dtype=torch.float32).permute((0, 3, 1, 2)).to(clip_model.device).cpu().detach().numpy()
    img = np.round(img).astype(np.uint8)
    print(img[0].shape)
    img1 = Image.fromarray(img[0]) 
    img1.save("/home/ubuntu/gujiahao/lctgen/pic/im.jpg")
    #img1.save(inter_jpg_path, "JPEG")
    logits_per_image, logits_per_text = clip_model(img, queries)

    probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()
    print(f"{np.diag(probs)=}")
    return probs




def sigmoid(x):
    import math
    return 1.0 / (1.0 + math.exp(-x))



def get_model_results(cfg_path, queries, inter_llm_model, lct_llm_model, inter_model, lct_model, clip_model): 
    cfg = get_config(cfg_path)
    lct_result = []
    inter_result = []
    import pickle
    inter_result_path = "./inter_result.pkl"
    saved_path = "./gpt_results/"
    new_saved_path = "./gpt_results/new/"
    feature_saved_path = "./gpt_results/feat/"

    saved_map_id = ['1_7093.pkl 7093', '0_2894.pkl 2894', '9_4330.pkl 4330', '3_3245.pkl 3245', \
                    '9_4330.pkl 4330', '5_5116.pkl 5116', '3_3245.pkl 3245', '6_6601.pkl 6601', \
                    '5_5116.pkl 5116', '4_413.pkl 413', '8_6426.pkl 6426', '1_7093.pkl 7093', \
                    '6_590.pkl 590', '2_1009.pkl 1009', '4_6971.pkl 6971', '8_6426.pkl 6426', \
                    '6_6601.pkl 6601', '5_5116.pkl 5116', '4_6822.pkl 6822', '4_413.pkl 413', \
                    '3_3245.pkl 3245', '5_587.pkl 587', '6_6601.pkl 6601', '8_6426.pkl 6426', \
                    '7_1812.pkl 1812', '4_413.pkl 413', '7_1812.pkl 1812', '6_590.pkl 590', \
                    '8_1306.pkl 1306', '4_4881.pkl 4881', '9_4330.pkl 4330', '5_5116.pkl 5116', \
                    '1_533.pkl 533', '1_7093.pkl 7093', '1_7093.pkl 7093', '6_6601.pkl 6601', \
                    '1_7093.pkl 7093', '6_6601.pkl 6601', '0_2894.pkl 2894', '2_1009.pkl 1009', \
                    '5_5116.pkl 5116', '5_5116.pkl 5116', '1_7093.pkl 7093', '5_5116.pkl 5116', \
                    '6_590.pkl 590', '9_4676.pkl 4676', '1_7093.pkl 7093', '5_5116.pkl 5116', \
                    '7_1812.pkl 1812', '4_4881.pkl 4881']

    inter_sigmoid = []
    inter_img_embeddings = []
    inter_imgs = []
    inter_prevs = []


    for i, query in enumerate(queries):
        
        print(f"Sending new message: {i}")     
        inter_path = saved_path + f"/inter_{i}.txt"
        
        '''
        # get reply from api
        inter_llm_result = inter_llm_model.forward(query)
        with open(inter_path, "w") as f:
            f.writelines(inter_llm_result)
            f.close()
        
        '''
        # load from saved files        
        with open(inter_path, 'rb') as f:
            llm_result = f.read()
        inter_llm_result = llm_result.decode('utf-8')
        save_name = new_saved_path + f"/inter_{i}.pkl"
        inter_jpg_path = new_saved_path + f"/inter_{i}.jpg"
        

        batch, agent_vector, event_vector, agent_num, map_id = gen_scenario_from_gpt_text(inter_llm_result, cfg, map_id=saved_map_id[i])
        # saved_map_id.append(map_id)
        # inter_embed, _, inter_jpg = get_embeddings(batch, query, agent_vector, event_vector, inter_model, clip_model, save_name)
        inter_embed, inter_jpg, inter_prev, inter_img, inter_img_embed = get_embeddings(batch, query, agent_vector, event_vector, agent_num, inter_model, clip_model, save_name)
        inter_img_embeddings.append(inter_img_embed)
        inter_imgs.append(inter_img)

        inter_img = inter_img[0].cpu().numpy()
        inter_img = np.transpose(inter_img, (1, 2, 0))  # 转换为 (224, 224, 3)
        inter_img = np.clip(inter_img * 255, 0, 255).astype(np.uint8)  # 这里假设数据范围是 [0, 1]，乘以255后转为 uint8
        inter_img_path = feature_saved_path + f"/inter_{i}.jpg"
        image = Image.fromarray(inter_img)
        image.save(inter_img_path)
        
        inter_jpg.save(inter_jpg_path, "JPEG")

        inter_embed = inter_embed[0].cpu().detach().item()
        # print(f"{inter_embed=}")
        inter_result.append(inter_embed)

        inter_sigmoid.append(sigmoid(inter_embed))

    print(f"inter binary score: {np.mean(inter_sigmoid)}")



    
    print("inter message done")
    
    lct_sigmoid = []
    lct_result_path = "./lct_result.pkl"  
    lct_img_embeddings = []
    lct_imgs = []
    lct_prevs = []

    for i, query in enumerate(queries):
        print(f"Sending new message: {i}")
        lct_path = saved_path + f"/lct_{i}.txt"

        '''
        # get reply from api
        lct_llm_result = lct_llm_model.forward(query)
        with open(lct_path, "w") as f:
            f.writelines(lct_llm_result)
            f.close()
        
        '''
        # load from saved files        
        with open(lct_path, 'rb') as f:
            llm_result = f.read()
        lct_llm_result = llm_result.decode('utf-8')
        
        
      
      
        save_name = new_saved_path + f"/lct_{i}.pkl"
        lct_jpg_path = new_saved_path + f"/lct_{i}.jpg"


        batch, agent_vector, _, agent_num, map_id = gen_scenario_from_gpt_text(lct_llm_result, cfg, map_id=saved_map_id[i])
        # lct_embed, _, lct_jpg = get_embeddings(batch, query, agent_vector, event_vector=None, model=lct_model, clip_model=clip_model, file_name=save_name)

        lct_embed, lct_jpg, lct_prev, lct_img, lct_img_embed = get_embeddings(batch, query, agent_vector, event_vector=None, agent_num=agent_num, model=lct_model, clip_model=clip_model, file_name=save_name)
        lct_img_embeddings.append(lct_img_embed)
        lct_imgs.append(lct_img)

        lct_img = lct_img[0].cpu().numpy()
        lct_img = np.transpose(lct_img, (1, 2, 0))  # 转换为 (224, 224, 3)
        lct_img = np.clip(lct_img * 255, 0, 255).astype(np.uint8)  # 这里假设数据范围是 [0, 1]，乘以255后转为 uint8
        lct_img_path = feature_saved_path + f"/lct_{i}.jpg"
        image = Image.fromarray(lct_img)
        image.save(lct_img_path)

        lct_jpg.save(lct_jpg_path, "JPEG")

        lct_embed = lct_embed[0].cpu().detach().item()
        # print(f"{lct_embed=}")
        lct_result.append(lct_embed)
        lct_sigmoid.append(sigmoid(lct_embed))

    print(f"lct binary score: {np.mean(lct_sigmoid)}")

    img_loss = [torch.norm(i.flatten()-j.flatten(), p=1).cpu().item() for i,j in zip(inter_imgs, lct_imgs)]
    print(f"{img_loss=}")

    embedding_loss = [torch.norm(i-j).cpu().item() for i,j in zip(inter_img_embeddings, lct_img_embeddings)]
    print(f"{embedding_loss=}")
    
    return inter_result, lct_result

def pairwise(model_result):
    print(model_result)

def read_csv(path):
    query = pd.read_csv(path, header=[0,1], index_col=[0])
    lct_result = query.LCTGen
    inter_result = query.InteractTraj
    lct_percentage= lct_result.percentage.values.tolist()
    inter_percentage= inter_result.percentage.values.tolist()
    return inter_percentage, lct_percentage

def relationship(model_result, human_result):
    from scipy.stats import pearsonr, spearmanr
    A_flat = model_result#.flatten() #.numpy()
    B_flat = np.array(human_result) #.flatten()#.numpy()
    pearson_corr, _ = pearsonr(A_flat, B_flat)

    spearman_corr, _ = spearmanr(A_flat, B_flat)
    return pearson_corr, spearman_corr

if __name__ == "__main__":
    import pickle
    lctgen_cfg = "/home/ubuntu/xiajunkai/new/lctgen/cfgs/inference.yaml"
    inter_cfg = "/home/ubuntu/xiajunkai/lctgen/cfgs/inference.yaml"
    clip_ckpt = "/home/ubuntu/gujiahao/lctgen/best_clip_model_1107.pth"

    #openai.api_key = # 
    #openai.base_url = # 

    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    inter_percentage, lct_percentage = read_csv("/home/ubuntu/xiajunkai/lctgen/query_res.csv") 

    lctgen_model = get_eval_model(lctgen_cfg, "lctgen")
    inter_model = get_eval_model(inter_cfg, "inter")
    clip_model = load_clip_model(clip_ckpt, device)

    inter_llm_cfg = get_config('/home/ubuntu/xiajunkai/lctgen/lctgen/gpt/cfgs/attr_ind_motion/new.yaml') # new.yaml')
    inter_llm_model = registry.get_llm('codex')(inter_llm_cfg)

    lct_llm_cfg = get_config('/home/ubuntu/xiajunkai/network/lctgen/lctgen/gpt/cfgs/attr_ind_motion/non_api_cot_attr_20m.yaml') # new.yaml')
    lct_llm_model = registry.get_llm('codex')(lct_llm_cfg)

    dataset = build_dataset(inter_cfg)

    query_path = "/home/ubuntu/xiajunkai/lctgen/query.txt"
    queries, labels = get_query(query_path) # label = 1: prefer inter

    
    tot = 0
    # inter_result_path = "./inter_result.pkl"
    # lct_result_path = "./inter_result.pkl"
    # with open(inter_result_path, "wb") as f:
    #     sim_inter = pickle.load(f)
    #     f.close()
    # with open(lct_result_path, "wb") as g:
    #     sim_lct = pickle.load(g)
    #     g.close()

    '''
    eval_batch_size = 10
    inter_score = batch_saved_embeddings(num_samples=eval_batch_size, queries=queries[:eval_batch_size], model=inter_model, clip_model=clip_model, data_type="inter")
    inter_prediction = np.argmax(inter_score, axis=1)
    print(f"{inter_prediction=}")
    
    lct_score = batch_saved_embeddings(num_samples=eval_batch_size, queries=queries[:eval_batch_size], model=lctgen_model, clip_model=clip_model, data_type="lct")
    lct_prediction = np.argmax(lct_score, axis=1)
    print(f"{lct_prediction=}")
    '''


    sim_inter, sim_lct = get_model_results(inter_cfg, queries, inter_llm_model, lct_llm_model, inter_model, lctgen_model, clip_model)
    print(f"{sim_inter=}")
    print(f"{sim_lct=}")


    # inter_percentage, lct_percentage = read_csv("/home/ubuntu/xiajunkai/lctgen/query_res.csv") 




    inter_param = sim_inter
    lct_param = sim_lct

    tot = 0
    tot_g = 0
    for idx, score in enumerate(sim_inter):
        score_lct = sim_lct[idx]
        if score >= score_lct:
            tot += 1
        if score > score_lct:# + 0.1:
            tot_g += 1

    print(f"winning rate: {tot/len(sim_inter)}")
    print(f"true winning rate: {tot_g/len(sim_inter)}")

    
    inter_relationship_p, inter_relationship_s = relationship(inter_param, inter_percentage)
    print(inter_relationship_p)
    print(inter_relationship_s)
    lct_relationship_p, lct_relationship_s = relationship(lct_param, lct_percentage)
    print(lct_relationship_p)
    print(lct_relationship_s) 
    param = inter_param + lct_param
    percentage = inter_percentage + lct_percentage
    relationship_p, relationship_s = relationship(param, percentage)
    print(relationship_p)
    print(relationship_s) 
    
