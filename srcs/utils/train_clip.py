import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from srcs.models.traj2prompt import generate_prompt
import torch.nn as nn
import numpy
from srcs.utils.utils import visualize_input_seq, visualize_map
import pickle
import torch
from torch.utils.data import DataLoader
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
from lctgen.models.traj2prompt import generate_prompt

def cleanup():
    dist_.destroy_process_group()



def convert_models_to_fp32(model): 
    for name, p in model.named_parameters(): 
        p.data = p.data.float() 
        try:
            p.grad.data = p.grad.data.float() 
        except:
            print(name)

def setup(rank, world_size):
    # os.environ['LOCAL_RANK']
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '65534'
    dist_.init_process_group("nccl") #, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train(rank, world_size):
    # init_process(rank, world_size)
    setup(rank, world_size)
    device = torch.device("cuda", rank) #if torch.cuda.is_available() else "cpu")

    cfg_file = '/home/ubuntu/xiajunkai/InteractTraj/srcs/configs/inference.yaml'
    cfg = get_config(cfg_file)

    dataset_type = cfg.DATASET.TYPE
    cfg.DATASET['CACHE'] = False
    dataset = registry.get_dataset(dataset_type)(cfg, 'train')

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    bz = 2 * world_size #torch.cuda.device_count()

    collate_fn = fc_collate_fn
    loader = DataLoader(dataset, batch_size = bz, shuffle=False, pin_memory = False,
                    drop_last=False, num_workers=0, collate_fn=collate_fn, sampler=sampler)


# model



# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIP(device = device).to(torch.float32).to(rank)
    # model.share_memory()
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    # model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # the lr is smaller, more safe for fine tuning to new dataset
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    num_epochs = 30
    best_loss = float('inf')
    save_interval = 5  # 每隔 5 个 epoch 保存一次
    model_save_path = 'new_best_clip_model.pth'

    for epoch in range(num_epochs):
        print(f"{epoch=}")
        model.train()
        # sampler.set_epoch(epoch)
        pbar = tqdm(loader, total=len(loader))
        epoch_loss = []        
        loader.sampler.set_epoch(epoch)
        for batch in pbar:            
            optimizer.zero_grad()
            prompt = generate_prompt(batch)
            logits_per_image, logits_per_text = model(batch, prompt)


            ground_truth = torch.arange(logits_per_image.shape[1], device=device, dtype=torch.long)
            # ground_truth = ground_truth.repeat([world_size,1]).view((-1,1)).squeeze()
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2          
            # print(f"{total_loss=}")
            # Backward pass
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else : 
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")
            epoch_loss.append(total_loss)
            
            # if (epoch + 1) % save_interval == 0:
            #     model.eval()  # 设置为评估模式
            #     with torch.no_grad():
            #         val_outputs = model(batch)
            #         val_loss = criterion(val_outputs, y_val)
                
            #     print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss.item():.4f}')
            
            # 保存表现最好的模型
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Best model saved with loss: {best_loss:.4f}')

    cleanup()


# for i, batch in enumerate(loader):
#     data = batch
#     prompt = generate_prompt(data)

if __name__ == "__main__":
    import os

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # mp.spawn(
    #     train,
    #     args=(world_size,),
    #     nprocs=world_size
    # )

    train(local_rank, world_size)

    
    
    # freeze_support()
    #   # 获取可用 GPU 数量
    # print(f"{world_size=}")

    # # 使用 multiprocessing 启动多个进程
    # import multiprocessing
    # processes = []
    # for rank in range(world_size):
    #     p = multiprocessing.Process(target=train, args=(rank, world_size), name=f'Process-{rank}')
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()

    # train(rank, world_size)
