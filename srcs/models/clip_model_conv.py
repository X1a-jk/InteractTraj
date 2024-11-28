import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import os


from trafficgen.utils.model_utils import CG_stacked
from .blocks import MLP, pos2posemb, PositionalEncoding
from .att_fuse import ScaledDotProductAttention, MultiHeadAttention
from lctgen.models.traj2prompt import generate_prompt

import clip
from transformers import CLIPProcessor, CLIPModel

from transformers import AutoProcessor

copy_func = copy.deepcopy

class ScenarioNet(nn.Module):

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

        self.target_dim = [256, 256]
        
        self.hidden_dim = self.target_dim[1]

        self._init_encoder()
        self._init_decoder()


    def _init_encoder(self):
        # self.CG_line = CG_stacked(5, self.hidden_dim)
        self.traj_encode = nn.Sequential(
            nn.Linear(50 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 9),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.line_encode = MLP([4, 256, 512, self.hidden_dim])        
        self.type_embedding = nn.Embedding(20, self.hidden_dim)
        self.traf_embedding = nn.Embedding(4, self.hidden_dim)

    def _init_decoder(self):
        mlp_dim = 512
        d_model = self.target_dim[1] #256 * 2

        query_dim = 504 #self.model_cfg.ATTR_QUERY.POS_ENCODING_DIM
        self.query_embedding_layer = nn.Sequential(
            nn.Linear(query_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.agent_type_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )       

        self.upsample = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size = 1),
            nn.ReLU(),
        )

        self.lane_downsample = nn.Sequential(
            nn.Conv1d(in_channels=736, out_channels=self.target_dim[0]-32, kernel_size=1, stride=1),
            nn.ReLU(),
        )


        

    def _map_lane_encode(self, lane_inp):
        polyline = lane_inp[..., :4]
        polyline_type = lane_inp[..., 4].to(int)
        polyline_traf = lane_inp[..., 5].to(int)

        self.type_embedding.to(torch.float32)
        polyline_type_embed = self.type_embedding(polyline_type)
        # print(f"{polyline_type_embed.shape=}")
        self.traf_embedding.to(torch.float32)
        polyline_traf_embed = self.traf_embedding(polyline_traf)
        # print(f"{polyline_traf_embed.shape=}")
        self.line_encode.to(torch.float32)
        try:
            poly_embed = self.line_encode(polyline)
        except Exception as e:
            print(f"{polyline.dtype=}")
            print(f"{list(self.line_encode.parameters())=}")
            for param in self.line_encode.parameters():
                print(f"{param.dtype=}")
            raise(e)

        line_enc = poly_embed + polyline_traf_embed + polyline_type_embed
        
        self.lane_downsample.to(torch.float32)
        line_enc = self.lane_downsample(line_enc)
        return line_enc

    def _map_feature_extract(self, line_enc, line_mask, context_agent):
        # map information fusion with CG block
        line_enc, context_line = self.CG_line(line_enc, context_agent, line_mask)
        # map context feature
        feature = torch.cat([line_enc, context_line.unsqueeze(1).repeat(1, line_enc.shape[1], 1)], dim=-1)

        return feature, context_line

    def forward(self, data):
        
        pos_enc_dim = 512
        print(f"{data['veh_type']=}")
        type_traj = data['veh_type'].float().to(self.device) 
        batch_size, max_agent_num, text_dim = data['text'].shape
        # Map Encoder 
        b = data['lane_inp'].shape[0]
        line_enc = self._map_lane_encode(data['lane_inp'].to(torch.float32).to(self.device)) 
        # print(f"{line_enc.shape=}") 
        # empty_context = torch.ones([b, line_enc.shape[-1]]).to(self.device)  
        # line_enc, context_feat = self._map_feature_extract(line_enc, data['lane_mask'].to(self.device), empty_context)
        # line_enc = line_enc[:, :data['center_mask'].shape[1]]    

        # Agent Query
        try:
            traj_info = data['traj'].float().to(self.device).permute((0,2,1,3)).contiguous().view((batch_size, max_agent_num, -1))
        except:
            # print(torch.tensor(data['traj']).shape)
            traj_info = torch.tensor(data['traj']).unsqueeze(0).float().to(self.device).permute((0,2,1,3)).contiguous().view((batch_size, max_agent_num, -1))
        self.traj_encode.to(torch.float32)
        attr_query_input = self.traj_encode(traj_info)


        
        # attr_query_input = data['text'].to(self.device)  
        attr_dim = attr_query_input.shape[-1]
        attr_query_encoding = pos2posemb(attr_query_input, pos_enc_dim//attr_dim)
        self.query_embedding_layer.to(torch.float32)
        attr_query_encoding = self.query_embedding_layer(attr_query_encoding)
        
        
        self.agent_type_embedding.to(torch.float32)
        agent_type_encoding = self.agent_type_embedding(type_traj)  
        attr_query_encoding = torch.cat([attr_query_encoding, agent_type_encoding], dim=-1)
        # context_feat = context_feat.unsqueeze(1).repeat(1, attr_query_encoding.shape[1], 1)
        self.upsample.to(torch.float32)
        scenario_embedding = self.upsample(torch.cat([attr_query_encoding, line_enc], dim=1).unsqueeze(1))
        print(f"{scenario_embedding.shape=}")
        return scenario_embedding

class ConvResizer(nn.Module):
    def __init__(self):
        super(ConvResizer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=0)
        

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=1)
        

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
    
        self.conv1.to(torch.float32)
        x = self.conv1(x)       
        x = F.relu(x)
        
        self.conv2.to(torch.float32)
        x = self.conv2(x)
        x = F.relu(x)
        
        self.conv3.to(torch.float32)
        x = self.conv3(x)
        x = F.relu(x)
        
        return x
def convert_list_to_tensor(data_list):
    """Convert a list of data elements to a tensor"""
    return torch.stack([torch.tensor(item) if not isinstance(item, torch.Tensor) else item for item in data_list])


def reset_parameters(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.fill_(0.0)
        elif isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.fill_(0.0)
        elif isinstance(layer, torch.nn.Embedding):
            torch.nn.init.constant_(layer.weight, 0.0)  # Set weights to zero
        elif isinstance(layer, torch.nn.LayerNorm):
            if hasattr(layer, 'weight') and layer.weight is not None:
                torch.nn.init.constant_(layer.weight, 0.0)  # Set weights to zero
            if hasattr(layer, 'bias') and layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)  # Set biases to zero
        elif isinstance(layer, torch.nn.MultiheadAttention):
            if hasattr(layer, 'out_proj'):
                # Initialize out_proj weights and biases to zero
                torch.nn.init.constant_(layer.out_proj.weight, 0.0)
                if layer.out_proj.bias is not None:
                    torch.nn.init.constant_(layer.out_proj.bias, 0.0)



class CLIP(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.hidden_dim = 256
        self._init_encoder()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip, self.preprocess = clip.load("ViT-B/32", jit=False)
        
        for param in self.clip.parameters():
            param.requires_grad = True

        # for k, v in self.clip.state_dict().items():
        #     print(f"{k}: {torch.norm(v, p=1)}")

        # reset_parameters(self.clip)


        self.clip.to(self.device)
        
        #self.resizer = ConvResizer()
        
        #self.resizer.to(self.device)

        # self.clip.to(self.device)
        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _init_encoder(self):
        self.scenario_encoder = ScenarioNet(device=self.device) #.to(self.device)


    def forward(self, data, prompt): 
        data = convert_list_to_tensor(data).float().to(self.device).permute(0,3,1,2)
        #print(data.shape)
        img_resized = self.image_processor(images=data, return_tensors="pt")["pixel_values"].to(self.device) # 8 * 3 * 224 * 224
        # print(f"{img_resized.shape=}")
        '''
        img_resized = self.resizer(data)
        '''
        '''
        save_dir = "/home/ubuntu/gujiahao/lctgen/saved_images"
        os.makedirs(save_dir, exist_ok=True)
        
        for i, img in enumerate(img_resized):
            img = img.cpu()
            vutils.save_image(img, os.path.join(save_dir, f"image_{i}.jpeg"))
        '''
        scenario_encoding = self.scenario_encoder(data).to(self.device)
        print(f"{scenario_encoding.shape=}")
        # prompt = generate_prompt(data)
        # inputs = self.processor(text=prompt, images=scenario_encoding, return_tensors="pt", padding=True)

        # scenario_encoding = self.preprocess(scenario_encoding).unsqueeze(0).to(self.device)

        text = clip.tokenize(prompt, context_length=77, truncate=True).to(self.device)
        logits_per_image, logits_per_text, img_embedding, text_embedding = self.clip(img_resized, text)
        # print(f"{logits_per_image.shape=}")
        # print(f"{logits_per_text.shape=}")

        return logits_per_image, logits_per_text, img_resized, img_embedding