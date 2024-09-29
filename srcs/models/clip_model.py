import copy
import torch
import torch.nn as nn

from srcs.utils.model_utils import CG_stacked
from .blocks import MLP, pos2posemb, PositionalEncoding
from .att_fuse import ScaledDotProductAttention, MultiHeadAttention
from srcs.models.traj2prompt import generate_prompt

import clip
from transformers import CLIPProcessor, CLIPModel

copy_func = copy.deepcopy

class ScenarioNet(nn.Module):

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

        self.target_dim = [224,224]
        
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

        traj_info = data['traj'].float().to(self.device).permute((0,2,1,3)).contiguous().view((batch_size, max_agent_num, -1))
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
        # print(f"{scenario_embedding.shape=}")
        return scenario_embedding

        
class CLIP(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.hidden_dim = 256
        self._init_encoder()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip, self.preprocess = clip.load("ViT-B/32", jit=False)
        # self.clip.to(self.device)

    def _init_encoder(self):
        self.scenario_encoder = ScenarioNet(device=self.device) #.to(self.device)


    def forward(self, data, prompt): 
        scenario_encoding = self.scenario_encoder(data) #.to(self.device)
        # print(f"{scenario_encoding.shape=}")
        # prompt = generate_prompt(data)
        # inputs = self.processor(text=prompt, images=scenario_encoding, return_tensors="pt", padding=True)

        # scenario_encoding = self.preprocess(scenario_encoding).unsqueeze(0).to(self.device)

        text = clip.tokenize(prompt, context_length=77, truncate=True).to(self.device)
        logits_per_image, logits_per_text = self.clip(scenario_encoding, text)

        # print(f"{logits_per_image.shape=}")
        # print(f"{logits_per_text.shape=}")

        return logits_per_image, logits_per_text
