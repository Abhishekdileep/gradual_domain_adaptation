import torch
from torch import nn
import torch.nn.functional as F


class PasstBasicWrapper2(nn.Module):
    def __init__(self,  net: nn.Module, max_model_window=10000, timestamp_window=160, timestamp_hop=50,
                 scene_hop=2500, scene_embedding_size=1295, timestamp_embedding_size=1295, mode="all", **kwargs):
        """
        @param mel: spectrogram extractor
        @param net: network module
        @param max_model_window: maximum clip length allowed by the model (milliseconds).
        @param timestamp_hop: the hop lengh for timestamp embeddings (milliseconds).
        @param scene_hop: the hop lengh for scene embeddings (milliseconds).
        @param scene_embedding_size:
        @param timestamp_embedding_size:
        @param mode: "all", "embed_only", "logits"
        """
        torch.nn.Module.__init__(self)
        # self.mel = mel
        self.net = net
        self.device_proxy = nn.Parameter(torch.zeros((1)))
        # self.sample_rate = mel.sr
        # self.timestamp_window = int(timestamp_window * self.sample_rate / 1000)
        # self.max_model_window = int(max_model_window * self.sample_rate / 1000)
        # self.timestamp_hop = int(timestamp_hop * self.sample_rate / 1000)
        # self.scene_hop = int(scene_hop * self.sample_rate / 1000)
        # self.scene_embedding_size = scene_embedding_size
        # self.timestamp_embedding_size = timestamp_embedding_size
        self.mode = mode

    def device(self):
        return self.device_proxy.device

    def forward(self, x):
        # specs = self.mel(x_src , x_tgt , alpha)
        # specs = specs.unsqueeze(1)

        x, features = self.net(x)
        if self.mode == "all":
            embed = torch.cat([x, features], dim=1)
        elif self.mode == "embed_only":
            embed = features
        elif self.mode == "logits":
            embed = x
        else:
            raise RuntimeError(f"mode='{self.mode}' is not recognized not in: all, embed_only, logits")
        return embed


    