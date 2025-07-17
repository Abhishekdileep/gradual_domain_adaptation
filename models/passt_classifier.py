import torch
import torch.nn as nn 
import torch.nn.functional as F1
from hear21passt.base import get_basic_model
from hear21passt.models.passt import get_model as get_model_passt



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

class Classifier_old(nn.Module):
    """
    A simple classifier model that can be used for classification tasks.
    This is a placeholder and should be replaced with the actual classifier model.
    """
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.thoda_rolu = nn.ReLU(inplace=True)
        self.dropme = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.thoda_rolu(x)
        x = self.dropme(x)
        
        return self.fc2(x)

class PaSST_wrapper(nn.Module):
    """
    A wrapper for the PaSST model, which is a pre-trained model for audio processing.
    This is a placeholder and should be replaced with the actual PaSST model.
    """
    def __init__(self, num_classes=10):
        super(PaSST_wrapper, self).__init__()
        # Placeholder for the actual PaSST model
        self.net = get_model_passt(arch="passt_s_swa_p16_128_ap476" )
        self.model = PasstBasicWrapper2(net=self.net, mode="embed_only")  # Example layer
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        output = self.model(x)
        logits = self.classifier(output)
        return logits 

class Classifier(nn.Module):
    """Simplified Classifier"""
    def __init__(self, input_size=768, num_classes=10):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, h):
        return self.layer(h)

def Passt_wrap():
    net = get_model_passt('passt_s_swa_p16_128_ap476' )
    classifier = Classifier()
    return net, classifier
