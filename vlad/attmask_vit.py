import sys
import os



import utils  
import models
import torch


def get_model(patch_size, checkpoint_path, type, device):
    num_channels = torch.load(checkpoint_path, map_location="cpu")[type]["backbone.patch_embed.proj.weight"].shape[1]
    architecture="vit_small"
    model = models.__dict__[architecture](
        patch_size=patch_size, 
        num_classes=0,
        in_chans = num_channels,
        use_mean_pooling=False,
        return_all_tokens=True)
    print(f"Model built.")
    
    model.to(device)
    utils.load_pretrained_weights(model, checkpoint_path, type, architecture, patch_size)
    model.eval()

    return model, num_channels

def get_teacher(patch_size, checkpoint_path, device):
    return get_model(patch_size, checkpoint_path, "teacher", device)

def get_student(patch_size, checkpoint_path, device):
    return get_model(patch_size, checkpoint_path, "student", device)
