import utils  
import models


def get_teacher(patch_size, pretrained_weights, architecture="vit_small"):
    model = models.__dict__[architecture](
        patch_size=patch_size, 
        num_classes=0,
        use_mean_pooling=False,
        return_all_tokens=True)
    print(f"Model built.")
    
    model.cuda()
    utils.load_pretrained_weights(model, pretrained_weights, "teacher", architecture, patch_size)
    model.eval()

    return model