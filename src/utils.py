from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn

def get_vgg_model(device):
    vgg_19_model = vgg19(
        weights=VGG19_Weights.IMAGENET1K_V1
    ).features
    vgg_19_model = vgg_19_model.to(device = device)
    vgg_19_model.eval()
    
    # CRUCIAL: Swap MaxPool with AvgPool to stop the pixelated blurring
    new_vgg = nn.Sequential()
    for name, layer in vgg_19_model.named_children():
        if isinstance(layer, nn.MaxPool2d):
            new_vgg.add_module(name, nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            new_vgg.add_module(name, layer)
            
    # Freeze all parameters
    for param in new_vgg.parameters():
        param.requires_grad = False
        
    return new_vgg


def get_content_layers(config):
    if config["use_relu_content"]:
        original_content_layer = "relu4_2"
    else:
        original_content_layer = "conv4_2"
    return original_content_layer

def get_style_layers(config):

    use_relu = config["use_relu_style"]
    use_deeper = config["use_deeper_style_layers"]

    if use_relu:
        if use_deeper:
            return {
                "relu1_1": 0.25,
                "relu2_1": 0.25,
                "relu3_1": 0.25,
                "relu4_1": 0.20,
                "relu5_1": 0.20,
                "relu5_2": 0.10,
                "relu5_3": 0.10,
                "relu5_4": 0.10,
            }
        return {
            "relu1_1": 1/5,
            "relu2_1": 1/5,
            "relu3_1": 1/5,
            "relu4_1": 1/5,
            "relu5_1": 1/5,
        }

    else:
        if use_deeper:
            return {
                "conv1_1": 0.20,
                "conv2_1": 0.20,
                "conv3_1": 0.20,
                "conv4_1": 0.20,
                "conv5_1": 0.20,
                "conv5_2": 0.01,
                "conv5_3": 0.01,
                "conv5_4": 0.01,
            }
        return {
            "conv1_1": 1/5,
            "conv2_1": 1/5,
            "conv3_1": 1/5,
            "conv4_1": 1/5,
            "conv5_1": 1/5,
        }