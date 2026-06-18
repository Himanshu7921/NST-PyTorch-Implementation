import torch.nn as nn

class ContentReconstructor(nn.Module):
    """
    > This Implements the Original Content Reconstruction Algorithm discussed in the NST Paper, the implementation is responsible to
    reconstruct the input image from white noise, so that it have same effect on the learned activation maps
    of the VGG-19 Network [self-implemented or Original Implementation]

    > Optimizes the MSE Loss b/w feature maps produced by the Content Image and White Noise

    > Layer used for Content Reconstruction is:
        > Conv4_2 [A single loss from the final layer of Con4_2]
    
    NOTE: These layers are configurable via swapping from Custom VGG-19 Implementation to Original VGG-19 Implementaions

    Paper Name: A Neural Algorithm of Artistic Style
    Paper Link: https://arxiv.org/pdf/1508.06576
    """
    def __init__(self, layers, config):
        super().__init__()
        self.layers = layers
        self.loss_fn = nn.MSELoss(reduction = "mean")

    def build_target_feature_cache(self, target_feature_map):
        self.target_feature_maps = target_feature_map.detach()

    def forward(self, generated_features):
        generated_conv4_2 = generated_features["conv4_2"]
        return self.loss_fn(self.target_feature_maps, generated_conv4_2)