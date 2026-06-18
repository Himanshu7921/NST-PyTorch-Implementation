import torch.nn as nn

class VGGFeatureExtractor(nn.Module):
    """
    Performs a single, sequential forward pass through a VGG backbone to extract 
    intermediate feature maps for Neural Style Transfer (NST).

    Instead of passing an image through the network multiple times to extract 
    features layer-by-layer—which is incredibly redundant and wasteful—this class 
    computes everything once in a single execution pass and captures the required 
    activations on the fly. 

    While optimizing layer execution sequentially is completely optional if you can 
    afford the computational overhead, capturing everything in one single forward 
    pass is essential for high-performance NST implementations to drastically 
    reduce GPU compute and training time.

    Extracted target layers mapping (based on standard torchvision VGG-19 features):
        - idx  0 -> conv1_1 (Style)
        - idx  5 -> conv2_1 (Style)
        - idx 10 -> conv3_1 (Style)
        - idx 19 -> conv4_1 (Style)
        - idx 21 -> conv4_2 (Content)
        - idx 28 -> conv5_1 (Style)
    """
    def __init__(
        self,
        vgg,
        use_relu: bool = True,
        use_deeper_style_layers: bool = False
    ):
        super().__init__()

        self.vgg = vgg
        self.use_relu = use_relu
        self.use_deeper_style_layers = use_deeper_style_layers

    def forward(self, x):
        features = {}

        for idx, layer in enumerate(self.vgg):
            x = layer(x)

            if self.use_relu:
                if idx == 1:
                    features["relu1_1"] = x
                elif idx == 6:
                    features["relu2_1"] = x
                elif idx == 11:
                    features["relu3_1"] = x
                elif idx == 20:
                    features["relu4_1"] = x
                elif idx == 21:
                    features["conv4_2"] = x
                elif idx == 25:
                    features["relu5_1"] = x

                if self.use_deeper_style_layers:
                    if idx == 27:
                        features["relu5_2"] = x
                    elif idx == 29:
                        features["relu5_3"] = x
                    elif idx == 31:
                        features["relu5_4"] = x

            else:
                if idx == 0:
                    features["conv1_1"] = x
                elif idx == 5:
                    features["conv2_1"] = x
                elif idx == 10:
                    features["conv3_1"] = x
                elif idx == 19:
                    features["conv4_1"] = x
                elif idx == 21:
                    features["conv4_2"] = x
                elif idx == 24:
                    features["conv5_1"] = x

                if self.use_deeper_style_layers:
                    if idx == 26:
                        features["conv5_2"] = x
                    elif idx == 28:
                        features["conv5_3"] = x
                    elif idx == 30:
                        features["conv5_4"] = x
        return features