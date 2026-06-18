import torch
import torch.nn as nn


class StyleReconstructor(nn.Module):
    """
    > This Implements the Original Style Reconstruction Algorithm discussed in the NST Paper, the implementation is responsible to
    construct an image from pure white noise, such that it matches the style of the provided style image, this is done via
    minimizing the Entries of Gram Matrix b/w Style Image and White Noise Image

    > Optimizes the MSE Loss b/w Entries of Gram Matrix produced by the Style Image and White Noise

    > Layer used for Style Reconstruction are:
        > Conv1_1, Conv2_1, Conv3_1, Conv4_1, Conv5_1 [Each layer contributes to the Loss Independently]

    NOTE: These layers are configurable via swapping from Custom VGG-19 Implementation to Original VGG-19 Implementaions
    
    NOTE: This implementation computes Gram Matrices directly as G = FFᵀ
        without applying Gram Matrix normalization. Consequently, style
        losses have a significantly larger magnitude than content losses.
        To balance optimization, a smaller style weighting coefficient
        (β = 1e-6) is used during Neural Style Transfer experiments.

    Paper Name: A Neural Algorithm of Artistic Style
    Paper Link: https://arxiv.org/pdf/1508.06576
    """
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.loss_fn = nn.MSELoss(reduction="sum")
        self.cached_grams = {}

    def forward(self, generated_features):
        return self._get_total_style_loss(generated_features)

    def _get_gram_matrix(self, feature_maps):
        C, HW = feature_maps.shape
        return (feature_maps @ feature_maps.T)

    @torch.no_grad()
    def build_gram_cache(self, style_features):
        for layer_name in self.layers:

            target_feature_maps = (
                style_features[layer_name]
                .squeeze(0)
                .flatten(1)
            )

            self.cached_grams[layer_name] = (
                self._get_gram_matrix(target_feature_maps)
                .detach()
            )

    def _compute_style_loss_per_layer(self, layer_name, generated_features):
        noise_feature_maps = generated_features[layer_name].squeeze(0).flatten(1)

        C, HW = noise_feature_maps.shape
        img_gram_matrix = self.cached_grams[layer_name]

        noise_gram_matrix = self._get_gram_matrix(
            noise_feature_maps
        )

        loss = self.loss_fn(img_gram_matrix, noise_gram_matrix) / (4 * (C ** 2) * (HW ** 2))
        return loss

    def _get_total_style_loss(
        self,
        generated_features
    ):
        total_style_loss = 0
        for layer_name, weight in self.layers.items():
            current_loss = (
                self._compute_style_loss_per_layer(
                    layer_name,
                    generated_features
                )
            )
            total_style_loss += (
                current_loss * weight
            )
        return total_style_loss