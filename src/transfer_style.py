from nst import NST
import torch.nn as nn
import torch
from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class TransferStyle(nn.Module):
    def __init__(self, vgg_19_model,
                content_layers,
                style_layers,
                config,
                wandb_config,
                content_img_path,
                style_img_path,
                content_transform,
                style_transform,
                device):
        super().__init__()

        self.content_img_path = content_img_path
        self.style_img_path = style_img_path

        self.content_img = Image.open(self.content_img_path)
        self.style_img = Image.open(self.style_img_path)
        self.vgg_19_model = vgg_19_model

        self.content_transform = content_transform
        self.style_transform = style_transform

        self.device = device
        self.nst = NST(vgg_model = self.vgg_19_model, content_layers = content_layers, style_layers = style_layers, config = config, wandb_config = wandb_config, device = self.device)

    def forward(self):
        return self._get_styled_img()

    def _get_styled_img(self):
        # self._plot_img(path = self.content_img_path, title = "Content Image")
        # self._plot_img(path = self.style_img_path, title = "Style Image")

        content_img_tensor = self.content_transform(self.content_img).unsqueeze(0).to(self.device)
        style_img_tensor = self.style_transform(self.style_img).unsqueeze(0).to(self.device)

        generated_image = self.nst(
                content_img_tensor = content_img_tensor,
                style_img_tensor = style_img_tensor
        )
        output_img = self._pre_process_img(generated_image)
        colored_img = self._preserve_original_color(output_img)

        # Save both Images
        os.makedirs("outputs", exist_ok=True)
        output_img.save("outputs/stylized.png")
        colored_img.save("outputs/stylized_preserved_color.png")

        return generated_image
    
    def _plot_img(self, path, title = None):
        img = Image.open(path)
        plt.imshow(img)
        if title is not None:
            plt.title(title)
        plt.axis("off")
        plt.show()

    def _pre_process_img(self, img_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        img = img_tensor.detach().cpu()

        img = img * std + mean
        img = img.clamp(0, 1)

        img = (
            img.squeeze(0)
            .permute(1, 2, 0)
            .numpy()
        )

        img = Image.fromarray((img * 255).astype("uint8"))
        img = img.resize(
            self.content_img.size,
            Image.Resampling.BICUBIC
        )
        return img

    def _plot_noise(self, noise_tensor, title=""):
        img = self._pre_process_img(noise_tensor)

        plt.figure(figsize=(6, 6))
        plt.imshow(img)

        if title:
            plt.title(title)

        plt.axis("off")
        plt.show()

    def _preserve_original_color(self, nst_output):
        content_img = Image.open(self.content_img_path).convert("RGB")
        content_rgb = np.array(content_img)
        content_lab = cv2.cvtColor(content_rgb, cv2.COLOR_RGB2LAB)

        # nst_output_img = Image.open(nst_output_path).convert("RGB")
        nst_output_img_rgb = np.array(nst_output.convert("RGB"))
        nst_output_lab = cv2.cvtColor(nst_output_img_rgb, cv2.COLOR_RGB2LAB)
        result_lab = nst_output_lab.copy()

        # Keep texture/lightness from stylized image
        result_lab[:, :, 0] = nst_output_lab[:, :, 0]

        # Keep colors from content image
        result_lab[:, :, 1] = content_lab[:, :, 1]
        result_lab[:, :, 2] = content_lab[:, :, 2]

        result_rgb = cv2.cvtColor(
            result_lab,
            cv2.COLOR_LAB2RGB
        )
        result_img = Image.fromarray(result_rgb)
        return result_img