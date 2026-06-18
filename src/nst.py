import torch.nn as nn
import torch
from vgg_feature_extractor import VGGFeatureExtractor
from content_reconstructor import ContentReconstructor
from style_reconstructor import StyleReconstructor
import os
import wandb
from config import config
from PIL import Image

class NST(nn.Module):
    """
    This class performs Neural Style Transfer by synthesizing a new image
    from a content image and a style image.

    Paper Name: A Neural Algorithm of Artistic Style
    Paper Link: https://arxiv.org/pdf/1508.06576
    """
    def __init__(self, vgg_model, content_layers, style_layers, config, wandb_config, device):
        super().__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.epochs = config["epochs"]
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.gamma = config["gamma"]
        self.img_size = config["img_size"]
        self.use_relu = config["use_relu_style"]
        self.use_relu = config["use_relu_content"]
        self.use_deeper_style_layers = config["use_deeper_style_layers"]
        self.initial_img = config["initial_img"] # initial_img = which img to start with? --> [content_img, white noise, style_img]
        self.wandb_config = wandb_config
        self.vgg = vgg_model
        self.device = device

        self.feature_extractor = VGGFeatureExtractor(self.vgg, use_relu = self.use_relu, use_deeper_style_layers = self.use_deeper_style_layers)
        self.content_reconstructor = ContentReconstructor(layers = self.content_layers, config = config)
        self.style_reconstructor = StyleReconstructor(layers = self.style_layers)
    
    def forward(self, content_img_tensor, style_img_tensor):
        self.eval()
        return self._gradient_descent(content_img_tensor, style_img_tensor)
    
    def _total_variation_loss(self, img):

        loss_h = torch.mean(
            (img[:, :, 1:, :] - img[:, :, :-1, :]) ** 2
        )

        loss_w = torch.mean(
            (img[:, :, :, 1:] - img[:, :, :, :-1]) ** 2
        )

        tv_loss = loss_h + loss_w
        return tv_loss

    
    def _gradient_descent(self, content_img_tensor, style_img_tensor):
        if config["enable_wandb"]:
            wandb.init(
                project="neural-style-transfer",
                name="nst_vgg19",
                config = self.wandb_config,
                tags=[
                    "neural-style-transfer",
                    "nst",
                    "vgg19",
                    "custom-vgg19",
                    "style-transfer",
                    "gram-matrix",
                    "paper-reproduction",
                    "gatys-et-al"
                ]
            )

        # Get the Initial Starting Image to Optimize on [content, random, style]
        if self.initial_img == "content":
            noise_tensor = content_img_tensor.clone().requires_grad_(True)
        elif self.initial_img == "random":
            noise_tensor = torch.randn_like(content_img_tensor).requires_grad_(True)
        elif self.initial_img == "style":
            # Don't worry this won't break, because i've already resized the content and style img to match same spatial resolution,
            # i've applied same "_val_transform" under 'TransferStyle' to both images
            noise_tensor = style_img_tensor.clone().requires_grad_(True)
        else: 
            noise_tensor = content_img_tensor.clone().requires_grad_(True)

        # optimizer = torch.optim.Adam(params = [noise_tensor], lr = 0.02)

        optimizer = torch.optim.LBFGS(
            [noise_tensor],
            lr = config["lr"],
            max_iter = 20,
            history_size = 20,
            line_search_fn = "strong_wolfe",
            tolerance_grad = 1e-7,
            tolerance_change = 1e-9
        )

        with torch.no_grad():
            # content_img_tensor = preprocess_gatys(content_img_tensor, self.device)
            # style_img_tensor = preprocess_gatys(style_img_tensor, self.device)

            content_features = self.feature_extractor(
                content_img_tensor
            )

            style_features = self.feature_extractor(
                style_img_tensor
            )

        self.content_reconstructor.build_target_feature_cache(
            content_features["conv4_2"]
        )
            
        self.style_reconstructor.build_gram_cache(
            style_features
        )

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=self.device
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=self.device
        ).view(1, 3, 1, 1)

        prev_loss = float('inf')
        plateau_count = 0
        for epoch in range(self.epochs):
            # -----------------------------------    Adam Optimization    ----------------------------
            # generated_features = self.feature_extractor(noise_tensor)
            # content_loss = self.content_reconstructor(
            #     generated_features
            # )
            # style_loss = self.style_reconstructor(
            #     generated_features
            # )

            # tv_loss = self._total_variation_loss(noise_tensor)

            # total_loss = self.alpha * content_loss + self.beta * style_loss + self.gamma * tv_loss

            # optimizer.zero_grad()
            # total_loss.backward()
            # optimizer.step()
            # grad_norm = noise_tensor.grad.norm().item()
            # grad_mean = noise_tensor.grad.abs().mean().item()
            # grad_max = noise_tensor.grad.abs().max().item()
            # grad_std = noise_tensor.grad.std().item()
            # ------------------------------------------------------------------------------------------


            # -----------------------------------    L-BFGS Optimization    ----------------------------
            container = {}
            def closure():
                optimizer.zero_grad()

                # generated_features = self.feature_extractor(preprocess_gatys(noise_tensor, self.device))
                generated_features = self.feature_extractor(noise_tensor)
                content_loss = self.content_reconstructor(
                    generated_features
                )
                style_loss = self.style_reconstructor(
                    generated_features
                )
                tv_loss = self._total_variation_loss(noise_tensor)
                total_loss = self.alpha * content_loss + self.beta * style_loss + self.gamma * tv_loss
                total_loss.backward()
                container["content_loss"] = content_loss
                container["style_loss"] = style_loss
                container["tv_loss"] = tv_loss
                container["total_loss"] = total_loss
                return total_loss
            
            optimizer.step(closure)
            with torch.no_grad():
                noise_tensor.clamp_(
                    (-mean / std).to(noise_tensor.device),
                    ((1 - mean) / std).to(noise_tensor.device)
                )

            content_loss = container["content_loss"]
            style_loss = container["style_loss"]
            tv_loss = container["tv_loss"]
            total_loss = container["total_loss"]
            current_loss = total_loss.item()

            if abs(prev_loss - current_loss) < 1e-6:
                plateau_count += 1
                if plateau_count >= 3:  # stuck for 5 epochs
                    optimizer = torch.optim.LBFGS(
                        [noise_tensor], lr=1.0, max_iter = 5,
                        history_size = 20, line_search_fn="strong_wolfe"
                    )
                    plateau_count = 0
            else:
                plateau_count = 0
            prev_loss = current_loss
            
            grad_norm = noise_tensor.grad.norm().item()
            grad_mean = noise_tensor.grad.abs().mean().item()
            grad_max = noise_tensor.grad.abs().max().item()
            grad_std = noise_tensor.grad.std().item()

            # --------------------------------------------------------------------------------------

            # content_losses.append(content_loss.item())
            # style_losses.append(style_loss.item())
            # total_losses.append(total_loss.item())
            # tv_losses.append(tv_loss.item())

            if config["enable_wandb"]:
                wandb.log({
                    "content_loss": content_loss.item(),
                    "style_loss": style_loss.item(),
                    "tv_loss": tv_loss.item(),
                    "total_loss": total_loss.item(),
                    "grad_norm": grad_norm,
                    "grad_mean": grad_mean,
                    "grad_max": grad_max,
                    "grad_std": grad_std,
                    "epoch": epoch
                })

            if epoch % 10 == 0:
                print(f"Epoch = [{epoch}/{self.epochs}] | Content Loss = {content_loss.item():.5f} | Style Loss = {style_loss.item():.10f} | Total Variation Loss = {tv_loss.item():.4f} | Total Loss = {total_loss.item():.5f} | Content Contribution = {self.alpha * content_loss} | Style Contribution = {self.beta * style_loss} | Contribution Ration (c/s): {(self.alpha * content_loss) / (self.beta * style_loss)} | Grad Norm: {grad_norm:.4f} | Grad mean: {grad_mean:.4f} | Grad std: {grad_std:.4f}")
            
            if config["enable_wandb"]:
                if epoch % 10 == 0:
                    img_path = self._plot_noise_in_wandb(
                        noise_tensor = noise_tensor,
                        img_size = self.img_size,
                        title = f"Epoch {epoch}"
                    )
                    wandb.log({
                        "stylized_preview": wandb.Image(img_path, caption=f"Epoch {epoch}"),
                    })
                

        if config["enable_wandb"]:
            wandb.finish()
        return noise_tensor
    
    def _plot_noise_in_wandb(self, noise_tensor, img_size, title=""):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        img = noise_tensor.detach().cpu()

        img = img * std + mean
        img = img.clamp(0, 1)

        img = (
            img.squeeze(0)
            .permute(1, 2, 0)
            .numpy()
        )

        img = Image.fromarray((img * 255).astype("uint8"))
        img = img.resize(
            (img_size, img_size),
            Image.Resampling.BICUBIC
        )

        # plt.figure(figsize=(6, 6))
        # plt.imshow(img)

        # if title:
        #     plt.title(title)

        # plt.axis("off")
        # plt.show()

        os.makedirs("./wandb_images", exist_ok=True)
        save_path = f"./wandb_images/wandb_preview_epoch_{title}.png"
        img.save(save_path)
        return save_path