import torch
from config import config
from torchvision import transforms
from utils import get_content_layers, get_style_layers, get_vgg_model
from transfer_style import TransferStyle

def main():

    content_path = "content_13.jpg"
    style_path = "style_04.png"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMAGE_SIZE = 512
    STYLE_SIZE = 512

    original_content_layer = get_content_layers(config)
    original_style_layers = get_style_layers(config)

    # for wandb experiment tracking
    if config["enable_wandb"]:
        wandb_config = {
            "epochs": config["epochs"],
            "initial_img": config["initial_img"],
            "img_size": config["img_size"],
            "alpha": config["alpha"],
            "beta": config["beta"],
            "gamma": config["gamma"],
            "lr": config["lr"],
            "content_layer": original_content_layer,
            "style_layers": original_style_layers,
            "use_relu_content": config["use_relu_content"],
            "use_relu_style": config["use_relu_style"]
        }
    else:
        wandb_config = None

    content_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    style_transform = transforms.Compose([
        transforms.Resize((STYLE_SIZE, STYLE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    vgg19 = get_vgg_model(device = device)
    style_transfer = TransferStyle(
        vgg_19_model = vgg19,
        content_layers = original_content_layer,
        style_layers = original_style_layers,
        config = config,
        wandb_config = wandb_config,
        content_img_path = content_path,
        style_img_path = style_path,
        content_transform = content_transform,
        style_transform = style_transform,
        device = device
    )
    print(f"Optimizing the Image on Device = {device}")

    print("\n" + "=" * 50)
    print("Neural Style Transfer Configuration")
    print("=" * 50)

    for key, value in config.items():
        print(f"{key:<25}: {value}")

    print("=" * 50 + "\n")
    generated_img_tensor = style_transfer()

if __name__ == "__main__":
    main()