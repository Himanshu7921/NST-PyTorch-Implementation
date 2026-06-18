# ----------------------------------- Notes: [Best Config for my setup] -----------------------------------
# 1. Content: alpha: 1e2, beta: 1e5, gamma: 1e-4
# 2. random:  alpha: 1e3, beta: 1e5, gamma: 1e-4
# 3. style: alpha: 1e3, beta: 1e4, gamma: 1e-4

config = {
    "epochs": 1,
    "alpha": 1e2,                                                                     
    "beta": 1e5,                                                           
    "gamma": 1e-4,                                                                   
    "img_size": 512,
    "lr": 1.0,
    "use_relu_content": True,
    "use_relu_style": True,
    "use_deeper_style_layers": False,
    "initial_img": "content", # [content, random, style]
    "enable_wandb": False # enable this for experiment tracking
}