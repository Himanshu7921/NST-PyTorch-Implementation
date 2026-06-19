# NST-PyTorch-Implementation
A clean, research-faithful PyTorch implementation of **Neural Style Transfer** based on the seminal paper by Gatys et al. (2016). This repository reproduces the core algorithm using a pre-trained VGG-19 feature extractor and the L-BFGS optimizer, with support for color-preserving transfer, multiple style blending, and systematic hyperparameter experimentation.


<p align="center">
  <img src="assets/output/dead_poets_society_03.png" width="32%">
  <img src="assets/output/dead_poets_society_02.png" width="32%">
  <img src="assets/output/dead_poets_society_01.png" width="32%">
</p>

<p align="center">
  <em>Neural Style Transfer with VGG19 — transforming cinematic scenes into painterly artworks through feature reconstruction and style optimization.</em>
</p>

<p align="center">
  <a href="https://medium.com/@himanshusr451tehs/implementing-neural-style-transfer-from-scratch-the-project-that-started-it-all-5e80eb774de2">Read the Blog</a> •
  <a href="https://arxiv.org/abs/1508.06576">Original Paper</a>
</p>

---


## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Results](#results)
  - [Baseline](#baseline)
  - [Initialization Comparison](#initialization-comparison)
  - [TV Loss (Stroke Weight) Ablation](#tv-loss-stroke-weight-ablation)
  - [Style Weight Ablation](#style-weight-ablation)
  - [Color-Preserving Transfer](#color-preserving-transfer)
  - [Multiple Style Blending](#multiple-style-blending)
  - [L-BFGS vs Adam](#l-bfgs-vs-adam)
- [Experiment Log](#experiment-log)
- [References](#references)

---

## Overview

Neural Style Transfer (NST) is an optimization-based technique that synthesizes a new image — referred to as a *pastiche* — by jointly minimizing a content loss and a style loss computed against a target content image and a target style image respectively. The stylized output preserves the semantic structure of the content image while adopting the low-level textural patterns of the style image.

This implementation strictly follows the Gatys et al. formulation, including:

- **Content representation** via `conv4_2` activations of VGG-19
- **Style representation** via Gram matrices computed at `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`
- **Total Variation regularization** for spatial smoothness
- **L-BFGS** optimization (default), consistent with the original paper

---

## How It Works
<img src="assets/architecture_diagram.png" width="700">

---

### Reproduction of Figure 3 from the Original NST Paper

To validate the correctness of the implementation, I reproduced the famous Figure 3 experiment from Gatys et al. (2016), where the same content image is stylized using multiple artworks. The generated outputs closely match the qualitative behavior reported in the original paper.

<p align="center">
  <img src="assets/results/reproduced_img.png" width="900">
</p>

<p align="center">
  <em>Figure 3 reproduced from the original Neural Style Transfer paper using this implementation. The content image is combined with multiple artistic styles, demonstrating successful separation of content and style representations.</em>
</p>

---

### Mathematical Formulation

The optimization target of Neural Style Transfer models the translation of artistic styles as a multi-objective loss minimization problem.

#### The Baseline Gatys Formulation

In the seminal paper by Gatys et al., the objective function relies strictly on balancing structural reconstruction against texture matching:

$$L_{\text{total}}(\vec{p}, \vec{a}, \vec{x}) = \alpha L_{\text{content}}(\vec{p}, \vec{x}) + \beta L_{\text{style}}(\vec{a}, \vec{x})$$

While this elegantly transfers statistical textures, minimizing this unconstrained loss vector via raw gradient descent frequently introduces high-frequency high-variance noise, pixel-level grit, and localized checkerboard artifacts.

#### My Enhanced Formulation (With TV Regularization)

To enforce spatial continuity, smoother transitions, and superior perceptual quality, this implementation introduces an isotropic **Total Variation (TV) Loss** regularizer ($\gamma L_{\text{tv}}$) to the joint objective function:

$$L_{\text{total}}(\vec{p}, \vec{a}, \vec{x}) = \alpha L_{\text{content}}(\vec{p}, \vec{x}) + \beta L_{\text{style}}(\vec{a}, \vec{x}) + \gamma L_{\text{tv}}(\vec{x})$$

Where $\alpha$, $\beta$, and $\gamma$ denote the structural reconstruction, stylistic texture, and spatial denoising weights respectively.

---

### 1. Content Representation Loss

Let $F^l, P^l \in \mathbb{R}^{N_l \times M_l}$ be the activation feature maps of the generated pastiche $\vec{x}$ and the original content image $\vec{p}$ respectively, extracted from layer $l$ (specifically `conv4_2`). Here, $N_l$ is the number of distinct filters (channels), and $M_l$ is the spatial volume (width $\times$ height). The content loss is defined via the squared error loss:

$$L_{\text{content}}(\vec{p}, \vec{x}) = \frac{1}{2} \sum_{i, j} \left( F_{ij}^l - P_{ij}^l \right)^2$$

---

### 2. Style Representation Loss

To capture the static texture signature of the style artwork $\vec{a}$, feature map activations are mapped to a spatial summary statistic via a **Gram Matrix** $G^l \in \mathbb{R}^{N_l \times N_l}$. The inner product of the vectorized feature channels computes cross-layer spatial correlations:

$$G_{ij}^l = \sum_{k=1}^{M_l} F_{ik}^l F_{jk}^l$$

Let $A^l$ be the stationary Gram matrix representation of the style target $\vec{a}$ in layer $l$. The total style loss across the targeted network ensemble $\mathcal{L}$ is a weighted sum of the mean squared error contributions of individual layers:

$$L_{\text{style}}(\vec{a}, \vec{x}) = \sum_{l \in \mathcal{L}} w_l \left( \frac{1}{4 N_l^2 M_l^2} \sum_{i, j} \left( G_{ij}^l - A_{ij}^l \right)^2 \right)$$

Where $w_l$ denotes the localized scaling factor assigned to each target encoding layer $l$.

---

### 3. Total Variation (TV) Regularization

To penalize pixel-level discontinuities, the added TV loss functions as an inline denoiser over the pixel grid of the generated pastiche $\vec{x}$. It bounds the variance between adjacent pixels to encourage piecewise smoothness and eliminate optimization grit:

$$L_{\text{tv}}(\vec{x}) = \sum_{i, j} \left[ \left( x_{i+1, j} - x_{i, j} \right)^2 + \left( x_{i, j+1} - x_{i, j} \right)^2 \right]$$

---

## Architecture

VGG-19 is used purely as a **frozen feature extractor**. No layers are fine-tuned. Average pooling replaces max pooling throughout the network, consistent with the recommendation in Gatys et al. for smoother gradient flow during stylization.

| Component              | Detail                                      |
|------------------------|---------------------------------------------|
| Backbone               | VGG-19 (pretrained on ImageNet)             |
| Pooling                | Average Pooling (replaces Max Pooling)      |
| Content Layer          | `conv4_2`                                   |
| Style Layers           | `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1` |
| Optimizer              | L-BFGS                                      |
| Color Transfer (opt.)  | CIE L\*a\*b\* color space                  |

---

## File Structure

```
NST/
├── assets/                        # Input images (content + style)
├── outputs/
│   ├── stylized.png               # Standard stylized output
│   └── stylized_preserved_color.png  # Color-preserving output
├── src/
│   ├── config.py                  # All hyperparameters and run settings
│   ├── main.py                    # Entry point
│   ├── nst.py                     # Core NST optimization loop
│   ├── vgg_feature_extractor.py   # VGG-19 wrapper (frozen, avg pooling)
│   ├── transfer_style.py          # Style transfer pipeline
│   ├── content_reconstructor.py   # Content reconstruction utility
│   ├── style_reconstructor.py     # Style reconstruction utility
│   └── utils.py                   # Image I/O, preprocessing, Gram matrix
├── wandb/                         # W&B experiment logs (optional)
├── wandb_images/                  # Saved images from W&B runs
├── content_13.jpg                 # Example content image
├── style_04.png                   # Example style image
├── .gitignore
└── README.md
```

---

## Getting Started

**Prerequisites**

- Python 3.9+
- CUDA-capable GPU recommended (tested on NVIDIA RTX 3050, 6GB VRAM)

**Installation**

```bash
git clone https://github.com/Himanshu7921/NST-PyTorch-Implementation.git
cd NST-PyTorch-Implementation
pip install -r requirements.txt
```

**Running**

All configuration is handled through `src/config.py`. Set your content image, style image, and hyperparameters there, then run:

```bash
python src/main.py
```

Output images are saved to the `outputs/` directory. For color-preserving transfer, the output `stylized_preserved_color.png` is generated alongside the standard stylized result.

**Experiment Tracking (Optional)**

Set `"enable_wandb": True` in `config.py` to enable W&B logging. Disable it (default) for a clean local run.

---

## Configuration

All hyperparameters are centralized in `src/config.py`:

```python
config = {
    "epochs": 1,
    "alpha": 1e2,           # Content loss weight
    "beta": 1e5,            # Style loss weight
    "gamma": 1e-4,          # TV loss weight
    "img_size": 512,
    "lr": 1.0,              # L-BFGS learning rate
    "use_relu_content": True,
    "use_relu_style": True,
    "use_deeper_style_layers": False,
    "initial_img": "content",    # Options: content | random | style
    "enable_wandb": False
}
```

**Best-found configurations per initialization strategy:**

| Init Image | alpha  | beta  | gamma | Notes                        |
|------------|--------|-------|-------|------------------------------|
| `content`  | `1e2`  | `1e5` | `1e-4` | Recommended default          |
| `random`   | `1e3`  | `1e5` | `1e-4` | Higher content weight needed |
| `style`    | `1e3`  | `1e4` | `1e-4` | Lower style weight sufficient|

**Execution time:** ~15–40 seconds for a 512×512 image on an NVIDIA RTX 3050 (6GB VRAM).

---

## Results

### Baseline

The baseline configuration uses equal style layer weights (1/5 per layer), standard hyperparameters, and content image initialization. This serves as the reference point for all subsequent ablations.

**Config:** `alpha=1e2`, `beta=1e5`, `gamma=1e-4`, `epochs=300`, `init=content`, `style layers weighted equally at 1/5`, `optimizer: L-BFGS`

|                     Content Image                     |                    Style Image                    |                   Generated Output                  |
| :---------------------------------------------------: | :-----------------------------------------------: | :-------------------------------------------------: |
| <img src="assets/content/content_01.jpg" width="300"> | <img src="assets/style/style_01.jpg" width="300"> | <img src="assets/output/output_01.png" width="300"> |
| <img src="assets/content/content_01.jpg" width="300"> | <img src="assets/style/style_02.jpg" width="300"> | <img src="assets/output/output_02.png" width="300"> |
| <img src="assets/content/content_01.jpg" width="300"> | <img src="assets/style/style_03.png" width="300"> | <img src="assets/output/cat_beta_1e6.png" width="300"> |
| <img src="assets/content/content_01.jpg" width="300"> | <img src="assets/style/style_04.jpg" width="300"> | <img src="assets/output/output_04.png" width="300"> |
| <img src="assets/content/content_01.jpg" width="300"> | <img src="assets/style/style_05.jpg" width="300"> | <img src="assets/output/output_05.png" width="300"> |
| <img src="assets/content/content_01.jpg" width="300"> | <img src="assets/style/style_08.jpg" height = "400" width="300"> | <img src="assets/output/output_08.png" width="300"> |
| <img src="assets/content/content_02.jpg" width="300"> | <img src="assets/style/style_03.png" height = "400" width="300"> | <img src="assets/output/output_07.png" width="300"> |



---


### Initialization Comparison

The choice of initialization image affects the convergence behavior and the character of the final output. Content initialization reliably converges to the highest quality results under the configurations tested.

|                 Content Initialization                |                 Random Initialization                |                 Style Initialization                |
| :---------------------------------------------------: | :--------------------------------------------------: | :-------------------------------------------------: |
| <img src="assets/output/cat_content.png" width="300"> | <img src="assets/output/cat_random.png" width="300"> | <img src="assets/output/cat_style.png" width="300"> |
|                 `α=1e2, β=1e5, γ=1e-4`                |                `α=1e3, β=1e5, γ=1e-4`                |                `α=1e3, β=1e4, γ=1e-4`               |


---

### Effect of Total Variation Weight (γ)

|                       γ = 1e-3                       |                       γ = 1e-2                       |                       γ = 1e-1                       |
| :--------------------------------------------------: | :--------------------------------------------------: | :--------------------------------------------------: |
| <img src="assets/output/gamma_1e-3.png" width="300"> | <img src="assets/output/gamma_1e-2.png" width="300"> | <img src="assets/output/gamma_1e-1.png" width="300"> |

|                       γ = 1e0                       |                       γ = 1e1                       |                       γ = 1e2                       |
| :-------------------------------------------------: | :-------------------------------------------------: | :-------------------------------------------------: |
| <img src="assets/output/gamma_1e0.png" width="300"> | <img src="assets/output/gamma_1e1.png" width="300"> | <img src="assets/output/gamma_1e2.png" width="300"> |

|                       γ = 1e3                       |                       γ = 1e4                       |                       γ = 1e5                       |
| :-------------------------------------------------: | :-------------------------------------------------: | :-------------------------------------------------: |
| <img src="assets/output/gamma_1e3.png" width="300"> | <img src="assets/output/gamma_1e4.png" width="300"> | <img src="assets/output/gamma_1e5.png" width="300"> |



**Observation:** Lower `gamma` values produce stronger, more expressive strokes. Higher values collapse the stylistic texture into a flat, homogeneous appearance. `gamma=1e-4` strikes the best balance between stroke expressiveness and visual coherence.

---

### Style Weight Ablation

Increasing `beta` strengthens the style signal relative to content. Below a certain threshold, stylization is visually weak; beyond a certain threshold, the content structure is overwhelmed.

> **Config:** `alpha=1e2`, `gamma=1e-4`, `init=content`, `beta` varied

|                       β = 1e1                      |                       β = 1e2                      |                       β = 1e3                      |
| :------------------------------------------------: | :------------------------------------------------: | :------------------------------------------------: |
| <img src="assets/output/beta_1e1.png" width="300"> | <img src="assets/output/beta_1e2.png" width="300"> | <img src="assets/output/beta_1e3.png" width="300"> |

|                       β = 1e4                      |                       β = 1e5                      |                       β = 1e6                      |
| :------------------------------------------------: | :------------------------------------------------: | :------------------------------------------------: |
| <img src="assets/output/beta_1e4.png" width="300"> | <img src="assets/output/beta_1e5.png" width="300"> | <img src="assets/output/beta_1e6.png" width="300"> |

|                       β = 1e7                      |                       β = 1e8                      |                       β = 1e9                      |
| :------------------------------------------------: | :------------------------------------------------: | :------------------------------------------------: |
| <img src="assets/output/beta_1e7.png" width="300"> | <img src="assets/output/beta_1e8.png" width="300"> | <img src="assets/output/beta_1e9.png" width="300"> |


---

### Color-Preserving Transfer

Standard NST transfers both texture and color from the style image. The color-preserving variant decomposes the generated image in the **CIE L\*a\*b\*** color space and replaces the chrominance (a\*, b\*) channels with those of the original content image, retaining only the luminance component from the stylized output.

|                     Standard NST                    |                      Color-Preserving NST                     |
| :-------------------------------------------------: | :-----------------------------------------------------------: |
| <img src="assets/output/output_07.png" width="300"> | <img src="assets/output/color_preserving_01.png" width="300"> |
| <img src="assets/output/homosapien.png" width="300"> | <img src="assets/output/color_preserving_02.png" width="300"> |
| <img src="assets/output/dead_poets_society_01.png" width="300"> | <img src="assets/output/color_preserving_03.png" width="300"> |
| <img src="assets/output/dead_poets_society_03.png" width="300"> | <img src="assets/output/color_preserving_04.png" width="300"> |


---

### Sequential Style Transfer

This experiment explores the effect of applying Neural Style Transfer in multiple stages. The content image is first stylized using one artwork, and the resulting output is then used as the content image for a second stylization pass with a different artwork.

Unlike direct style blending, where multiple style targets are optimized simultaneously, this approach progressively transforms the image through successive stylization steps, producing a unique combination of artistic characteristics from both styles.

> Pipeline: `Content → Style A → Intermediate Output → Style B → Final Output`

|                      Style 1                      |                      Style 2                      |                          After Style 1                          |                   After Style 2 (Final Output)                  |
| :-----------------------------------------------: | :-----------------------------------------------: | :-------------------------------------------------------------: | :-------------------------------------------------------------: |
| <img src="assets/style/style_03.png" width="300"> | <img src="assets/style/style_01.jpg" width="300"> | <img src="assets/output/cat_beta_1e6.png" width="300"> | <img src="assets/output/cat_multiple_style_01.png" width="300"> |
| <img src="assets/style/style_03.png" width="300"> | <img src="assets/style/style_06.png" width="250"> | <img src="assets/output/jon_snow_style_01.png" width="300"> | <img src="assets/output/jon_snow_multiple_style.png" height = 500 width="300"> |



---

### L-BFGS vs Adam

L-BFGS is the optimizer used in the original Gatys et al. paper and is the default in this implementation. Adam was also benchmarked for comparison. L-BFGS produces sharper, more faithful stylization in fewer effective iterations, while Adam requires more tuning to reach comparable quality.

| L-BFGS | Adam |
|:---:|:---:|
| <img src="assets/output/lbfgs_01.png" width="300"> | <img src="assets/output/adam_01.png" width="300"> |
| <img src="assets/output/lbfgs_02.jpg" width="300"> | <img src="assets/output/adam_02.jpg" width="300"> |

**Observation:** L-BFGS converges faster and produces visually superior results for this problem. Adam is more memory-efficient and can be preferable on hardware with limited VRAM.

---

## Experiment Log

A complete record of all configurations tested during development.

| Exp | alpha | beta | gamma | Style Layer Weights | Init | Epochs | Observation |
|-----|-------|------|-------|---------------------|------|--------|-------------|
| Baseline | 1 | 1e2 | 1e-4 | Equal (1/5) | content | 300 | Good quality reference output. Weak style texture. |
| Exp-01 | 1 | 1e2 | 1e-4 | [0.1, 0.15, 0.2, 0.25, 0.3] | content | 300 | Visually indistinguishable from baseline. |
| Exp-02 | 1 | 2e2 | 1e-4 | Equal (1/5) | content | 300 | No noticeable improvement over baseline. |
| Exp-03 | 1 | 1e3 | 1e-4 | Equal (1/5) | content | 300 | No noticeable improvement. |
| Exp-04 | 1 | 1e4 | 1e-4 | Equal (1/5) | content | 300 | No noticeable improvement. |
| **Best (content)** | **1e2** | **1e5** | **1e-4** | Equal (1/5) | **content** | — | **Best visual quality. Recommended default.** |
| **Best (random)** | **1e3** | **1e5** | **1e-4** | Equal (1/5) | **random** | — | **Best with random init.** |
| **Best (style)** | **1e3** | **1e4** | **1e-4** | Equal (1/5) | **style** | — | **Best with style init.** |

---

## References

```bibtex
@article{gatys2016image,
  title={Image Style Transfer Using Convolutional Neural Networks},
  author={Gatys, Leon A and Ecker, Alexander S and Bethge, Matthias},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2016}
}

@article{simonyan2014very,
  title={Very Deep Convolutional Networks for Large-Scale Image Recognition},
  author={Simonyan, Karen and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1409.1556},
  year={2014}
}
```

- Gatys et al., [*Image Style Transfer Using Convolutional Neural Networks*](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), CVPR 2016
- Simonyan & Zisserman, [*Very Deep Convolutional Networks for Large-Scale Image Recognition*](https://arxiv.org/abs/1409.1556), ICLR 2015
- PyTorch [Neural Style Transfer Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)