<div align="center">
<p align="center">
  <h2>MMAudio</h2>
  <a href="">Paper (Soon)</a> | <a href="https://hkchengrex.github.io/MMAudio">Webpage</a> | <a href="https://huggingface.co/hkchengrex/MMAudio/tree/main">Models</a> | <a href="https://huggingface.co/spaces/hkchengrex/MMAudio"> Huggingface Demo</a> | <a href="https://colab.research.google.com/drive/1TAaXCY2-kPk4xE4PwKB3EqFbSnkUuzZ8?usp=sharing">Colab Demo</a> | <a href="https://replicate.com/zsxkib/mmaudio">Replicate Demo</a>
</p>
</div>

## [Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis](https://hkchengrex.github.io/MMAudio)

[Ho Kei Cheng](https://hkchengrex.github.io/), [Masato Ishii](https://scholar.google.co.jp/citations?user=RRIO1CcAAAAJ), [Akio Hayakawa](https://scholar.google.com/citations?user=sXAjHFIAAAAJ), [Takashi Shibuya](https://scholar.google.com/citations?user=XCRO260AAAAJ), [Alexander Schwing](https://www.alexander-schwing.de/), [Yuki Mitsufuji](https://www.yukimitsufuji.com/)

University of Illinois Urbana-Champaign, Sony AI, and Sony Group Corporation

**Note: This repository is still under construction. Single-example inference should work as expected. The training code will be added. Code is subject to non-backward-compatible changes.**

## Highlight

MMAudio generates synchronized audio given video and/or text inputs.
Our key innovation is multimodal joint training which allows training on a wide range of audio-visual and audio-text datasets.
Moreover, a synchronization module aligns the generated audio with the video frames.

## Results

(All audio from our algorithm MMAudio)

Videos from Sora:

https://github.com/user-attachments/assets/82afd192-0cee-48a1-86ca-bd39b8c8f330

Videos from MovieGen/Hunyuan Video/VGGSound:

https://github.com/user-attachments/assets/29230d4e-21c1-4cf8-a221-c28f2af6d0ca

For more results, visit https://hkchengrex.com/MMAudio/video_main.html.

## Installation

We have only tested this on Ubuntu.

### Prerequisites

We recommend using a [miniforge](https://github.com/conda-forge/miniforge) environment.

- Python 3.9+
- PyTorch **2.5.1+** and corresponding torchvision/torchaudio (pick your CUDA version https://pytorch.org/, pip install recommended)
- ffmpeg<7 ([this is required by torchaudio](https://pytorch.org/audio/master/installation.html#optional-dependencies), you can install it in a miniforge environment with `conda install -c conda-forge 'ffmpeg<7'`)

**1. Install prerequisite if not yet met:**
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
```
(Or any other CUDA versions that your GPUs/driver support)

```
conda install -c conda-forge 'ffmpeg<7
```
(Optional, if you use miniforge and don't already have the appropriate ffmpeg)

**2. Clone our repository:**

```bash
git clone https://github.com/hkchengrex/MMAudio.git
```

**3. Install with pip (install pytorch first before attempting this!):**

```bash
cd MMAudio
pip install -e .
```

(If you encounter the File "setup.py" not found error, upgrade your pip with pip install --upgrade pip)


**Pretrained models:**

The models will be downloaded automatically when you run the demo script. MD5 checksums are provided in `mmaudio/utils/download_utils.py`
The models are also available at https://huggingface.co/hkchengrex/MMAudio/tree/main

| Model    | Download link | File size |
| -------- | ------- | ------- |
| Flow prediction network, small 16kHz | <a href="https://databank.illinois.edu/datafiles/k6jve/download" download="mmaudio_small_16k.pth">mmaudio_small_16k.pth</a> | 601M |
| Flow prediction network, small 44.1kHz | <a href="https://databank.illinois.edu/datafiles/864ya/download" download="mmaudio_small_44k.pth">mmaudio_small_44k.pth</a> | 601M |
| Flow prediction network, medium 44.1kHz | <a href="https://databank.illinois.edu/datafiles/pa94t/download" download="mmaudio_medium_44k.pth">mmaudio_medium_44k.pth</a> | 2.4G |
| Flow prediction network, large 44.1kHz | <a href="https://databank.illinois.edu/datafiles/4jx76/download" download="mmaudio_large_44k.pth">mmaudio_large_44k.pth</a> | 3.9G |
| Flow prediction network, large 44.1kHz, v2 **(recommended)** | <a href="https://databank.illinois.edu/datafiles/i1pd9/download" download="mmaudio_large_44k_v2.pth">mmaudio_large_44k_v2.pth</a> | 3.9G |
| 16kHz VAE | <a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-16.pth">v1-16.pth</a> | 655M |
| 16kHz BigVGAN vocoder |<a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/best_netG.pt">best_netG.pt</a> | 429M |
| 44.1kHz VAE |<a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-44.pth">v1-44.pth</a> | 1.2G | 
| Synchformer visual encoder |<a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth">synchformer_state_dict.pth</a> | 907M |

The 44.1kHz vocoder will be downloaded automatically.

The expected directory structure (full):

```bash
MMAudio
├── ext_weights
│   ├── best_netG.pt
│   ├── synchformer_state_dict.pth
│   ├── v1-16.pth
│   └── v1-44.pth
├── weights
│   ├── mmaudio_small_16k.pth
│   ├── mmaudio_small_44k.pth
│   ├── mmaudio_medium_44k.pth
│   ├── mmaudio_large_44k.pth
│   └── mmaudio_large_44k_v2.pth
└── ...
```

The expected directory structure (minimal, for the recommended model only):

```bash
MMAudio
├── ext_weights
│   ├── synchformer_state_dict.pth
│   └── v1-44.pth
├── weights
│   └── mmaudio_large_44k_v2.pth
└── ...
```

## Demo

By default, these scripts use the `large_44k` model. 
In our experiments, inference only takes around 6GB of GPU memory (in 16-bit mode) which should fit in most modern GPUs.

### Command-line interface

With `demo.py`
```bash
python demo.py --duration=8 --video=<path to video> --prompt "your prompt" 
```
The output (audio in `.flac` format, and video in `.mp4` format) will be saved in `./output`.
See the file for more options.
Simply omit the `--video` option for text-to-audio synthesis.
The default output (and training) duration is 8 seconds. Longer/shorter durations could also work, but a large deviation from the training duration may result in a lower quality.


### Gradio interface

Supports video-to-audio and text-to-audio synthesis. Use [port forwarding](https://unix.stackexchange.com/questions/115897/whats-ssh-port-forwarding-and-whats-the-difference-between-ssh-local-and-remot) if necessary. Our default port is `7860` which you can change in `gradio_demo.py`.

```
python gradio_demo.py
```

### Known limitations

1. The model sometimes generates undesired unintelligible human speech-like sounds
2. The model sometimes generates undesired background music
3. The model struggles with unfamiliar concepts, e.g., it can generate "gunfires" but not "RPG firing".

We believe all of these three limitations can be addressed with more high-quality training data.

## Training
Work in progress.

## Evaluation
Work in progress.

## Acknowledgement
Many thanks to:
- [Make-An-Audio 2](https://github.com/bytedance/Make-An-Audio-2) for the 16kHz BigVGAN pretrained model and the VAE architecture
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [Synchformer](https://github.com/v-iashin/Synchformer) 
- [EDM2](https://github.com/NVlabs/edm2) for the magnitude-preserving network architecture
