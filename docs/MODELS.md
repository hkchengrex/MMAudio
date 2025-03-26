# Pretrained models

The models will be downloaded automatically when you run the demo script. MD5 checksums are provided in `mmaudio/utils/download_utils.py`.
The models are also available at https://huggingface.co/hkchengrex/MMAudio/tree/main

| Model    | Download link | File size |
| -------- | ------- | ------- |
| Flow prediction network, small 16kHz | <a href="https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_small_16k.pth" download="mmaudio_small_16k.pth">mmaudio_small_16k.pth</a> | 601M |
| Flow prediction network, small 44.1kHz | <a href="https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_small_44k.pth" download="mmaudio_small_44k.pth">mmaudio_small_44k.pth</a> | 601M |
| Flow prediction network, medium 44.1kHz | <a href="https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_medium_44k.pth" download="mmaudio_medium_44k.pth">mmaudio_medium_44k.pth</a> | 2.4G |
| Flow prediction network, large 44.1kHz | <a href="https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_large_44k.pth" download="mmaudio_large_44k.pth">mmaudio_large_44k.pth</a> | 3.9G |
| Flow prediction network, large 44.1kHz, v2 **(recommended)** | <a href="https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_large_44k_v2.pth" download="mmaudio_large_44k_v2.pth">mmaudio_large_44k_v2.pth</a> | 3.9G |
| 16kHz VAE | <a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-16.pth">v1-16.pth</a> | 655M |
| 16kHz BigVGAN vocoder (from Make-An-Audio 2) |<a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/best_netG.pt">best_netG.pt</a> | 429M |
| 44.1kHz VAE |<a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-44.pth">v1-44.pth</a> | 1.2G | 
| Synchformer visual encoder |<a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth">synchformer_state_dict.pth</a> | 907M |

To run the model, you need four components: a flow prediction network, visual feature extractors (Synchformer and CLIP, CLIP will be downloaded automatically), a VAE, and a vocoder. VAEs and vocoders are specific to the sampling rate (16kHz or 44.1kHz) and not model sizes.
The 44.1kHz vocoder will be downloaded automatically.
The `_v2` model performs worse in benchmarking (e.g., in  Fréchet distance), but, in my experience, generalizes better to new data.

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
