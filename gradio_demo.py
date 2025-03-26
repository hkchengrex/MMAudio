import gc
import logging
from argparse import ArgumentParser
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import gradio as gr
import torch
import torchaudio

from mmaudio.eval_utils import (ModelConfig, VideoInfo, all_model_cfg, generate, load_image,
                                load_video, make_video, setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.sequence_config import SequenceConfig
from mmaudio.model.utils.features_utils import FeaturesUtils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    log.warning('CUDA/MPS are not available, running on CPU')
dtype = torch.bfloat16

model: ModelConfig = all_model_cfg['large_44k_v2']
model.download_if_needed()
output_dir = Path('./output/gradio')

setup_eval_logging()


def get_model() -> tuple[MMAudio, FeaturesUtils, SequenceConfig]:
    seq_cfg = model.seq_cfg

    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model.model_path}')

    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()

    return net, feature_utils, seq_cfg


net, feature_utils, seq_cfg = get_model()


@torch.inference_mode()
def video_to_audio(video: gr.Video, prompt: str, negative_prompt: str, seed: int, num_steps: int,
                   cfg_strength: float, duration: float):

    rng = torch.Generator(device=device)
    if seed >= 0:
        rng.manual_seed(seed)
    else:
        rng.seed()
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    video_info = load_video(video, duration)
    clip_frames = video_info.clip_frames
    sync_frames = video_info.sync_frames
    duration = video_info.duration_sec
    clip_frames = clip_frames.unsqueeze(0)
    sync_frames = sync_frames.unsqueeze(0)
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(clip_frames,
                      sync_frames, [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength)
    audio = audios.float().cpu()[0]

    current_time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(exist_ok=True, parents=True)
    video_save_path = output_dir / f'{current_time_string}.mp4'
    make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
    gc.collect()
    return video_save_path


@torch.inference_mode()
def image_to_audio(image: gr.Image, prompt: str, negative_prompt: str, seed: int, num_steps: int,
                   cfg_strength: float, duration: float):

    rng = torch.Generator(device=device)
    if seed >= 0:
        rng.manual_seed(seed)
    else:
        rng.seed()
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    image_info = load_image(image)
    clip_frames = image_info.clip_frames
    sync_frames = image_info.sync_frames
    clip_frames = clip_frames.unsqueeze(0)
    sync_frames = sync_frames.unsqueeze(0)
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(clip_frames,
                      sync_frames, [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength,
                      image_input=True)
    audio = audios.float().cpu()[0]

    current_time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(exist_ok=True, parents=True)
    video_save_path = output_dir / f'{current_time_string}.mp4'
    video_info = VideoInfo.from_image_info(image_info, duration, fps=Fraction(1))
    make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
    gc.collect()
    return video_save_path


@torch.inference_mode()
def text_to_audio(prompt: str, negative_prompt: str, seed: int, num_steps: int, cfg_strength: float,
                  duration: float):

    rng = torch.Generator(device=device)
    if seed >= 0:
        rng.manual_seed(seed)
    else:
        rng.seed()
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    clip_frames = sync_frames = None
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(clip_frames,
                      sync_frames, [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength)
    audio = audios.float().cpu()[0]

    current_time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(exist_ok=True, parents=True)
    audio_save_path = output_dir / f'{current_time_string}.flac'
    torchaudio.save(audio_save_path, audio, seq_cfg.sampling_rate)
    gc.collect()
    return audio_save_path


video_to_audio_tab = gr.Interface(
    fn=video_to_audio,
    description="""
    Project page: <a href="https://hkchengrex.com/MMAudio/">https://hkchengrex.com/MMAudio/</a><br>
    Code: <a href="https://github.com/hkchengrex/MMAudio">https://github.com/hkchengrex/MMAudio</a><br>

    NOTE: It takes longer to process high-resolution videos (>384 px on the shorter side). 
    Doing so does not improve results.
    """,
    inputs=[
        gr.Video(),
        gr.Text(label='Prompt'),
        gr.Text(label='Negative prompt', value='music'),
        gr.Number(label='Seed (-1: random)', value=-1, precision=0, minimum=-1),
        gr.Number(label='Num steps', value=25, precision=0, minimum=1),
        gr.Number(label='Guidance Strength', value=4.5, minimum=1),
        gr.Number(label='Duration (sec)', value=8, minimum=1),
    ],
    outputs='playable_video',
    cache_examples=False,
    title='MMAudio — Video-to-Audio Synthesis',
    examples=[
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_beach.mp4',
            'waves, seagulls',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_serpent.mp4',
            '',
            'music',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_seahorse.mp4',
            'bubbles',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_india.mp4',
            'Indian holy music',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_galloping.mp4',
            'galloping',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_kraken.mp4',
            'waves, storm',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/mochi_storm.mp4',
            'storm',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/hunyuan_spring.mp4',
            '',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/hunyuan_typing.mp4',
            'typing',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/hunyuan_wake_up.mp4',
            '',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_nyc.mp4',
            '',
            '',
            0,
            25,
            4.5,
            10,
        ],
    ])

text_to_audio_tab = gr.Interface(
    fn=text_to_audio,
    description="""
    Project page: <a href="https://hkchengrex.com/MMAudio/">https://hkchengrex.com/MMAudio/</a><br>
    Code: <a href="https://github.com/hkchengrex/MMAudio">https://github.com/hkchengrex/MMAudio</a><br>
    """,
    inputs=[
        gr.Text(label='Prompt'),
        gr.Text(label='Negative prompt'),
        gr.Number(label='Seed (-1: random)', value=-1, precision=0, minimum=-1),
        gr.Number(label='Num steps', value=25, precision=0, minimum=1),
        gr.Number(label='Guidance Strength', value=4.5, minimum=1),
        gr.Number(label='Duration (sec)', value=8, minimum=1),
    ],
    outputs='audio',
    cache_examples=False,
    title='MMAudio — Text-to-Audio Synthesis',
)

image_to_audio_tab = gr.Interface(
    fn=image_to_audio,
    description="""
    Project page: <a href="https://hkchengrex.com/MMAudio/">https://hkchengrex.com/MMAudio/</a><br>
    Code: <a href="https://github.com/hkchengrex/MMAudio">https://github.com/hkchengrex/MMAudio</a><br>

    NOTE: It takes longer to process high-resolution images (>384 px on the shorter side). 
    Doing so does not improve results.
    """,
    inputs=[
        gr.Image(type='filepath'),
        gr.Text(label='Prompt'),
        gr.Text(label='Negative prompt'),
        gr.Number(label='Seed (-1: random)', value=-1, precision=0, minimum=-1),
        gr.Number(label='Num steps', value=25, precision=0, minimum=1),
        gr.Number(label='Guidance Strength', value=4.5, minimum=1),
        gr.Number(label='Duration (sec)', value=8, minimum=1),
    ],
    outputs='playable_video',
    cache_examples=False,
    title='MMAudio — Image-to-Audio Synthesis (experimental)',
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()

    gr.TabbedInterface([video_to_audio_tab, text_to_audio_tab, image_to_audio_tab],
                       ['Video-to-Audio', 'Text-to-Audio', 'Image-to-Audio (experimental)']).launch(
                           server_port=args.port, allowed_paths=[output_dir])
