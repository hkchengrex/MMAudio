# Prediction interface for Cog ⚙️
# https://cog.run/python

import os

MODEL_CACHE = "cache"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE
WEIGHTS_BASE_URL = "https://weights.replicate.delivery/default/mmaudio"
MODEL_FILES = ["weights.tar", "ext_weights.tar", "cache.tar"]

import torch
import torchaudio
from pathlib import Path
from datetime import datetime
from cog import BasePredictor, Input, Path as CogPath
import time
import subprocess

# Import model utilities
try:
    import mmaudio
except ImportError:
    os.system("pip install -e .")
    import mmaudio

from mmaudio.eval_utils import (
    ModelConfig,
    all_model_cfg,
    generate,
    load_video,
    make_video,
)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)

    if ".tar" in dest:
        dest = os.path.dirname(dest)

    command = ["pget", "-vfx", url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self):
        # Add download logic at the start of setup
        for model_file in MODEL_FILES:
            url = WEIGHTS_BASE_URL + "/" + model_file
            dest_path = model_file

            dir_name = dest_path.replace(".tar", "")
            if os.path.exists(dir_name):
                print(f"[+] Directory {dir_name} already exists, skipping download")
                continue

            download_weights(url, dest_path)

        # Load the recommended large_44k_v2 model
        self.model_cfg: ModelConfig = all_model_cfg["large_44k_v2"]
        self.model_cfg.download_if_needed()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self.seq_cfg = self.model_cfg.seq_cfg

        # Load network
        self.net: MMAudio = (
            get_my_mmaudio(self.model_cfg.model_name).to(self.device, self.dtype).eval()
        )
        self.net.load_weights(
            torch.load(
                self.model_cfg.model_path, map_location=self.device, weights_only=True
            )
        )

        # Load feature utilities
        self.feature_utils = (
            FeaturesUtils(
                tod_vae_ckpt=self.model_cfg.vae_path,
                synchformer_ckpt=self.model_cfg.synchformer_ckpt,
                enable_conditions=True,
                mode=self.model_cfg.mode,
                bigvgan_vocoder_ckpt=self.model_cfg.bigvgan_16k_path,
            )
            .to(self.device, self.dtype)
            .eval()
        )

        self.output_dir = Path("./output")
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def predict(
        self,
        prompt: str = Input(description="Text prompt for generated audio", default=""),
        negative_prompt: str = Input(
            description="Negative prompt to avoid certain sounds", default="music"
        ),
        video: CogPath = Input(
            description="Optional video file for video-to-audio generation",
            default=None,
        ),
        duration: float = Input(
            description="Duration of output in seconds", default=8.0
        ),
        num_steps: int = Input(description="Number of inference steps", default=25),
        cfg_strength: float = Input(description="Guidance strength (CFG)", default=4.5),
        seed: int = Input(
            description="Random seed. Use -1 to randomize the seed", default=-1
        ),
    ) -> CogPath:
        """
        If `video` is provided, generates audio that syncs with the given video and returns an MP4.
        If `video` is not provided, generates audio from text and returns a FLAC file.
        """

        # Cast video to str if it's not None
        video = str(video) if video is not None else None

        if seed == -1:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        rng = torch.Generator(device=self.device).manual_seed(seed)
        fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

        # Prepare inputs
        if video:
            # Load video frames and sync frames
            clip_frames, sync_frames, duration = load_video(video, duration)
            # Detach tensors and add batch dimension
            clip_frames = (
                clip_frames.detach().unsqueeze(0) if clip_frames is not None else None
            )
            sync_frames = (
                sync_frames.detach().unsqueeze(0) if sync_frames is not None else None
            )
        else:
            clip_frames = sync_frames = None

        self.seq_cfg.duration = duration
        self.net.update_seq_lengths(
            self.seq_cfg.latent_seq_len,
            self.seq_cfg.clip_seq_len,
            self.seq_cfg.sync_seq_len,
        )

        # Generate audio with no_grad
        with torch.no_grad():
            audios = generate(
                clip_frames,
                sync_frames,
                [prompt],
                negative_text=[negative_prompt],
                feature_utils=self.feature_utils,
                net=self.net,
                fm=fm,
                rng=rng,
                cfg_strength=cfg_strength,
            )
        audio = audios.float().cpu()[0]

        current_time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        if video:
            # Combine video and audio into an MP4
            video_save_path = self.output_dir / f"{current_time_string}.mp4"
            make_video(
                video,
                video_save_path,
                audio,
                sampling_rate=self.seq_cfg.sampling_rate,
                duration_sec=self.seq_cfg.duration,
            )
            return CogPath(video_save_path)
        else:
            # Just save audio as FLAC
            audio_save_path = self.output_dir / f"{current_time_string}.flac"
            torchaudio.save(
                audio_save_path, audio.unsqueeze(0), self.seq_cfg.sampling_rate
            )
            return CogPath(audio_save_path)
